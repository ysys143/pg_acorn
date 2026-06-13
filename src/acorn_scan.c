/*
 * acorn_scan.c — ACORN-1 predicate subgraph traversal
 *
 * Implements Algorithm 3 from "ACORN: Performant Filtered Vector Search
 * via Segment-Graph Navigation" (Patel et al., arxiv 2403.04871).
 *
 * Key invariant: filter-failing neighbors are excluded from W (result set)
 * but kept in C (traversal candidates), preserving graph connectivity and
 * guaranteeing top-k at any filter selectivity — unlike prefilter or naive
 * inline filtering approaches that fail below the percolation threshold.
 *
 * This file is shared by Tier 1 (CustomScan executor) and Tier 2 (AM scan).
 */

#include "postgres.h"

#include <float.h>
#include <math.h>

#include "executor/executor.h"
#include "executor/tuptable.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/fmgroids.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "access/tableam.h"
#include "catalog/pg_type.h"

#include "pg_acorn.h"
#include "acorn_scan.h"
#include "hnsw_compat.h"
#include "acorn_t2_page.h"
#include "acorn_dist.h"
#include "acorn_codecache.h"

/* -----------------------------------------------------------------------
 * Code-cache read pipelining
 *
 * The shared-memory code table (~40 MB at 250K/dim=128) overflows CPU cache,
 * so each per-neighbor entry read is a cold DRAM miss — the measured penalty
 * vs the inline layout, whose codes are co-located in an already-pinned page.
 * We hide that latency by software-prefetching neighbor (i+K)'s resolved
 * entry while neighbor i's already-prefetched entry is read+scored.  Prefetch
 * is a pure hint: a prefetched address that turns out invalid/evicted just
 * wastes the hint; acorn_codecache_read still performs every G4 safety check.
 *
 * K is the pipeline depth: how many neighbors ahead to prefetch.  At dim=128
 * the entry spans ~3 cache lines after the 40-byte header; K in [4,8] covers
 * a DRAM miss (~80-100 ns) behind one SQ8 distance compute (~few hundred ns).
 */
#define ACORN_CC_PREFETCH_AHEAD	6

#if defined(__GNUC__) || defined(__clang__)
#define ACORN_PREFETCH_R(addr)	__builtin_prefetch((addr), 0, 3)
#else
#define ACORN_PREFETCH_R(addr)	((void) (addr))
#endif

/*
 * Prefetch a resolved code-cache entry into L1 for an imminent read: the
 * entry header (version/heaptid/.../scale/offset, 40 B) plus the SQ8 code
 * (dim bytes spanning ~3 lines at dim=128).  NULL-safe; touches at most
 * 1 (header) + 4 (up to 256 code bytes) cache lines.  A read hint only.
 */
static inline void
acorn_cc_prefetch_entry(const AcornCodeCacheEntry *e, int dim)
{
	const char *p;
	int			off;

	if (e == NULL)
		return;
	ACORN_PREFETCH_R(e);					/* header + first code bytes */
	p = (const char *) e->code;
	for (off = 64; off < dim; off += 64)	/* remaining code cache lines */
		ACORN_PREFETCH_R(p + off);
}

/* -----------------------------------------------------------------------
 * Internal element reference
 * ----------------------------------------------------------------------- */

typedef struct AcornElem
{
	ItemPointerData indextid;	/* index page (blkno, offno) */
	ItemPointerData heaptid;	/* heap TID — for predicate evaluation */
	double			distance;

	/*
	 * Lower bound on the EXACT distance.  Equal to `distance` everywhere
	 * except inline-vector (SQ8) discoveries, where it subtracts the
	 * worst-case quantization error.  The T2 result heap R orders on this,
	 * so the exact re-rank provably pulls every candidate that could still
	 * beat the current exact head before anything is emitted.
	 */
	double			lb;

	/*
	 * T2 single-read fast path: neighbor-tuple location + level captured at
	 * discovery (acorn_t2_load_node), so expansion does not re-read the
	 * element page.  Valid only when has_nbr; tier 1 paths leave it false.
	 */
	ItemPointerData nbrtid;		/* index TID of the node's neighbor tuple */
	uint8			level;		/* node's highest layer */
	bool			has_nbr;	/* nbrtid/level above are valid */
} AcornElem;

/* -----------------------------------------------------------------------
 * Priority queue nodes for pairingheap
 * ----------------------------------------------------------------------- */

typedef struct AcornPQNode
{
	pairingheap_node ph_node;
	AcornElem		elem;
} AcornPQNode;

/* max-heap: furthest element at top (used for W, the result set) */
static int
pq_cmp_max(const pairingheap_node *a, const pairingheap_node *b,
		   void *arg)
{
	double da = ((const AcornPQNode *) a)->elem.distance;
	double db = ((const AcornPQNode *) b)->elem.distance;
	return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* min-heap: closest element at top (used for C, the candidate set) */
static int
pq_cmp_min(const pairingheap_node *a, const pairingheap_node *b,
		   void *arg)
{
	return -pq_cmp_max(a, b, arg);
}

/* min-heap on the exact-distance LOWER BOUND (T2 result heap R only) */
static int
pq_cmp_lb_min(const pairingheap_node *a, const pairingheap_node *b,
			  void *arg)
{
	double la = ((const AcornPQNode *) a)->elem.lb;
	double lbv = ((const AcornPQNode *) b)->elem.lb;
	return (la < lbv) ? 1 : (la > lbv) ? -1 : 0;
}

/* -----------------------------------------------------------------------
 * Visited set — hash table keyed on index TID
 * ----------------------------------------------------------------------- */

typedef struct VisitedEntry
{
	ItemPointerData indextid;
	char			status;
} VisitedEntry;

static HTAB *
create_visited_set(void)
{
	HASHCTL info;
	memset(&info, 0, sizeof(info));
	info.keysize = sizeof(ItemPointerData);
	info.entrysize = sizeof(VisitedEntry);
	return hash_create("acorn_visited", 512, &info,
					   HASH_ELEM | HASH_BLOBS);
}

static bool
is_visited(HTAB *visited, const ItemPointerData *tid)
{
	bool found;
	hash_search(visited, tid, HASH_FIND, &found);
	return found;
}

static void
mark_visited(HTAB *visited, const ItemPointerData *tid)
{
	bool found;
	hash_search(visited, (void *) tid, HASH_ENTER, &found);
}

/*
 * Single-probe visit: enter the TID, return true iff it was NOT already
 * present (first visit).  Identical semantics to is_visited+mark_visited
 * with one hash probe instead of two.
 */
static inline bool
visit_once(HTAB *visited, const ItemPointerData *tid)
{
	bool found;
	hash_search(visited, (void *) tid, HASH_ENTER, &found);
	return !found;
}

/* -----------------------------------------------------------------------
 * pgvector 0.8.x page reading helpers
 * ----------------------------------------------------------------------- */

/*
 * Read meta page, return true if index has an entry point.
 *
 * dims_out / acorn_flags_out (nullable) read the Tier 2 meta extension; on
 * pgvector-written meta pages (Tier 1) and pre-extension Tier 2 indexes the
 * flag bytes are zero (PageInit zeroes the page), i.e. inline vectors off.
 */
static bool
acorn_meta_read(Relation index, BlockNumber *entry_blkno,
				OffsetNumber *entry_offno, int *entry_level, int *m_out,
				int *dims_out, uint16 *acorn_flags_out)
{
	Buffer		buf;
	Page		page;
	AcornT2MetaPage meta;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	meta = AcornT2PageGetMeta(page);

	*entry_blkno = meta->hnsw.entryBlkno;
	*entry_offno = meta->hnsw.entryOffno;
	*entry_level = (int) meta->hnsw.entryLevel;
	*m_out = (int) meta->hnsw.m;
	if (dims_out)
		*dims_out = (int) meta->hnsw.dimensions;
	if (acorn_flags_out)
		*acorn_flags_out = meta->acorn_flags;

	UnlockReleaseBuffer(buf);
	return (*entry_blkno != InvalidBlockNumber);
}

/*
 * Load element tuple header from index page.
 *
 * Returns the primary heaptid, the node's level, deleted flag, and the index
 * TID (page/offset) of the node's neighbor tuple — which in pgvector lives on
 * a separate index tuple, not adjacent to the element.
 *
 * Pins and returns the buffer — caller must UnlockReleaseBuffer (it can read
 * the inline vector first).
 */
static Buffer
acorn_load_element(Relation index, BlockNumber blkno, OffsetNumber offno,
				   ItemPointerData *heaptid_out, int *level_out,
				   bool *deleted_out,
				   BlockNumber *nbr_blkno_out, OffsetNumber *nbr_offno_out)
{
	Buffer				buf;
	Page				page;
	ItemId				iid;
	HnswElementTuple	etup;

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	iid = PageGetItemId(page, offno);
	etup = (HnswElementTuple) PageGetItem(page, iid);

	Assert(etup->type == HNSW_ELEMENT_TUPLE_TYPE);
	*heaptid_out   = etup->heaptids[0];		/* primary TID (first in HOT chain) */
	*level_out     = (int) etup->level;
	*deleted_out   = (etup->deleted != 0);
	*nbr_blkno_out = ItemPointerGetBlockNumber(&etup->neighbortid);
	*nbr_offno_out = ItemPointerGetOffsetNumber(&etup->neighbortid);

	/* Buffer remains locked — caller reads vector then calls UnlockReleaseBuffer */
	return buf;
}

/*
 * Get neighbor index TIDs for an element at layer `layer`.
 *
 * The neighbor tuple is read from (nbr_blkno, nbr_offno) — the separate index
 * tuple referenced by the element's neighbortid.  `level` is the element's own
 * highest layer, needed to compute the per-layer start offset.
 */
static int
acorn_get_neighbors(Relation index, BlockNumber nbr_blkno, OffsetNumber nbr_offno,
					int level, int layer, int m,
					ItemPointerData *neighbors_out, int max_neighbors)
{
	Buffer				buf;
	Page				page;
	ItemId				iid;
	HnswNeighborTuple	ntup;
	ItemPointerData	   *tids;
	int					start;
	int					count;

	buf = ReadBuffer(index, nbr_blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	iid = PageGetItemId(page, nbr_offno);
	ntup = (HnswNeighborTuple) PageGetItem(page, iid);

	Assert(ntup->type == HNSW_NEIGHBOR_TUPLE_TYPE);

	tids  = HnswNeighborTupleGetTids(ntup);
	start = HnswNeighborStart(m, level, layer);
	count = Min(HnswNeighborCount(m, layer), max_neighbors);

	/*
	 * Skip (not stop at) invalid slots: with acorn_payload_edges the layer-0
	 * slot array is split into a global half and a payload half, so a
	 * partially-filled global half may precede valid payload slots.
	 */
	{
		int n = 0;

		for (int i = 0; i < count; i++)
		{
			if (!ItemPointerIsValid(&tids[start + i]))
				continue;
			neighbors_out[n++] = tids[start + i];
		}
		count = n;
	}

	UnlockReleaseBuffer(buf);
	return count;
}

/*
 * Fetch vector from index page and compute distance to query using the
 * index's built-in distance function (support function 1 of the opclass).
 *
 * Note: `query` is a Datum holding the query vector as passed to acorn_scan.
 */
static double
acorn_distance(Relation index, BlockNumber blkno, OffsetNumber offno,
			   Datum query, FmgrInfo *dist_proc)
{
	Buffer				buf;
	Page				page;
	ItemId				iid;
	HnswElementTuple	etup;
	void			   *vec;
	Datum				result;

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	iid  = PageGetItemId(page, offno);
	etup = (HnswElementTuple) PageGetItem(page, iid);
	vec  = HnswElementTupleGetVector(etup);

	/* Call opclass distance function: distance(vec, query) */
	result = FunctionCall2(dist_proc, PointerGetDatum(vec), query);

	UnlockReleaseBuffer(buf);
	return DatumGetFloat8(result);
}

/*
 * Load the distance function from the index's opclass.
 * Support function 1 for HNSW is the distance function (e.g. l2, ip, cosine).
 */
static void
acorn_load_dist_proc(Relation index, FmgrInfo *finfo)
{
	/*
	 * rd_support is a RegProcedure[] indexed by
	 * (attno-1)*nproc + (procno-1).  For a single-column HNSW index,
	 * support proc 1 (the distance function) is at index 0.
	 */
	RegProcedure proc = index->rd_support[0];
	fmgr_info((Oid) proc, finfo);
}

/*
 * Resolve a direct C kernel for the opclass distance function (fmgr bypass).
 *
 * Matched by function name + signature (2 args, float8 return).  The kernels
 * in acorn_dist.c are compiled with pgvector's flags and replicate its loops,
 * so results are numerically identical to the fmgr path.  Returns NULL for
 * unknown opclasses — callers must keep the fmgr fallback.
 */
AcornDistFn
acorn_resolve_direct_dist(Relation index)
{
	Oid			procOid = (Oid) index->rd_support[0];
	char	   *name;
	AcornDistFn fn = NULL;

	if (get_func_rettype(procOid) != FLOAT8OID)
		return NULL;

	name = get_func_name(procOid);
	if (name == NULL)
		return NULL;

	if (strcmp(name, "vector_l2_squared_distance") == 0)
		fn = acorn_dist_l2sq;
	else if (strcmp(name, "vector_negative_inner_product") == 0)
		fn = acorn_dist_neg_ip;

	pfree(name);
	return fn;
}

/*
 * Single page read for tier1 (pgvector-format) scan nodes: computes the
 * distance from the inline vector and extracts heaptid / level / deleted /
 * neighbor-tuple TID under one buffer pin.  Replaces the previous
 * acorn_distance + acorn_load_element pair, which pinned and locked the
 * SAME element page twice per neighbor.
 */
static double
acorn_load_node(Relation index, BlockNumber blkno, OffsetNumber offno,
				Datum query, FmgrInfo *dist_proc, AcornDistFn dist_direct,
				ItemPointerData *heaptid_out, int *level_out,
				bool *deleted_out,
				BlockNumber *nbr_blkno_out, OffsetNumber *nbr_offno_out)
{
	Buffer				buf;
	Page				page;
	HnswElementTuple	etup;
	void			   *vec;
	double				dist;

	buf  = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	vec  = HnswElementTupleGetVector(etup);

	if (dist_direct)
	{
		AcornPgVector *v = (AcornPgVector *) vec;
		AcornPgVector *q = (AcornPgVector *) DatumGetPointer(query);

		if (likely(v->dim == q->dim))
			dist = dist_direct((int) v->dim, v->x, q->x);
		else
			dist = DatumGetFloat8(
				FunctionCall2(dist_proc, PointerGetDatum(vec), query));
	}
	else
		dist = DatumGetFloat8(
			FunctionCall2(dist_proc, PointerGetDatum(vec), query));

	*heaptid_out   = etup->heaptids[0];
	*level_out     = (int) etup->level;
	*deleted_out   = (etup->deleted != 0);
	*nbr_blkno_out = ItemPointerGetBlockNumber(&etup->neighbortid);
	*nbr_offno_out = ItemPointerGetOffsetNumber(&etup->neighbortid);

	UnlockReleaseBuffer(buf);
	return dist;
}

/*
 * Evaluate predicate on the heap tuple identified by heaptid.
 * Returns true if the row passes the filter (or if predicate is NULL).
 */
static bool
acorn_eval_predicate(Relation heap, ItemPointer heaptid,
					 Snapshot snapshot,
					 ExprState *predicate, ExprContext *econtext)
{
	TupleTableSlot *slot;
	bool			matches;

	if (predicate == NULL)
		return true;

	slot = MakeSingleTupleTableSlot(RelationGetDescr(heap),
									table_slot_callbacks(heap));

	if (!table_tuple_fetch_row_version(heap, heaptid, snapshot, slot))
	{
		ExecDropSingleTupleTableSlot(slot);
		return false;
	}

	econtext->ecxt_scantuple = slot;
	matches = ExecQual(predicate, econtext);

	ExecDropSingleTupleTableSlot(slot);
	return matches;
}

/* -----------------------------------------------------------------------
 * Greedy descent: traverse layers `start_level` down to `target_level`.
 * Returns the closest element at target_level as (blkno, offno).
 * ----------------------------------------------------------------------- */

static void
acorn_greedy_descent(Relation index, int m,
					 BlockNumber *cur_blkno, OffsetNumber *cur_offno,
					 int start_level, int target_level,
					 Datum query, FmgrInfo *dist_proc)
{
	double cur_dist = acorn_distance(index, *cur_blkno, *cur_offno,
									 query, dist_proc);

	for (int lc = start_level; lc > target_level; lc--)
	{
		bool improved;

		do {
			ItemPointerData neighbors[HNSW_MAX_NEIGHBORS];
			ItemPointerData cur_heaptid;
			int				cur_level;
			bool			cur_deleted;
			BlockNumber		nbr_blkno;
			OffsetNumber	nbr_offno;
			Buffer			ebuf;
			int				n_count;

			improved = false;

			/* Load current element for its level + neighbor tuple location */
			ebuf = acorn_load_element(index, *cur_blkno, *cur_offno,
									  &cur_heaptid, &cur_level, &cur_deleted,
									  &nbr_blkno, &nbr_offno);
			UnlockReleaseBuffer(ebuf);

			n_count = acorn_get_neighbors(index, nbr_blkno, nbr_offno,
										  cur_level, lc, m, neighbors,
										  HNSW_MAX_NEIGHBORS);

			for (int i = 0; i < n_count; i++)
			{
				BlockNumber  nblkno  = ItemPointerGetBlockNumber(&neighbors[i]);
				OffsetNumber noffno  = ItemPointerGetOffsetNumber(&neighbors[i]);
				double       nd;

				if (!ItemPointerIsValid(&neighbors[i]))
					continue;

				nd = acorn_distance(index, nblkno, noffno, query, dist_proc);
				if (nd < cur_dist)
				{
					cur_dist  = nd;
					*cur_blkno = nblkno;
					*cur_offno = noffno;
					improved   = true;
				}
			}
		} while (improved);
	}
}

/* -----------------------------------------------------------------------
 * ACORN-1 layer-0 traversal (the core algorithm)
 * ----------------------------------------------------------------------- */

static int
acorn_layer0_search(Relation index, Relation heap, int m,
					BlockNumber entry_blkno, OffsetNumber entry_offno,
					Datum query, FmgrInfo *dist_proc, AcornDistFn dist_direct,
					Snapshot snapshot,
					ExprState *predicate, ExprContext *econtext,
					int k, int ef_search,
					ItemPointerData *result_tids_out)
{
	pairingheap    *C;		/* min-heap: candidates (closest at top) */
	pairingheap    *W;		/* max-heap: results    (furthest at top) */
	HTAB		   *visited;
	MemoryContext	tmp_ctx;
	MemoryContext	old_ctx;
	int				n_results = 0;

	tmp_ctx = AllocSetContextCreate(CurrentMemoryContext,
									"AcornScan", ALLOCSET_DEFAULT_SIZES);
	old_ctx = MemoryContextSwitchTo(tmp_ctx);

	C = pairingheap_allocate(pq_cmp_min, NULL);
	W = pairingheap_allocate(pq_cmp_max, NULL);
	visited = create_visited_set();

	/* Seed with entry point */
	{
		AcornPQNode *node;
		ItemPointerData entry_tid;
		ItemPointerSet(&entry_tid, entry_blkno, entry_offno);

		mark_visited(visited, &entry_tid);

		node = palloc(sizeof(AcornPQNode));
		ItemPointerCopy(&entry_tid, &node->elem.indextid);
		node->elem.distance = acorn_distance(index, entry_blkno,
											 entry_offno, query, dist_proc);
		/* heaptid will be loaded when we process this candidate */
		ItemPointerSetInvalid(&node->elem.heaptid);
		pairingheap_add(C, &node->ph_node);
	}

	/* Main ACORN-1 traversal loop */
	while (!pairingheap_is_empty(C))
	{
		AcornPQNode    *c_node;
		AcornElem		c;
		double			f_dist;		/* distance of furthest result in W */
		ItemPointerData neighbors[HNSW_MAX_NEIGHBORS];
		int				n_count;
		ItemPointerData c_heaptid;
		int				c_level;
		bool			c_deleted;
		BlockNumber		c_nbr_blkno;
		OffsetNumber	c_nbr_offno;
		Buffer			c_buf;

		/* Pop closest candidate */
		c_node = (AcornPQNode *) pairingheap_remove_first(C);
		c = c_node->elem;
		pfree(c_node);

		/* Termination: if candidate is farther than worst result and W is full */
		if (pairingheap_is_empty(W))
			f_dist = DBL_MAX;
		else
			f_dist = ((AcornPQNode *) pairingheap_first(W))->elem.distance;

		if (c.distance > f_dist && n_results >= k)
			break;

		/* Load candidate element: heaptid, level, and neighbor tuple location */
		c_buf = acorn_load_element(index,
								   ItemPointerGetBlockNumber(&c.indextid),
								   ItemPointerGetOffsetNumber(&c.indextid),
								   &c_heaptid, &c_level, &c_deleted,
								   &c_nbr_blkno, &c_nbr_offno);
		UnlockReleaseBuffer(c_buf);

		/* Get layer-0 neighbors of c */
		n_count = acorn_get_neighbors(index, c_nbr_blkno, c_nbr_offno,
									  c_level, 0, m, neighbors,
									  HNSW_MAX_NEIGHBORS);

		for (int i = 0; i < n_count; i++)
		{
			BlockNumber  nblkno;
			OffsetNumber noffno;
			AcornPQNode *nc;
			double		 nd;
			ItemPointerData nheaptid;
			int			 nlevel;
			bool		 ndeleted;
			BlockNumber	 n_nbr_blkno;	/* unused here; n's neighbors are */
			OffsetNumber n_nbr_offno;	/* loaded when n is popped from C */
			Buffer		 nbuf;

			if (!ItemPointerIsValid(&neighbors[i]))
				continue;

			if (is_visited(visited, &neighbors[i]))
				continue;

			mark_visited(visited, &neighbors[i]);

			nblkno = ItemPointerGetBlockNumber(&neighbors[i]);
			noffno = ItemPointerGetOffsetNumber(&neighbors[i]);

			if (acorn_scan_single_read)
			{
				/* One pin: distance + heaptid/deleted from the same page */
				nd = acorn_load_node(index, nblkno, noffno,
									 query, dist_proc, dist_direct,
									 &nheaptid, &nlevel, &ndeleted,
									 &n_nbr_blkno, &n_nbr_offno);
			}
			else
			{
				nd = acorn_distance(index, nblkno, noffno, query, dist_proc);

				/* Load heaptid to evaluate predicate (second pin, same page) */
				nbuf = acorn_load_element(index, nblkno, noffno,
										  &nheaptid, &nlevel, &ndeleted,
										  &n_nbr_blkno, &n_nbr_offno);
				UnlockReleaseBuffer(nbuf);
			}

			/* Always add to C (ACORN-1: preserve connectivity) */
			nc = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&neighbors[i], &nc->elem.indextid);
			nc->elem.distance = nd;
			ItemPointerSetInvalid(&nc->elem.heaptid);
			pairingheap_add(C, &nc->ph_node);

			if (ndeleted)
				continue;

			/* Add to W only if predicate matches */
			if (acorn_eval_predicate(heap, &nheaptid, snapshot,
									 predicate, econtext))
			{
				f_dist = pairingheap_is_empty(W)
					? DBL_MAX
					: ((AcornPQNode *) pairingheap_first(W))->elem.distance;

				if (n_results < k || nd < f_dist)
				{
					AcornPQNode *wn = palloc(sizeof(AcornPQNode));
					ItemPointerCopy(&neighbors[i], &wn->elem.indextid);
					ItemPointerCopy(&nheaptid, &wn->elem.heaptid);
					wn->elem.distance = nd;
					pairingheap_add(W, &wn->ph_node);
					n_results++;

					/* Trim W if over k */
					if (n_results > k)
					{
						AcornPQNode *evicted =
							(AcornPQNode *) pairingheap_remove_first(W);
						pfree(evicted);
						n_results--;
					}
				}
			}

			/*
			 * Bounded C (ef_search cap) is an optimization, not required for
			 * correctness.  pairingheap has no O(1) size query, so we skip
			 * the cap here; Tier 2 (acorn_am) will implement it with a
			 * counted candidate set.
			 */
		}
	}

	/* Extract results from W into output array */
	{
		int idx = n_results - 1;		/* fill back-to-front (W is max-heap) */
		while (!pairingheap_is_empty(W) && idx >= 0)
		{
			AcornPQNode *wn = (AcornPQNode *) pairingheap_remove_first(W);
			result_tids_out[idx--] = wn->elem.heaptid;
			pfree(wn);
		}
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(tmp_ctx);

	return n_results;
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

int
acorn_scan_execute(AcornScanState *state, Relation index, Relation heap,
				   Datum query, Snapshot snapshot,
				   ItemPointerData *result_tids_out)
{
	FmgrInfo		dist_proc;
	AcornDistFn		dist_direct;
	BlockNumber		entry_blkno;
	OffsetNumber	entry_offno;
	int				entry_level;
	int				m;

	acorn_load_dist_proc(index, &dist_proc);
	dist_direct = acorn_scan_direct_dist
		? acorn_resolve_direct_dist(index) : NULL;

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m,
						 NULL, NULL))
		return 0;			/* empty index */

	/* Greedy descent to layer 1 (skipping to base layer entry point) */
	if (entry_level > 0)
		acorn_greedy_descent(index, m, &entry_blkno, &entry_offno,
							 entry_level, 0, query, &dist_proc);

	return acorn_layer0_search(index, heap, m,
							   entry_blkno, entry_offno,
							   query, &dist_proc, dist_direct,
							   snapshot,
							   state->predicate, state->econtext,
							   state->k, state->ef_search,
							   result_tids_out);
}

/* -----------------------------------------------------------------------
 * Resumable (streaming) scan — Tier 2
 *
 * Maintains a persistent frontier across calls:
 *   C — candidates awaiting expansion (min-heap, nearest first)
 *   R — discovered live nodes awaiting emission (min-heap, nearest first)
 *   visited — index TIDs already discovered (dedup)
 *
 * Each node is added to C and R exactly once (guarded by `visited`), expanded
 * at most once (popped from C), and emitted at most once (popped from R).  The
 * executor post-filters, so no predicate is evaluated here.
 * ----------------------------------------------------------------------- */

struct AcornStreamScan
{
	Relation		index;
	FmgrInfo		dist_proc;
	Datum			query;
	int				m;
	pairingheap	   *C;			/* candidates to expand */
	pairingheap	   *R;			/* discovered, awaiting emission */
	HTAB		   *visited;
	MemoryContext	mcxt;
	bool			exhausted;
};

/* Expand one candidate: add its unvisited layer-0 neighbors to C and (if live) R. */
static void
acorn_stream_expand(AcornStreamScan *s, const AcornElem *ce)
{
	ItemPointerData neighbors[HNSW_MAX_NEIGHBORS];
	ItemPointerData ce_heaptid;
	int				ce_level;
	bool			ce_deleted;
	BlockNumber		nbr_blkno;
	OffsetNumber	nbr_offno;
	Buffer			ebuf;
	int				n_count;

	/* Load the candidate's level + neighbor-tuple location */
	ebuf = acorn_load_element(s->index,
							  ItemPointerGetBlockNumber(&ce->indextid),
							  ItemPointerGetOffsetNumber(&ce->indextid),
							  &ce_heaptid, &ce_level, &ce_deleted,
							  &nbr_blkno, &nbr_offno);
	UnlockReleaseBuffer(ebuf);

	n_count = acorn_get_neighbors(s->index, nbr_blkno, nbr_offno,
								  ce_level, 0, s->m, neighbors,
								  HNSW_MAX_NEIGHBORS);

	for (int i = 0; i < n_count; i++)
	{
		BlockNumber		nblkno;
		OffsetNumber	noffno;
		double			nd;
		ItemPointerData nheaptid;
		int				nlevel;
		bool			ndeleted;
		BlockNumber		n_nbr_blkno;
		OffsetNumber	n_nbr_offno;
		AcornPQNode	   *cn;

		if (!ItemPointerIsValid(&neighbors[i]))
			continue;
		if (is_visited(s->visited, &neighbors[i]))
			continue;
		mark_visited(s->visited, &neighbors[i]);

		nblkno = ItemPointerGetBlockNumber(&neighbors[i]);
		noffno = ItemPointerGetOffsetNumber(&neighbors[i]);

		/* One pin: distance + heaptid/deleted from the same page (was an
		 * acorn_distance + acorn_load_element pair pinning the page twice) */
		nd = acorn_load_node(s->index, nblkno, noffno,
							 s->query, &s->dist_proc, NULL,
							 &nheaptid, &nlevel, &ndeleted,
							 &n_nbr_blkno, &n_nbr_offno);

		/* Always a candidate for expansion (preserve connectivity) */
		cn = palloc(sizeof(AcornPQNode));
		ItemPointerCopy(&neighbors[i], &cn->elem.indextid);
		ItemPointerSetInvalid(&cn->elem.heaptid);
		cn->elem.distance = nd;
		pairingheap_add(s->C, &cn->ph_node);

		if (!ndeleted)
		{
			AcornPQNode *rn = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&neighbors[i], &rn->elem.indextid);
			ItemPointerCopy(&nheaptid, &rn->elem.heaptid);
			rn->elem.distance = nd;
			pairingheap_add(s->R, &rn->ph_node);
		}
	}
}

AcornStreamScan *
acorn_stream_begin(Relation index, Datum query, Snapshot snapshot,
				   MemoryContext mcxt)
{
	MemoryContext	old = MemoryContextSwitchTo(mcxt);
	AcornStreamScan *s  = palloc0(sizeof(AcornStreamScan));
	BlockNumber		entry_blkno;
	OffsetNumber	entry_offno;
	int				entry_level;
	int				m;
	HASHCTL			info;

	(void) snapshot;				/* visibility handled by executor post-fetch */

	s->index     = index;
	s->query     = query;
	s->mcxt      = mcxt;
	s->exhausted = false;
	acorn_load_dist_proc(index, &s->dist_proc);

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m,
						 NULL, NULL))
	{
		s->exhausted = true;		/* empty index */
		MemoryContextSwitchTo(old);
		return s;
	}
	s->m = m;

	/* Descend upper layers to the base-layer entry point */
	if (entry_level > 0)
		acorn_greedy_descent(index, m, &entry_blkno, &entry_offno,
							 entry_level, 0, query, &s->dist_proc);

	s->C = pairingheap_allocate(pq_cmp_min, NULL);
	s->R = pairingheap_allocate(pq_cmp_min, NULL);

	/* visited set anchored to mcxt so it is freed with the scan */
	memset(&info, 0, sizeof(info));
	info.keysize   = sizeof(ItemPointerData);
	info.entrysize = sizeof(VisitedEntry);
	info.hcxt      = mcxt;
	s->visited = hash_create("acorn_stream_visited", 512, &info,
							 HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	/* Seed the frontier with the base-layer entry point */
	{
		ItemPointerData entry_tid;
		ItemPointerData heaptid;
		int				lvl;
		bool			deleted;
		BlockNumber		nb;
		OffsetNumber	no;
		Buffer			eb;
		double			d;
		AcornPQNode	   *cn;

		ItemPointerSet(&entry_tid, entry_blkno, entry_offno);
		mark_visited(s->visited, &entry_tid);

		d  = acorn_distance(index, entry_blkno, entry_offno, query, &s->dist_proc);
		eb = acorn_load_element(index, entry_blkno, entry_offno,
								&heaptid, &lvl, &deleted, &nb, &no);
		UnlockReleaseBuffer(eb);

		cn = palloc(sizeof(AcornPQNode));
		ItemPointerCopy(&entry_tid, &cn->elem.indextid);
		ItemPointerSetInvalid(&cn->elem.heaptid);
		cn->elem.distance = d;
		pairingheap_add(s->C, &cn->ph_node);

		if (!deleted)
		{
			AcornPQNode *rn = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&entry_tid, &rn->elem.indextid);
			ItemPointerCopy(&heaptid, &rn->elem.heaptid);
			rn->elem.distance = d;
			pairingheap_add(s->R, &rn->ph_node);
		}
	}

	MemoryContextSwitchTo(old);
	return s;
}

bool
acorn_stream_next(AcornStreamScan *s, ItemPointerData *heaptid_out)
{
	MemoryContext old;

	if (s->exhausted)
		return false;

	old = MemoryContextSwitchTo(s->mcxt);

	for (;;)
	{
		double r_dist = pairingheap_is_empty(s->R) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->R))->elem.distance;
		double c_dist = pairingheap_is_empty(s->C) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->C))->elem.distance;

		/*
		 * While the nearest unexpanded candidate could be closer than (or tie)
		 * the nearest discovered result, expand it first — the standard
		 * streaming-HNSW emission-safety condition.
		 */
		if (!pairingheap_is_empty(s->C) && c_dist <= r_dist)
		{
			AcornPQNode *cn = (AcornPQNode *) pairingheap_remove_first(s->C);
			AcornElem	 ce = cn->elem;
			pfree(cn);
			acorn_stream_expand(s, &ce);
			continue;
		}

		if (pairingheap_is_empty(s->R))
		{
			s->exhausted = true;		/* graph fully streamed */
			MemoryContextSwitchTo(old);
			return false;
		}

		{
			AcornPQNode *rn = (AcornPQNode *) pairingheap_remove_first(s->R);
			*heaptid_out = rn->elem.heaptid;
			pfree(rn);
			MemoryContextSwitchTo(old);
			return true;
		}
	}
}

/* -----------------------------------------------------------------------
 * Tier 2 in-filter traversal
 *
 * Uses AcornT2 element tuple format (acorn_t2_page.h).  Evaluates scalar
 * ScanKey predicates against the inline filter_val — no heap fetch.
 *
 * ACORN invariant preserved: filter-failing nodes go to C only (connectivity);
 * filter-passing nodes go to both C and R (candidate + result).
 * ----------------------------------------------------------------------- */

/*
 * Exact re-rank lookahead for inline-vector scans: before emitting, the top
 * window of approx-ordered results is re-scored with exact distances (one
 * element-page read each), and emission order within the window is exact.
 * 40 = max(4k, 40) for the k <= 10 regime the AM streams for; beyond the
 * window the scan keeps sliding (one re-rank per emit), so larger LIMITs
 * stay covered.
 */
#define ACORN_T2_RERANK_WINDOW	40

/* Persistent in-filter streaming frontier for Tier 2. */
struct AcornT2StreamScan
{
	Relation		index;
	FmgrInfo		dist_proc;
	AcornDistFn		dist_direct;	/* fmgr bypass kernel; NULL = use dist_proc */
	Datum			query;
	int				m;
	int				ef_search;		/* expansion budget: max node expansions */
	int				n_expansions;	/* expansions used so far */
	pairingheap	   *C;			/* failing candidates to expand (min-heap) */
	pairingheap	   *Cm;			/* passing candidates (member_first mode) */
	pairingheap	   *R;			/* discovered passing nodes, awaiting emit */
	HTAB		   *visited;
	MemoryContext	mcxt;
	bool			exhausted;
	bool			member_first;	/* spend budget on passing candidates first */
	bool			buffered;	/* finish expansion before first emission */
	ScanKey			keys;		/* filter ScanKeys (lives in caller's memory) */
	int				nkeys;

	/*
	 * Vector co-location (acorn_inline_vectors index + scan_inline_vectors
	 * GUC).  Discovery distances come from co-located SQ8 codes (approx);
	 * C/Cm are approx-ordered, R is ordered by the exact-distance lower
	 * bound, and Rx holds exact-rescored results.
	 */
	bool			inline_on;
	int				inline_dim;		/* index dimensions (= code length) */
	Size			inline_esz;		/* on-disk inline entry stride */
	AcornSq8DistFn	sq8_direct;		/* NULL = dequantize + fmgr */
	AcornPgVector  *dequant_scratch;	/* lazily allocated, in mcxt */
	double			q_l1;			/* L1 norm of the query (IP error bound) */
	pairingheap	   *Rx;			/* exact-reranked results, awaiting emit */
	int				rx_count;

	/*
	 * Shared-memory SQ8 code cache (acorn_codecache.c) — NON-inline indexes
	 * only.  Discovery hits provide the same fields an inline entry does
	 * (SQ8 code for approx ordering, filter_val, nbrtid/level, heaptid);
	 * misses use the element-page read.  `approx` is true whenever discovery
	 * distances can be SQ8 approximations (inline_on or cc), enabling the
	 * lb-ordered R + exact re-rank emission machinery.
	 */
	AcornCodeCacheScan *cc;		/* NULL = cache not serving this scan */
	bool			approx;		/* inline_on || cc != NULL */

#ifdef ACORN_CC_DEBUG
	/* temporary M1.5 instrumentation */
	uint64			dbg_neigh_iters;	/* per-neighbor loop iterations */
	uint64			dbg_discoveries;	/* unique (unvisited) neighbors */
	uint64			dbg_cc_hits;		/* cache hits */
	uint64			dbg_loads;			/* element-page load_node calls */
	uint64			dbg_reranks;		/* rerank_one calls */
	uint64			dbg_emits;			/* tuples emitted */
	uint64			dbg_next_iters;		/* for(;;) iterations in stream_next */
	bool			dbg_dumped;
#endif
};

/*
 * Load a Tier-2 element tuple.  Mirrors acorn_load_element but casts to
 * AcornT2ElementTuple and also returns filter_val (may be NULL).
 */
static Buffer
acorn_t2_load_element(Relation index, BlockNumber blkno, OffsetNumber offno,
					   ItemPointerData *heaptid_out, int *level_out,
					   bool *deleted_out,
					   BlockNumber *nbr_blkno_out, OffsetNumber *nbr_offno_out,
					   int64 *filter_val_out)
{
	Buffer				buf;
	Page				page;
	ItemId				iid;
	AcornT2ElementTuple etup;

	buf  = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	iid  = PageGetItemId(page, offno);
	etup = (AcornT2ElementTuple) PageGetItem(page, iid);

	*heaptid_out   = etup->heaptids[0];
	*level_out     = (int) etup->level;
	*deleted_out   = (etup->deleted != 0);
	*nbr_blkno_out = ItemPointerGetBlockNumber(&etup->neighbortid);
	*nbr_offno_out = ItemPointerGetOffsetNumber(&etup->neighbortid);
	if (filter_val_out)
		*filter_val_out = etup->filter_val;

	/* Buffer remains locked — caller must UnlockReleaseBuffer */
	return buf;
}

/*
 * Compute distance(stored vector, query) for a T2 element tuple.
 *
 * Uses the direct C kernel when one was resolved for the opclass (fmgr
 * bypass; numerically identical), falling back to the opclass support
 * function through fmgr otherwise.  Caller holds the buffer lock.
 */
static inline double
acorn_t2_etup_distance(AcornT2StreamScan *s, AcornT2ElementTuple etup)
{
	if (s->dist_direct)
	{
		AcornPgVector *v = (AcornPgVector *) AcornT2ElementTupleGetVector(etup);
		AcornPgVector *q = (AcornPgVector *) DatumGetPointer(s->query);

		if (likely(v->dim == q->dim))
			return s->dist_direct((int) v->dim, v->x, q->x);
	}

	return DatumGetFloat8(
		FunctionCall2(&s->dist_proc,
					  PointerGetDatum(AcornT2ElementTupleGetVector(etup)),
					  s->query));
}

/*
 * Distance using the T2 vector accessor (offset 80, not 72).
 */
static double
acorn_t2_distance(AcornT2StreamScan *s, BlockNumber blkno, OffsetNumber offno)
{
	Buffer				buf;
	Page				page;
	ItemId				iid;
	AcornT2ElementTuple etup;
	double				result;

	buf  = ReadBuffer(s->index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	iid  = PageGetItemId(page, offno);
	etup = (AcornT2ElementTuple) PageGetItem(page, iid);

	result = acorn_t2_etup_distance(s, etup);

	UnlockReleaseBuffer(buf);
	return result;
}

/*
 * Single page read for T2 scan nodes: extracts distance (from inline vector)
 * and all metadata (heaptid, filter_val, nbr_tid, deleted) in one buffer pin.
 * Halves element-page I/O vs the previous two-call pattern.
 */
static double
acorn_t2_load_node(AcornT2StreamScan *s,
				   BlockNumber blkno, OffsetNumber offno,
				   ItemPointerData *heaptid_out,
				   bool            *deleted_out,
				   int64           *filter_val_out,
				   ItemPointerData *nbrtid_out,
				   uint8           *level_out)
{
	Buffer				buf;
	Page				page;
	AcornT2ElementTuple etup;
	double				dist;

	buf  = ReadBuffer(s->index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (AcornT2ElementTuple)
		   PageGetItem(page, PageGetItemId(page, offno));

	dist = acorn_t2_etup_distance(s, etup);

	*heaptid_out    = etup->heaptids[0];
	*deleted_out    = (etup->deleted != 0);
	*filter_val_out = etup->filter_val;
	if (nbrtid_out)
		*nbrtid_out = etup->neighbortid;
	if (level_out)
		*level_out  = etup->level;

	UnlockReleaseBuffer(buf);
	return dist;
}

/*
 * Evaluate all ScanKey quals against stored_val.
 * sk_func is the opclass compare function (btint4cmp, etc.).
 * Strategy numbers: 1=< 2=<= 3== 4=>= 5=>.
 * Returns true if all quals pass, or when nkeys == 0.
 */
static bool
acorn_t2_eval_filter(ScanKey keys, int nkeys, int64 stored_val)
{
	int i;

	for (i = 0; i < nkeys; i++)
	{
		/*
		 * Fast path: known btree int4 predicates compared directly in C
		 * (fmgr bypass; integer compare is trivially identical to the
		 * int4lt/int4le/... implementations).
		 */
		if (acorn_scan_direct_filter)
		{
			int32	a = DatumGetInt32((Datum) stored_val);
			int32	b = DatumGetInt32(keys[i].sk_argument);
			bool	known = true;
			bool	ok = false;

			switch (keys[i].sk_func.fn_oid)
			{
				case F_INT4LT: ok = (a <  b); break;
				case F_INT4LE: ok = (a <= b); break;
				case F_INT4EQ: ok = (a == b); break;
				case F_INT4GE: ok = (a >= b); break;
				case F_INT4GT: ok = (a >  b); break;
				default: known = false; break;
			}

			if (known)
			{
				if (!ok)
					return false;
				continue;
			}
		}

		/*
		 * sk_func is the operator's own predicate (e.g. int4lt for <),
		 * which returns a boolean Datum directly — not a comparison integer.
		 */
		if (!DatumGetBool(
			FunctionCall2Coll(&keys[i].sk_func,
							   keys[i].sk_collation,
							   (Datum) stored_val,
							   keys[i].sk_argument)))
			return false;
	}
	return true;
}

/* -----------------------------------------------------------------------
 * Vector co-location: inline-entry discovery (acorn_inline_vectors)
 * ----------------------------------------------------------------------- */

/*
 * Approximate distance from SQ8 codes to the query.
 * Uses the direct asymmetric kernel when the opclass distance resolved to a
 * known function; otherwise dequantizes into a scratch vector and calls the
 * opclass support function through fmgr (correct for any opclass).
 * Shared by the inline-entry path and the shared-memory code cache path.
 */
static double
acorn_t2_sq8_distance(AcornT2StreamScan *s, const uint8 *code,
					  float scale, float offset)
{
	if (s->sq8_direct)
	{
		AcornPgVector *q = (AcornPgVector *) DatumGetPointer(s->query);

		return s->sq8_direct(s->inline_dim, code, scale, offset, q->x);
	}

	if (s->dequant_scratch == NULL)
	{
		Size		sz = offsetof(AcornPgVector, x)
			+ sizeof(float) * s->inline_dim;

		s->dequant_scratch = (AcornPgVector *)
			MemoryContextAllocZero(s->mcxt, sz);
		SET_VARSIZE(s->dequant_scratch, sz);
		s->dequant_scratch->dim = (int16) s->inline_dim;
	}
	acorn_sq8_decode(s->inline_dim, code, scale, offset,
					 s->dequant_scratch->x);
	return DatumGetFloat8(FunctionCall2(&s->dist_proc,
										PointerGetDatum(s->dequant_scratch),
										s->query));
}

/* Approximate distance from a co-located SQ8 entry to the query. */
static inline double
acorn_t2_inline_distance(AcornT2StreamScan *s, const AcornT2InlineEntry *e)
{
	return acorn_t2_sq8_distance(s, e->code, e->scale, e->offset);
}

/*
 * Worst-case lower bound on the exact distance given an SQ8 approximation.
 *
 * Per-coordinate reconstruction error is bounded by scale/2 (round to
 * nearest), so:
 *   neg IP:  |d_exact - d_approx| <= (scale/2) * ||q||_1
 *   L2^2:    ||e||_2 <= (scale/2) * sqrt(dim)  ->
 *            sqrt(d_exact) >= sqrt(d_approx) - ||e||_2
 * Unknown opclasses (fmgr dequant path) have no usable bound; the re-rank
 * then falls back to the fixed lookahead window.
 */
static inline double
acorn_t2_inline_lb(AcornT2StreamScan *s, double dist, float scale)
{
	if (s->sq8_direct == acorn_dist_neg_ip_sq8)
		return dist - 0.5 * (double) scale * s->q_l1;
	if (s->sq8_direct == acorn_dist_l2sq_sq8)
	{
		double		en = 0.5 * (double) scale * sqrt((double) s->inline_dim);
		double		lo = sqrt(Max(dist, 0.0)) - en;

		return (lo > 0.0) ? lo * lo : 0.0;
	}
	return dist;				/* no bound available */
}

/* Discovery snapshot of one neighbor slot (stack-local during expansion) */
typedef struct AcornInlineCand
{
	ItemPointerData indextid;
	ItemPointerData heaptid;
	ItemPointerData nbrtid;
	uint8			level;
	bool			deleted;
	bool			from_inline;	/* false: stale entry — read element page */
	float			scale;			/* SQ8 scale (error bound), from_inline only */
	int64			fval;
	double			dist;
} AcornInlineCand;

/*
 * Expand one candidate using the co-located layer-0 entries: read the
 * neighbor-tuple chunk chain (2 pages at the dim=128/gamma=2 design point),
 * compute approximate SQ8 distances for every unvisited slot, and push
 * candidates carrying heaptid/filter_val/nbrtid WITHOUT visiting their
 * element pages.  Slots whose inline entry is stale (indextid mismatch after
 * a torn reverse-edge update) or missing are resolved with a per-slot
 * element-page read — the classic cost, never a wrong result.
 *
 * ACORN invariant unchanged: every neighbor stays expandable (C/Cm);
 * filter-passing ones also enter R (approx-ordered; exact re-rank at emit).
 */
static void
acorn_t2_stream_expand_inline(AcornT2StreamScan *s, const AcornElem *ce)
{
	AcornInlineCand cands[HNSW_MAX_NEIGHBORS];
	ItemPointerData tids_local[HNSW_MAX_NEIGHBORS];
	bool			covered[HNSW_MAX_NEIGHBORS] = {false};
	int				n_c = 0;
	int				layer0_n = HnswGetLayerM(s->m, 0);
	BlockNumber		nbr_blkno;
	OffsetNumber	nbr_offno;
	ItemPointerData next_tid;
	bool			first = true;

	Assert(layer0_n <= HNSW_MAX_NEIGHBORS);

	if (ce->has_nbr)
	{
		nbr_blkno = ItemPointerGetBlockNumber(&ce->nbrtid);
		nbr_offno = ItemPointerGetOffsetNumber(&ce->nbrtid);
	}
	else
	{
		ItemPointerData ce_heaptid;
		int				ce_level;
		bool			ce_deleted;
		Buffer			ebuf;

		ebuf = acorn_t2_load_element(s->index,
									  ItemPointerGetBlockNumber(&ce->indextid),
									  ItemPointerGetOffsetNumber(&ce->indextid),
									  &ce_heaptid, &ce_level, &ce_deleted,
									  &nbr_blkno, &nbr_offno, NULL);
		UnlockReleaseBuffer(ebuf);
	}

	/* Walk the chunk chain: primary neighbor tuple, then continuations */
	ItemPointerSet(&next_tid, nbr_blkno, nbr_offno);
	while (ItemPointerIsValid(&next_tid))
	{
		Buffer				buf;
		Page				page;
		BlockNumber			cblk = ItemPointerGetBlockNumber(&next_tid);
		OffsetNumber		coff = ItemPointerGetOffsetNumber(&next_tid);
		AcornT2InlineHdr	hdr;
		char			   *entries;

		buf  = ReadBuffer(s->index, cblk);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (first)
		{
			HnswNeighborTuple	ntup = (HnswNeighborTuple)
				PageGetItem(page, PageGetItemId(page, coff));
			int					level  = (int) ntup->count / s->m - 2;
			int					start0 = HnswNeighborStart(s->m, level, 0);
			ItemPointerData	   *tids   = HnswNeighborTupleGetTids(ntup);

			Assert(ntup->type == HNSW_NEIGHBOR_TUPLE_TYPE);
			memcpy(tids_local, &tids[start0],
				   layer0_n * sizeof(ItemPointerData));
			hdr     = AcornT2NeighborInlineHdr(ntup);
			entries = AcornT2InlineHdrEntries(hdr);
		}
		else
		{
			AcornT2InlineCont	cont = (AcornT2InlineCont)
				PageGetItem(page, PageGetItemId(page, coff));

			if (cont->type != ACORN_T2_INLINE_TUPLE_TYPE)
			{
				/* broken chain: stop; uncovered slots are resolved below */
				UnlockReleaseBuffer(buf);
				break;
			}
			hdr     = &cont->hdr;
			entries = AcornT2ContEntries(cont);
		}

		for (int j = hdr->start;
			 j < (int) hdr->start + (int) hdr->n_here && j < layer0_n;
			 j++)
		{
			const AcornT2InlineEntry *e;
			AcornInlineCand *c;

#ifdef ACORN_CC_DEBUG
			s->dbg_neigh_iters++;
#endif
			if (covered[j])
				continue;
			covered[j] = true;

			if (!ItemPointerIsValid(&tids_local[j]))
				continue;
			if (!visit_once(s->visited, &tids_local[j]))
				continue;

			e = (const AcornT2InlineEntry *)
				AcornT2InlineEntryAt(entries, j - hdr->start, s->inline_esz);

			c = &cands[n_c++];
			c->indextid = tids_local[j];

			if ((e->flags & ACORN_T2_INLINE_VALID) &&
				ItemPointerEquals((ItemPointer) &e->indextid, &tids_local[j]))
			{
				c->heaptid     = e->heaptid;
				c->nbrtid      = e->nbrtid;
				c->level       = e->level;
				c->deleted     = (e->flags & ACORN_T2_INLINE_DELETED) != 0;
				c->from_inline = true;
				c->scale       = e->scale;
				c->fval        = e->filter_val;
				c->dist        = acorn_t2_inline_distance(s, e);
			}
			else
				c->from_inline = false;
		}

		next_tid = hdr->next;
		UnlockReleaseBuffer(buf);
		first = false;
	}

	/* Slots no chunk covered (defensive — well-formed chains cover all) */
	for (int j = 0; j < layer0_n; j++)
	{
		if (covered[j] || !ItemPointerIsValid(&tids_local[j]))
			continue;
		if (!visit_once(s->visited, &tids_local[j]))
			continue;
		cands[n_c].indextid    = tids_local[j];
		cands[n_c].from_inline = false;
		n_c++;
	}

	/* Resolve stale slots, then push candidates/results */
#ifdef ACORN_CC_DEBUG
	s->dbg_discoveries += n_c;
#endif
	for (int i = 0; i < n_c; i++)
	{
		AcornInlineCand *c = &cands[i];
		AcornPQNode	   *cn;
		bool			passes;

		if (!c->from_inline)
		{
#ifdef ACORN_CC_DEBUG
			s->dbg_loads++;
#endif
			ItemPointerData nbrtid;
			uint8			level;

			c->dist = acorn_t2_load_node(s,
								 ItemPointerGetBlockNumber(&c->indextid),
								 ItemPointerGetOffsetNumber(&c->indextid),
								 &c->heaptid, &c->deleted, &c->fval,
								 &nbrtid, &level);
			c->nbrtid = nbrtid;
			c->level  = level;
		}

		passes = !c->deleted &&
			acorn_t2_eval_filter(s->keys, s->nkeys, c->fval);

		cn = palloc(sizeof(AcornPQNode));
		ItemPointerCopy(&c->indextid, &cn->elem.indextid);
		ItemPointerSetInvalid(&cn->elem.heaptid);
		cn->elem.distance = c->dist;
		cn->elem.lb = c->from_inline
			? acorn_t2_inline_lb(s, c->dist, c->scale)
			: c->dist;
		ItemPointerCopy(&c->nbrtid, &cn->elem.nbrtid);
		cn->elem.level   = c->level;
		cn->elem.has_nbr = true;
		pairingheap_add((s->member_first && passes) ? s->Cm : s->C,
						&cn->ph_node);

		if (passes)
		{
			AcornPQNode *rn = palloc(sizeof(AcornPQNode));

			ItemPointerCopy(&c->indextid, &rn->elem.indextid);
			ItemPointerCopy(&c->heaptid, &rn->elem.heaptid);
			rn->elem.distance = c->dist;
			rn->elem.lb = c->from_inline
				? acorn_t2_inline_lb(s, c->dist, c->scale)
				: c->dist;		/* element-page read: exact already */
			rn->elem.has_nbr  = false;
			pairingheap_add(s->R, &rn->ph_node);
		}
	}
}

/*
 * Pop the nearest approx-ordered result, re-score it with the EXACT distance
 * (one element-page read, which also refreshes the deleted flag), and move
 * it to Rx.  Vacuum-deleted nodes are dropped here — same outcome as the
 * non-inline scan's discovery-time deleted check.
 */
static void
acorn_t2_rerank_one(AcornT2StreamScan *s)
{
	AcornPQNode	   *rn = (AcornPQNode *) pairingheap_remove_first(s->R);
	ItemPointerData heaptid;
	bool			deleted;
	int64			fval;
	double			exact;

	exact = acorn_t2_load_node(s,
							   ItemPointerGetBlockNumber(&rn->elem.indextid),
							   ItemPointerGetOffsetNumber(&rn->elem.indextid),
							   &heaptid, &deleted, &fval, NULL, NULL);
	if (deleted)
	{
		pfree(rn);
		return;
	}
	rn->elem.distance = exact;
	rn->elem.heaptid  = heaptid;
	pairingheap_add(s->Rx, &rn->ph_node);
	s->rx_count++;
}

/*
 * Greedy descent for T2 format — mirrors acorn_greedy_descent using T2
 * element loader and distance function.
 */
static void
acorn_t2_greedy_descent(AcornT2StreamScan *s,
						  BlockNumber *cur_blkno, OffsetNumber *cur_offno,
						  int start_level, int target_level)
{
	Relation	index = s->index;
	int			m = s->m;
	double		cur_dist = acorn_t2_distance(s, *cur_blkno, *cur_offno);

	for (int lc = start_level; lc > target_level; lc--)
	{
		bool improved;

		do {
			ItemPointerData neighbors[HNSW_MAX_NEIGHBORS];
			ItemPointerData cur_heaptid;
			int				cur_level;
			bool			cur_deleted;
			BlockNumber		nbr_blkno;
			OffsetNumber	nbr_offno;
			Buffer			ebuf;
			int				n_count;

			improved = false;

			ebuf = acorn_t2_load_element(index, *cur_blkno, *cur_offno,
										  &cur_heaptid, &cur_level, &cur_deleted,
										  &nbr_blkno, &nbr_offno, NULL);
			UnlockReleaseBuffer(ebuf);

			n_count = acorn_get_neighbors(index, nbr_blkno, nbr_offno,
										   cur_level, lc, m, neighbors,
										   HNSW_MAX_NEIGHBORS);

			for (int i = 0; i < n_count; i++)
			{
				BlockNumber  nblkno  = ItemPointerGetBlockNumber(&neighbors[i]);
				OffsetNumber noffno  = ItemPointerGetOffsetNumber(&neighbors[i]);
				double		 nd;

				if (!ItemPointerIsValid(&neighbors[i]))
					continue;

				nd = acorn_t2_distance(s, nblkno, noffno);
				if (nd < cur_dist)
				{
					cur_dist   = nd;
					*cur_blkno = nblkno;
					*cur_offno = noffno;
					improved   = true;
				}
			}
		} while (improved);
	}
}

/*
 * Expand one T2 candidate: add unvisited layer-0 neighbors to C; add to R
 * only those that pass the filter.  Filter-failing nodes go to C only —
 * this is the ACORN invariant that preserves graph connectivity.
 */
static void
acorn_t2_stream_expand(AcornT2StreamScan *s, const AcornElem *ce)
{
	ItemPointerData neighbors[HNSW_MAX_NEIGHBORS];
	int				ce_level;
	BlockNumber		nbr_blkno;
	OffsetNumber	nbr_offno;
	int				n_count;
	/* code-cache read pipeline: resolved entries prefetched K ahead of use */
	const AcornCodeCacheEntry *pf_entry[HNSW_MAX_NEIGHBORS];

	if (ce->has_nbr)
	{
		/*
		 * Single-read fast path: neighbor-tuple location + level were
		 * captured when this node's element page was read at discovery —
		 * no element page re-read here.
		 */
		ce_level  = (int) ce->level;
		nbr_blkno = ItemPointerGetBlockNumber(&ce->nbrtid);
		nbr_offno = ItemPointerGetOffsetNumber(&ce->nbrtid);
	}
	else
	{
		ItemPointerData ce_heaptid;
		bool			ce_deleted;
		Buffer			ebuf;

		ebuf = acorn_t2_load_element(s->index,
									  ItemPointerGetBlockNumber(&ce->indextid),
									  ItemPointerGetOffsetNumber(&ce->indextid),
									  &ce_heaptid, &ce_level, &ce_deleted,
									  &nbr_blkno, &nbr_offno, NULL);
		UnlockReleaseBuffer(ebuf);
	}

	n_count = acorn_get_neighbors(s->index, nbr_blkno, nbr_offno,
								   ce_level, 0, s->m, neighbors,
								   HNSW_MAX_NEIGHBORS);

	/*
	 * Prefetch distinct unvisited neighbor pages before the per-neighbor
	 * loop, so the buffer manager can issue I/O ahead of the sequential
	 * load+distance pass below.  PrefetchBuffer does not pin.
	 */
	if (acorn_scan_prefetch)
	{
		BlockNumber seen[HNSW_MAX_NEIGHBORS];
		int			n_seen = 0;

		for (int i = 0; i < n_count; i++)
		{
			BlockNumber blk;
			int			j;

			if (!ItemPointerIsValid(&neighbors[i]))
				continue;
			if (is_visited(s->visited, &neighbors[i]))
				continue;

			blk = ItemPointerGetBlockNumber(&neighbors[i]);
			for (j = 0; j < n_seen; j++)
			{
				if (seen[j] == blk)
					break;
			}
			if (j < n_seen)
				continue;
			seen[n_seen++] = blk;
			PrefetchBuffer(s->index, MAIN_FORKNUM, blk);
		}
	}

	/*
	 * Code-cache read pipeline (see acorn_cc_prefetch_entry).  pf_entry[j]
	 * holds neighbor j's resolved cache entry pointer, pre-prefetched K
	 * iterations before it is read+scored so its DRAM line is warm by then.
	 * Resolving is cheap (two dependent loads, no lock); the heavy read
	 * (seqlock copy of the cold code) happens only at the consume site.
	 * Pre-resolution is a hint with no semantic effect: a visited/skipped
	 * neighbor just wastes its prefetch, and the consume site still does the
	 * full resolve-if-needed + version-checked read + G4 fallback.  Only the
	 * leading edge needs priming; the rolling prefetch inside the loop covers
	 * the rest.
	 */
	if (s->cc)
	{
		int		prime = Min(ACORN_CC_PREFETCH_AHEAD, n_count);

		for (int j = 0; j < n_count; j++)
			pf_entry[j] = NULL;
		for (int j = 0; j < prime; j++)
		{
			if (!ItemPointerIsValid(&neighbors[j]))
				continue;
			pf_entry[j] = acorn_codecache_resolve(s->cc,
								ItemPointerGetBlockNumber(&neighbors[j]),
								ItemPointerGetOffsetNumber(&neighbors[j]));
			acorn_cc_prefetch_entry(pf_entry[j], s->inline_dim);
		}
	}

	for (int i = 0; i < n_count; i++)
	{
		BlockNumber		nblkno;
		OffsetNumber	noffno;
		double			nd;
		ItemPointerData nheaptid;
		bool			ndeleted;
		int64			nfilter_val;
		ItemPointerData n_nbrtid;
		uint8			n_level;
		AcornPQNode	   *cn;

#ifdef ACORN_CC_DEBUG
		s->dbg_neigh_iters++;
#endif
		/*
		 * Rolling prefetch: resolve + prefetch neighbor i+K's entry now so it
		 * is warm when we reach it.  Done before the visited check so the
		 * pipeline depth stays constant regardless of which neighbors get
		 * skipped (a skipped neighbor's prefetch is simply unused).
		 */
		if (s->cc)
		{
			int		pf = i + ACORN_CC_PREFETCH_AHEAD;

			if (pf < n_count && ItemPointerIsValid(&neighbors[pf]))
			{
				pf_entry[pf] = acorn_codecache_resolve(s->cc,
									ItemPointerGetBlockNumber(&neighbors[pf]),
									ItemPointerGetOffsetNumber(&neighbors[pf]));
				acorn_cc_prefetch_entry(pf_entry[pf], s->inline_dim);
			}
		}

		if (!ItemPointerIsValid(&neighbors[i]))
			continue;
		if (acorn_scan_visited_oneprobe)
		{
			if (!visit_once(s->visited, &neighbors[i]))
				continue;
		}
		else
		{
			if (is_visited(s->visited, &neighbors[i]))
				continue;
			mark_visited(s->visited, &neighbors[i]);
		}

		nblkno = ItemPointerGetBlockNumber(&neighbors[i]);
		noffno = ItemPointerGetOffsetNumber(&neighbors[i]);

#ifdef ACORN_CC_DEBUG
		s->dbg_discoveries++;
#endif

		/*
		 * Shared-memory code cache hit: everything the element page would
		 * provide comes from the cached entry instead — filter_val for the
		 * predicate, SQ8 distance for (approximate) ordering, nbrtid/level
		 * for expansion, heaptid for emission — exactly what the inline
		 * path reads from its co-located entries.  Approx distances carry
		 * an exact-distance lower bound; emission re-ranks exactly.
		 * A miss (or no cache) takes the element-page read below (G4:
		 * correctness never depends on cache state).
		 *
		 * The entry was resolved + prefetched K iterations earlier, so the
		 * read here finds its code line already warm.  If pre-resolution
		 * returned NULL (TID not yet cached at prefetch time, or a growth
		 * refresh moved the snapshot), re-resolve inline — the result is
		 * identical, just without the prefetch head start.
		 */
		if (s->cc)
		{
			AcornCodeCacheHit hit;
			const AcornCodeCacheEntry *e = pf_entry[i];

			if (e == NULL)
				e = acorn_codecache_resolve(s->cc, nblkno, noffno);

			if (acorn_codecache_read(s->cc, e, &hit))
			{
				double	ad = acorn_t2_sq8_distance(s, hit.code,
												   hit.scale, hit.offset);

#ifdef ACORN_CC_DEBUG
				s->dbg_cc_hits++;
#endif
				double	alb = acorn_t2_inline_lb(s, ad, hit.scale);
				bool	cpasses = (hit.flags & ACORN_CC_DELETED) == 0 &&
					acorn_t2_eval_filter(s->keys, s->nkeys, hit.filter_val);

				cn = palloc(sizeof(AcornPQNode));
				ItemPointerCopy(&neighbors[i], &cn->elem.indextid);
				ItemPointerSetInvalid(&cn->elem.heaptid);
				cn->elem.distance = ad;
				cn->elem.lb       = alb;
				ItemPointerCopy(&hit.nbrtid, &cn->elem.nbrtid);
				cn->elem.level   = hit.level;
				cn->elem.has_nbr = true;
				pairingheap_add((s->member_first && cpasses) ? s->Cm : s->C,
								&cn->ph_node);

				if (cpasses)
				{
					AcornPQNode *rn = palloc(sizeof(AcornPQNode));

					ItemPointerCopy(&neighbors[i], &rn->elem.indextid);
					ItemPointerCopy(&hit.heaptid, &rn->elem.heaptid);
					rn->elem.distance = ad;
					rn->elem.lb       = alb;
					rn->elem.has_nbr  = false;
					pairingheap_add(s->R, &rn->ph_node);
				}
				continue;
			}
		}

#ifdef ACORN_CC_DEBUG
		s->dbg_loads++;
#endif
		nd = acorn_t2_load_node(s, nblkno, noffno,
								 &nheaptid, &ndeleted, &nfilter_val,
								 &n_nbrtid, &n_level);

		{
			bool passes = !ndeleted &&
				acorn_t2_eval_filter(s->keys, s->nkeys, nfilter_val);

			/*
			 * Every neighbor stays expandable — ACORN invariant: preserve
			 * connectivity.  In member_first mode passing candidates go to
			 * Cm so the expansion budget prefers the predicate subgraph.
			 * Either way the node carries its neighbor-tuple location when
			 * scan_single_read is on, so expansion skips the element-page
			 * re-read regardless of which heap it came from.
			 */
			cn = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&neighbors[i], &cn->elem.indextid);
			ItemPointerSetInvalid(&cn->elem.heaptid);
			cn->elem.distance = nd;
			cn->elem.lb       = nd;
			if (acorn_scan_single_read)
			{
				ItemPointerCopy(&n_nbrtid, &cn->elem.nbrtid);
				cn->elem.level   = n_level;
				cn->elem.has_nbr = true;
			}
			else
				cn->elem.has_nbr = false;
			pairingheap_add((s->member_first && passes) ? s->Cm : s->C,
							&cn->ph_node);

			if (passes)
			{
				AcornPQNode *rn = palloc(sizeof(AcornPQNode));

				ItemPointerCopy(&neighbors[i], &rn->elem.indextid);
				ItemPointerCopy(&nheaptid, &rn->elem.heaptid);
				rn->elem.distance = nd;
				rn->elem.lb       = nd;		/* exact (element page) */
				pairingheap_add(s->R, &rn->ph_node);
			}
		}
	}

}

AcornT2StreamScan *
acorn_t2_stream_begin(Relation index, Datum query,
					   ScanKey keys, int nkeys, int ef_search,
					   Snapshot snapshot, MemoryContext mcxt)
{
	MemoryContext		old = MemoryContextSwitchTo(mcxt);
	AcornT2StreamScan  *s  = palloc0(sizeof(AcornT2StreamScan));
	BlockNumber			entry_blkno;
	OffsetNumber		entry_offno;
	int					entry_level;
	int					m;
	int					meta_dims = 0;
	uint16				meta_flags = 0;
	HASHCTL				info;

	(void) snapshot;			/* visibility handled by executor post-fetch */

	s->index        = index;
	s->query        = query;
	s->mcxt         = mcxt;
	s->keys         = keys;
	s->nkeys        = nkeys;
	s->ef_search    = ef_search;
	s->n_expansions = 0;
	s->exhausted    = false;
	s->member_first = acorn_member_first;
	s->buffered     = acorn_buffered_emission;
	acorn_load_dist_proc(index, &s->dist_proc);
	s->dist_direct = acorn_scan_direct_dist
		? acorn_resolve_direct_dist(index) : NULL;

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m,
						 &meta_dims, &meta_flags))
	{
		s->exhausted = true;		/* empty index */
		MemoryContextSwitchTo(old);
		return s;
	}
	s->m = m;

	/*
	 * Vector co-location: the layout flag comes from the META PAGE (set at
	 * build time), not rd_options.  The SQ8 kernel is resolved independently
	 * of the scan_direct_dist GUC — approx distances have no fmgr equivalent;
	 * unknown opclasses dequantize and go through fmgr.
	 */
	s->inline_on = (meta_flags & ACORN_T2_META_INLINE_VECTORS) != 0
		&& acorn_scan_inline_vectors
		&& meta_dims > 0;

	/*
	 * Shared-memory SQ8 code cache — NON-inline indexes only (the meta flag,
	 * not the GUC, decides: inline indexes carry their own co-located codes
	 * and never consult the cache).  NULL whenever the cache cannot serve
	 * (GUC off, budget 0, directory full, slot LOADING in another backend);
	 * the scan then runs the classic element-page path unchanged.
	 */
	s->cc = NULL;
	if ((meta_flags & ACORN_T2_META_INLINE_VECTORS) == 0
		&& acorn_scan_code_cache
		&& meta_dims > 0)
		s->cc = acorn_codecache_begin_scan(index, meta_dims);

	s->approx = s->inline_on || (s->cc != NULL);
	if (s->approx)
	{
		AcornPgVector  *q  = (AcornPgVector *) DatumGetPointer(query);
		AcornDistFn		fn = acorn_resolve_direct_dist(index);

		s->inline_dim = meta_dims;
		s->inline_esz = ACORN_T2_INLINE_ENTRY_SIZE(meta_dims);
		s->sq8_direct = NULL;
		if ((int) q->dim == meta_dims)
		{
			if (fn == acorn_dist_l2sq)
				s->sq8_direct = acorn_dist_l2sq_sq8;
			else if (fn == acorn_dist_neg_ip)
				s->sq8_direct = acorn_dist_neg_ip_sq8;
		}

		/* ||q||_1 for the neg-IP quantization error bound */
		s->q_l1 = 0.0;
		for (int i = 0; i < (int) q->dim; i++)
			s->q_l1 += fabs((double) q->x[i]);
	}

	if (entry_level > 0)
		acorn_t2_greedy_descent(s, &entry_blkno, &entry_offno,
								 entry_level, 0);

	s->C  = pairingheap_allocate(pq_cmp_min, NULL);	/* candidates, nearest-first */
	s->Cm = pairingheap_allocate(pq_cmp_min, NULL);	/* passing candidates */
	s->R  = pairingheap_allocate(pq_cmp_lb_min, NULL);	/* results, by lower bound */
	s->Rx = pairingheap_allocate(pq_cmp_min, NULL);	/* exact-reranked results */
	s->rx_count = 0;

	memset(&info, 0, sizeof(info));
	info.keysize   = sizeof(ItemPointerData);
	info.entrysize = sizeof(VisitedEntry);
	info.hcxt      = mcxt;
	s->visited = hash_create("acorn_t2_stream_visited", 512, &info,
							  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);

	/* Seed the frontier with the base-layer entry point */
	{
		ItemPointerData entry_tid;
		ItemPointerData heaptid;
		bool			deleted;
		int64			fval;
		ItemPointerData nbrtid;
		uint8			level;
		double			d;
		AcornPQNode	   *cn;

		ItemPointerSet(&entry_tid, entry_blkno, entry_offno);
		mark_visited(s->visited, &entry_tid);

		d = acorn_t2_load_node(s, entry_blkno, entry_offno,
							   &heaptid, &deleted, &fval, &nbrtid, &level);

		{
			bool passes = !deleted &&
				acorn_t2_eval_filter(keys, nkeys, fval);

			cn = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&entry_tid, &cn->elem.indextid);
			ItemPointerSetInvalid(&cn->elem.heaptid);
			cn->elem.distance = d;
			cn->elem.lb       = d;
			if (acorn_scan_single_read || s->approx)
			{
				ItemPointerCopy(&nbrtid, &cn->elem.nbrtid);
				cn->elem.level   = level;
				cn->elem.has_nbr = true;
			}
			else
				cn->elem.has_nbr = false;
			pairingheap_add((s->member_first && passes) ? s->Cm : s->C,
							&cn->ph_node);

			if (passes)
			{
				AcornPQNode *rn = palloc(sizeof(AcornPQNode));

				ItemPointerCopy(&entry_tid, &rn->elem.indextid);
				ItemPointerCopy(&heaptid, &rn->elem.heaptid);
				rn->elem.distance = d;
				rn->elem.lb       = d;		/* exact (element page) */
				pairingheap_add(s->R, &rn->ph_node);
			}
		}
	}

	MemoryContextSwitchTo(old);
	return s;
}

void
acorn_t2_stream_end(AcornT2StreamScan *s)
{
	if (s == NULL || s->cc == NULL)
		return;
	acorn_codecache_end_scan(s->cc);
	s->cc = NULL;				/* idempotent: a second call is a no-op */
}

/*
 * Emit the next nearest filter-passing heap TID, expanding the frontier lazily.
 *
 * Pops the nearest candidate from C and expands it; filter-failing nodes stay in
 * C for connectivity while passing nodes go to R.  The executor post-filters and
 * keeps pulling until its LIMIT is satisfied.
 *
 * ef_search bounds the number of node expansions (the work budget).  With
 * buffered emission (pg_acorn.buffered_emission, default on) the scan spends
 * the whole budget best-first BEFORE the first emission, then drains the
 * discovered results in exact-distance order (lb-ordered R + exact re-rank) —
 * the emitted sequence has zero inversions over the discovered set.  With
 * eager emission (off; legacy) it emits as soon as no unexpanded candidate is
 * closer than the nearest discovered result, which can emit a greedy local
 * minimum before a truly closer node is discovered.
 * Larger ef_search explores more of the predicate subgraph — higher recall and
 * more page reads; smaller ef_search stops sooner.  This is the counted
 * candidate budget that the Tier 1 bounded search (acorn_layer0_search) deferred
 * to Tier 2: it works where the standard "nearest candidate beyond worst result"
 * HNSW termination does not, because ACORN keeps near filter-failing nodes in C.
 *
 * The Tier 1 stream (acorn_stream_next above) keeps eager emission: it has no
 * expansion budget, so "buffering" there would expand the entire graph.
 */
bool
acorn_t2_stream_next(AcornT2StreamScan *s, ItemPointerData *heaptid_out)
{
	MemoryContext old;

	if (s->exhausted)
		return false;

	old = MemoryContextSwitchTo(s->mcxt);

	for (;;)
	{
#ifdef ACORN_CC_DEBUG
		s->dbg_next_iters++;
#endif
		/*
		 * R is ordered by the exact-distance lower bound.  The EXPANSION
		 * safety bound stays at the approx level (head's `distance`): using
		 * the lb there stops exploration early and wastes the ef budget
		 * (measured: -0.05 recall at 40% selectivity).  The lb is consulted
		 * only for the re-rank refill decision below.
		 */
		AcornPQNode *rhead = pairingheap_is_empty(s->R) ? NULL
			: (AcornPQNode *) pairingheap_first(s->R);
		double r_dist = rhead ? rhead->elem.distance : DBL_MAX;
		double r_lb   = rhead ? rhead->elem.lb       : DBL_MAX;
		double rx_dist = (s->approx && !pairingheap_is_empty(s->Rx))
			? ((AcornPQNode *) pairingheap_first(s->Rx))->elem.distance
			: DBL_MAX;
		double r_bound = Min(r_dist, rx_dist);
		double cf_dist = pairingheap_is_empty(s->C) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->C))->elem.distance;
		double cm_dist = pairingheap_is_empty(s->Cm) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->Cm))->elem.distance;
		double c_dist = Min(cf_dist, cm_dist);

		/*
		 * Expansion trigger uses the candidates' exact-distance lower bound
		 * (== distance when not SQ8-approximated): keep exploring while a
		 * candidate COULD truly beat the nearest result, so quantization
		 * over-estimates cannot truncate discovery.  Budget-capped as ever.
		 */
		double cf_lb = pairingheap_is_empty(s->C) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->C))->elem.lb;
		double cm_lb = pairingheap_is_empty(s->Cm) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->Cm))->elem.lb;
		double c_lb = Min(cf_lb, cm_lb);

		/*
		 * Expansion gate.
		 *
		 * Buffered emission (default): run the expansion phase to completion
		 * FIRST — the ef_search budget (or an empty frontier) is the only
		 * terminator.  The eager gate `c_lb <= r_bound` settles in a greedy
		 * local minimum: it stops exploring as soon as no unexpanded
		 * candidate is closer than the nearest DISCOVERED result, so a truly
		 * closer node can still be undiscovered when the first row is
		 * emitted (measured exact-distance inversions: rank-0 d=0.654 while
		 * an undiscovered d=0.000 node existed).  Because expansion finishes
		 * before anything is emitted, no later discovery can undercut an
		 * already-emitted row — combined with the lb-ordered R / exact
		 * re-rank below, the emitted sequence is exactly ordered.
		 *
		 * Eager (buffered_emission=off, legacy A/B path): explore only while
		 * a candidate may still beat the nearest result, within the budget.
		 *
		 * Expansion choice: in member_first mode, spend the budget on the
		 * nearest PASSING candidate whenever one is queued — the payload
		 * subgraph keeps same-partition nodes connected, so this drains the
		 * predicate subgraph instead of the (mostly failing) global
		 * neighborhood.  Buffering only changes WHEN emission starts; the
		 * member_first/Cm routing of WHICH nodes get expanded is unchanged.
		 */
		if (c_dist != DBL_MAX
			&& s->n_expansions < s->ef_search
			&& (s->buffered || c_lb <= r_bound))
		{
			pairingheap *src;
			AcornPQNode *cn;
			AcornElem	 ce;

			CHECK_FOR_INTERRUPTS();

			if (s->member_first && !pairingheap_is_empty(s->Cm))
				src = s->Cm;
			else if (!pairingheap_is_empty(s->C) && cf_dist <= cm_dist)
				src = s->C;
			else
				src = s->Cm;

			cn = (AcornPQNode *) pairingheap_remove_first(src);
			ce = cn->elem;
			pfree(cn);
			if (s->inline_on)
				acorn_t2_stream_expand_inline(s, &ce);
			else
				acorn_t2_stream_expand(s, &ce);
			s->n_expansions++;
			continue;
		}

		/*
		 * Approx mode (inline entries or code cache): exact re-rank before
		 * emission.  R is ordered by the exact-distance lower bound, so
		 * re-ranking while lb(R head) <= exact(Rx head) provably moves
		 * every candidate that could still beat the current exact head
		 * into Rx — emission order among emitted results is then exact,
		 * with no quantization misses.  The fixed lookahead window only
		 * matters for opclasses without a usable error bound (fmgr dequant
		 * path, where lb == approx).
		 */
		if (s->approx)
		{
			bool	have_bound = (s->sq8_direct != NULL);

			if (rhead != NULL &&
				(r_lb <= rx_dist ||
				 (!have_bound && s->rx_count < ACORN_T2_RERANK_WINDOW)))
			{
#ifdef ACORN_CC_DEBUG
				s->dbg_reranks++;
#endif
				acorn_t2_rerank_one(s);
				continue;		/* re-derive all bounds */
			}

			if (pairingheap_is_empty(s->Rx))
			{
				s->exhausted = true;
				MemoryContextSwitchTo(old);
				return false;
			}

			{
				AcornPQNode *rn = (AcornPQNode *) pairingheap_remove_first(s->Rx);

				s->rx_count--;
				*heaptid_out = rn->elem.heaptid;
				pfree(rn);
#ifdef ACORN_CC_DEBUG
				s->dbg_emits++;
				if (!s->dbg_dumped)
				{
					s->dbg_dumped = true;
					elog(NOTICE, "acorn_cc_dbg: exp=%d next_iters=" UINT64_FORMAT
						 " neigh=" UINT64_FORMAT " disc=" UINT64_FORMAT
						 " hits=" UINT64_FORMAT " loads=" UINT64_FORMAT
						 " reranks=" UINT64_FORMAT,
						 s->n_expansions, s->dbg_next_iters, s->dbg_neigh_iters,
						 s->dbg_discoveries, s->dbg_cc_hits, s->dbg_loads,
						 s->dbg_reranks);
				}
#endif
				MemoryContextSwitchTo(old);
				return true;
			}
		}

		if (pairingheap_is_empty(s->R))
		{
			s->exhausted = true;
			MemoryContextSwitchTo(old);
			return false;
		}

		{
			AcornPQNode *rn = (AcornPQNode *) pairingheap_remove_first(s->R);
			*heaptid_out = rn->elem.heaptid;
			pfree(rn);
#ifdef ACORN_CC_DEBUG
			s->dbg_emits++;
			if (!s->dbg_dumped)
			{
				s->dbg_dumped = true;
				elog(NOTICE, "acorn_cc_dbg: exp=%d next_iters=" UINT64_FORMAT
					 " neigh=" UINT64_FORMAT " disc=" UINT64_FORMAT
					 " hits=" UINT64_FORMAT " loads=" UINT64_FORMAT
					 " reranks=" UINT64_FORMAT,
					 s->n_expansions, s->dbg_next_iters, s->dbg_neigh_iters,
					 s->dbg_discoveries, s->dbg_cc_hits, s->dbg_loads,
					 s->dbg_reranks);
			}
#endif
			MemoryContextSwitchTo(old);
			return true;
		}
	}
}
