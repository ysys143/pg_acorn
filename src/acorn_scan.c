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

#include "executor/executor.h"
#include "executor/tuptable.h"
#include "lib/pairingheap.h"
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

/* -----------------------------------------------------------------------
 * Internal element reference
 * ----------------------------------------------------------------------- */

typedef struct AcornElem
{
	ItemPointerData indextid;	/* index page (blkno, offno) */
	ItemPointerData heaptid;	/* heap TID — for predicate evaluation */
	double			distance;

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
 */
static bool
acorn_meta_read(Relation index, BlockNumber *entry_blkno,
				OffsetNumber *entry_offno, int *entry_level, int *m_out)
{
	Buffer		buf;
	Page		page;
	HnswMetaPage meta;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	meta = HnswPageGetMeta(page);

	*entry_blkno = meta->entryBlkno;
	*entry_offno = meta->entryOffno;
	*entry_level = (int) meta->entryLevel;
	*m_out = (int) meta->m;

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

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m))
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

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m))
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
	ScanKey			keys;		/* filter ScanKeys (lives in caller's memory) */
	int				nkeys;
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
	acorn_load_dist_proc(index, &s->dist_proc);
	s->dist_direct = acorn_scan_direct_dist
		? acorn_resolve_direct_dist(index) : NULL;

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m))
	{
		s->exhausted = true;		/* empty index */
		MemoryContextSwitchTo(old);
		return s;
	}
	s->m = m;

	if (entry_level > 0)
		acorn_t2_greedy_descent(s, &entry_blkno, &entry_offno,
								 entry_level, 0);

	s->C  = pairingheap_allocate(pq_cmp_min, NULL);	/* candidates, nearest-first */
	s->Cm = pairingheap_allocate(pq_cmp_min, NULL);	/* passing candidates */
	s->R  = pairingheap_allocate(pq_cmp_min, NULL);	/* results, nearest-first */

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
			if (acorn_scan_single_read)
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
				pairingheap_add(s->R, &rn->ph_node);
			}
		}
	}

	MemoryContextSwitchTo(old);
	return s;
}

/*
 * Emit the next nearest filter-passing heap TID, expanding the frontier lazily.
 *
 * Pops the nearest candidate from C and expands it; filter-failing nodes stay in
 * C for connectivity while passing nodes go to R.  The executor post-filters and
 * keeps pulling until its LIMIT is satisfied.
 *
 * ef_search bounds the number of node expansions (the work budget).  While the
 * budget remains, the scan explores best-first and only emits a result once no
 * unexpanded candidate is closer (exact ordering).  Once the budget is spent it
 * stops exploring and drains the discovered results in nearest-first order.
 * Larger ef_search explores more of the predicate subgraph — higher recall and
 * more page reads; smaller ef_search stops sooner.  This is the counted
 * candidate budget that the Tier 1 bounded search (acorn_layer0_search) deferred
 * to Tier 2: it works where the standard "nearest candidate beyond worst result"
 * HNSW termination does not, because ACORN keeps near filter-failing nodes in C.
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
		double r_dist = pairingheap_is_empty(s->R) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->R))->elem.distance;
		double cf_dist = pairingheap_is_empty(s->C) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->C))->elem.distance;
		double cm_dist = pairingheap_is_empty(s->Cm) ? DBL_MAX
			: ((AcornPQNode *) pairingheap_first(s->Cm))->elem.distance;
		double c_dist = Min(cf_dist, cm_dist);

		/*
		 * Explore while a candidate may still beat the nearest result, but
		 * only within the ef_search expansion budget.
		 *
		 * Expansion choice: in member_first mode, spend the budget on the
		 * nearest PASSING candidate whenever one is queued — the payload
		 * subgraph keeps same-partition nodes connected, so this drains the
		 * predicate subgraph instead of the (mostly failing) global
		 * neighborhood.  Emission safety is unaffected: c_dist still covers
		 * both heaps, so a result is never emitted past a closer unexpanded
		 * candidate of either kind.
		 */
		if (c_dist != DBL_MAX && c_dist <= r_dist
			&& s->n_expansions < s->ef_search)
		{
			pairingheap *src;
			AcornPQNode *cn;
			AcornElem	 ce;

			if (s->member_first && !pairingheap_is_empty(s->Cm))
				src = s->Cm;
			else if (!pairingheap_is_empty(s->C) && cf_dist <= cm_dist)
				src = s->C;
			else
				src = s->Cm;

			cn = (AcornPQNode *) pairingheap_remove_first(src);
			ce = cn->elem;
			pfree(cn);
			acorn_t2_stream_expand(s, &ce);
			s->n_expansions++;
			continue;
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
			MemoryContextSwitchTo(old);
			return true;
		}
	}
}
