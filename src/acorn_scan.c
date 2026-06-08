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
#include "utils/memutils.h"
#include "utils/rel.h"
#include "access/tableam.h"

#include "acorn_scan.h"
#include "hnsw_compat.h"

/* -----------------------------------------------------------------------
 * Internal element reference
 * ----------------------------------------------------------------------- */

typedef struct AcornElem
{
	ItemPointerData indextid;	/* index page (blkno, offno) */
	ItemPointerData heaptid;	/* heap TID — for predicate evaluation */
	double			distance;
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

	for (int i = 0; i < count; i++)
	{
		if (!ItemPointerIsValid(&tids[start + i]))
		{
			count = i;
			break;
		}
		neighbors_out[i] = tids[start + i];
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
					Datum query, FmgrInfo *dist_proc,
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
			nd     = acorn_distance(index, nblkno, noffno, query, dist_proc);

			/* Always add to C (ACORN-1: preserve connectivity) */
			nc = palloc(sizeof(AcornPQNode));
			ItemPointerCopy(&neighbors[i], &nc->elem.indextid);
			nc->elem.distance = nd;
			ItemPointerSetInvalid(&nc->elem.heaptid);
			pairingheap_add(C, &nc->ph_node);

			/* Load heaptid to evaluate predicate */
			nbuf = acorn_load_element(index, nblkno, noffno,
									  &nheaptid, &nlevel, &ndeleted,
									  &n_nbr_blkno, &n_nbr_offno);
			UnlockReleaseBuffer(nbuf);

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
	BlockNumber		entry_blkno;
	OffsetNumber	entry_offno;
	int				entry_level;
	int				m;

	acorn_load_dist_proc(index, &dist_proc);

	if (!acorn_meta_read(index, &entry_blkno, &entry_offno, &entry_level, &m))
		return 0;			/* empty index */

	/* Greedy descent to layer 1 (skipping to base layer entry point) */
	if (entry_level > 0)
		acorn_greedy_descent(index, m, &entry_blkno, &entry_offno,
							 entry_level, 0, query, &dist_proc);

	return acorn_layer0_search(index, heap, m,
							   entry_blkno, entry_offno,
							   query, &dist_proc,
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
		Buffer			nbuf;
		AcornPQNode	   *cn;

		if (!ItemPointerIsValid(&neighbors[i]))
			continue;
		if (is_visited(s->visited, &neighbors[i]))
			continue;
		mark_visited(s->visited, &neighbors[i]);

		nblkno = ItemPointerGetBlockNumber(&neighbors[i]);
		noffno = ItemPointerGetOffsetNumber(&neighbors[i]);
		nd     = acorn_distance(s->index, nblkno, noffno, s->query, &s->dist_proc);

		/* Always a candidate for expansion (preserve connectivity) */
		cn = palloc(sizeof(AcornPQNode));
		ItemPointerCopy(&neighbors[i], &cn->elem.indextid);
		ItemPointerSetInvalid(&cn->elem.heaptid);
		cn->elem.distance = nd;
		pairingheap_add(s->C, &cn->ph_node);

		/* Emit only live nodes — load heaptid + deleted flag */
		nbuf = acorn_load_element(s->index, nblkno, noffno,
								  &nheaptid, &nlevel, &ndeleted,
								  &n_nbr_blkno, &n_nbr_offno);
		UnlockReleaseBuffer(nbuf);

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
