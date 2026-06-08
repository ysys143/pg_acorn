/*
 * acorn_build.c — ACORN-gamma index build + incremental insert (Tier 2)
 *
 * Writes index pages in pgvector 0.8.0's on-disk format (hnsw_compat.h) so
 * the shared traversal in acorn_scan.c reads them unchanged.  Design:
 *
 *   - Multi-layer HNSW graph.  Level assignment: l = floor(-ln(u) / ln(m_eff))
 *     where u ~ pg_erand48.  The entry point tracks the highest-level element.
 *     acorn_scan.c's greedy descent already handles multi-layer reads.
 *   - ACORN-gamma: store m_eff = m * gamma neighbors; record m_eff as meta->m.
 *     Layer 0 holds 2*m_eff connections; upper layers hold m_eff connections.
 *   - Neighbor selection uses a bounded ef_construction beam search at each
 *     layer (O(n log n) total), replacing the previous brute-force O(n²) scan.
 *   - Bidirectional edges with FIXED-SLOT RETRY: when a neighbor's slots are
 *     full, replace its furthest neighbor if the new node is closer.  The
 *     two-phase locking protocol in acorn_add_reverse_edge_at_layer avoids
 *     self-deadlock (element + neighbor tuples share pages).
 *
 * Vectors are stored as-is (no normalization); the distance kernel is opclass
 * support function 1, identical to what acorn_scan.c uses.
 */

#include "postgres.h"

#include <float.h>
#include <math.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/datum.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pg_acorn.h"
#include "acorn_am.h"
#include "hnsw_compat.h"

/*
 * Cap random levels so neighbor tuples remain page-sized.
 * With ACORN_MAX_LEVEL=6 and the maximum m_eff=100:
 *   HNSW_NEIGHBOR_TUPLE_SIZE(6, 100) = 4 + 8*100*6 = 4804 bytes < HNSW_MAX_SIZE
 */
#define ACORN_MAX_LEVEL  6

/* -----------------------------------------------------------------------
 * Options
 * ----------------------------------------------------------------------- */

static int
acorn_opt_m(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->m : ACORN_DEFAULT_M;
}

static int
acorn_opt_ef_construction(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->efConstruction : ACORN_DEFAULT_EF_CONSTRUCTION;
}

static int
acorn_opt_gamma(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->gamma : acorn_default_gamma;
}

/* Effective neighbor multiplier: m * gamma, capped so neighbor tuples fit a page */
static int
acorn_m_eff(Relation index)
{
	int m_eff = acorn_opt_m(index) * acorn_opt_gamma(index);
	if (m_eff > HNSW_MAX_M)
		m_eff = HNSW_MAX_M;
	return m_eff;
}

/* -----------------------------------------------------------------------
 * Distance kernel (opclass support function 1)
 * ----------------------------------------------------------------------- */

static FmgrInfo *
acorn_dist_proc(Relation index)
{
	return index_getprocinfo(index, 1, 1);
}

static inline double
acorn_dist(FmgrInfo *proc, Datum stored_vec, Datum query)
{
	return DatumGetFloat8(FunctionCall2Coll(proc, InvalidOid, stored_vec, query));
}

/* -----------------------------------------------------------------------
 * Level assignment
 * ----------------------------------------------------------------------- */

/*
 * Assign a random HNSW level: floor(-ln(u) / ln(m_eff)) where u ~ U(0,1).
 * Expected level = 1 / ln(m_eff) ≈ 0.36 for m_eff=16.
 * Capped at ACORN_MAX_LEVEL to guarantee neighbor tuples fit on a page.
 */
static int
acorn_assign_level(int m_eff, unsigned short rand_state[3])
{
	double r;
	int    level;

	if (m_eff <= 1)
		return 0;

	do {
		r = pg_erand48(rand_state);
	} while (r == 0.0);

	level = (int) floor(-log(r) / log((double) m_eff));
	return Min(level, ACORN_MAX_LEVEL);
}

/* -----------------------------------------------------------------------
 * Page helpers (mirror pgvector HnswNewBuffer / HnswInitPage / CreateMetaPage)
 * ----------------------------------------------------------------------- */

static Buffer
acorn_new_buffer(Relation index, ForkNumber forkNum)
{
	Buffer buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

static void
acorn_init_page(Buffer buf, Page page)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(HnswPageOpaqueData));
	HnswPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	HnswPageGetOpaque(page)->page_id   = HNSW_PAGE_ID;
}

static void
acorn_create_meta_page(Relation index, ForkNumber forkNum, int m_eff,
					   int efConstruction, int dimensions)
{
	Buffer		 buf;
	Page		 page;
	HnswMetaPage metap;

	buf  = acorn_new_buffer(index, forkNum);	/* becomes block 0 */
	page = BufferGetPage(buf);
	acorn_init_page(buf, page);

	metap                 = HnswPageGetMeta(page);
	metap->magicNumber    = HNSW_MAGIC_NUMBER;
	metap->version        = HNSW_VERSION;
	metap->dimensions     = dimensions;
	metap->m              = m_eff;
	metap->efConstruction = efConstruction;
	metap->entryBlkno     = InvalidBlockNumber;
	metap->entryOffno     = InvalidOffsetNumber;
	metap->entryLevel     = -1;
	metap->insertPage     = InvalidBlockNumber;

	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(HnswMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Read meta: returns m (= m_eff).  Optional out-params may be NULL.
 */
static int
acorn_read_meta(Relation index, BlockNumber *entry_blkno, OffsetNumber *entry_offno,
				int *entry_level_out)
{
	Buffer		 buf;
	Page		 page;
	HnswMetaPage metap;
	int			 m;

	buf   = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page  = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);
	m     = (int) metap->m;
	if (entry_blkno)
		*entry_blkno = metap->entryBlkno;
	if (entry_offno)
		*entry_offno = metap->entryOffno;
	if (entry_level_out)
		*entry_level_out = (int) metap->entryLevel;
	UnlockReleaseBuffer(buf);
	return m;
}

int
acorn_index_m(Relation index)
{
	return acorn_read_meta(index, NULL, NULL, NULL);
}

/*
 * Update the meta entry point to (blkno, offno, level) if no entry exists yet
 * or if the new element has a strictly higher level than the current entry.
 */
static void
acorn_maybe_update_entry(Relation index, BlockNumber blkno, OffsetNumber offno,
						 int level)
{
	Buffer		 buf;
	Page		 page;
	HnswMetaPage metap;

	buf   = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page  = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (!BlockNumberIsValid(metap->entryBlkno) || level > (int) metap->entryLevel)
	{
		metap->entryBlkno = blkno;
		metap->entryOffno = offno;
		metap->entryLevel = (int16) level;
		MarkBufferDirty(buf);
	}
	UnlockReleaseBuffer(buf);
}

/*
 * Append a tuple to the index, returning its on-disk location.
 * Tries the last page, extends the relation with a fresh page if needed.
 */
static void
acorn_append_tuple(Relation index, ForkNumber forkNum, Item tup, Size size,
				   BlockNumber *blkno_out, OffsetNumber *off_out)
{
	BlockNumber  nblocks = RelationGetNumberOfBlocksInFork(index, forkNum);
	Buffer		 buf;
	Page		 page;
	OffsetNumber off;

	/* block 0 is the meta page; data starts at block 1 */
	if (nblocks > 1)
	{
		buf  = ReadBufferExtended(index, forkNum, nblocks - 1, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		if (PageGetFreeSpace(page) >= size)
		{
			off = PageAddItem(page, tup, size, InvalidOffsetNumber, false, false);
			if (off == InvalidOffsetNumber)
				elog(ERROR, "acorn_hnsw: failed to add index tuple");
			MarkBufferDirty(buf);
			*blkno_out = nblocks - 1;
			*off_out   = off;
			UnlockReleaseBuffer(buf);
			return;
		}
		UnlockReleaseBuffer(buf);
	}

	LockRelationForExtension(index, ExclusiveLock);
	buf = acorn_new_buffer(index, forkNum);
	UnlockRelationForExtension(index, ExclusiveLock);
	page = BufferGetPage(buf);
	acorn_init_page(buf, page);

	off = PageAddItem(page, tup, size, InvalidOffsetNumber, false, false);
	if (off == InvalidOffsetNumber)
		elog(ERROR, "acorn_hnsw: tuple too large for page");
	MarkBufferDirty(buf);
	*blkno_out = BufferGetBlockNumber(buf);
	*off_out   = off;
	UnlockReleaseBuffer(buf);
}

/* -----------------------------------------------------------------------
 * Element / neighbor reading helpers
 * ----------------------------------------------------------------------- */

/*
 * Read an element tuple; optionally return level, neighbor tuple location,
 * and a palloc'd copy of the inline vector.  All out-params may be NULL.
 */
static void
acorn_read_element(Relation index, ForkNumber forkNum, ItemPointer tid,
				   int *level_out,
				   BlockNumber *nbr_blkno, OffsetNumber *nbr_offno,
				   Datum *vec_copy)
{
	Buffer			 buf;
	Page			 page;
	HnswElementTuple etup;
	Pointer			 vec;
	Size			 vsize;

	buf  = ReadBufferExtended(index, forkNum,
							  ItemPointerGetBlockNumber(tid), RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (HnswElementTuple) PageGetItem(page,
				PageGetItemId(page, ItemPointerGetOffsetNumber(tid)));

	if (level_out)
		*level_out = (int) etup->level;
	if (nbr_blkno)
		*nbr_blkno = ItemPointerGetBlockNumber(&etup->neighbortid);
	if (nbr_offno)
		*nbr_offno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	if (vec_copy)
	{
		vec   = (Pointer) HnswElementTupleGetVector(etup);
		vsize = VARSIZE_ANY(vec);
		*vec_copy = PointerGetDatum(memcpy(palloc(vsize), vec, vsize));
	}
	UnlockReleaseBuffer(buf);
}

/* Compute distance from the inline vector of (blkno, offno) to query. */
static double
acorn_node_distance(Relation index, ForkNumber forkNum,
					BlockNumber blkno, OffsetNumber offno,
					Datum query, FmgrInfo *proc)
{
	Buffer			 buf;
	Page			 page;
	HnswElementTuple etup;
	double			 dist;

	buf  = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	dist = acorn_dist(proc, PointerGetDatum(HnswElementTupleGetVector(etup)), query);
	UnlockReleaseBuffer(buf);
	return dist;
}

/*
 * Read neighbor TIDs at a specific layer for an element whose neighbor tuple
 * is at (nbr_blkno, nbr_offno).  elem_level is the element's own highest layer
 * (needed to compute the slot start offset via HnswNeighborStart).
 * Returns count of valid TIDs written to tids_out.
 */
static int
acorn_read_nbr_tids_at_layer(Relation index, ForkNumber forkNum,
							  BlockNumber nbr_blkno, OffsetNumber nbr_offno,
							  int elem_level, int layer, int m_eff,
							  ItemPointerData *tids_out, int max_tids)
{
	Buffer			  buf;
	Page			  page;
	HnswNeighborTuple ntup;
	ItemPointerData  *tids;
	int				  start = HnswNeighborStart(m_eff, elem_level, layer);
	int				  count = Min(HnswNeighborCount(m_eff, layer), max_tids);
	int				  n     = 0;

	buf  = ReadBufferExtended(index, forkNum, nbr_blkno, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, nbr_offno));
	tids = HnswNeighborTupleGetTids(ntup);

	for (int i = 0; i < count; i++)
	{
		if (!ItemPointerIsValid(&tids[start + i]))
			break;
		tids_out[n++] = tids[start + i];
	}

	UnlockReleaseBuffer(buf);
	return n;
}

/* -----------------------------------------------------------------------
 * Construction beam search infrastructure (pairingheap + visited set)
 * ----------------------------------------------------------------------- */

typedef struct BuildPQNode
{
	pairingheap_node ph_node;
	ItemPointerData  tid;
	double			 distance;
} BuildPQNode;

/* max-heap: furthest element at top (W = working result set) */
static int
build_cmp_max(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	double da = ((const BuildPQNode *) a)->distance;
	double db = ((const BuildPQNode *) b)->distance;
	return (da > db) ? 1 : (da < db) ? -1 : 0;
}

/* min-heap: closest element at top (C = candidate set) */
static int
build_cmp_min(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	return -build_cmp_max(a, b, arg);
}

static HTAB *
build_create_visited(void)
{
	HASHCTL info;

	memset(&info, 0, sizeof(info));
	info.keysize   = sizeof(ItemPointerData);
	info.entrysize = sizeof(ItemPointerData) + 1;
	return hash_create("acorn_build_visited", 512, &info,
					   HASH_ELEM | HASH_BLOBS);
}

static bool
build_is_visited(HTAB *visited, const ItemPointerData *tid)
{
	bool found;
	hash_search(visited, tid, HASH_FIND, &found);
	return found;
}

static void
build_mark_visited(HTAB *visited, const ItemPointerData *tid)
{
	bool found;
	hash_search(visited, (void *) tid, HASH_ENTER, &found);
}

/* -----------------------------------------------------------------------
 * Greedy descent for build (ef=1 upper-layer traversal)
 * ----------------------------------------------------------------------- */

/*
 * Greedy descent: from (cur_blkno, cur_offno) at from_level, descend to
 * to_level+1 following single-closest neighbors at each layer.
 * Updates *cur_blkno / *cur_offno in place.
 *
 * This is the "zoom-in" phase of HNSW insertion: find the best entry point
 * at layer to_level before the full ef_construction beam search.
 */
static void
acorn_greedy_descend_build(Relation index, ForkNumber forkNum,
						   BlockNumber *cur_blkno, OffsetNumber *cur_offno,
						   Datum query, FmgrInfo *proc, int m_eff,
						   int from_level, int to_level)
{
	for (int lc = from_level; lc > to_level; lc--)
	{
		bool improved;

		do {
			ItemPointerData c_tid;
			int				c_level;
			BlockNumber		c_nbr_blk;
			OffsetNumber	c_nbr_off;
			ItemPointerData nbrs[HNSW_MAX_NEIGHBORS];
			int				n_nbrs;
			double			cur_dist;

			improved = false;
			cur_dist = acorn_node_distance(index, forkNum,
										   *cur_blkno, *cur_offno,
										   query, proc);
			ItemPointerSet(&c_tid, *cur_blkno, *cur_offno);
			acorn_read_element(index, forkNum, &c_tid, &c_level,
							   &c_nbr_blk, &c_nbr_off, NULL);

			n_nbrs = acorn_read_nbr_tids_at_layer(index, forkNum,
												   c_nbr_blk, c_nbr_off,
												   c_level, lc, m_eff,
												   nbrs, HNSW_MAX_NEIGHBORS);

			for (int i = 0; i < n_nbrs; i++)
			{
				double nd = acorn_node_distance(index, forkNum,
												 ItemPointerGetBlockNumber(&nbrs[i]),
												 ItemPointerGetOffsetNumber(&nbrs[i]),
												 query, proc);
				if (nd < cur_dist)
				{
					cur_dist   = nd;
					*cur_blkno = ItemPointerGetBlockNumber(&nbrs[i]);
					*cur_offno = ItemPointerGetOffsetNumber(&nbrs[i]);
					improved   = true;
				}
			}
		} while (improved);
	}
}

/* -----------------------------------------------------------------------
 * Bounded beam search at a single layer for construction
 * ----------------------------------------------------------------------- */

/*
 * (tid, distance) pair used to return sorted candidates from beam search.
 */
typedef struct AcornCand
{
	ItemPointerData tid;
	double			distance;
} AcornCand;

/*
 * Bounded beam search at `layer` for construction (no predicate).
 *
 * Implements the standard HNSW Algorithm 2 "SEARCH-LAYER" with ef-bounded
 * candidate set.  Returns up to ef candidates in out_cands[] sorted
 * nearest-first; out_cands must have ef slots.  Returns actual count.
 */
static int
acorn_search_layer_construction(Relation index, ForkNumber forkNum,
								  BlockNumber entry_blkno, OffsetNumber entry_offno,
								  Datum query, FmgrInfo *proc, int m_eff,
								  int layer, int ef,
								  AcornCand *out_cands)
{
	pairingheap    *C;			/* min-heap: candidates (closest at top) */
	pairingheap    *W;			/* max-heap: current best set (furthest at top) */
	HTAB		   *visited;
	int				W_count = 0;
	MemoryContext	tmp_ctx, old_ctx;
	int				n_out;

	tmp_ctx = AllocSetContextCreate(CurrentMemoryContext,
									"acorn_build_search",
									ALLOCSET_DEFAULT_SIZES);
	old_ctx = MemoryContextSwitchTo(tmp_ctx);

	C       = pairingheap_allocate(build_cmp_min, NULL);
	W       = pairingheap_allocate(build_cmp_max, NULL);
	visited = build_create_visited();

	/* Seed with entry point */
	{
		ItemPointerData ep_tid;
		double			ep_dist;
		BuildPQNode    *cn, *wn;

		ItemPointerSet(&ep_tid, entry_blkno, entry_offno);
		build_mark_visited(visited, &ep_tid);
		ep_dist = acorn_node_distance(index, forkNum,
									   entry_blkno, entry_offno,
									   query, proc);

		cn           = palloc(sizeof(BuildPQNode));
		cn->tid      = ep_tid;
		cn->distance = ep_dist;
		pairingheap_add(C, &cn->ph_node);

		wn           = palloc(sizeof(BuildPQNode));
		wn->tid      = ep_tid;
		wn->distance = ep_dist;
		pairingheap_add(W, &wn->ph_node);
		W_count = 1;
	}

	while (!pairingheap_is_empty(C))
	{
		BuildPQNode    *c_node;
		double			c_dist;
		ItemPointerData c_tid;
		double			f_dist;
		int				c_level;
		BlockNumber		c_nbr_blk;
		OffsetNumber	c_nbr_off;
		ItemPointerData nbrs[HNSW_MAX_NEIGHBORS];
		int				n_nbrs;

		c_node = (BuildPQNode *) pairingheap_remove_first(C);
		c_dist = c_node->distance;
		c_tid  = c_node->tid;
		pfree(c_node);

		/* HNSW termination: candidate farther than worst result and W is full */
		f_dist = ((BuildPQNode *) pairingheap_first(W))->distance;
		if (c_dist > f_dist && W_count >= ef)
			break;

		/* Load c's level + neighbor tuple location (no lock held at this point) */
		acorn_read_element(index, forkNum, &c_tid, &c_level,
						   &c_nbr_blk, &c_nbr_off, NULL);

		n_nbrs = acorn_read_nbr_tids_at_layer(index, forkNum,
											   c_nbr_blk, c_nbr_off,
											   c_level, layer, m_eff,
											   nbrs, HNSW_MAX_NEIGHBORS);

		for (int i = 0; i < n_nbrs; i++)
		{
			double		 nd;
			BuildPQNode *cn, *wn;

			if (!ItemPointerIsValid(&nbrs[i]))
				continue;
			if (build_is_visited(visited, &nbrs[i]))
				continue;
			build_mark_visited(visited, &nbrs[i]);

			nd = acorn_node_distance(index, forkNum,
									  ItemPointerGetBlockNumber(&nbrs[i]),
									  ItemPointerGetOffsetNumber(&nbrs[i]),
									  query, proc);

			f_dist = ((BuildPQNode *) pairingheap_first(W))->distance;
			if (nd < f_dist || W_count < ef)
			{
				cn           = palloc(sizeof(BuildPQNode));
				cn->tid      = nbrs[i];
				cn->distance = nd;
				pairingheap_add(C, &cn->ph_node);

				wn           = palloc(sizeof(BuildPQNode));
				wn->tid      = nbrs[i];
				wn->distance = nd;
				pairingheap_add(W, &wn->ph_node);
				W_count++;

				if (W_count > ef)
				{
					BuildPQNode *evicted =
						(BuildPQNode *) pairingheap_remove_first(W);
					pfree(evicted);
					W_count--;
				}
			}
		}
	}

	/*
	 * Extract W into out_cands sorted nearest-first.
	 * W is a max-heap so pairingheap_remove_first gives the furthest; fill
	 * the output array back-to-front so out_cands[0] = nearest.
	 */
	n_out = W_count;
	for (int i = W_count - 1; i >= 0; i--)
	{
		BuildPQNode *wn = (BuildPQNode *) pairingheap_remove_first(W);
		out_cands[i].tid      = wn->tid;
		out_cands[i].distance = wn->distance;
		pfree(wn);
	}

	/* Drain remaining candidates from C */
	while (!pairingheap_is_empty(C))
	{
		BuildPQNode *cn = (BuildPQNode *) pairingheap_remove_first(C);
		pfree(cn);
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(tmp_ctx);
	return n_out;
}

/* -----------------------------------------------------------------------
 * Fixed-slot retry reverse edge (generalized to any layer)
 * ----------------------------------------------------------------------- */

/*
 * Add element E (e_blk, e_off) as a neighbor of N (n_tid) at `layer`.
 * If N's layer slots are full, replace N's furthest neighbor when E is closer.
 *
 * IMPORTANT: two-phase locking to avoid self-deadlock on the buffer LWLock.
 * Element and neighbor tuples share pages.  Never read another page while
 * holding a lock on the neighbor page:
 *   Phase 1: read N's element + neighbor slots under SHARE (copy + release).
 *   Phase 2: write the chosen slot under EXCLUSIVE.
 * Build/insert is serialized per index so the phase-1 snapshot remains valid.
 */
static void
acorn_add_reverse_edge_at_layer(Relation index, ForkNumber forkNum,
								 ItemPointer n_tid,
								 BlockNumber e_blk, OffsetNumber e_off,
								 Datum e_value, FmgrInfo *proc,
								 int m_eff, int layer)
{
	int				  n_level;
	int				  layer_m = HnswGetLayerM(m_eff, layer);
	BlockNumber		  n_nbr_blk;
	OffsetNumber	  n_nbr_off;
	Datum			  n_vec;
	Buffer			  buf;
	Page			  page;
	HnswNeighborTuple ntup;
	ItemPointerData  *tids;
	ItemPointerData	  tids_copy[HNSW_MAX_NEIGHBORS];
	int				  start;
	int				  len    = 0;
	int				  target = -1;

	Assert(layer_m <= HNSW_MAX_NEIGHBORS);

	/* --- Phase 1: read N's element tuple (level + neighbor location + vector) --- */
	acorn_read_element(index, forkNum, n_tid, &n_level,
					   &n_nbr_blk, &n_nbr_off, &n_vec);
	start = HnswNeighborStart(m_eff, n_level, layer);

	/* --- Phase 1a: copy N's layer slots under SHARE (release before any write) --- */
	buf  = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
	tids = HnswNeighborTupleGetTids(ntup);
	memcpy(tids_copy, &tids[start], layer_m * sizeof(ItemPointerData));
	UnlockReleaseBuffer(buf);

	/* Find the first empty slot; bail if E is already connected */
	for (len = 0; len < layer_m; len++)
	{
		if (!ItemPointerIsValid(&tids_copy[len]))
			break;
		if (ItemPointerGetBlockNumber(&tids_copy[len]) == e_blk &&
			ItemPointerGetOffsetNumber(&tids_copy[len]) == e_off)
		{
			pfree(DatumGetPointer(n_vec));
			return;
		}
	}

	if (len < layer_m)
	{
		/* Free slot available */
		target = len;
	}
	else
	{
		/* --- Phase 1b: full — retry: find furthest, replace if E is closer --- */
		double furthest_d = -DBL_MAX;
		int    furthest_j = -1;
		double dEN        = acorn_dist(proc, n_vec, e_value);

		for (int j = 0; j < layer_m; j++)
		{
			Datum  vj;
			double d;

			acorn_read_element(index, forkNum, &tids_copy[j],
							   NULL, NULL, NULL, &vj);
			d = acorn_dist(proc, n_vec, vj);
			pfree(DatumGetPointer(vj));
			if (d > furthest_d)
			{
				furthest_d = d;
				furthest_j = j;
			}
		}

		if (dEN < furthest_d)
			target = furthest_j;
	}

	pfree(DatumGetPointer(n_vec));

	if (target < 0)
		return;		/* E is not closer than any existing neighbor */

	/* --- Phase 2: write the chosen slot under EXCLUSIVE --- */
	buf  = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
	tids = HnswNeighborTupleGetTids(ntup);
	ItemPointerSet(&tids[start + target], e_blk, e_off);
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/* -----------------------------------------------------------------------
 * Insert one element (shared by build and aminsert)
 * ----------------------------------------------------------------------- */

/*
 * Full multi-layer HNSW insert (HNSW Algorithm 1 "INSERT"):
 *
 *  1. Assign random level l_new = floor(-ln(u) / ln(m_eff)).
 *  2. Greedy-descend from the current entry level to l_new+1 (ef=1).
 *  3. For each layer lc from min(entry_level, l_new) down to 0:
 *     a. Beam-search with ef_construction candidates.
 *     b. Select min(layer_m, ncands) nearest as E's neighbors at lc.
 *     c. Update entry point to the closest candidate (for the next layer).
 *  4. Write neighbor tuple ((l_new+2)*m_eff slots) then element tuple.
 *  5. Update meta entry point if l_new > entry_level.
 *  6. Add reverse edges at each layer with fixed-slot retry.
 */
static void
acorn_insert_element(Relation index, ForkNumber forkNum, Datum value,
					 ItemPointer heaptid, unsigned short rand_state[3])
{
	int				  m_eff    = acorn_m_eff(index);
	int				  efc      = acorn_opt_ef_construction(index);
	FmgrInfo		 *proc     = acorn_dist_proc(index);
	Size			  vsize    = VARSIZE_ANY(DatumGetPointer(value));
	Size			  etupSize = HNSW_ELEMENT_TUPLE_SIZE(vsize);

	int				  l_new;
	Size			  ntupSize;
	HnswNeighborTuple ntup;
	HnswElementTuple  etup;
	ItemPointerData  *ntids;
	BlockNumber		  e_blk, n_blk;
	OffsetNumber	  e_off, n_off;
	BlockNumber		  entry_blkno;
	OffsetNumber	  entry_offno;
	int				  entry_level;
	bool			  has_entry;

	/* Step 1: assign random level */
	l_new = acorn_assign_level(m_eff, rand_state);

	/* Step 2: read current entry point */
	acorn_read_meta(index, &entry_blkno, &entry_offno, &entry_level);
	has_entry = BlockNumberIsValid(entry_blkno);

	/*
	 * Step 3: allocate neighbor tuple.
	 * Total slots = (l_new + 2) * m_eff: layers l_new..1 use m_eff each,
	 * layer 0 uses 2*m_eff.  HnswGetLayerM / HnswNeighborStart encode this.
	 */
	ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(l_new, m_eff);
	Assert(ntupSize <= HNSW_MAX_SIZE);
	ntup           = palloc0(ntupSize);
	ntup->type     = HNSW_NEIGHBOR_TUPLE_TYPE;
	ntup->version  = HNSW_VERSION;
	ntup->count    = (l_new + 2) * m_eff;
	ntids = HnswNeighborTupleGetTids(ntup);
	for (int i = 0; i < (l_new + 2) * m_eff; i++)
		ItemPointerSetInvalid(&ntids[i]);

	/* Step 4: find neighbors at each layer */
	if (has_entry)
	{
		BlockNumber  ep_blkno = entry_blkno;
		OffsetNumber ep_offno = entry_offno;

		/* Greedy descent from entry_level down to l_new+1 (ef=1 per layer) */
		if (entry_level > l_new)
			acorn_greedy_descend_build(index, forkNum, &ep_blkno, &ep_offno,
									   value, proc, m_eff,
									   entry_level, l_new);

		/* Beam search from min(entry_level, l_new) down to 0 */
		for (int lc = Min(entry_level, l_new); lc >= 0; lc--)
		{
			int		   layer_m = HnswGetLayerM(m_eff, lc);
			int		   start   = HnswNeighborStart(m_eff, l_new, lc);
			int		   n_cands;
			AcornCand *cands   = palloc(sizeof(AcornCand) * efc);

			n_cands = acorn_search_layer_construction(index, forkNum,
													   ep_blkno, ep_offno,
													   value, proc, m_eff,
													   lc, efc, cands);

			/* Store min(layer_m, n_cands) nearest as E's neighbors at layer lc */
			for (int i = 0; i < Min(n_cands, layer_m); i++)
				ntids[start + i] = cands[i].tid;

			/* Use closest result as entry point for the next (lower) layer */
			if (n_cands > 0)
			{
				ep_blkno = ItemPointerGetBlockNumber(&cands[0].tid);
				ep_offno = ItemPointerGetOffsetNumber(&cands[0].tid);
			}
			pfree(cands);
		}
	}

	/* Step 5: write neighbor tuple to disk */
	acorn_append_tuple(index, forkNum, (Item) ntup, ntupSize, &n_blk, &n_off);

	/* Step 6: write element tuple (level, heaptids, neighbortid, inline vector) */
	etup           = palloc0(etupSize);
	etup->type     = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level    = (uint8) l_new;
	etup->deleted  = 0;
	etup->version  = HNSW_VERSION;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&etup->heaptids[i]);
	etup->heaptids[0] = *heaptid;
	ItemPointerSet(&etup->neighbortid, n_blk, n_off);
	etup->unused   = 0;
	memcpy(HnswElementTupleGetVector(etup), DatumGetPointer(value), vsize);

	acorn_append_tuple(index, forkNum, (Item) etup, etupSize, &e_blk, &e_off);

	/* Step 7: update meta entry point if this element has the highest level */
	acorn_maybe_update_entry(index, e_blk, e_off, l_new);

	/* Step 8: add reverse edges at each layer with fixed-slot retry */
	if (has_entry)
	{
		for (int lc = Min(entry_level, l_new); lc >= 0; lc--)
		{
			int layer_m = HnswGetLayerM(m_eff, lc);
			int start   = HnswNeighborStart(m_eff, l_new, lc);

			for (int i = 0; i < layer_m; i++)
			{
				if (!ItemPointerIsValid(&ntids[start + i]))
					break;
				acorn_add_reverse_edge_at_layer(index, forkNum,
												 &ntids[start + i],
												 e_blk, e_off,
												 value, proc, m_eff, lc);
			}
		}
	}

	pfree(ntup);
	pfree(etup);
}

/* -----------------------------------------------------------------------
 * Build callback + entry points
 * ----------------------------------------------------------------------- */

typedef struct AcornBuildState
{
	ForkNumber		forkNum;
	double			ntuples;
	MemoryContext	tmpCtx;
	unsigned short	rand_state[3];	/* pg_erand48 state for level assignment */
} AcornBuildState;

static void
acorn_build_callback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *state)
{
	AcornBuildState *bs = (AcornBuildState *) state;
	MemoryContext	 oldCtx;
	Datum			 value;

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(bs->tmpCtx);

	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	acorn_insert_element(index, bs->forkNum, value, tid, bs->rand_state);
	bs->ntuples += 1;

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(bs->tmpCtx);
}

static void
acorn_build_internal(Relation heap, Relation index, IndexInfo *indexInfo,
					 ForkNumber forkNum, double *heap_tuples, double *index_tuples)
{
	int				m_eff = acorn_m_eff(index);
	int				efc   = acorn_opt_ef_construction(index);
	int				dims  = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;
	AcornBuildState bs;
	double			reltuples = 0;

	if (dims < 0)
		dims = 0;

	acorn_create_meta_page(index, forkNum, m_eff, efc, dims);

	bs.forkNum       = forkNum;
	bs.ntuples       = 0;
	bs.tmpCtx        = AllocSetContextCreate(CurrentMemoryContext,
											  "acorn build temp",
											  ALLOCSET_DEFAULT_SIZES);
	/* Seed from process ID; gives different level sequences per build */
	bs.rand_state[0] = (unsigned short) (MyProcPid & 0xFFFF);
	bs.rand_state[1] = (unsigned short) (MyProcPid >> 16);
	bs.rand_state[2] = 0x1234;

	if (heap != NULL)
		reltuples = table_index_build_scan(heap, index, indexInfo, true, true,
										   acorn_build_callback, (void *) &bs, NULL);

	MemoryContextDelete(bs.tmpCtx);

	if (heap_tuples)
		*heap_tuples  = reltuples;
	if (index_tuples)
		*index_tuples = bs.ntuples;
}

IndexBuildResult *
acorn_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	double			  heap_tuples = 0, index_tuples = 0;

	acorn_build_internal(heap, index, indexInfo, MAIN_FORKNUM,
						  &heap_tuples, &index_tuples);

	result->heap_tuples  = heap_tuples;
	result->index_tuples = index_tuples;
	return result;
}

void
acorn_buildempty(Relation index)
{
	IndexInfo *indexInfo = BuildIndexInfo(index);

	acorn_build_internal(NULL, index, indexInfo, INIT_FORKNUM, NULL, NULL);
}

bool
acorn_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			 Relation heap, IndexUniqueCheck checkUnique,
			 bool indexUnchanged, IndexInfo *indexInfo)
{
	MemoryContext  oldCtx, insertCtx;
	Datum		   value;
	unsigned short rand_state[3];

	if (isnull[0])
		return false;

	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "acorn insert temp",
									   ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	/*
	 * Seed PRNG from heap TID so each row gets a deterministic level even when
	 * inserts are replayed (e.g. during recovery).
	 */
	rand_state[0] = (unsigned short) (ItemPointerGetBlockNumber(heap_tid) & 0xFFFF);
	rand_state[1] = (unsigned short) (ItemPointerGetBlockNumber(heap_tid) >> 16);
	rand_state[2] = (unsigned short)  ItemPointerGetOffsetNumber(heap_tid);

	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	acorn_insert_element(index, MAIN_FORKNUM, value, heap_tid, rand_state);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
