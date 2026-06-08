/*
 * acorn_build.c — ACORN-gamma index build + incremental insert (Tier 2)
 *
 * Writes index pages in pgvector 0.8.0's on-disk format (hnsw_compat.h) so the
 * shared traversal in acorn_scan.c reads them unchanged.  Design:
 *
 *   - Single-layer graph (all nodes at level 0).  The reader's greedy descent
 *     is a no-op (entryLevel = 0) and layer-0 search does the work.
 *   - ACORN-gamma: store m_eff = m * gamma neighbors; record m_eff as meta->m.
 *     Layer 0 therefore holds HnswGetLayerM(m_eff, 0) = 2*m_eff connections.
 *   - Neighbor selection is brute force (exact nearest among inserted nodes).
 *     O(n^2) build — fine for the test/benchmark sizes here and gives higher
 *     recall than approximate construction.
 *   - Bidirectional edges with FIXED-SLOT RETRY: when a neighbor's slots are
 *     full, replace its furthest neighbor if the new node is closer.  This is
 *     the algorithm unit-tested in test/unit/test_acorn_build.c and fixes
 *     pgvector's "TODO Retry updating connections if not".
 *
 * Vectors are stored as-is (no cosine normalization); the distance kernel is
 * the opclass support function 1, identical to what acorn_scan.c uses, so the
 * graph is internally consistent under whichever metric the opclass defines.
 */

#include "postgres.h"

#include <float.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pg_acorn.h"
#include "acorn_am.h"
#include "hnsw_compat.h"

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

/* Effective neighbor multiplier: m * gamma, capped so 2*m_eff fits scan buffers */
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
	/* Support proc 1, attribute 1 — the distance function */
	return index_getprocinfo(index, 1, 1);
}

/* distance(stored_vec_ptr, query_value) following acorn_scan.c's arg order */
static inline double
acorn_dist(FmgrInfo *proc, Datum stored_vec, Datum query)
{
	return DatumGetFloat8(FunctionCall2Coll(proc, InvalidOid, stored_vec, query));
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
	HnswPageGetOpaque(page)->page_id = HNSW_PAGE_ID;
}

static void
acorn_create_meta_page(Relation index, ForkNumber forkNum, int m_eff,
					   int efConstruction, int dimensions)
{
	Buffer			buf;
	Page			page;
	HnswMetaPage	metap;

	buf = acorn_new_buffer(index, forkNum);	/* becomes block 0 */
	page = BufferGetPage(buf);
	acorn_init_page(buf, page);

	metap = HnswPageGetMeta(page);
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

/* Read meta: returns m (= m_eff); *entry_blkno set to entry point (may be Invalid) */
static int
acorn_read_meta(Relation index, BlockNumber *entry_blkno, OffsetNumber *entry_offno)
{
	Buffer			buf;
	Page			page;
	HnswMetaPage	metap;
	int				m;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);
	m = (int) metap->m;
	if (entry_blkno)
		*entry_blkno = metap->entryBlkno;
	if (entry_offno)
		*entry_offno = metap->entryOffno;
	UnlockReleaseBuffer(buf);
	return m;
}

int
acorn_index_m(Relation index)
{
	return acorn_read_meta(index, NULL, NULL);
}

/* Set the meta entry point to (blkno, offno) at level 0 if currently unset */
static void
acorn_set_entry_if_unset(Relation index, BlockNumber blkno, OffsetNumber offno)
{
	Buffer			buf;
	Page			page;
	HnswMetaPage	metap;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	metap = HnswPageGetMeta(page);

	if (!BlockNumberIsValid(metap->entryBlkno))
	{
		metap->entryBlkno = blkno;
		metap->entryOffno = offno;
		metap->entryLevel = 0;
		MarkBufferDirty(buf);
	}
	UnlockReleaseBuffer(buf);
}

/*
 * Append a tuple to the index, returning its location.  Tries the last page,
 * else extends the relation with a fresh page.
 */
static void
acorn_append_tuple(Relation index, ForkNumber forkNum, Item tup, Size size,
				   BlockNumber *blkno_out, OffsetNumber *off_out)
{
	BlockNumber nblocks = RelationGetNumberOfBlocksInFork(index, forkNum);
	Buffer		buf;
	Page		page;
	OffsetNumber off;

	/* block 0 is the meta page; data starts at block 1 */
	if (nblocks > 1)
	{
		buf = ReadBufferExtended(index, forkNum, nblocks - 1, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		if (PageGetFreeSpace(page) >= size)
		{
			off = PageAddItem(page, tup, size, InvalidOffsetNumber, false, false);
			if (off == InvalidOffsetNumber)
				elog(ERROR, "acorn_hnsw: failed to add index tuple");
			MarkBufferDirty(buf);
			*blkno_out = nblocks - 1;
			*off_out = off;
			UnlockReleaseBuffer(buf);
			return;
		}
		UnlockReleaseBuffer(buf);
	}

	/* Need a new page */
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
	*off_out = off;
	UnlockReleaseBuffer(buf);
}

/* -----------------------------------------------------------------------
 * Nearest-neighbor collection (brute force over inserted elements)
 * ----------------------------------------------------------------------- */

typedef struct AcornCand
{
	ItemPointerData tid;		/* index TID of the element */
	double			distance;
} AcornCand;

static int
acorn_cmp_cand(const void *a, const void *b)
{
	double da = ((const AcornCand *) a)->distance;
	double db = ((const AcornCand *) b)->distance;
	return (da < db) ? -1 : (da > db) ? 1 : 0;
}

/*
 * Collect distances from `query` to every live element in the index.
 * Returns a palloc'd array (caller pfrees) and sets *n.
 */
static AcornCand *
acorn_collect(Relation index, ForkNumber forkNum, Datum query,
			  FmgrInfo *proc, int *n)
{
	BlockNumber nblocks = RelationGetNumberOfBlocksInFork(index, forkNum);
	AcornCand  *cands = NULL;
	int			count = 0;
	int			cap = 0;

	for (BlockNumber blk = 1; blk < nblocks; blk++)
	{
		Buffer		buf = ReadBufferExtended(index, forkNum, blk, RBM_NORMAL, NULL);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);

		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId			iid = PageGetItemId(page, off);
			HnswElementTuple etup;

			if (!ItemIdIsUsed(iid))
				continue;
			etup = (HnswElementTuple) PageGetItem(page, iid);
			if (etup->type != HNSW_ELEMENT_TUPLE_TYPE || etup->deleted)
				continue;

			if (count == cap)
			{
				cap = cap ? cap * 2 : 64;
				cands = cands ? repalloc(cands, cap * sizeof(AcornCand))
							  : palloc(cap * sizeof(AcornCand));
			}
			ItemPointerSet(&cands[count].tid, blk, off);
			cands[count].distance =
				acorn_dist(proc, PointerGetDatum(HnswElementTupleGetVector(etup)), query);
			count++;
		}
		UnlockReleaseBuffer(buf);
	}

	*n = count;
	return cands;
}

/* Read an element's neighbor-tuple location and copy its vector into palloc'd mem */
static void
acorn_read_element(Relation index, ForkNumber forkNum, ItemPointer tid,
				   BlockNumber *nbr_blkno, OffsetNumber *nbr_offno,
				   Datum *vec_copy)
{
	Buffer			buf;
	Page			page;
	HnswElementTuple etup;
	Pointer			vec;
	Size			vsize;

	buf = ReadBufferExtended(index, forkNum,
							 ItemPointerGetBlockNumber(tid), RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (HnswElementTuple) PageGetItem(page, PageGetItemId(page, ItemPointerGetOffsetNumber(tid)));

	if (nbr_blkno)
		*nbr_blkno = ItemPointerGetBlockNumber(&etup->neighbortid);
	if (nbr_offno)
		*nbr_offno = ItemPointerGetOffsetNumber(&etup->neighbortid);

	if (vec_copy)
	{
		vec = (Pointer) HnswElementTupleGetVector(etup);
		vsize = VARSIZE_ANY(vec);
		*vec_copy = PointerGetDatum(memcpy(palloc(vsize), vec, vsize));
	}
	UnlockReleaseBuffer(buf);
}

/* -----------------------------------------------------------------------
 * Fixed-slot retry reverse edge
 * ----------------------------------------------------------------------- */

/*
 * Add element E (e_blk,e_off) as a neighbor of N (n_tid).  If N's layer-0
 * slots are full, replace N's furthest neighbor when E is closer (retry).
 *
 * IMPORTANT: we must never read another element/neighbor page while holding a
 * lock on N's neighbor page — element and neighbor tuples share pages, so that
 * would self-deadlock on the buffer LWLock.  So this runs in two phases:
 *   1. read N's neighbor tuple (SHARE, copy + release), then read whatever
 *      vectors we need (SHARE, released) to pick the target slot;
 *   2. take EXCLUSIVE on N's neighbor page only to write the single slot.
 * Build/insert is serialized per index, so the snapshot from phase 1 is still
 * valid in phase 2.
 */
static void
acorn_add_reverse_edge(Relation index, ForkNumber forkNum, ItemPointer n_tid,
					   BlockNumber e_blk, OffsetNumber e_off, Datum e_value,
					   FmgrInfo *proc, int m_eff)
{
	int				layer0_m = HnswGetLayerM(m_eff, 0);
	BlockNumber		n_nbr_blk;
	OffsetNumber	n_nbr_off;
	Datum			n_vec;
	Buffer			buf;
	Page			page;
	HnswNeighborTuple ntup;
	ItemPointerData *tids;
	ItemPointerData	tids_copy[HNSW_MAX_NEIGHBORS];
	int				len = 0;
	int				target = -1;

	/* --- read N's element (vector + neighbor location) --- */
	acorn_read_element(index, forkNum, n_tid, &n_nbr_blk, &n_nbr_off, &n_vec);

	/* --- phase 1a: copy N's neighbor slots under a SHARE lock --- */
	buf = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
	tids = HnswNeighborTupleGetTids(ntup);
	memcpy(tids_copy, tids, layer0_m * sizeof(ItemPointerData));
	UnlockReleaseBuffer(buf);

	/* Slots fill contiguously, so the first invalid slot is the length. */
	for (len = 0; len < layer0_m; len++)
	{
		if (!ItemPointerIsValid(&tids_copy[len]))
			break;
		/* Already connected? */
		if (ItemPointerGetBlockNumber(&tids_copy[len]) == e_blk &&
			ItemPointerGetOffsetNumber(&tids_copy[len]) == e_off)
		{
			pfree(DatumGetPointer(n_vec));
			return;
		}
	}

	if (len < layer0_m)
	{
		/* Free slot available */
		target = len;
	}
	else
	{
		/* --- phase 1b: full — find furthest neighbor (no page lock held) --- */
		double	furthest_d = -DBL_MAX;
		int		furthest_j = -1;
		double	dEN = acorn_dist(proc, n_vec, e_value);

		for (int j = 0; j < layer0_m; j++)
		{
			Datum	vj;
			double	d;

			acorn_read_element(index, forkNum, &tids_copy[j], NULL, NULL, &vj);
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
		return;					/* E is not closer than any existing neighbor */

	/* --- phase 2: write the chosen slot under EXCLUSIVE --- */
	buf = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
	tids = HnswNeighborTupleGetTids(ntup);
	ItemPointerSet(&tids[target], e_blk, e_off);
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/* -----------------------------------------------------------------------
 * Insert one element (shared by build and aminsert)
 * ----------------------------------------------------------------------- */

static void
acorn_insert_element(Relation index, ForkNumber forkNum, Datum value,
					 ItemPointer heaptid)
{
	int				m_eff = acorn_m_eff(index);
	int				layer0_m = HnswGetLayerM(m_eff, 0);
	FmgrInfo	   *proc = acorn_dist_proc(index);
	Size			vsize = VARSIZE_ANY(DatumGetPointer(value));
	Size			etupSize = HNSW_ELEMENT_TUPLE_SIZE(vsize);
	Size			ntupSize = HNSW_NEIGHBOR_TUPLE_SIZE(0, m_eff);
	HnswElementTuple etup;
	HnswNeighborTuple ntup;
	ItemPointerData *ntids;
	AcornCand	   *cands;
	int				ncand = 0;
	int				nsel;
	BlockNumber		e_blk,
					n_blk;
	OffsetNumber	e_off,
					n_off;

	/* 1. find nearest existing neighbors (brute force) */
	cands = acorn_collect(index, forkNum, value, proc, &ncand);
	if (ncand > 0)
		qsort(cands, ncand, sizeof(AcornCand), acorn_cmp_cand);
	nsel = Min(ncand, layer0_m);

	/* 2. neighbor tuple: count = (level+2)*m_eff = 2*m_eff = layer0_m */
	ntup = palloc0(ntupSize);
	ntup->type = HNSW_NEIGHBOR_TUPLE_TYPE;
	ntup->version = HNSW_VERSION;
	ntup->count = layer0_m;
	ntids = HnswNeighborTupleGetTids(ntup);
	for (int i = 0; i < layer0_m; i++)
		ItemPointerSetInvalid(&ntids[i]);
	for (int i = 0; i < nsel; i++)
		ntids[i] = cands[i].tid;

	acorn_append_tuple(index, forkNum, (Item) ntup, ntupSize, &n_blk, &n_off);

	/* 3. element tuple (vector inline), pointing at its neighbor tuple */
	etup = palloc0(etupSize);
	etup->type = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level = 0;
	etup->deleted = 0;
	etup->version = HNSW_VERSION;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&etup->heaptids[i]);
	etup->heaptids[0] = *heaptid;
	ItemPointerSet(&etup->neighbortid, n_blk, n_off);
	etup->unused = 0;
	memcpy(HnswElementTupleGetVector(etup), DatumGetPointer(value), vsize);

	acorn_append_tuple(index, forkNum, (Item) etup, etupSize, &e_blk, &e_off);

	/* 4. entry point if this is the first element */
	acorn_set_entry_if_unset(index, e_blk, e_off);

	/* 5. reverse edges with fixed-slot retry */
	for (int i = 0; i < nsel; i++)
		acorn_add_reverse_edge(index, forkNum, &cands[i].tid,
							   e_blk, e_off, value, proc, m_eff);

	if (cands)
		pfree(cands);
	pfree(ntup);
	pfree(etup);
}

/* -----------------------------------------------------------------------
 * Build callback + entry points
 * ----------------------------------------------------------------------- */

typedef struct AcornBuildState
{
	ForkNumber	forkNum;
	double		ntuples;
	MemoryContext tmpCtx;
} AcornBuildState;

static void
acorn_build_callback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *state)
{
	AcornBuildState *bs = (AcornBuildState *) state;
	MemoryContext	oldCtx;
	Datum			value;

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(bs->tmpCtx);

	/* Detoast the vector; store as-is (no normalization) */
	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	acorn_insert_element(index, bs->forkNum, value, tid);
	bs->ntuples += 1;

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(bs->tmpCtx);
}

static void
acorn_build_internal(Relation heap, Relation index, IndexInfo *indexInfo,
					 ForkNumber forkNum, double *heap_tuples, double *index_tuples)
{
	int				m_eff = acorn_m_eff(index);
	int				efc = acorn_opt_ef_construction(index);
	int				dims = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;
	AcornBuildState bs;
	double			reltuples = 0;

	if (dims < 0)
		dims = 0;

	/* Meta page (block 0) records m_eff so the reader picks up the dense graph */
	acorn_create_meta_page(index, forkNum, m_eff, efc, dims);

	bs.forkNum = forkNum;
	bs.ntuples = 0;
	bs.tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "acorn build temp",
									  ALLOCSET_DEFAULT_SIZES);

	if (heap != NULL)
		reltuples = table_index_build_scan(heap, index, indexInfo, true, true,
										   acorn_build_callback, (void *) &bs, NULL);

	MemoryContextDelete(bs.tmpCtx);

	if (heap_tuples)
		*heap_tuples = reltuples;
	if (index_tuples)
		*index_tuples = bs.ntuples;
}

IndexBuildResult *
acorn_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result = palloc0(sizeof(IndexBuildResult));
	double			heap_tuples = 0,
					index_tuples = 0;

	acorn_build_internal(heap, index, indexInfo, MAIN_FORKNUM,
						 &heap_tuples, &index_tuples);

	result->heap_tuples = heap_tuples;
	result->index_tuples = index_tuples;
	return result;
}

void
acorn_buildempty(Relation index)
{
	IndexInfo *indexInfo = BuildIndexInfo(index);

	/* Empty index for an unlogged table: just the meta page in INIT_FORKNUM */
	acorn_build_internal(NULL, index, indexInfo, INIT_FORKNUM, NULL, NULL);
}

bool
acorn_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			 Relation heap, IndexUniqueCheck checkUnique,
			 bool indexUnchanged, IndexInfo *indexInfo)
{
	MemoryContext	oldCtx;
	MemoryContext	insertCtx;
	Datum			value;

	if (isnull[0])
		return false;

	insertCtx = AllocSetContextCreate(CurrentMemoryContext,
									  "acorn insert temp",
									  ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(insertCtx);

	value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	acorn_insert_element(index, MAIN_FORKNUM, value, heap_tid);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
