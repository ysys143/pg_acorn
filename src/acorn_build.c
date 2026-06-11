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
#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/parallel.h"
#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "optimizer/optimizer.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "storage/condition_variable.h"
#include "storage/lmgr.h"
#include "storage/lwlock.h"
#include "storage/shmem.h"
#include "storage/spin.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/wait_event.h"

#include "pg_acorn.h"
#include "acorn_am.h"
#include "acorn_dist.h"
#include "acorn_scan.h"
#include "acorn_t2_page.h"
#include "acorn_dist.h"

/*
 * Cap random levels so neighbor tuples remain page-sized.
 * With ACORN_MAX_LEVEL=6 and the maximum m_eff=100:
 *   HNSW_NEIGHBOR_TUPLE_SIZE(6, 100) = 4 + 8*100*6 = 4804 bytes < HNSW_MAX_SIZE
 */
#define ACORN_MAX_LEVEL  6

/* shm_toc keys for the parallel build (mirrors pgvector hnswbuild.c) */
#define PARALLEL_KEY_ACORN_SHARED	UINT64CONST(0xB000000000000001)
#define PARALLEL_KEY_ACORN_AREA		UINT64CONST(0xB000000000000002)
#define PARALLEL_KEY_QUERY_TEXT		UINT64CONST(0xB000000000000003)

/*
 * LWLock tranche for parallel-build locks.  The tranche id lives in a tiny
 * shared-memory struct (allocated from PostgreSQL's small-allocation slop,
 * exactly like pgvector's HnswInitLockTranche) so that one id is shared by
 * every backend; each backend additionally registers the id locally.
 */
static int acorn_lock_tranche_id = 0;

static void
acorn_init_lock_tranche(void)
{
	int		   *tranche_ids;
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tranche_ids = ShmemInitStruct("pg_acorn LWLock ids",
								  sizeof(int) * 1,
								  &found);
	if (!found)
		tranche_ids[0] = LWLockNewTrancheId();
	acorn_lock_tranche_id = tranche_ids[0];
	LWLockRelease(AddinShmemInitLock);

	/* Per-backend registration of the tranche ID */
	LWLockRegisterTranche(acorn_lock_tranche_id, "AcornBuild");
}

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

static bool
acorn_opt_payload_edges(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->payloadEdges : false;
}

static bool
acorn_opt_diversify(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->diversify : false;
}

static bool
acorn_opt_inline_vectors(Relation index)
{
	AcornOptions *opts = (AcornOptions *) index->rd_options;
	return opts ? opts->inlineVectors : false;
}

/* -----------------------------------------------------------------------
 * Payload partitions (acorn_payload_edges)
 *
 * Nodes are grouped into ACORN_PAYLOAD_PARTITIONS partitions by filter_val.
 * The hash is identity mod 256: for low-cardinality int4 payloads (e.g.
 * bucket 0..99) partition == value, mirroring Qdrant's per-value sub-HNSW.
 * When the option is on, each node's layer-0 neighbor slots (2*m_eff) are
 * split: the first m_eff hold global nearest neighbors, the last m_eff hold
 * nearest neighbors within the node's own partition.  The scan needs no
 * changes — it walks all stored slots and ACORN keeps filter-failing nodes
 * in the candidate heap, so same-partition edges make the predicate
 * subgraph navigable.
 * ----------------------------------------------------------------------- */

#define ACORN_PAYLOAD_PARTITIONS 256

static inline int
acorn_payload_partition(int64 filter_val)
{
	return (int) (((uint64) filter_val) & (ACORN_PAYLOAD_PARTITIONS - 1));
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
 *
 * Resolved ONCE per build/insert: a direct C kernel (acorn_dist.c, fmgr
 * bypass) for known pgvector distance functions, with the fmgr FunctionCall2
 * path as fallback for unknown opclasses or dimension mismatch.  Mirrors the
 * scan path's acorn_resolve_direct_dist usage — the bulk in-memory build is
 * distance-dominated, so bypassing fmgr here is the build-time speedup lever.
 * ----------------------------------------------------------------------- */

typedef struct AcornDistCtx
{
	FmgrInfo	*proc;		/* fmgr fallback (always valid) */
	AcornDistFn	 direct;	/* direct C kernel; NULL = always use proc */
} AcornDistCtx;

static void
acorn_dist_ctx_init(Relation index, AcornDistCtx *dist)
{
	dist->proc   = index_getprocinfo(index, 1, 1);
	dist->direct = acorn_build_direct_dist
		? acorn_resolve_direct_dist(index) : NULL;
}

static inline double
acorn_dist(const AcornDistCtx *dist, Datum stored_vec, Datum query)
{
	if (dist->direct)
	{
		AcornPgVector *v = (AcornPgVector *) DatumGetPointer(stored_vec);
		AcornPgVector *q = (AcornPgVector *) DatumGetPointer(query);

		if (likely(v->dim == q->dim))
			return dist->direct((int) v->dim, v->x, q->x);
	}
	return DatumGetFloat8(FunctionCall2Coll(dist->proc, InvalidOid,
											stored_vec, query));
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
		r = erand48(rand_state);
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
					   int efConstruction, int dimensions, uint16 acorn_flags)
{
	Buffer		 buf;
	Page		 page;
	AcornT2MetaPage metap;

	buf  = acorn_new_buffer(index, forkNum);	/* becomes block 0 */
	page = BufferGetPage(buf);
	acorn_init_page(buf, page);

	metap                      = AcornT2PageGetMeta(page);
	metap->hnsw.magicNumber    = HNSW_MAGIC_NUMBER;
	metap->hnsw.version        = HNSW_VERSION;
	metap->hnsw.dimensions     = dimensions;
	metap->hnsw.m              = m_eff;
	metap->hnsw.efConstruction = efConstruction;
	metap->hnsw.entryBlkno     = InvalidBlockNumber;
	metap->hnsw.entryOffno     = InvalidOffsetNumber;
	metap->hnsw.entryLevel     = -1;
	metap->hnsw.insertPage     = InvalidBlockNumber;
	metap->acorn_flags         = acorn_flags;
	metap->reserved            = 0;

	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(AcornT2MetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Read meta: returns m (= m_eff).  Optional out-params may be NULL.
 *
 * acorn_flags/dims read the Tier 2 meta extension; indexes built before the
 * extension existed return zeros there (PageInit zeroes the page).
 */
static int
acorn_read_meta(Relation index, BlockNumber *entry_blkno, OffsetNumber *entry_offno,
				int *entry_level_out, int *dims_out, uint16 *acorn_flags_out)
{
	Buffer		 buf;
	Page		 page;
	AcornT2MetaPage metap;
	int			 m;

	buf   = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page  = BufferGetPage(buf);
	metap = AcornT2PageGetMeta(page);
	m     = (int) metap->hnsw.m;
	if (entry_blkno)
		*entry_blkno = metap->hnsw.entryBlkno;
	if (entry_offno)
		*entry_offno = metap->hnsw.entryOffno;
	if (entry_level_out)
		*entry_level_out = (int) metap->hnsw.entryLevel;
	if (dims_out)
		*dims_out = (int) metap->hnsw.dimensions;
	if (acorn_flags_out)
		*acorn_flags_out = metap->acorn_flags;
	UnlockReleaseBuffer(buf);
	return m;
}

int
acorn_index_m(Relation index)
{
	return acorn_read_meta(index, NULL, NULL, NULL, NULL, NULL);
}

/*
 * Update the meta entry point to (blkno, offno, level) if no entry exists yet
 * or if the new element has a strictly higher level than the current entry.
 */
static void
acorn_maybe_update_entry(Relation index, BlockNumber blkno, OffsetNumber offno,
						 int level, bool use_wal)
{
	Buffer			  buf;
	Page			  page;
	HnswMetaPage	  metap;
	GenericXLogState *state = NULL;

	buf = ReadBuffer(index, HNSW_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (use_wal)
	{
		state = GenericXLogStart(index);
		page  = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
	}
	else
		page = BufferGetPage(buf);

	metap = HnswPageGetMeta(page);

	if (!BlockNumberIsValid(metap->entryBlkno) || level > (int) metap->entryLevel)
	{
		metap->entryBlkno = blkno;
		metap->entryOffno = offno;
		metap->entryLevel = (int16) level;
		if (state)
			GenericXLogFinish(state);
		else
			MarkBufferDirty(buf);
	}
	else if (state)
		GenericXLogAbort(state);

	UnlockReleaseBuffer(buf);
}

/*
 * Append a tuple to the index, returning its on-disk location.
 * Tries the last page, extends the relation with a fresh page if needed.
 */
static void
acorn_append_tuple(Relation index, ForkNumber forkNum, Item tup, Size size,
				   BlockNumber *blkno_out, OffsetNumber *off_out, bool use_wal)
{
	BlockNumber		  nblocks = RelationGetNumberOfBlocksInFork(index, forkNum);
	Buffer			  buf;
	Page			  page;
	OffsetNumber	  off;
	GenericXLogState *state;

	/* block 0 is the meta page; data starts at block 1 */
	if (nblocks > 1)
	{
		buf = ReadBufferExtended(index, forkNum, nblocks - 1, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (use_wal)
		{
			state = GenericXLogStart(index);
			page  = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
		}
		else
			page = BufferGetPage(buf);

		if (PageGetFreeSpace(page) >= size)
		{
			off = PageAddItem(page, tup, size, InvalidOffsetNumber, false, false);
			if (off == InvalidOffsetNumber)
				elog(ERROR, "acorn_hnsw: failed to add index tuple");
			if (use_wal)
				GenericXLogFinish(state);
			else
				MarkBufferDirty(buf);
			*blkno_out = nblocks - 1;
			*off_out   = off;
			UnlockReleaseBuffer(buf);
			return;
		}
		if (use_wal)
			GenericXLogAbort(state);
		UnlockReleaseBuffer(buf);
	}

	LockRelationForExtension(index, ExclusiveLock);
	buf = acorn_new_buffer(index, forkNum);
	UnlockRelationForExtension(index, ExclusiveLock);

	if (use_wal)
	{
		state = GenericXLogStart(index);
		page  = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
	}
	else
		page = BufferGetPage(buf);

	acorn_init_page(buf, page);

	off = PageAddItem(page, tup, size, InvalidOffsetNumber, false, false);
	if (off == InvalidOffsetNumber)
		elog(ERROR, "acorn_hnsw: tuple too large for page");

	if (use_wal)
		GenericXLogFinish(state);
	else
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
 * a palloc'd copy of the inline vector, the inline filter value, the primary
 * heap TID, and the deleted flag.  All out-params may be NULL.
 */
static void
acorn_read_element(Relation index, ForkNumber forkNum, ItemPointer tid,
				   int *level_out,
				   BlockNumber *nbr_blkno, OffsetNumber *nbr_offno,
				   Datum *vec_copy, int64 *filter_out,
				   ItemPointerData *heaptid_out, bool *deleted_out)
{
	Buffer				buf;
	Page				page;
	AcornT2ElementTuple etup;
	Pointer				vec;
	Size				vsize;

	buf  = ReadBufferExtended(index, forkNum,
							  ItemPointerGetBlockNumber(tid), RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (AcornT2ElementTuple) PageGetItem(page,
				PageGetItemId(page, ItemPointerGetOffsetNumber(tid)));

	if (level_out)
		*level_out = (int) etup->level;
	if (nbr_blkno)
		*nbr_blkno = ItemPointerGetBlockNumber(&etup->neighbortid);
	if (nbr_offno)
		*nbr_offno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	if (vec_copy)
	{
		vec   = (Pointer) AcornT2ElementTupleGetVector(etup);
		vsize = VARSIZE_ANY(vec);
		*vec_copy = PointerGetDatum(memcpy(palloc(vsize), vec, vsize));
	}
	if (filter_out)
		*filter_out = etup->filter_val;
	if (heaptid_out)
		*heaptid_out = etup->heaptids[0];
	if (deleted_out)
		*deleted_out = (etup->deleted != 0);
	UnlockReleaseBuffer(buf);
}

/*
 * Compute distance from the inline vector of (blkno, offno) to query.
 * Optionally returns the inline filter value in the same page read.
 */
static double
acorn_node_distance(Relation index, ForkNumber forkNum,
					BlockNumber blkno, OffsetNumber offno,
					Datum query, const AcornDistCtx *dist, int64 *filter_out)
{
	Buffer				buf;
	Page				page;
	AcornT2ElementTuple etup;
	double				d;

	buf  = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	etup = (AcornT2ElementTuple) PageGetItem(page, PageGetItemId(page, offno));
	d    = acorn_dist(dist, PointerGetDatum(AcornT2ElementTupleGetVector(etup)), query);
	if (filter_out)
		*filter_out = etup->filter_val;
	UnlockReleaseBuffer(buf);
	return d;
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
			continue;	/* payload split may leave gaps between halves */
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
	info.hcxt      = CurrentMemoryContext;

	/*
	 * HASH_CONTEXT is essential: without it dynahash parks the table under
	 * TopMemoryContext, so it outlives the per-search temp context and leaks
	 * ~100KB per on-disk insert (measured 1.26GB peak RssAnon during a 60K
	 * maintenance_work_mem spill tail vs 87MB for the full in-memory build).
	 */
	return hash_create("acorn_build_visited", 512, &info,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
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
 * In-memory node representation for scalable bulk build (work item D)
 *
 * The entire HNSW graph is constructed in memory using integer node IDs,
 * eliminating per-insert random disk I/O.  After construction, a two-pass
 * flush writes all tuples sequentially:
 *
 *   Pass 1: write neighbor tuples (TID slots left invalid)
 *   Pass 2: write element tuples  (neighbortid filled from pre-assign)
 *   Pass 3: patch neighbor tuples with element TIDs
 *
 * A page-packing simulation (acorn_mem_preassign_tids) assigns every tuple's
 * on-disk location before any write, breaking the circular TID dependency.
 *
 * MEMORY BUDGET (maintenance_work_mem): the graph may only grow while its
 * total allocation fits maintenance_work_mem.  When the next node would not
 * fit, the graph built so far is flushed and every remaining tuple goes
 * through the per-element on-disk insert path (pgvector's two-phase build
 * pattern).  In a serial build a running byte counter tracks the actual
 * allocations; in a parallel build the graph lives in one fixed-size shared
 * memory arena, so exhaustion of the arena is the same condition.
 *
 * PARALLEL BUILD: the node array + vectors + neighbor arrays live in a DSM
 * arena mapped at a different address in each participant, so graph-internal
 * references are stored as arena offsets (AcornBuildPtr dual-mode pointers,
 * pgvector's relptr pattern).  Concurrency control mirrors pgvector
 * hnswbuild.c:
 *
 *   - allocatorLock:  node-id + arena bump allocation
 *   - entryLock/entryWaitLock: entry point reads (shared) and updates
 *     (exclusive, with the wait-lock dance to avoid starvation)
 *   - flushLock:      shared while inserting in memory; exclusive to flush
 *     and for each (serialized) on-disk insert after spill — the acorn
 *     on-disk insert path is NOT concurrency-safe, so correctness over
 *     speed for the spilled portion
 *   - partLock:       payload partition entry map
 *   - per-node lock:  a node's neighbor slots are read (slot-range copy)
 *     under LW_SHARED and mutated under LW_EXCLUSIVE
 *
 * A new node is published to other participants only by the first reverse
 * edge that stores its id (taken under the target's exclusive lock), so its
 * own level/vector/forward slots are complete before anyone can reach it.
 * ----------------------------------------------------------------------- */

/* Dual-mode graph pointer: absolute in serial builds, arena offset in parallel */
typedef union AcornBuildPtr
{
	void	   *ptr;
	Size		off;
} AcornBuildPtr;

typedef struct AcornMemNode
{
	int				 level;
	int64			 filter_val;
	AcornBuildPtr	 vec;			/* vector varlena copy */
	Size			 vsize;
	AcornBuildPtr	 nbr;			/* flat (level+2)*m_eff int IDs; -1=empty */
	ItemPointerData  heaptid;
	LWLock			 lock;			/* parallel build: protects nbr slots */
	BlockNumber		 nbr_blkno;		/* assigned by acorn_mem_preassign_tids */
	OffsetNumber	 nbr_offno;
	BlockNumber		 elem_blkno;
	OffsetNumber	 elem_offno;
	ItemPointerData *cont_tids;		/* inline-vector continuation chunk TIDs
									 * (flusher-private allocation) */
	int				 n_conts;
} AcornMemNode;

/*
 * Shared state for a parallel build.  Lives at the start of the DSM segment;
 * the ParallelTableScanDesc follows at a BUFFERALIGN'd offset, and the graph
 * arena is a separate shm_toc chunk.
 */
typedef struct AcornShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;
	Size		arena_size;		/* total bytes of the graph arena */
	int			max_nodes;		/* capacity of the node array */

	/* Worker progress (under mutex) */
	slock_t		mutex;
	int			nparticipantsdone;
	double		reltuples;
	double		indtuples;
	ConditionVariable workersdonecv;

	/* Graph state */
	LWLock		allocatorLock;
	LWLock		entryLock;
	LWLock		entryWaitLock;
	LWLock		flushLock;
	LWLock		partLock;
	int			n_nodes;		/* under allocatorLock */
	Size		arena_used;		/* bump offset, under allocatorLock */
	int			entry_id;		/* under entryLock; -1 = none */
	int			entry_level;
	bool		flushed;		/* under flushLock */
	int			part_entry[ACORN_PAYLOAD_PARTITIONS];	/* under partLock */
} AcornShared;

#define ParallelTableScanFromAcornShared(shared) \
	((ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(AcornShared))))

typedef struct AcornMemBuild
{
	AcornMemNode   *nodes;			/* serial: palloc'd; parallel: in arena */
	char		   *base;			/* arena base; NULL = serial (absolute ptrs) */
	AcornShared	   *shared;			/* NULL = serial build */
	int				n_nodes;		/* serial only (parallel: shared->n_nodes) */
	int				capacity;		/* serial: growable; parallel: max_nodes */
	int				entry_id;		/* serial only; -1 = no entry yet */
	int				entry_level;
	uint32		   *visit_gen;		/* per-process generation-based visited set */
	uint32			cur_gen;
	bool			payload_edges;	/* split layer-0 slots global/partition halves */
	bool			inline_vectors;	/* co-locate SQ8 vectors in neighbor lists */
	int				dims;			/* vector dimensions (inline mode) */
	Size			entry_size;		/* inline entry stride (inline mode) */
	int			   *part_entry;		/* partition -> first member node id, -1 = none
									 * (parallel: points at shared->part_entry) */
	MemoryContext	build_ctx;
	/* maintenance_work_mem accounting (serial; parallel uses the arena) */
	Size			mem_used;		/* bytes allocated for the graph so far */
	Size			mem_total;		/* budget = maintenance_work_mem in bytes */
} AcornMemBuild;

/* Graph-internal pointer access (relptr pattern; see block comment above) */
static inline Datum
acorn_node_vec(const AcornMemBuild *mb, const AcornMemNode *node)
{
	return mb->base ? PointerGetDatum(mb->base + node->vec.off)
		: PointerGetDatum(node->vec.ptr);
}

static inline int *
acorn_node_nbr(const AcornMemBuild *mb, const AcornMemNode *node)
{
	return mb->base ? (int *) (mb->base + node->nbr.off)
		: (int *) node->nbr.ptr;
}

/* Per-node neighbor-slot locks (no-ops in a serial build) */
static inline void
acorn_node_lock(AcornMemBuild *mb, AcornMemNode *node, LWLockMode mode)
{
	if (mb->shared)
		LWLockAcquire(&node->lock, mode);
}

static inline void
acorn_node_unlock(AcornMemBuild *mb, AcornMemNode *node)
{
	if (mb->shared)
		LWLockRelease(&node->lock);
}

/*
 * Copy `count` neighbor slots of `node` starting at flat slot index `start`
 * into out[].  Readers must never chase another node's slot array in place:
 * a concurrent insert may be mutating it (parallel build), so the slots are
 * snapshotted under the node's shared lock.
 */
static inline void
acorn_mem_copy_slots(AcornMemBuild *mb, AcornMemNode *node,
					 int start, int count, int *out)
{
	acorn_node_lock(mb, node, LW_SHARED);
	memcpy(out, acorn_node_nbr(mb, node) + start, count * sizeof(int));
	acorn_node_unlock(mb, node);
}

/* Min/max-heap comparators for in-memory beam search (integer node IDs) */
typedef struct MemPQNode
{
	pairingheap_node ph;
	int				 id;
	double			 dist;
} MemPQNode;

static int
mem_cmp_max(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	double da = ((const MemPQNode *) a)->dist;
	double db = ((const MemPQNode *) b)->dist;
	return (da > db) ? 1 : (da < db) ? -1 : 0;
}

static int
mem_cmp_min(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	return -mem_cmp_max(a, b, arg);
}

static AcornMemBuild *
acorn_mem_build_init(MemoryContext ctx)
{
	MemoryContext  old = MemoryContextSwitchTo(ctx);
	AcornMemBuild *mb  = palloc0(sizeof(AcornMemBuild));

	mb->capacity    = 1024;
	mb->nodes       = palloc0(mb->capacity * sizeof(AcornMemNode));
	mb->visit_gen   = palloc0(mb->capacity * sizeof(uint32));
	mb->entry_id    = -1;
	mb->entry_level = -1;
	mb->cur_gen     = 1;
	mb->build_ctx   = ctx;
	mb->payload_edges = false;
	mb->part_entry  = palloc(ACORN_PAYLOAD_PARTITIONS * sizeof(int));
	for (int p = 0; p < ACORN_PAYLOAD_PARTITIONS; p++)
		mb->part_entry[p] = -1;
	mb->mem_used    = mb->capacity * (sizeof(AcornMemNode) + sizeof(uint32))
		+ sizeof(AcornMemBuild) + ACORN_PAYLOAD_PARTITIONS * sizeof(int);
	mb->mem_total   = (Size) maintenance_work_mem * 1024;
	MemoryContextSwitchTo(old);
	return mb;
}

/*
 * Bytes the graph grows by when node `level` with a `vsize` vector is pushed.
 * Used both for the serial maintenance_work_mem check (against the running
 * counter of the same charges) and for the next-node projection.
 */
static inline Size
acorn_mem_node_cost(const AcornMemBuild *mb, int level, Size vsize, int m_eff)
{
	Size cost = MAXALIGN(vsize)
		+ MAXALIGN((Size) (level + 2) * m_eff * sizeof(int));

	/* serial: account the capacity doubling this push would trigger */
	if (!mb->shared && mb->n_nodes >= mb->capacity)
		cost += (Size) mb->capacity * (sizeof(AcornMemNode) + sizeof(uint32));
	return cost;
}

/*
 * Append a node to the graph: node slot + vector copy + neighbor array
 * (all slots initialized to -1).  The level must already be assigned.
 * Returns the new node id, or -1 when a parallel build's shared arena
 * cannot fit the node (the caller flushes and spills to the disk path).
 * Serial builds never return -1: the caller checks the budget first.
 */
static int
acorn_mem_push_node(AcornMemBuild *mb, Datum vec, Size vsize, int level,
					int64 filter_val, ItemPointer heaptid, int m_eff)
{
	AcornMemNode  *node;
	int			   id;
	int			   n_slots = (level + 2) * m_eff;
	int			  *nbr;
	void		  *vdst;

	if (mb->shared)
	{
		AcornShared *sh = mb->shared;
		Size		 vbytes = MAXALIGN(vsize);
		Size		 nbytes = MAXALIGN((Size) n_slots * sizeof(int));

		LWLockAcquire(&sh->allocatorLock, LW_EXCLUSIVE);
		if (sh->n_nodes >= sh->max_nodes ||
			sh->arena_used + vbytes + nbytes > sh->arena_size)
		{
			LWLockRelease(&sh->allocatorLock);
			return -1;
		}
		id = sh->n_nodes++;
		node = &mb->nodes[id];
		MemSet(node, 0, sizeof(AcornMemNode));
		node->vec.off = sh->arena_used;
		sh->arena_used += vbytes;
		node->nbr.off = sh->arena_used;
		sh->arena_used += nbytes;
		LWLockRelease(&sh->allocatorLock);

		LWLockInitialize(&node->lock, acorn_lock_tranche_id);
		vdst = mb->base + node->vec.off;
		nbr  = (int *) (mb->base + node->nbr.off);
	}
	else
	{
		MemoryContext old = MemoryContextSwitchTo(mb->build_ctx);

		if (mb->n_nodes >= mb->capacity)
		{
			int old_cap   = mb->capacity;
			mb->capacity *= 2;
			mb->nodes     = repalloc(mb->nodes,
									  mb->capacity * sizeof(AcornMemNode));
			mb->visit_gen = repalloc(mb->visit_gen,
									  mb->capacity * sizeof(uint32));
			memset(mb->visit_gen + old_cap, 0,
				   (mb->capacity - old_cap) * sizeof(uint32));
			mb->mem_used += (Size) old_cap *
				(sizeof(AcornMemNode) + sizeof(uint32));
		}

		id = mb->n_nodes++;
		node = &mb->nodes[id];
		MemSet(node, 0, sizeof(AcornMemNode));
		vdst = palloc(vsize);
		node->vec.ptr = vdst;
		nbr = palloc(n_slots * sizeof(int));
		node->nbr.ptr = nbr;
		mb->mem_used += MAXALIGN(vsize)
			+ MAXALIGN((Size) n_slots * sizeof(int));
		MemoryContextSwitchTo(old);
	}

	node->level      = level;
	node->filter_val = filter_val;
	node->vsize      = vsize;
	node->heaptid    = *heaptid;
	node->nbr_blkno  = InvalidBlockNumber;
	node->nbr_offno  = InvalidOffsetNumber;
	node->elem_blkno = InvalidBlockNumber;
	node->elem_offno = InvalidOffsetNumber;
	node->cont_tids  = NULL;
	node->n_conts    = 0;
	memcpy(vdst, DatumGetPointer(vec), vsize);
	for (int k = 0; k < n_slots; k++)
		nbr[k] = -1;

	return id;
}

/* In-memory greedy descent from from_level down to to_level+1 */
static void
acorn_mem_greedy_descend(AcornMemBuild *mb, const AcornDistCtx *dist, int m_eff,
						  int *ep_id, Datum query,
						  int from_level, int to_level)
{
	int slots[HNSW_MAX_NEIGHBORS];

	for (int lc = from_level; lc > to_level; lc--)
	{
		bool improved;

		do {
			int           cur      = *ep_id;
			AcornMemNode *node     = &mb->nodes[cur];
			double        cur_dist = acorn_dist(dist, acorn_node_vec(mb, node),
												query);
			int           start    = HnswNeighborStart(m_eff, node->level, lc);
			int           layer_m  = HnswGetLayerM(m_eff, lc);

			acorn_mem_copy_slots(mb, node, start, layer_m, slots);

			improved = false;
			for (int j = 0; j < layer_m; j++)
			{
				int    nbr_id = slots[j];
				double nd;

				if (nbr_id < 0)
					continue;	/* payload split may leave gaps between halves */
				nd = acorn_dist(dist, acorn_node_vec(mb, &mb->nodes[nbr_id]),
								query);
				if (nd < cur_dist)
				{
					cur_dist = nd;
					*ep_id   = nbr_id;
					improved = true;
				}
			}
		} while (improved);
	}
}

/*
 * In-memory beam search at `layer` with ef candidates.
 * Returns up to ef results in out_ids[]/out_dists[], nearest-first.
 */
static int
acorn_mem_search_layer(AcornMemBuild *mb, const AcornDistCtx *dist, int m_eff,
					   int entry_id, Datum query, int layer, int ef,
					   int *out_ids, double *out_dists)
{
	MemoryContext  tmp_ctx, old_ctx;
	pairingheap   *C, *W;
	int			   W_count = 0;
	int			   n_out;

	tmp_ctx = AllocSetContextCreate(mb->build_ctx, "acorn_mem_search",
									ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(tmp_ctx);

	mb->cur_gen++;
	C = pairingheap_allocate(mem_cmp_min, NULL);
	W = pairingheap_allocate(mem_cmp_max, NULL);

	/* Seed with entry_id */
	{
		double    ep_dist = acorn_dist(dist, acorn_node_vec(mb, &mb->nodes[entry_id]),
									   query);
		MemPQNode *cn, *wn;

		mb->visit_gen[entry_id] = mb->cur_gen;
		cn = palloc(sizeof(MemPQNode)); cn->id = entry_id; cn->dist = ep_dist;
		pairingheap_add(C, &cn->ph);
		wn = palloc(sizeof(MemPQNode)); wn->id = entry_id; wn->dist = ep_dist;
		pairingheap_add(W, &wn->ph);
		W_count = 1;
	}

	while (!pairingheap_is_empty(C))
	{
		MemPQNode    *c_node = (MemPQNode *) pairingheap_remove_first(C);
		int			  c_id   = c_node->id;
		double		  c_dist = c_node->dist;
		AcornMemNode *c_data = &mb->nodes[c_id];
		double		  f_dist;
		int			  start, layer_m;
		int			  slots[HNSW_MAX_NEIGHBORS];

		f_dist = ((MemPQNode *) pairingheap_first(W))->dist;
		if (c_dist > f_dist && W_count >= ef)
			break;

		if (c_data->level < layer)
			continue;	/* safety: node doesn't appear at this layer */

		start   = HnswNeighborStart(m_eff, c_data->level, layer);
		layer_m = HnswGetLayerM(m_eff, layer);

		acorn_mem_copy_slots(mb, c_data, start, layer_m, slots);

		for (int j = 0; j < layer_m; j++)
		{
			int    nbr_id = slots[j];
			double nd;

			if (nbr_id < 0)
				continue;	/* payload split may leave gaps between halves */
			if (mb->visit_gen[nbr_id] == mb->cur_gen)
				continue;
			if (mb->nodes[nbr_id].level < layer)
				continue;
			mb->visit_gen[nbr_id] = mb->cur_gen;

			nd     = acorn_dist(dist, acorn_node_vec(mb, &mb->nodes[nbr_id]),
								query);
			f_dist = ((MemPQNode *) pairingheap_first(W))->dist;

			if (nd < f_dist || W_count < ef)
			{
				MemPQNode *cn, *wn;

				cn = palloc(sizeof(MemPQNode)); cn->id = nbr_id; cn->dist = nd;
				pairingheap_add(C, &cn->ph);
				wn = palloc(sizeof(MemPQNode)); wn->id = nbr_id; wn->dist = nd;
				pairingheap_add(W, &wn->ph);
				W_count++;

				if (W_count > ef)
				{
					(void) pairingheap_remove_first(W);
					W_count--;
				}
			}
		}
	}

	/* Extract W nearest-first */
	n_out = W_count;
	for (int i = W_count - 1; i >= 0; i--)
	{
		MemPQNode *wn  = (MemPQNode *) pairingheap_remove_first(W);
		out_ids[i]   = wn->id;
		out_dists[i] = wn->dist;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(tmp_ctx);
	return n_out;
}

/*
 * In-memory layer-0 beam search restricted to one payload partition.
 *
 * Starts at entry_id (a member of `part`) and explores only nodes whose
 * filter_val maps to `part`.  Expansion walks ALL of a member's layer-0
 * slots (both the global and the payload half) — non-members are skipped,
 * members are kept.  Because every partition member is wired to the nearest
 * existing members on insert (bidirectionally, payload half), the partition
 * subgraph stays connected and this search behaves like a per-partition
 * single-layer HNSW (Qdrant's "Filterable HNSW" merged sub-graph).
 *
 * Returns up to ef same-partition node ids nearest-first.
 */
static int
acorn_mem_search_partition(AcornMemBuild *mb, const AcornDistCtx *dist, int m_eff,
						   int entry_id, Datum query, int part, int ef,
						   int *out_ids, double *out_dists)
{
	MemoryContext  tmp_ctx, old_ctx;
	pairingheap   *C, *W;
	int			   W_count = 0;
	int			   n_out;

	tmp_ctx = AllocSetContextCreate(mb->build_ctx, "acorn_mem_part_search",
									ALLOCSET_SMALL_SIZES);
	old_ctx = MemoryContextSwitchTo(tmp_ctx);

	mb->cur_gen++;
	C = pairingheap_allocate(mem_cmp_min, NULL);
	W = pairingheap_allocate(mem_cmp_max, NULL);

	{
		double     ep_dist = acorn_dist(dist, acorn_node_vec(mb, &mb->nodes[entry_id]),
										query);
		MemPQNode *cn, *wn;

		Assert(acorn_payload_partition(mb->nodes[entry_id].filter_val) == part);
		mb->visit_gen[entry_id] = mb->cur_gen;
		cn = palloc(sizeof(MemPQNode)); cn->id = entry_id; cn->dist = ep_dist;
		pairingheap_add(C, &cn->ph);
		wn = palloc(sizeof(MemPQNode)); wn->id = entry_id; wn->dist = ep_dist;
		pairingheap_add(W, &wn->ph);
		W_count = 1;
	}

	while (!pairingheap_is_empty(C))
	{
		MemPQNode    *c_node = (MemPQNode *) pairingheap_remove_first(C);
		int			  c_id   = c_node->id;
		double		  c_dist = c_node->dist;
		AcornMemNode *c_data = &mb->nodes[c_id];
		double		  f_dist;
		int			  start, layer_m;
		int			  slots[HNSW_MAX_NEIGHBORS];

		f_dist = ((MemPQNode *) pairingheap_first(W))->dist;
		if (c_dist > f_dist && W_count >= ef)
			break;

		start   = HnswNeighborStart(m_eff, c_data->level, 0);
		layer_m = HnswGetLayerM(m_eff, 0);

		acorn_mem_copy_slots(mb, c_data, start, layer_m, slots);

		for (int j = 0; j < layer_m; j++)
		{
			int    nbr_id = slots[j];
			double nd;

			if (nbr_id < 0)
				continue;
			if (mb->visit_gen[nbr_id] == mb->cur_gen)
				continue;
			mb->visit_gen[nbr_id] = mb->cur_gen;

			if (acorn_payload_partition(mb->nodes[nbr_id].filter_val) != part)
				continue;	/* restricted to the partition subgraph */

			nd     = acorn_dist(dist, acorn_node_vec(mb, &mb->nodes[nbr_id]),
								query);
			f_dist = ((MemPQNode *) pairingheap_first(W))->dist;

			if (nd < f_dist || W_count < ef)
			{
				MemPQNode *cn, *wn;

				cn = palloc(sizeof(MemPQNode)); cn->id = nbr_id; cn->dist = nd;
				pairingheap_add(C, &cn->ph);
				wn = palloc(sizeof(MemPQNode)); wn->id = nbr_id; wn->dist = nd;
				pairingheap_add(W, &wn->ph);
				W_count++;

				if (W_count > ef)
				{
					(void) pairingheap_remove_first(W);
					W_count--;
				}
			}
		}
	}

	n_out = W_count;
	for (int i = W_count - 1; i >= 0; i--)
	{
		MemPQNode *wn  = (MemPQNode *) pairingheap_remove_first(W);
		out_ids[i]   = wn->id;
		out_dists[i] = wn->dist;
	}

	MemoryContextSwitchTo(old_ctx);
	MemoryContextDelete(tmp_ctx);
	return n_out;
}

/*
 * HNSW neighbor-selection diversity heuristic (Malkov Alg. 4 / pgvector
 * HnswSelectNeighbors, with keepPrunedConnections).
 *
 * cand_ids/cand_dists hold candidates sorted ASCENDING by distance to the
 * base vector.  A candidate is kept only if it is closer to the base than to
 * every already-kept neighbor (dist(c, kept) > dist(c, base)); pruned
 * candidates then refill remaining slots in ascending order so slots stay
 * full.  Returns the number kept (<= m_max), written to kept_out as node ids.
 *
 * Without this, nearest-only selection on clustered data wires each node
 * exclusively into its own cluster: layer 0 fragments into per-cluster
 * components and whole buckets become unreachable (W1 root cause).
 */
static int
acorn_mem_select_diverse(AcornMemBuild *mb, const AcornDistCtx *dist,
						 const int *cand_ids, const double *cand_dists,
						 int n_cands, int m_max, int *kept_out)
{
	int  n_kept   = 0;
	int  n_pruned = 0;
	int *pruned;

	if (n_cands <= m_max)
	{
		/* keepPrunedConnections refills everything anyway — keep all */
		memcpy(kept_out, cand_ids, n_cands * sizeof(int));
		return n_cands;
	}

	pruned = palloc(n_cands * sizeof(int));

	for (int i = 0; i < n_cands && n_kept < m_max; i++)
	{
		bool keep = true;

		for (int j = 0; j < n_kept; j++)
		{
			double d = acorn_dist(dist,
								  acorn_node_vec(mb, &mb->nodes[cand_ids[i]]),
								  acorn_node_vec(mb, &mb->nodes[kept_out[j]]));

			if (d <= cand_dists[i])
			{
				keep = false;
				break;
			}
		}
		if (keep)
			kept_out[n_kept++] = cand_ids[i];
		else
			pruned[n_pruned++] = cand_ids[i];
	}

	/* keepPrunedConnections: refill remaining slots from pruned, in order */
	for (int p = 0; p < n_pruned && n_kept < m_max; p++)
		kept_out[n_kept++] = pruned[p];

	pfree(pruned);
	return n_kept;
}

/*
 * Bidirectional edge with FIXED-SLOT RETRY restricted to a slot range
 * [slot_base, slot_base + slot_count) of nbr_id's flat slot array.
 * With payload edges off the range covers a whole layer; with payload edges
 * on, layer-0 global and payload halves are maintained independently so a
 * flood of global candidates can never evict same-partition edges.
 *
 * If part >= 0 (payload half), slots occupied by nodes OUTSIDE the partition
 * (graceful-degradation backfill) are evicted preferentially: a same-
 * partition edge is strictly more valuable there.  Otherwise, with diversify
 * on, the eviction choice is made by the diversity heuristic over
 * existing + new (pgvector HnswUpdateConnection semantics); with diversify
 * off the legacy replace-furthest-if-closer retry applies.
 */
static void
acorn_mem_add_reverse_edge(AcornMemBuild *mb, const AcornDistCtx *dist,
						   int nbr_id, int new_id,
						   int slot_base, int slot_count, int part,
						   bool diversify)
{
	AcornMemNode *nbr_node = &mb->nodes[nbr_id];
	AcornMemNode *node     = &mb->nodes[new_id];
	int			 *nbr_slots;
	int			  free_slot = -1;
	int			  nonmember_slot = -1;
	double		  nonmember_d = -DBL_MAX;

	/*
	 * The whole read-modify-write runs under the target's exclusive lock
	 * (parallel build); reads of other nodes' vectors below are lock-free
	 * (vectors are immutable after publication).
	 */
	acorn_node_lock(mb, nbr_node, LW_EXCLUSIVE);
	nbr_slots = acorn_node_nbr(mb, nbr_node) + slot_base;

	for (int j = 0; j < slot_count; j++)
	{
		int eid = nbr_slots[j];

		if (eid == new_id)
			goto done;			/* already connected */
		if (eid < 0)
		{
			if (free_slot < 0)
				free_slot = j;
			continue;
		}
		if (part >= 0 &&
			acorn_payload_partition(mb->nodes[eid].filter_val) != part)
		{
			double d = acorn_dist(dist, acorn_node_vec(mb, nbr_node),
								  acorn_node_vec(mb, &mb->nodes[eid]));

			if (d > nonmember_d)
			{
				nonmember_d    = d;
				nonmember_slot = j;
			}
		}
	}

	if (free_slot >= 0)
	{
		nbr_slots[free_slot] = new_id;
		goto done;
	}

	if (nonmember_slot >= 0)
	{
		nbr_slots[nonmember_slot] = new_id;
		goto done;
	}

	if (diversify)
	{
		/*
		 * Full: re-select with the diversity heuristic over existing + new.
		 * n_cands = slot_count + 1 while keepPrunedConnections refills to
		 * slot_count, so exactly one candidate is dropped: if it is the new
		 * node nothing changes, otherwise the new node takes its slot.
		 */
		int		n_cands = slot_count + 1;
		int	   *ids     = palloc(n_cands * sizeof(int));
		double *dists   = palloc(n_cands * sizeof(double));
		int	   *kept    = palloc(slot_count * sizeof(int));
		int		n_kept;
		bool	new_kept = false;

		for (int j = 0; j < slot_count; j++)
		{
			ids[j]   = nbr_slots[j];
			dists[j] = acorn_dist(dist, acorn_node_vec(mb, nbr_node),
								  acorn_node_vec(mb, &mb->nodes[ids[j]]));
		}
		ids[slot_count]   = new_id;
		dists[slot_count] = acorn_dist(dist, acorn_node_vec(mb, nbr_node),
									   acorn_node_vec(mb, node));

		/* insertion sort ascending by distance (slot_count <= 2*m_eff) */
		for (int i = 1; i < n_cands; i++)
		{
			int    tmp_id = ids[i];
			double tmp_d  = dists[i];
			int    k      = i - 1;

			while (k >= 0 && dists[k] > tmp_d)
			{
				ids[k + 1]   = ids[k];
				dists[k + 1] = dists[k];
				k--;
			}
			ids[k + 1]   = tmp_id;
			dists[k + 1] = tmp_d;
		}

		n_kept = acorn_mem_select_diverse(mb, dist, ids, dists, n_cands,
										  slot_count, kept);

		for (int i = 0; i < n_kept; i++)
		{
			if (kept[i] == new_id)
			{
				new_kept = true;
				break;
			}
		}
		if (new_kept)
		{
			/* replace the single existing occupant that was dropped */
			for (int j = 0; j < slot_count; j++)
			{
				int  eid     = nbr_slots[j];
				bool in_kept = false;

				for (int i = 0; i < n_kept; i++)
				{
					if (kept[i] == eid)
					{
						in_kept = true;
						break;
					}
				}
				if (!in_kept)
				{
					nbr_slots[j] = new_id;
					break;
				}
			}
		}
		pfree(ids);
		pfree(dists);
		pfree(kept);
		goto done;
	}

	/* Full: replace the furthest existing neighbor if the new node is closer */
	{
		double d_new      = acorn_dist(dist, acorn_node_vec(mb, nbr_node),
									   acorn_node_vec(mb, node));
		double furthest_d = -DBL_MAX;
		int    furthest_j = -1;

		for (int j = 0; j < slot_count; j++)
		{
			int    eid = nbr_slots[j];
			double d   = acorn_dist(dist, acorn_node_vec(mb, nbr_node),
									acorn_node_vec(mb, &mb->nodes[eid]));

			if (d > furthest_d)
			{
				furthest_d = d;
				furthest_j = j;
			}
		}
		if (d_new < furthest_d)
			nbr_slots[furthest_j] = new_id;
	}

done:
	acorn_node_unlock(mb, nbr_node);
}

/* Deferred reverse edge (applied after all of the new node's layers are wired) */
typedef struct AcornRevEdge
{
	int		target;			/* node receiving the reverse edge */
	int		slot_base;		/* flat slot index of the range start */
	int		slot_count;		/* range length */
	int		part;			/* payload partition, -1 = none */
} AcornRevEdge;

/* Register new_id as a partition entry point if its partition has none yet */
static void
acorn_mem_register_part_entry(AcornMemBuild *mb, AcornMemNode *node, int new_id)
{
	int part;

	if (!mb->payload_edges)
		return;
	part = acorn_payload_partition(node->filter_val);
	if (mb->shared)
		LWLockAcquire(&mb->shared->partLock, LW_EXCLUSIVE);
	if (mb->part_entry[part] < 0)
		mb->part_entry[part] = new_id;
	if (mb->shared)
		LWLockRelease(&mb->shared->partLock);
}

/*
 * In-memory HNSW insert for node new_id (level + neighbor array already
 * assigned by acorn_mem_push_node).  Runs greedy descent + beam search,
 * fills the node's own neighbor IDs layer by layer, then applies all reverse
 * edges (fixed-slot retry / diversity re-selection).
 *
 * Reverse edges are deferred until every layer of the new node is wired:
 * within one insert the orderings are equivalent (a reverse edge at layer lc
 * only touches layer-lc slot ranges, which no lower-layer search of the same
 * insert reads), and in a parallel build the first reverse edge is what
 * publishes the node to other participants — deferral guarantees the node's
 * own slots are complete before it becomes reachable.
 *
 * Parallel entry-point protocol (mirrors pgvector InsertTupleInMemory):
 * the entry is read under entryLock LW_SHARED held for the whole insert;
 * when this node will (or may) update the entry, the lock is taken
 * LW_EXCLUSIVE instead, with entryWaitLock preventing starvation.
 */
static void
acorn_mem_insert_node(AcornMemBuild *mb, const AcornDistCtx *dist,
					  int m_eff, int efc, bool diversify, int new_id)
{
	AcornMemNode *node    = &mb->nodes[new_id];
	int			  level   = node->level;
	int			 *my_nbr  = acorn_node_nbr(mb, node);
	Datum		  q       = acorn_node_vec(mb, node);
	int			 *out_ids;
	double		 *out_dists;
	int			  ep_id;
	int			  entry_id;
	int			  entry_level;
	AcornShared  *sh = mb->shared;
	AcornRevEdge *rev;
	int			  n_rev = 0;

	/* --- Read the entry point (locked dance in a parallel build) --- */
	if (sh)
	{
		LWLockAcquire(&sh->entryWaitLock, LW_EXCLUSIVE);
		LWLockRelease(&sh->entryWaitLock);

		LWLockAcquire(&sh->entryLock, LW_SHARED);
		entry_id    = sh->entry_id;
		entry_level = sh->entry_level;

		if (entry_id < 0 || level > entry_level)
		{
			LWLockRelease(&sh->entryLock);

			LWLockAcquire(&sh->entryWaitLock, LW_EXCLUSIVE);
			LWLockAcquire(&sh->entryLock, LW_EXCLUSIVE);
			LWLockRelease(&sh->entryWaitLock);

			entry_id    = sh->entry_id;
			entry_level = sh->entry_level;
		}
	}
	else
	{
		entry_id    = mb->entry_id;
		entry_level = mb->entry_level;
	}

	if (entry_id < 0)
	{
		/* First node: it becomes the entry point (exclusive lock held) */
		if (sh)
		{
			sh->entry_id    = new_id;
			sh->entry_level = level;
			LWLockRelease(&sh->entryLock);
		}
		else
		{
			mb->entry_id    = new_id;
			mb->entry_level = level;
		}
		acorn_mem_register_part_entry(mb, node, new_id);
		return;
	}

	ep_id = entry_id;

	if (entry_level > level)
		acorn_mem_greedy_descend(mb, dist, m_eff, &ep_id,
								  q, entry_level, level);

	out_ids   = palloc(sizeof(int)    * efc);
	out_dists = palloc(sizeof(double) * efc);
	rev       = palloc(sizeof(AcornRevEdge) * (Size) (level + 2) * m_eff);

	for (int lc = Min(entry_level, level); lc >= 0; lc--)
	{
		int layer_m = HnswGetLayerM(m_eff, lc);
		int start   = HnswNeighborStart(m_eff, level, lc);
		int n_cands = acorn_mem_search_layer(mb, dist, m_eff,
											  ep_id, q, lc, efc,
											  out_ids, out_dists);

		if (lc == 0 && mb->payload_edges)
		{
			/*
			 * Split layer-0 slots: [start, start+g_half) = global nearest,
			 * [start+g_half, start+layer_m) = nearest same-partition members.
			 */
			int  g_half   = layer_m / 2;	/* = m_eff */
			int  p_half   = layer_m - g_half;
			int  part     = acorn_payload_partition(node->filter_val);
			int  part_ep;
			int  n_global;
			int  p_filled = 0;
			int *g_sel    = palloc(sizeof(int) * g_half);

			/* Global half: diversity-select up to g_half from the beam */
			if (diversify)
				n_global = acorn_mem_select_diverse(mb, dist,
													out_ids, out_dists,
													n_cands, g_half, g_sel);
			else
			{
				n_global = Min(n_cands, g_half);
				memcpy(g_sel, out_ids, n_global * sizeof(int));
			}
			for (int i = 0; i < n_global; i++)
				my_nbr[start + i] = g_sel[i];

			if (sh)
				LWLockAcquire(&sh->partLock, LW_SHARED);
			part_ep = mb->part_entry[part];
			if (sh)
				LWLockRelease(&sh->partLock);

			if (part_ep >= 0)
			{
				int    *p_ids   = palloc(sizeof(int) * efc);
				double *p_dists = palloc(sizeof(double) * efc);
				int     n_part  = acorn_mem_search_partition(mb, dist, m_eff,
															 part_ep,
															 q, part, efc,
															 p_ids, p_dists);
				int     n_avail = 0;
				int    *p_sel   = palloc(sizeof(int) * p_half);
				int     n_sel;

				/* drop candidates already wired as global neighbors */
				for (int i = 0; i < n_part; i++)
				{
					bool dup = false;

					for (int j = 0; j < n_global; j++)
					{
						if (g_sel[j] == p_ids[i])
						{
							dup = true;
							break;
						}
					}
					if (dup)
						continue;	/* already a global neighbor of this node */
					p_ids[n_avail]   = p_ids[i];
					p_dists[n_avail] = p_dists[i];
					n_avail++;
				}

				/* Payload half: diversity within the partition too */
				if (diversify)
					n_sel = acorn_mem_select_diverse(mb, dist, p_ids, p_dists,
													 n_avail, p_half, p_sel);
				else
				{
					n_sel = Min(n_avail, p_half);
					memcpy(p_sel, p_ids, n_sel * sizeof(int));
				}

				for (int i = 0; i < n_sel; i++)
				{
					my_nbr[start + g_half + p_filled] = p_sel[i];
					p_filled++;

					/* bidirectional: reverse edge into the member's payload half */
					rev[n_rev].target     = p_sel[i];
					rev[n_rev].slot_base  =
						HnswNeighborStart(m_eff, mb->nodes[p_sel[i]].level, 0) + g_half;
					rev[n_rev].slot_count = p_half;
					rev[n_rev].part       = part;
					n_rev++;
				}
				pfree(p_sel);
				pfree(p_ids);
				pfree(p_dists);
			}

			/*
			 * Graceful degradation: if the partition is still too small, fill
			 * remaining payload slots with the next-best UNUSED global
			 * candidates (forward edges only — they are not partition members).
			 */
			for (int i = 0; i < n_cands && p_filled < p_half; i++)
			{
				bool used = false;

				for (int j = 0; j < n_global; j++)
				{
					if (g_sel[j] == out_ids[i])
					{
						used = true;
						break;
					}
				}
				for (int j = 0; !used && j < p_filled; j++)
				{
					if (my_nbr[start + g_half + j] == out_ids[i])
						used = true;
				}
				if (used)
					continue;
				my_nbr[start + g_half + p_filled] = out_ids[i];
				p_filled++;
			}

			/* Reverse edges for the global half into neighbors' global halves */
			for (int i = 0; i < n_global; i++)
			{
				rev[n_rev].target     = g_sel[i];
				rev[n_rev].slot_base  =
					HnswNeighborStart(m_eff, mb->nodes[g_sel[i]].level, 0);
				rev[n_rev].slot_count = g_half;
				rev[n_rev].part       = -1;
				n_rev++;
			}
			pfree(g_sel);
		}
		else
		{
			int  n_sel;
			int *sel = palloc(sizeof(int) * layer_m);

			if (diversify)
				n_sel = acorn_mem_select_diverse(mb, dist, out_ids, out_dists,
												 n_cands, layer_m, sel);
			else
			{
				n_sel = Min(n_cands, layer_m);
				memcpy(sel, out_ids, n_sel * sizeof(int));
			}

			for (int i = 0; i < n_sel; i++)
				my_nbr[start + i] = sel[i];

			/* Reverse edges with fixed-slot retry over the whole layer range */
			for (int i = 0; i < n_sel; i++)
			{
				rev[n_rev].target     = sel[i];
				rev[n_rev].slot_base  =
					HnswNeighborStart(m_eff, mb->nodes[sel[i]].level, lc);
				rev[n_rev].slot_count = layer_m;
				rev[n_rev].part       = -1;
				n_rev++;
			}
			pfree(sel);
		}

		if (n_cands > 0)
			ep_id = out_ids[0];
	}

	/*
	 * Apply the deferred reverse edges.  The first one published makes the
	 * node reachable by other participants; everything above is complete.
	 */
	for (int i = 0; i < n_rev; i++)
		acorn_mem_add_reverse_edge(mb, dist, rev[i].target, new_id,
								   rev[i].slot_base, rev[i].slot_count,
								   rev[i].part, diversify);

	pfree(rev);
	pfree(out_ids);
	pfree(out_dists);

	if (level > entry_level)
	{
		/* exclusive entryLock held in a parallel build (taken above) */
		if (sh)
		{
			sh->entry_id    = new_id;
			sh->entry_level = level;
		}
		else
		{
			mb->entry_id    = new_id;
			mb->entry_level = level;
		}
	}
	if (sh)
		LWLockRelease(&sh->entryLock);

	acorn_mem_register_part_entry(mb, node, new_id);
}

/* -----------------------------------------------------------------------
 * TID pre-assignment: simulate page packing without touching disk.
 * Must exactly mirror acorn_append_tuple's page-selection logic.
 * ----------------------------------------------------------------------- */

#define SIM_PAGE_LOWER_INIT  ((int) SizeOfPageHeaderData)
#define SIM_PAGE_UPPER_INIT  ((int) (BLCKSZ - MAXALIGN(sizeof(HnswPageOpaqueData))))

typedef struct SimPage
{
	BlockNumber blkno;
	int			pd_lower;
	int			pd_upper;
	int			nitems;
} SimPage;

static void
sim_page_init(SimPage *sp)
{
	sp->blkno    = 1;				/* block 0 = meta */
	sp->pd_lower = SIM_PAGE_LOWER_INIT;
	sp->pd_upper = SIM_PAGE_UPPER_INIT;
	sp->nitems   = 0;
}

/* True if an item of `size` (already MAXALIGN'd) fits on the current page. */
static bool
sim_page_fits(const SimPage *sp, Size size)
{
	int free_space = sp->pd_upper - sp->pd_lower - (int) sizeof(ItemIdData);
	return free_space >= (int) size;
}

static OffsetNumber
sim_page_alloc(SimPage *sp, Size size)
{
	sp->pd_lower += (int) sizeof(ItemIdData);
	sp->pd_upper -= (int) size;
	sp->nitems++;
	return (OffsetNumber) sp->nitems;
}

static void
sim_page_advance(SimPage *sp)
{
	sp->blkno++;
	sp->pd_lower = SIM_PAGE_LOWER_INIT;
	sp->pd_upper = SIM_PAGE_UPPER_INIT;
	sp->nitems   = 0;
}

/*
 * Pre-assign on-disk locations for all node tuples.
 * Simulates the two-pass write order: all neighbor tuples first, then all
 * element tuples, in the same sequential page stream.  With inline vectors
 * each node's continuation chunk tuples are appended directly after its
 * neighbor tuple (the flush must write in exactly this order).
 */
static void
acorn_mem_preassign_tids(AcornMemBuild *mb, int m_eff)
{
	SimPage sp;

	sim_page_init(&sp);

	/* Pass 1: neighbor tuples (+ inline continuation chunks) */
	for (int i = 0; i < mb->n_nodes; i++)
	{
		AcornMemNode *node = &mb->nodes[i];
		Size          ntup_sz;

		if (mb->inline_vectors)
		{
			int		layer0_n = HnswGetLayerM(m_eff, 0);
			int		n1 = Min(layer0_n,
							 acorn_t2_inline_primary_cap(node->level, m_eff,
														 mb->entry_size));
			int		cont_cap = acorn_t2_inline_cont_cap(mb->entry_size);
			int		rest = layer0_n - n1;

			ntup_sz = ACORN_T2_INLINE_NTUP_SIZE(node->level, m_eff, n1,
												mb->entry_size);
			if (!sim_page_fits(&sp, ntup_sz))
				sim_page_advance(&sp);
			node->nbr_blkno = sp.blkno;
			node->nbr_offno = sim_page_alloc(&sp, ntup_sz);

			node->n_conts = (cont_cap > 0)
				? (rest + cont_cap - 1) / cont_cap : 0;
			node->cont_tids = (node->n_conts > 0)
				? MemoryContextAlloc(mb->build_ctx,
									 node->n_conts * sizeof(ItemPointerData))
				: NULL;
			for (int c = 0; c < node->n_conts; c++)
			{
				int		n_here = Min(rest, cont_cap);
				Size	csz = ACORN_T2_INLINE_CONT_SIZE(n_here, mb->entry_size);

				if (!sim_page_fits(&sp, csz))
					sim_page_advance(&sp);
				{
					BlockNumber		cb = sp.blkno;
					OffsetNumber	co = sim_page_alloc(&sp, csz);

					ItemPointerSet(&node->cont_tids[c], cb, co);
				}
				rest -= n_here;
			}
		}
		else
		{
			ntup_sz = HNSW_NEIGHBOR_TUPLE_SIZE(node->level, m_eff);
			if (!sim_page_fits(&sp, ntup_sz))
				sim_page_advance(&sp);
			node->nbr_blkno = sp.blkno;
			node->nbr_offno = sim_page_alloc(&sp, ntup_sz);
		}
	}

	/* Pass 2: element tuples (page stream continues from pass 1) */
	for (int i = 0; i < mb->n_nodes; i++)
	{
		AcornMemNode *node    = &mb->nodes[i];
		Size          etup_sz = ACORN_T2_ELEMENT_TUPLE_SIZE(node->vsize);

		if (!sim_page_fits(&sp, etup_sz))
			sim_page_advance(&sp);
		node->elem_blkno = sp.blkno;
		node->elem_offno = sim_page_alloc(&sp, etup_sz);
	}
}

/*
 * Fill one inline entry from in-memory node nbr_id (the edge target).
 * Entry memory comes from a palloc0'd tuple, so padding bytes stay zero.
 */
static void
acorn_mem_fill_inline_entry(AcornMemBuild *mb, AcornT2InlineEntry *e, int nbr_id)
{
	AcornMemNode   *t = &mb->nodes[nbr_id];
	AcornPgVector  *v = (AcornPgVector *) DatumGetPointer(acorn_node_vec(mb, t));

	if ((int) v->dim != mb->dims)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("acorn_hnsw: vector dimension %d does not match index dimension %d",
						(int) v->dim, mb->dims)));

	ItemPointerSet(&e->indextid, t->elem_blkno, t->elem_offno);
	e->heaptid = t->heaptid;
	ItemPointerSet(&e->nbrtid, t->nbr_blkno, t->nbr_offno);
	e->level = (uint8) t->level;
	e->flags = ACORN_T2_INLINE_VALID;
	e->filter_val = t->filter_val;
	acorn_sq8_encode(mb->dims, v->x, e->code, &e->scale, &e->offset);
}

/*
 * Write node i's neighbor tuple with co-located inline entries, plus its
 * continuation chunk tuples.  TID slots stay invalid (patched in pass 3 with
 * the same element TIDs the inline entries carry); inline entries are
 * complete here because acorn_mem_preassign_tids already fixed every tuple's
 * on-disk location.
 */
static void
acorn_mem_flush_inline_node(AcornMemBuild *mb, Relation index,
							ForkNumber forkNum, int m_eff, int i)
{
	AcornMemNode *node     = &mb->nodes[i];
	int			 *my_nbr   = acorn_node_nbr(mb, node);
	Size		  esz      = mb->entry_size;
	int			  layer0_n = HnswGetLayerM(m_eff, 0);
	int			  n1       = Min(layer0_n,
								 acorn_t2_inline_primary_cap(node->level, m_eff, esz));
	int			  cont_cap = acorn_t2_inline_cont_cap(esz);
	int			  start0   = HnswNeighborStart(m_eff, node->level, 0);
	int			  done;
	BlockNumber	  blkno_out;
	OffsetNumber  off_out;

	/* Primary: header + TID slots + first inline chunk */
	{
		Size				sz   = ACORN_T2_INLINE_NTUP_SIZE(node->level, m_eff,
															 n1, esz);
		HnswNeighborTuple	ntup = palloc0(sz);
		AcornT2InlineHdr	hdr;
		char			   *entries;

		ntup->type    = HNSW_NEIGHBOR_TUPLE_TYPE;
		ntup->version = HNSW_VERSION;
		ntup->count   = (uint16) ((node->level + 2) * m_eff);

		hdr = AcornT2NeighborInlineHdr(ntup);
		hdr->n_here     = (uint16) n1;
		hdr->start      = 0;
		hdr->entry_size = (uint32) esz;
		if (node->n_conts > 0)
			hdr->next = node->cont_tids[0];
		else
			ItemPointerSetInvalid(&hdr->next);

		entries = AcornT2InlineHdrEntries(hdr);
		for (int j = 0; j < n1; j++)
		{
			int nbr_id = my_nbr[start0 + j];

			if (nbr_id < 0)
				continue;		/* palloc0: flags == 0 == invalid entry */
			acorn_mem_fill_inline_entry(mb,
										AcornT2InlineEntryAt(entries, j, esz),
										nbr_id);
		}

		acorn_append_tuple(index, forkNum, (Item) ntup, sz,
						   &blkno_out, &off_out, false);
		Assert(blkno_out == node->nbr_blkno && off_out == node->nbr_offno);
		pfree(ntup);
	}

	/* Continuation chunks */
	done = n1;
	for (int c = 0; c < node->n_conts; c++)
	{
		int					n_here = Min(layer0_n - done, cont_cap);
		Size				sz     = ACORN_T2_INLINE_CONT_SIZE(n_here, esz);
		AcornT2InlineCont	cont   = palloc0(sz);
		char			   *entries = AcornT2ContEntries(cont);

		cont->type           = ACORN_T2_INLINE_TUPLE_TYPE;
		cont->version        = HNSW_VERSION;
		cont->hdr.n_here     = (uint16) n_here;
		cont->hdr.start      = (uint16) done;
		cont->hdr.entry_size = (uint32) esz;
		if (c + 1 < node->n_conts)
			cont->hdr.next = node->cont_tids[c + 1];
		else
			ItemPointerSetInvalid(&cont->hdr.next);

		for (int j = 0; j < n_here; j++)
		{
			int nbr_id = my_nbr[start0 + done + j];

			if (nbr_id < 0)
				continue;
			acorn_mem_fill_inline_entry(mb,
										AcornT2InlineEntryAt(entries, j, esz),
										nbr_id);
		}

		acorn_append_tuple(index, forkNum, (Item) cont, sz,
						   &blkno_out, &off_out, false);
		Assert(blkno_out == ItemPointerGetBlockNumber(&node->cont_tids[c]) &&
			   off_out  == ItemPointerGetOffsetNumber(&node->cont_tids[c]));
		pfree(cont);
		done += n_here;
	}
}

/*
 * Flush in-memory graph to disk.
 *
 * Pass 1: write neighbor tuples (TID slots all invalid — patched in pass 3;
 *         with inline vectors the co-located entries + continuation chunks
 *         are written complete here, from pre-assigned TIDs).
 * Pass 2: write element tuples  (neighbortid valid from pre-assign).
 * Pass 3: patch neighbor TID slots with element TIDs.
 * Then update meta entry point.
 */
static void
acorn_mem_flush(AcornMemBuild *mb, Relation index, ForkNumber forkNum, int m_eff)
{
	BlockNumber  blkno_out;
	OffsetNumber off_out;
	int			 n_nodes  = mb->shared ? mb->shared->n_nodes : mb->n_nodes;
	int			 entry_id = mb->shared ? mb->shared->entry_id : mb->entry_id;

	if (n_nodes == 0)
		return;

	/* keep the serial fields coherent for the helpers below */
	mb->n_nodes = n_nodes;

	acorn_mem_preassign_tids(mb, m_eff);

	/* Pass 1: write all neighbor tuples */
	for (int i = 0; i < n_nodes; i++)
	{
		if (mb->inline_vectors)
		{
			acorn_mem_flush_inline_node(mb, index, forkNum, m_eff, i);
			continue;
		}

		{
			AcornMemNode     *node    = &mb->nodes[i];
			Size              ntup_sz = HNSW_NEIGHBOR_TUPLE_SIZE(node->level, m_eff);
			HnswNeighborTuple ntup    = palloc0(ntup_sz);

			ntup->type    = HNSW_NEIGHBOR_TUPLE_TYPE;
			ntup->version = HNSW_VERSION;
			ntup->count   = (uint16) ((node->level + 2) * m_eff);

			acorn_append_tuple(index, forkNum, (Item) ntup, ntup_sz,
							   &blkno_out, &off_out, false);
			Assert(blkno_out == node->nbr_blkno && off_out == node->nbr_offno);
			pfree(ntup);
		}
	}

	/* Pass 2: write all element tuples */
	for (int i = 0; i < n_nodes; i++)
	{
		AcornMemNode       *node    = &mb->nodes[i];
		Size                etup_sz = ACORN_T2_ELEMENT_TUPLE_SIZE(node->vsize);
		AcornT2ElementTuple etup    = palloc0(etup_sz);

		etup->type    = HNSW_ELEMENT_TUPLE_TYPE;
		etup->level   = (uint8) node->level;
		etup->version = HNSW_VERSION;
		for (int k = 0; k < HNSW_HEAPTIDS; k++)
			ItemPointerSetInvalid(&etup->heaptids[k]);
		etup->heaptids[0] = node->heaptid;
		ItemPointerSet(&etup->neighbortid, node->nbr_blkno, node->nbr_offno);
		etup->filter_val  = node->filter_val;
		memcpy(AcornT2ElementTupleGetVector(etup),
			   DatumGetPointer(acorn_node_vec(mb, node)), node->vsize);

		acorn_append_tuple(index, forkNum, (Item) etup, etup_sz,
						   &blkno_out, &off_out, false);
		Assert(blkno_out == node->elem_blkno && off_out == node->elem_offno);
		pfree(etup);
	}

	/* Pass 3: patch neighbor TID slots with element TIDs */
	for (int i = 0; i < n_nodes; i++)
	{
		AcornMemNode *node   = &mb->nodes[i];
		int			 *my_nbr = acorn_node_nbr(mb, node);

		for (int lc = node->level; lc >= 0; lc--)
		{
			int start   = HnswNeighborStart(m_eff, node->level, lc);
			int layer_m = HnswGetLayerM(m_eff, lc);

			for (int j = 0; j < layer_m; j++)
			{
				int               nbr_id = my_nbr[start + j];
				Buffer            buf;
				Page              page;
				HnswNeighborTuple ntup;

				if (nbr_id < 0)
					continue;	/* payload split may leave gaps between halves */

				buf  = ReadBufferExtended(index, forkNum,
										  node->nbr_blkno, RBM_NORMAL, NULL);
				LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
				page = BufferGetPage(buf);
				ntup = (HnswNeighborTuple)
					PageGetItem(page, PageGetItemId(page, node->nbr_offno));
				ItemPointerSet(
					&HnswNeighborTupleGetTids(ntup)[start + j],
					mb->nodes[nbr_id].elem_blkno,
					mb->nodes[nbr_id].elem_offno);
				MarkBufferDirty(buf);
				UnlockReleaseBuffer(buf);
			}
		}
	}

	/* Update meta entry point */
	if (entry_id >= 0)
	{
		AcornMemNode *ep = &mb->nodes[entry_id];
		acorn_maybe_update_entry(index, ep->elem_blkno, ep->elem_offno,
								  ep->level, false);
	}
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
						   Datum query, const AcornDistCtx *dist, int m_eff,
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
										   query, dist, NULL);
			ItemPointerSet(&c_tid, *cur_blkno, *cur_offno);
			acorn_read_element(index, forkNum, &c_tid, &c_level,
							   &c_nbr_blk, &c_nbr_off, NULL, NULL, NULL, NULL);

			n_nbrs = acorn_read_nbr_tids_at_layer(index, forkNum,
												   c_nbr_blk, c_nbr_off,
												   c_level, lc, m_eff,
												   nbrs, HNSW_MAX_NEIGHBORS);

			for (int i = 0; i < n_nbrs; i++)
			{
				double nd = acorn_node_distance(index, forkNum,
												 ItemPointerGetBlockNumber(&nbrs[i]),
												 ItemPointerGetOffsetNumber(&nbrs[i]),
												 query, dist, NULL);
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
 * Bounded beam search at `layer` for construction.
 *
 * Implements the standard HNSW Algorithm 2 "SEARCH-LAYER" with ef-bounded
 * candidate set.  Returns up to ef candidates in out_cands[] sorted
 * nearest-first; out_cands must have ef slots.  Returns actual count.
 *
 * If part >= 0 the result set W is restricted to nodes whose filter_val maps
 * to that payload partition while exploration (C) remains unrestricted —
 * the ACORN-style search used by the incremental-insert payload-edge path,
 * where no in-memory partition entry map exists.  max_expansions caps the
 * work in that mode (a sparse partition might otherwise force a full
 * traversal); pass 0 for the standard unrestricted search.
 */
static int
acorn_search_layer_construction(Relation index, ForkNumber forkNum,
								  BlockNumber entry_blkno, OffsetNumber entry_offno,
								  Datum query, const AcornDistCtx *dist, int m_eff,
								  int layer, int ef,
								  AcornCand *out_cands,
								  int part, int max_expansions)
{
	pairingheap    *C;			/* min-heap: candidates (closest at top) */
	pairingheap    *W;			/* max-heap: current best set (furthest at top) */
	HTAB		   *visited;
	int				W_count = 0;
	MemoryContext	tmp_ctx, old_ctx;
	int				n_out;
	int				n_expansions = 0;

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
		int64			ep_fval;
		BuildPQNode    *cn, *wn;

		ItemPointerSet(&ep_tid, entry_blkno, entry_offno);
		build_mark_visited(visited, &ep_tid);
		ep_dist = acorn_node_distance(index, forkNum,
									   entry_blkno, entry_offno,
									   query, dist, &ep_fval);

		cn           = palloc(sizeof(BuildPQNode));
		cn->tid      = ep_tid;
		cn->distance = ep_dist;
		pairingheap_add(C, &cn->ph_node);

		if (part < 0 || acorn_payload_partition(ep_fval) == part)
		{
			wn           = palloc(sizeof(BuildPQNode));
			wn->tid      = ep_tid;
			wn->distance = ep_dist;
			pairingheap_add(W, &wn->ph_node);
			W_count = 1;
		}
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
		f_dist = pairingheap_is_empty(W) ? DBL_MAX
			: ((BuildPQNode *) pairingheap_first(W))->distance;
		if (c_dist > f_dist && W_count >= ef)
			break;

		/* Restricted mode: bound total exploration work */
		if (max_expansions > 0 && n_expansions >= max_expansions)
			break;
		n_expansions++;

		/* Load c's level + neighbor tuple location (no lock held at this point) */
		acorn_read_element(index, forkNum, &c_tid, &c_level,
						   &c_nbr_blk, &c_nbr_off, NULL, NULL, NULL, NULL);

		n_nbrs = acorn_read_nbr_tids_at_layer(index, forkNum,
											   c_nbr_blk, c_nbr_off,
											   c_level, layer, m_eff,
											   nbrs, HNSW_MAX_NEIGHBORS);

		for (int i = 0; i < n_nbrs; i++)
		{
			double		 nd;
			int64		 nfval;
			bool		 is_member;
			BuildPQNode *cn, *wn;

			if (!ItemPointerIsValid(&nbrs[i]))
				continue;
			if (build_is_visited(visited, &nbrs[i]))
				continue;
			build_mark_visited(visited, &nbrs[i]);

			nd = acorn_node_distance(index, forkNum,
									  ItemPointerGetBlockNumber(&nbrs[i]),
									  ItemPointerGetOffsetNumber(&nbrs[i]),
									  query, dist, &nfval);
			is_member = (part < 0 || acorn_payload_partition(nfval) == part);

			f_dist = pairingheap_is_empty(W) ? DBL_MAX
				: ((BuildPQNode *) pairingheap_first(W))->distance;
			if (nd < f_dist || W_count < ef || part >= 0)
			{
				cn           = palloc(sizeof(BuildPQNode));
				cn->tid      = nbrs[i];
				cn->distance = nd;
				pairingheap_add(C, &cn->ph_node);

				if (is_member && (nd < f_dist || W_count < ef))
				{
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

/*
 * Diversity selection over disk candidates (AcornCand tid/distance pairs
 * sorted ascending by distance to the base).  Mirrors acorn_mem_select_diverse
 * (Malkov Alg. 4 + keepPrunedConnections); reads each candidate's inline
 * vector once for the pairwise checks.  Writes up to m_max kept candidates to
 * kept_out and returns the count.
 */
static int
acorn_select_diverse_tids(Relation index, ForkNumber forkNum,
						  const AcornDistCtx *dist,
						  const AcornCand *cands, int n_cands,
						  int m_max, AcornCand *kept_out)
{
	Datum	   *vecs;
	int		   *kept_idx;
	int		   *pruned;
	int			n_kept = 0;
	int			n_pruned = 0;

	if (n_cands <= m_max)
	{
		/* keepPrunedConnections refills everything anyway — keep all */
		memcpy(kept_out, cands, n_cands * sizeof(AcornCand));
		return n_cands;
	}

	vecs     = palloc(n_cands * sizeof(Datum));
	kept_idx = palloc(m_max * sizeof(int));
	pruned   = palloc(n_cands * sizeof(int));

	for (int i = 0; i < n_cands; i++)
	{
		ItemPointerData tid = cands[i].tid;

		acorn_read_element(index, forkNum, &tid, NULL, NULL, NULL,
						   &vecs[i], NULL, NULL, NULL);
	}

	for (int i = 0; i < n_cands && n_kept < m_max; i++)
	{
		bool keep = true;

		for (int j = 0; j < n_kept; j++)
		{
			double d = acorn_dist(dist, vecs[i], vecs[kept_idx[j]]);

			if (d <= cands[i].distance)
			{
				keep = false;
				break;
			}
		}
		if (keep)
			kept_idx[n_kept++] = i;
		else
			pruned[n_pruned++] = i;
	}

	/* keepPrunedConnections: refill remaining slots from pruned, in order */
	for (int p = 0; p < n_pruned && n_kept < m_max; p++)
		kept_idx[n_kept++] = pruned[p];

	for (int i = 0; i < n_kept; i++)
		kept_out[i] = cands[kept_idx[i]];

	for (int i = 0; i < n_cands; i++)
		pfree(DatumGetPointer(vecs[i]));
	pfree(vecs);
	pfree(kept_idx);
	pfree(pruned);
	return n_kept;
}

/* -----------------------------------------------------------------------
 * Fixed-slot retry reverse edge (generalized to any layer)
 * ----------------------------------------------------------------------- */

/*
 * Update the co-located inline entry for absolute layer-0 slot `slot` of the
 * element whose neighbor tuple is at (nbr_blk, nbr_off), setting it to
 * *e_entry (insert path: the reverse edge just stored E in that TID slot).
 *
 * Two-phase: walk the chunk chain under SHARE to find the covering chunk,
 * then rewrite the entry under EXCLUSIVE.  The caller updates the TID slot
 * BEFORE this; between the two writes (or after a crash that loses this
 * one) the scan sees entry.indextid != slot TID for this edge and reads the
 * target's element page instead — slower, never wrong.
 */
static void
acorn_t2_inline_write_entry(Relation index, ForkNumber forkNum,
							BlockNumber nbr_blk, OffsetNumber nbr_off,
							int slot, const AcornT2InlineEntry *e_entry,
							Size esz, bool use_wal)
{
	BlockNumber		chunk_blk = nbr_blk;
	OffsetNumber	chunk_off = nbr_off;
	bool			primary   = true;
	int				idx = -1;
	Size			entries_off = 0;

	/* Phase 1: locate the covering chunk (SHARE, one page at a time) */
	for (;;)
	{
		Buffer			buf;
		Page			page;
		ItemPointerData	next;
		uint16			start;
		uint16			n_here;

		buf  = ReadBufferExtended(index, forkNum, chunk_blk, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);

		if (primary)
		{
			HnswNeighborTuple ntup = (HnswNeighborTuple)
				PageGetItem(page, PageGetItemId(page, chunk_off));
			AcornT2InlineHdr  hdr  = AcornT2NeighborInlineHdr(ntup);

			start       = hdr->start;
			n_here      = hdr->n_here;
			next        = hdr->next;
			entries_off = ACORN_T2_INLINE_HDR_OFFSET(ntup->count)
				+ sizeof(AcornT2InlineHdrData);
		}
		else
		{
			AcornT2InlineCont cont = (AcornT2InlineCont)
				PageGetItem(page, PageGetItemId(page, chunk_off));

			Assert(cont->type == ACORN_T2_INLINE_TUPLE_TYPE);
			start       = cont->hdr.start;
			n_here      = cont->hdr.n_here;
			next        = cont->hdr.next;
			entries_off = sizeof(AcornT2InlineContData);
		}
		UnlockReleaseBuffer(buf);

		if (slot >= (int) start && slot < (int) start + (int) n_here)
		{
			idx = slot - (int) start;
			break;
		}
		if (!ItemPointerIsValid(&next))
			return;				/* slot not covered — reader uses element page */
		chunk_blk = ItemPointerGetBlockNumber(&next);
		chunk_off = ItemPointerGetOffsetNumber(&next);
		primary   = false;
	}

	/* Phase 2: write the entry under EXCLUSIVE */
	{
		Buffer	buf;
		Page	page;
		char   *tup;

		buf = ReadBufferExtended(index, forkNum, chunk_blk, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (use_wal)
		{
			GenericXLogState *state = GenericXLogStart(index);

			page = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
			tup  = (char *) PageGetItem(page, PageGetItemId(page, chunk_off));
			memcpy(tup + entries_off + (Size) idx * esz, e_entry, esz);
			GenericXLogFinish(state);
		}
		else
		{
			page = BufferGetPage(buf);
			tup  = (char *) PageGetItem(page, PageGetItemId(page, chunk_off));
			memcpy(tup + entries_off + (Size) idx * esz, e_entry, esz);
			MarkBufferDirty(buf);
		}
		UnlockReleaseBuffer(buf);
	}
}

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
 *
 * e_entry (nullable): E's pre-built inline entry; when set and layer == 0,
 * the co-located entry for the written slot is updated after the TID slot
 * (vector co-location, correct-first ordering documented above).
 */
static void
acorn_add_reverse_edge_at_layer(Relation index, ForkNumber forkNum,
								 ItemPointer n_tid,
								 BlockNumber e_blk, OffsetNumber e_off,
								 Datum e_value, const AcornDistCtx *dist,
								 int m_eff, int layer,
								 int range_off, int range_len, int part,
								 bool diversify,
								 const AcornT2InlineEntry *e_entry, Size e_esz,
								 bool use_wal)
{
	int				  n_level;
	BlockNumber		  n_nbr_blk;
	OffsetNumber	  n_nbr_off;
	Datum			  n_vec;
	Buffer			  buf;
	Page			  page;
	HnswNeighborTuple ntup;
	ItemPointerData  *tids;
	ItemPointerData	  tids_copy[HNSW_MAX_NEIGHBORS];
	int				  start;
	int				  target = -1;
	int				  nonmember_j = -1;
	double			  nonmember_d = -DBL_MAX;

	Assert(range_len <= HNSW_MAX_NEIGHBORS);
	Assert(range_off + range_len <= HnswGetLayerM(m_eff, layer));

	/* --- Phase 1: read N's element tuple (level + neighbor location + vector) --- */
	acorn_read_element(index, forkNum, n_tid, &n_level,
					   &n_nbr_blk, &n_nbr_off, &n_vec, NULL, NULL, NULL);
	start = HnswNeighborStart(m_eff, n_level, layer) + range_off;

	/* --- Phase 1a: copy N's range slots under SHARE (release before any write) --- */
	buf  = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
	tids = HnswNeighborTupleGetTids(ntup);
	memcpy(tids_copy, &tids[start], range_len * sizeof(ItemPointerData));
	UnlockReleaseBuffer(buf);

	/* Find the first empty slot; bail if E is already connected */
	for (int j = 0; j < range_len; j++)
	{
		if (!ItemPointerIsValid(&tids_copy[j]))
		{
			if (target < 0)
				target = j;
			continue;
		}
		if (ItemPointerGetBlockNumber(&tids_copy[j]) == e_blk &&
			ItemPointerGetOffsetNumber(&tids_copy[j]) == e_off)
		{
			pfree(DatumGetPointer(n_vec));
			return;
		}
	}

	if (target < 0 && part >= 0)
	{
		/*
		 * Payload half full: prefer evicting the furthest occupant that is NOT
		 * a partition member (graceful-degradation backfill) — a same-
		 * partition edge is strictly more valuable there.
		 */
		for (int j = 0; j < range_len; j++)
		{
			int64  fj;
			Datum  vj;
			double d;

			acorn_read_element(index, forkNum, &tids_copy[j],
							   NULL, NULL, NULL, &vj, &fj, NULL, NULL);
			d = acorn_dist(dist, n_vec, vj);
			pfree(DatumGetPointer(vj));
			if (acorn_payload_partition(fj) != part && d > nonmember_d)
			{
				nonmember_d = d;
				nonmember_j = j;
			}
		}
		if (nonmember_j >= 0)
			target = nonmember_j;
	}

	if (target < 0 && diversify)
	{
		/*
		 * --- Phase 1b (diversify): heuristic re-selection over existing + E.
		 * n_cands = range_len + 1 while keepPrunedConnections refills to
		 * range_len, so exactly one candidate is dropped: if it is an
		 * existing occupant E takes its slot, if it is E nothing changes.
		 * Candidate index range_len denotes E itself.
		 */
		int		n_cands  = range_len + 1;
		Datum  *vecs     = palloc(n_cands * sizeof(Datum));
		double *dists    = palloc(n_cands * sizeof(double));
		int	   *order    = palloc(n_cands * sizeof(int));
		int	   *kept     = palloc(range_len * sizeof(int));
		int	   *pruned   = palloc(n_cands * sizeof(int));
		int		n_kept   = 0;
		int		n_pruned = 0;
		bool	e_kept   = false;

		for (int j = 0; j < range_len; j++)
		{
			acorn_read_element(index, forkNum, &tids_copy[j],
							   NULL, NULL, NULL, &vecs[j], NULL, NULL, NULL);
			dists[j] = acorn_dist(dist, n_vec, vecs[j]);
			order[j] = j;
		}
		vecs[range_len]  = e_value;		/* not palloc'd here — do not pfree */
		dists[range_len] = acorn_dist(dist, n_vec, e_value);
		order[range_len] = range_len;

		/* insertion sort of order[] ascending by distance to N */
		for (int i = 1; i < n_cands; i++)
		{
			int    oi = order[i];
			double di = dists[oi];
			int    k  = i - 1;

			while (k >= 0 && dists[order[k]] > di)
			{
				order[k + 1] = order[k];
				k--;
			}
			order[k + 1] = oi;
		}

		for (int i = 0; i < n_cands && n_kept < range_len; i++)
		{
			int  ci   = order[i];
			bool keep = true;

			for (int j = 0; j < n_kept; j++)
			{
				double d = acorn_dist(dist, vecs[ci], vecs[kept[j]]);

				if (d <= dists[ci])
				{
					keep = false;
					break;
				}
			}
			if (keep)
				kept[n_kept++] = ci;
			else
				pruned[n_pruned++] = ci;
		}
		for (int p = 0; p < n_pruned && n_kept < range_len; p++)
			kept[n_kept++] = pruned[p];

		for (int i = 0; i < n_kept; i++)
		{
			if (kept[i] == range_len)
			{
				e_kept = true;
				break;
			}
		}
		if (e_kept)
		{
			/* find the single dropped existing occupant's slot */
			for (int j = 0; j < range_len; j++)
			{
				bool in_kept = false;

				for (int i = 0; i < n_kept; i++)
				{
					if (kept[i] == j)
					{
						in_kept = true;
						break;
					}
				}
				if (!in_kept)
				{
					target = j;
					break;
				}
			}
		}

		for (int j = 0; j < range_len; j++)
			pfree(DatumGetPointer(vecs[j]));
		pfree(vecs);
		pfree(dists);
		pfree(order);
		pfree(kept);
		pfree(pruned);
	}
	else if (target < 0)
	{
		/* --- Phase 1b: full — retry: find furthest, replace if E is closer --- */
		double furthest_d = -DBL_MAX;
		int    furthest_j = -1;
		double dEN        = acorn_dist(dist, n_vec, e_value);

		for (int j = 0; j < range_len; j++)
		{
			Datum  vj;
			double d;

			acorn_read_element(index, forkNum, &tids_copy[j],
							   NULL, NULL, NULL, &vj, NULL, NULL, NULL);
			d = acorn_dist(dist, n_vec, vj);
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
	buf = ReadBufferExtended(index, forkNum, n_nbr_blk, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (use_wal)
	{
		GenericXLogState *state;

		state = GenericXLogStart(index);
		page  = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
		ntup  = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
		tids  = HnswNeighborTupleGetTids(ntup);
		ItemPointerSet(&tids[start + target], e_blk, e_off);
		GenericXLogFinish(state);
	}
	else
	{
		page = BufferGetPage(buf);
		ntup = (HnswNeighborTuple) PageGetItem(page, PageGetItemId(page, n_nbr_off));
		tids = HnswNeighborTupleGetTids(ntup);
		ItemPointerSet(&tids[start + target], e_blk, e_off);
		MarkBufferDirty(buf);
	}

	UnlockReleaseBuffer(buf);

	/* Vector co-location: sync N's inline entry for the written slot */
	if (e_entry != NULL && layer == 0)
		acorn_t2_inline_write_entry(index, forkNum, n_nbr_blk, n_nbr_off,
									range_off + target, e_entry, e_esz,
									use_wal);
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
					 int64 filter_val,
					 ItemPointer heaptid, unsigned short rand_state[3],
					 bool use_wal)
{
	int					m_eff    = acorn_m_eff(index);
	int					efc      = acorn_opt_ef_construction(index);
	bool				diversify = acorn_opt_diversify(index);
	AcornDistCtx		dist_ctx;
	const AcornDistCtx *dist     = &dist_ctx;
	Size				vsize    = VARSIZE_ANY(DatumGetPointer(value));
	Size				etupSize = ACORN_T2_ELEMENT_TUPLE_SIZE(vsize);

	acorn_dist_ctx_init(index, &dist_ctx);

	if (etupSize > HNSW_MAX_SIZE)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("acorn_hnsw: vector too large to index"),
				 errdetail("Element tuple size %zu exceeds page limit %zu. "
						   "Reduce dimensions.", etupSize, HNSW_MAX_SIZE)));

	int					l_new;
	Size				ntupSize;
	HnswNeighborTuple	ntup;
	AcornT2ElementTuple etup;
	ItemPointerData  *ntids;
	BlockNumber		  e_blk, n_blk;
	OffsetNumber	  e_off, n_off;
	BlockNumber		  entry_blkno;
	OffsetNumber	  entry_offno;
	int				  entry_level;
	bool			  has_entry;
	bool			  payload_on;
	int				  part;
	int				  n_payload_real = 0;	/* partition-sourced layer-0 payload slots */
	int				  meta_dims;
	uint16			  meta_flags;
	bool			  inline_on;
	Size			  esz = 0;
	AcornT2InlineEntry *e_self = NULL;		/* E's own inline entry (reverse edges) */

	payload_on = acorn_opt_payload_edges(index) &&
				 RelationGetDescr(index)->natts > 1;
	part = acorn_payload_partition(filter_val);

	/* Step 1: assign random level */
	l_new = acorn_assign_level(m_eff, rand_state);

	/* Step 2: read current entry point (+ Tier 2 meta: inline layout flag) */
	acorn_read_meta(index, &entry_blkno, &entry_offno, &entry_level,
					&meta_dims, &meta_flags);
	has_entry = BlockNumberIsValid(entry_blkno);
	inline_on = (meta_flags & ACORN_T2_META_INLINE_VECTORS) != 0;
	if (inline_on)
	{
		AcornPgVector *v = (AcornPgVector *) DatumGetPointer(value);

		if ((int) v->dim != meta_dims)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("acorn_hnsw: vector dimension %d does not match index dimension %d",
							(int) v->dim, meta_dims)));
		esz = ACORN_T2_INLINE_ENTRY_SIZE(meta_dims);
	}

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
									   value, dist, m_eff,
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
													   value, dist, m_eff,
													   lc, efc, cands, -1, 0);

			if (lc == 0 && payload_on)
			{
				/*
				 * Split layer-0 slots: first half global nearest, second half
				 * nearest same-partition nodes found by an ACORN-style
				 * restricted search (W limited to the partition, exploration
				 * unrestricted, work capped).
				 */
				int			g_half   = layer_m / 2;		/* = m_eff */
				int			p_half   = layer_m - g_half;
				int			n_global;
				int			p_filled = 0;
				AcornCand  *pcands   = palloc(sizeof(AcornCand) * efc);
				AcornCand  *gsel     = palloc(sizeof(AcornCand) * g_half);
				int			n_part;
				int			n_avail  = 0;

				/* Global half: diversity-select up to g_half from the beam */
				if (diversify)
					n_global = acorn_select_diverse_tids(index, forkNum, dist,
														 cands, n_cands,
														 g_half, gsel);
				else
				{
					n_global = Min(n_cands, g_half);
					memcpy(gsel, cands, n_global * sizeof(AcornCand));
				}
				for (int i = 0; i < n_global; i++)
					ntids[start + i] = gsel[i].tid;

				n_part = acorn_search_layer_construction(index, forkNum,
														  ep_blkno, ep_offno,
														  value, dist, m_eff,
														  0, efc, pcands,
														  part, efc * 8);

				/* drop candidates already wired as global neighbors */
				for (int i = 0; i < n_part; i++)
				{
					bool dup = false;

					for (int j = 0; j < n_global; j++)
					{
						if (ItemPointerEquals(&ntids[start + j], &pcands[i].tid))
						{
							dup = true;
							break;
						}
					}
					if (dup)
						continue;
					pcands[n_avail++] = pcands[i];
				}

				/* Payload half: diversity within the partition too */
				if (diversify)
				{
					AcornCand  *psel  = palloc(sizeof(AcornCand) * p_half);
					int			n_sel = acorn_select_diverse_tids(index, forkNum,
																  dist, pcands,
																  n_avail,
																  p_half, psel);

					for (int i = 0; i < n_sel; i++)
					{
						ntids[start + g_half + p_filled] = psel[i].tid;
						p_filled++;
					}
					pfree(psel);
				}
				else
				{
					for (int i = 0; i < n_avail && p_filled < p_half; i++)
					{
						ntids[start + g_half + p_filled] = pcands[i].tid;
						p_filled++;
					}
				}
				n_payload_real = p_filled;

				/* graceful degradation: backfill with unused global candidates */
				for (int i = 0; i < n_cands && p_filled < p_half; i++)
				{
					bool used = false;

					for (int j = 0; j < n_global; j++)
					{
						if (ItemPointerEquals(&ntids[start + j], &cands[i].tid))
						{
							used = true;
							break;
						}
					}
					for (int j = 0; !used && j < p_filled; j++)
					{
						if (ItemPointerEquals(&ntids[start + g_half + j],
											  &cands[i].tid))
							used = true;
					}
					if (used)
						continue;
					ntids[start + g_half + p_filled] = cands[i].tid;
					p_filled++;
				}
				pfree(gsel);
				pfree(pcands);
			}
			else if (diversify)
			{
				/* Diversity-select up to layer_m of the beam candidates */
				AcornCand  *sel   = palloc(sizeof(AcornCand) * layer_m);
				int			n_sel = acorn_select_diverse_tids(index, forkNum,
															  dist, cands,
															  n_cands,
															  layer_m, sel);

				for (int i = 0; i < n_sel; i++)
					ntids[start + i] = sel[i].tid;
				pfree(sel);
			}
			else
			{
				/* Store min(layer_m, n_cands) nearest as E's neighbors at lc */
				for (int i = 0; i < Min(n_cands, layer_m); i++)
					ntids[start + i] = cands[i].tid;
			}

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
	if (inline_on)
	{
		/*
		 * Vector co-location: gather each selected layer-0 target's payload
		 * (one element-page read per target — correct-first; inserts are the
		 * slow path), then write continuation chunks in reverse chain order
		 * (so each knows its successor's TID) followed by the primary
		 * neighbor tuple carrying the TID slots + first chunk.
		 */
		int		layer0_n = HnswGetLayerM(m_eff, 0);
		int		start0   = HnswNeighborStart(m_eff, l_new, 0);
		int		n1       = Min(layer0_n,
							   acorn_t2_inline_primary_cap(l_new, m_eff, esz));
		int		cont_cap = acorn_t2_inline_cont_cap(esz);
		int		n_conts  = (cont_cap > 0)
			? (layer0_n - n1 + cont_cap - 1) / cont_cap : 0;
		char   *lentries = palloc0((Size) layer0_n * esz);
		ItemPointerData next_tid;

		for (int j = 0; j < layer0_n; j++)
		{
			AcornT2InlineEntry *e = AcornT2InlineEntryAt(lentries, j, esz);
			int				t_level;
			BlockNumber		t_nbr_blk;
			OffsetNumber	t_nbr_off;
			Datum			t_vec;
			int64			t_fval;
			ItemPointerData	t_heaptid;
			bool			t_deleted;
			AcornPgVector  *tv;

			if (!ItemPointerIsValid(&ntids[start0 + j]))
				continue;		/* palloc0: flags == 0 == invalid entry */

			acorn_read_element(index, forkNum, &ntids[start0 + j],
							   &t_level, &t_nbr_blk, &t_nbr_off,
							   &t_vec, &t_fval, &t_heaptid, &t_deleted);
			tv = (AcornPgVector *) DatumGetPointer(t_vec);

			e->indextid = ntids[start0 + j];
			e->heaptid  = t_heaptid;
			ItemPointerSet(&e->nbrtid, t_nbr_blk, t_nbr_off);
			e->level = (uint8) t_level;
			e->flags = ACORN_T2_INLINE_VALID
				| (t_deleted ? ACORN_T2_INLINE_DELETED : 0);
			e->filter_val = t_fval;
			acorn_sq8_encode(meta_dims, tv->x, e->code, &e->scale, &e->offset);
			pfree(tv);
		}

		/* Continuation chunks, last to first */
		ItemPointerSetInvalid(&next_tid);
		for (int c = n_conts - 1; c >= 0; c--)
		{
			int					chunk_start = n1 + c * cont_cap;
			int					n_here = Min(layer0_n - chunk_start, cont_cap);
			Size				csz    = ACORN_T2_INLINE_CONT_SIZE(n_here, esz);
			AcornT2InlineCont	cont   = palloc0(csz);
			BlockNumber			c_blk;
			OffsetNumber		c_off;

			cont->type           = ACORN_T2_INLINE_TUPLE_TYPE;
			cont->version        = HNSW_VERSION;
			cont->hdr.n_here     = (uint16) n_here;
			cont->hdr.start      = (uint16) chunk_start;
			cont->hdr.entry_size = (uint32) esz;
			cont->hdr.next       = next_tid;
			memcpy(AcornT2ContEntries(cont),
				   lentries + (Size) chunk_start * esz,
				   (Size) n_here * esz);

			acorn_append_tuple(index, forkNum, (Item) cont, csz,
							   &c_blk, &c_off, use_wal);
			ItemPointerSet(&next_tid, c_blk, c_off);
			pfree(cont);
		}

		/* Primary neighbor tuple: TID slots + chunk 0 */
		{
			Size				isz = ACORN_T2_INLINE_NTUP_SIZE(l_new, m_eff,
																n1, esz);
			HnswNeighborTuple	intup = palloc0(isz);
			AcornT2InlineHdr	hdr;

			memcpy(intup, ntup, HNSW_NEIGHBOR_TUPLE_SIZE(l_new, m_eff));
			hdr = AcornT2NeighborInlineHdr(intup);
			hdr->n_here     = (uint16) n1;
			hdr->start      = 0;
			hdr->entry_size = (uint32) esz;
			hdr->next       = next_tid;
			memcpy(AcornT2InlineHdrEntries(hdr), lentries, (Size) n1 * esz);

			acorn_append_tuple(index, forkNum, (Item) intup, isz,
							   &n_blk, &n_off, use_wal);
			pfree(intup);
		}
		pfree(lentries);
	}
	else
		acorn_append_tuple(index, forkNum, (Item) ntup, ntupSize, &n_blk, &n_off,
						   use_wal);

	/* Step 6: write element tuple (level, heaptids, neighbortid, filter_val, vector) */
	etup              = palloc0(etupSize);
	etup->type        = HNSW_ELEMENT_TUPLE_TYPE;
	etup->level       = (uint8) l_new;
	etup->deleted     = 0;
	etup->version     = HNSW_VERSION;
	for (int i = 0; i < HNSW_HEAPTIDS; i++)
		ItemPointerSetInvalid(&etup->heaptids[i]);
	etup->heaptids[0] = *heaptid;
	ItemPointerSet(&etup->neighbortid, n_blk, n_off);
	etup->unused      = 0;
	etup->filter_val  = filter_val;
	memcpy(AcornT2ElementTupleGetVector(etup), DatumGetPointer(value), vsize);

	acorn_append_tuple(index, forkNum, (Item) etup, etupSize, &e_blk, &e_off,
					   use_wal);

	/* Step 7: update meta entry point if this element has the highest level */
	acorn_maybe_update_entry(index, e_blk, e_off, l_new, use_wal);

	/* E's own inline entry, applied to reverse-edge targets' layer-0 chunks */
	if (inline_on)
	{
		AcornPgVector *v = (AcornPgVector *) DatumGetPointer(value);

		e_self = palloc0(esz);
		ItemPointerSet(&e_self->indextid, e_blk, e_off);
		e_self->heaptid = *heaptid;
		ItemPointerSet(&e_self->nbrtid, n_blk, n_off);
		e_self->level = (uint8) l_new;
		e_self->flags = ACORN_T2_INLINE_VALID;
		e_self->filter_val = filter_val;
		acorn_sq8_encode(meta_dims, v->x, e_self->code,
						 &e_self->scale, &e_self->offset);
	}

	/* Step 8: add reverse edges at each layer with fixed-slot retry */
	if (has_entry)
	{
		for (int lc = Min(entry_level, l_new); lc >= 0; lc--)
		{
			int layer_m = HnswGetLayerM(m_eff, lc);
			int start   = HnswNeighborStart(m_eff, l_new, lc);

			if (lc == 0 && payload_on)
			{
				int g_half = layer_m / 2;
				int p_half = layer_m - g_half;

				/* global half -> neighbors' global halves */
				for (int i = 0; i < g_half; i++)
				{
					if (!ItemPointerIsValid(&ntids[start + i]))
						continue;
					acorn_add_reverse_edge_at_layer(index, forkNum,
													 &ntids[start + i],
													 e_blk, e_off,
													 value, dist, m_eff, lc,
													 0, g_half, -1, diversify,
													 e_self, esz, use_wal);
				}

				/*
				 * Partition-sourced payload slots -> neighbors' payload
				 * halves.  Backfilled global leftovers (i >= n_payload_real)
				 * stay forward-only.
				 */
				for (int i = 0; i < n_payload_real && i < p_half; i++)
				{
					if (!ItemPointerIsValid(&ntids[start + g_half + i]))
						continue;
					acorn_add_reverse_edge_at_layer(index, forkNum,
													 &ntids[start + g_half + i],
													 e_blk, e_off,
													 value, dist, m_eff, lc,
													 g_half, p_half, part,
													 diversify, e_self, esz,
													 use_wal);
				}
			}
			else
			{
				for (int i = 0; i < layer_m; i++)
				{
					if (!ItemPointerIsValid(&ntids[start + i]))
						continue;
					acorn_add_reverse_edge_at_layer(index, forkNum,
													 &ntids[start + i],
													 e_blk, e_off,
													 value, dist, m_eff, lc,
													 0, layer_m, -1, diversify,
													 (lc == 0) ? e_self : NULL, esz,
													 use_wal);
				}
			}
		}
	}

	pfree(ntup);
	pfree(etup);
}

/* -----------------------------------------------------------------------
 * Build callback + entry points
 *
 * The build runs in two phases (pgvector hnswbuild.c pattern):
 *
 *   1. In-memory phase — each scanned tuple is inserted into the in-memory
 *      graph while the graph fits maintenance_work_mem.
 *   2. On-disk phase — once the budget is exhausted, the graph is flushed
 *      and every remaining tuple goes through the per-element on-disk
 *      insert path (acorn_insert_element), un-WAL-logged; the whole index
 *      is WAL-logged once at the end via log_newpage_range.
 *
 * A parallel build (amcanbuildparallel) holds the graph in a DSM arena and
 * runs the heap scan in the leader + max_parallel_maintenance_workers
 * workers; see the AcornShared comment for the locking protocol.
 * ----------------------------------------------------------------------- */

typedef struct AcornLeader
{
	ParallelContext *pcxt;
	int				 nparticipants;	/* launched workers + participating leader */
	AcornShared		*shared;
	Snapshot		 snapshot;
	char			*area;
} AcornLeader;

typedef struct AcornBuildState
{
	Relation		 heap;
	Relation		 index;
	IndexInfo		*indexInfo;
	ForkNumber		 forkNum;
	int				 m_eff;
	int				 efc;
	bool			 diversify;
	bool			 has_filter;	/* true if index has a scalar filter column */
	double			 reltuples;
	double			 indtuples;
	unsigned short	 rand_state[3];	/* pg_erand48 state for level assignment */
	AcornDistCtx	 dist;
	MemoryContext	 graph_ctx;		/* serial graph allocations (freed on spill) */
	MemoryContext	 tmp_ctx;		/* per-tuple transient allocations */
	AcornMemBuild	*mb;			/* in-memory graph accessor */
	bool			 flushed;		/* serial: on-disk phase from now on */
	bool			 mwm_warned;
	AcornShared		*shared;		/* NULL = serial build */
	AcornLeader		*leader;		/* set in the leader of a parallel build */
} AcornBuildState;

/*
 * Seed the level RNG.  participant 0 = serial build / parallel leader;
 * parallel workers use participant = ParallelWorkerNumber + 1 so each
 * participant draws an independent deterministic stream from build_seed.
 *
 * NOTE on determinism: with pg_acorn.build_seed >= 0, SERIAL builds are
 * fully deterministic (same data + same seed -> identical graph).  Parallel
 * builds are NOT: the parallel table scan hands out blocks dynamically, so
 * the insertion order (and therefore the graph) varies run to run even at a
 * fixed seed and worker count.  Set max_parallel_maintenance_workers = 0
 * when a reproducible graph is required.
 */
static void
acorn_build_init_rand(unsigned short rand_state[3], int participant)
{
	if (acorn_build_seed >= 0)
	{
		uint32 seed = (uint32) acorn_build_seed + (uint32) participant * 7919;

		rand_state[0] = (unsigned short) (seed & 0xFFFF);
		rand_state[1] = (unsigned short) ((seed >> 16) & 0xFFFF);
		rand_state[2] = 0x1234;
	}
	else
	{
		/* Legacy: seed from process ID; different level sequences per build */
		rand_state[0] = (unsigned short) (MyProcPid & 0xFFFF);
		rand_state[1] = (unsigned short) (MyProcPid >> 16);
		rand_state[2] = 0x1234;
	}
}

static void
acorn_init_build_state(AcornBuildState *bs, Relation heap, Relation index,
					   IndexInfo *indexInfo, ForkNumber forkNum, int participant)
{
	int dims = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;

	if (dims < 0)
		dims = 0;

	bs->heap       = heap;
	bs->index      = index;
	bs->indexInfo  = indexInfo;
	bs->forkNum    = forkNum;
	bs->m_eff      = acorn_m_eff(index);
	bs->efc        = acorn_opt_ef_construction(index);
	bs->diversify  = acorn_opt_diversify(index);
	bs->has_filter = (RelationGetDescr(index)->natts > 1);
	bs->reltuples  = 0;
	bs->indtuples  = 0;
	acorn_build_init_rand(bs->rand_state, participant);
	acorn_dist_ctx_init(index, &bs->dist);

	bs->graph_ctx = AllocSetContextCreate(CurrentMemoryContext,
										  "acorn build graph",
										  ALLOCSET_DEFAULT_SIZES);
	bs->tmp_ctx   = AllocSetContextCreate(CurrentMemoryContext,
										  "acorn build temp",
										  ALLOCSET_DEFAULT_SIZES);

	bs->mb = acorn_mem_build_init(bs->graph_ctx);
	/* payload edges need a filter column to partition on */
	bs->mb->payload_edges  = acorn_opt_payload_edges(index) && bs->has_filter;
	bs->mb->inline_vectors = acorn_opt_inline_vectors(index);
	bs->mb->dims           = dims;
	bs->mb->entry_size     = bs->mb->inline_vectors
		? ACORN_T2_INLINE_ENTRY_SIZE(dims) : 0;

	bs->flushed    = false;
	bs->mwm_warned = false;
	bs->shared     = NULL;
	bs->leader     = NULL;
}

/* Point a participant's graph accessor at the shared DSM graph */
static void
acorn_attach_shared(AcornBuildState *bs, AcornShared *shared, char *area)
{
	AcornMemBuild *mb = bs->mb;

	bs->shared     = shared;
	mb->shared     = shared;
	mb->base       = area;
	mb->nodes      = (AcornMemNode *) area;	/* node array at arena start */
	mb->capacity   = shared->max_nodes;
	mb->part_entry = shared->part_entry;
	/* per-process visited generations sized to the shared capacity */
	mb->visit_gen  = (uint32 *)
		MemoryContextAllocZero(mb->build_ctx,
							   (Size) shared->max_nodes * sizeof(uint32));
	mb->cur_gen    = 1;
}

/*
 * Size the shared node array: expected per-node bytes = node struct +
 * vector varlena + expected neighbor slots ((E[level] + 2) * m_eff ints).
 * Misestimation is safe in both directions — whichever resource (node
 * slots or bump area) runs out first triggers the on-disk spill.
 */
static int
acorn_estimate_max_nodes(Relation index, Size arena_size, int m_eff)
{
	int		dims = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;
	Size	vsize;
	Size	per_node;
	double	e_level = 1.0 / log((double) Max(m_eff, 2));

	if (dims <= 0)
		dims = 128;				/* untyped vector column: assume mid-size */
	vsize = offsetof(AcornPgVector, x) + sizeof(float) * (Size) dims;

	per_node = MAXALIGN(sizeof(AcornMemNode)) + MAXALIGN(vsize)
		+ MAXALIGN((Size) ((e_level + 2.1) * m_eff) * sizeof(int));

	return (int) Min((Size) INT_MAX, arena_size / per_node);
}

static void acorn_build_mwm_warning(double indtuples, Size used, Size total);

/*
 * Insert one scanned tuple: in-memory while the budget holds, on-disk after.
 */
static void
acorn_build_insert_tuple(AcornBuildState *bs, Datum value, Size vsize,
						 int64 filter_val, ItemPointer tid)
{
	AcornMemBuild *mb = bs->mb;
	int			   level;
	int			   id;

	if (bs->shared)
	{
		AcornShared *sh = bs->shared;

		LWLockAcquire(&sh->flushLock, LW_SHARED);
		if (!sh->flushed)
		{
			level = acorn_assign_level(bs->m_eff, bs->rand_state);
			id = acorn_mem_push_node(mb, value, vsize, level, filter_val,
									 tid, bs->m_eff);
			if (id >= 0)
			{
				acorn_mem_insert_node(mb, &bs->dist, bs->m_eff, bs->efc,
									  bs->diversify, id);
				LWLockRelease(&sh->flushLock);
				return;
			}
		}
		LWLockRelease(&sh->flushLock);

		/*
		 * Arena exhausted (or graph already flushed): on-disk phase.  The
		 * acorn on-disk insert path assumes a single writer, so every
		 * spilled insert runs under the exclusive flush lock — serialized,
		 * correctness over speed.
		 */
		LWLockAcquire(&sh->flushLock, LW_EXCLUSIVE);
		if (!sh->flushed)
		{
			double indtuples;

			SpinLockAcquire(&sh->mutex);
			indtuples = sh->indtuples;
			SpinLockRelease(&sh->mutex);

			acorn_build_mwm_warning(indtuples, sh->arena_used, sh->arena_size);
			acorn_mem_flush(mb, bs->index, bs->forkNum, bs->m_eff);
			sh->flushed = true;
		}
		acorn_insert_element(bs->index, bs->forkNum, value, filter_val, tid,
							 bs->rand_state, false);
		LWLockRelease(&sh->flushLock);
	}
	else
	{
		if (!bs->flushed)
		{
			Size needed;

			level  = acorn_assign_level(bs->m_eff, bs->rand_state);
			needed = acorn_mem_node_cost(mb, level, vsize, bs->m_eff);
			if (mb->mem_used + needed <= mb->mem_total)
			{
				id = acorn_mem_push_node(mb, value, vsize, level, filter_val,
										 tid, bs->m_eff);
				acorn_mem_insert_node(mb, &bs->dist, bs->m_eff, bs->efc,
									  bs->diversify, id);
				return;
			}

			/* Budget exhausted: flush, free the graph, switch to disk */
			acorn_build_mwm_warning(bs->indtuples, mb->mem_used, mb->mem_total);
			acorn_mem_flush(mb, bs->index, bs->forkNum, bs->m_eff);
			bs->flushed = true;
			bs->mb = NULL;
			MemoryContextDelete(bs->graph_ctx);
			bs->graph_ctx = NULL;
		}
		acorn_insert_element(bs->index, bs->forkNum, value, filter_val, tid,
							 bs->rand_state, false);
	}
}

static void
acorn_build_mwm_warning(double indtuples, Size used, Size total)
{
	ereport(WARNING,
			(errmsg("acorn_hnsw graph no longer fits in maintenance_work_mem after %.0f tuples: %zu kB used of %zu kB budget",
					indtuples, used / 1024, total / 1024),
			 errdetail("Remaining tuples will be inserted through the on-disk path; building will take significantly more time."),
			 errhint("Increase maintenance_work_mem to speed up builds.")));
}

static void
acorn_build_callback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *state)
{
	AcornBuildState *bs = (AcornBuildState *) state;
	int64			 filter_val = 0;
	Datum			 detoasted;
	Size			 vsize;
	MemoryContext	 oldCtx;

	if (isnull[0])
		return;

	if (bs->has_filter && !isnull[1])
		filter_val = (int64) values[1];

	oldCtx = MemoryContextSwitchTo(bs->tmp_ctx);

	detoasted = PointerGetDatum(PG_DETOAST_DATUM(values[0]));
	vsize     = VARSIZE_ANY(DatumGetPointer(detoasted));
	acorn_build_insert_tuple(bs, detoasted, vsize, filter_val, tid);

	if (bs->shared)
	{
		SpinLockAcquire(&bs->shared->mutex);
		bs->shared->indtuples += 1;
		SpinLockRelease(&bs->shared->mutex);
	}
	bs->indtuples += 1;

	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(bs->tmp_ctx);
}

/* -----------------------------------------------------------------------
 * Parallel build (mirrors pgvector 0.8.0 hnswbuild.c)
 * ----------------------------------------------------------------------- */

/*
 * Perform a participant's portion of the parallel scan + insert.
 */
static void
acorn_parallel_scan_and_insert(Relation heap, Relation index,
							   AcornShared *shared, char *area, bool progress)
{
	AcornBuildState bs;
	TableScanDesc	scan;
	double			reltuples;
	IndexInfo	   *indexInfo;
	int				participant =
		IsParallelWorker() ? ParallelWorkerNumber + 1 : 0;

	/* Join parallel scan */
	indexInfo = BuildIndexInfo(index);
	indexInfo->ii_Concurrent = shared->isconcurrent;
	acorn_init_build_state(&bs, heap, index, indexInfo, MAIN_FORKNUM,
						   participant);
	acorn_attach_shared(&bs, shared, area);

	scan = table_beginscan_parallel(heap,
									ParallelTableScanFromAcornShared(shared));
	reltuples = table_index_build_scan(heap, index, indexInfo, true, progress,
									   acorn_build_callback, (void *) &bs,
									   scan);

	/* Record statistics */
	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->reltuples += reltuples;
	SpinLockRelease(&shared->mutex);

	ereport(DEBUG1,
			(errmsg("acorn_hnsw %s processed %.0f tuples",
					progress ? "leader" : "worker", reltuples)));

	/* Notify leader */
	ConditionVariableSignal(&shared->workersdonecv);

	if (bs.graph_ctx)
		MemoryContextDelete(bs.graph_ctx);
	MemoryContextDelete(bs.tmp_ctx);
}

/*
 * Perform work within a launched parallel worker.
 */
PGDLLEXPORT void AcornParallelBuildMain(dsm_segment *seg, shm_toc *toc);

void
AcornParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	AcornShared *shared;
	char	   *area;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	/* Set debug_query_string for individual workers first */
	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;

	/* Report the query string from leader */
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	/* Look up shared state */
	shared = shm_toc_lookup(toc, PARALLEL_KEY_ACORN_SHARED, false);

	/* Open relations using lock modes known to be obtained by index.c */
	if (!shared->isconcurrent)
	{
		heapLockmode  = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode  = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	/* Open relations within worker */
	heapRel  = table_open(shared->heaprelid, heapLockmode);
	indexRel = index_open(shared->indexrelid, indexLockmode);

	area = shm_toc_lookup(toc, PARALLEL_KEY_ACORN_AREA, false);

	/* Register the LWLock tranche before touching any graph lock */
	acorn_init_lock_tranche();

	/* Perform inserts */
	acorn_parallel_scan_and_insert(heapRel, indexRel, shared, area, false);

	/* Close relations within worker */
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

/*
 * Within leader, wait for the end of the parallel heap scan.
 */
static double
acorn_parallel_heapscan(AcornBuildState *bs)
{
	AcornShared *shared = bs->leader->shared;
	int			 nparticipants = bs->leader->nparticipants;
	double		 reltuples;

	for (;;)
	{
		SpinLockAcquire(&shared->mutex);
		if (shared->nparticipantsdone == nparticipants)
		{
			bs->indtuples = shared->indtuples;
			reltuples = shared->reltuples;
			SpinLockRelease(&shared->mutex);
			break;
		}
		SpinLockRelease(&shared->mutex);

		ConditionVariableSleep(&shared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();

	return reltuples;
}

/*
 * End the parallel build.
 */
static void
acorn_end_parallel(AcornLeader *leader)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(leader->pcxt);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(leader->snapshot))
		UnregisterSnapshot(leader->snapshot);
	DestroyParallelContext(leader->pcxt);
	ExitParallelMode();
}

/*
 * Begin the parallel build: DSM segment = AcornShared + parallel scan desc
 * + the graph arena (sized from maintenance_work_mem, so (a)'s budget also
 * bounds the parallel graph).
 */
static void
acorn_begin_parallel(AcornBuildState *bs, bool isconcurrent, int request)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		estshared;
	Size		estarea;
	Size		estother;
	AcornShared *shared;
	char	   *area;
	AcornLeader *leader = (AcornLeader *) palloc0(sizeof(AcornLeader));
	bool		leaderparticipates = true;
	int			querylen;
	int			max_nodes;

	acorn_init_lock_tranche();

	/* Enter parallel mode and create context */
	EnterParallelMode();
	Assert(request > 0);
	pcxt = CreateParallelContext("pg_acorn", "AcornParallelBuildMain", request);

	/* Get snapshot for table scan */
	if (!isconcurrent)
		snapshot = SnapshotAny;
	else
		snapshot = RegisterSnapshot(GetTransactionSnapshot());

	/* Estimate size of workspaces */
	estshared = add_size(BUFFERALIGN(sizeof(AcornShared)),
						 table_parallelscan_estimate(bs->heap, snapshot));
	shm_toc_estimate_chunk(&pcxt->estimator, estshared);

	/*
	 * Leave space for other objects in shared memory (pgvector does the
	 * same: Docker's default 64MB shm_size equals the default
	 * maintenance_work_mem, so the arena must come in under the budget).
	 */
	estarea  = (Size) maintenance_work_mem * 1024;
	estother = 3 * 1024 * 1024;
	if (estarea > estother)
		estarea -= estother;

	shm_toc_estimate_chunk(&pcxt->estimator, estarea);
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	/* Finally, estimate PARALLEL_KEY_QUERY_TEXT space */
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	else
		querylen = 0;			/* keep compiler quiet */

	/* Everyone's had a chance to ask for space, so now create the DSM */
	InitializeParallelDSM(pcxt);

	/* If no DSM segment was available, back out (do serial build) */
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return;
	}

	/* Store shared build state, for which we reserved space */
	shared = (AcornShared *) shm_toc_allocate(pcxt->toc, estshared);
	shared->heaprelid    = RelationGetRelid(bs->heap);
	shared->indexrelid   = RelationGetRelid(bs->index);
	shared->isconcurrent = isconcurrent;
	ConditionVariableInit(&shared->workersdonecv);
	SpinLockInit(&shared->mutex);
	shared->nparticipantsdone = 0;
	shared->reltuples = 0;
	shared->indtuples = 0;
	table_parallelscan_initialize(bs->heap,
								  ParallelTableScanFromAcornShared(shared),
								  snapshot);

	area = (char *) shm_toc_allocate(pcxt->toc, estarea);
	/* Report less than allocated so initialization never overruns */
	shared->arena_size = (estarea > 1024 * 1024) ? estarea - 1024 * 1024 : 0;
	max_nodes = acorn_estimate_max_nodes(bs->index, shared->arena_size,
										 bs->m_eff);
	shared->max_nodes   = max_nodes;
	shared->n_nodes     = 0;
	shared->arena_used  = MAXALIGN((Size) max_nodes * sizeof(AcornMemNode));
	shared->entry_id    = -1;
	shared->entry_level = -1;
	shared->flushed     = false;
	LWLockInitialize(&shared->allocatorLock, acorn_lock_tranche_id);
	LWLockInitialize(&shared->entryLock, acorn_lock_tranche_id);
	LWLockInitialize(&shared->entryWaitLock, acorn_lock_tranche_id);
	LWLockInitialize(&shared->flushLock, acorn_lock_tranche_id);
	LWLockInitialize(&shared->partLock, acorn_lock_tranche_id);
	for (int p = 0; p < ACORN_PAYLOAD_PARTITIONS; p++)
		shared->part_entry[p] = -1;

	shm_toc_insert(pcxt->toc, PARALLEL_KEY_ACORN_SHARED, shared);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_ACORN_AREA, area);

	/* Store query string for workers */
	if (debug_query_string)
	{
		char	   *sharedquery;

		sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
	}

	/* Launch workers, saving status for leader/caller */
	LaunchParallelWorkers(pcxt);
	leader->pcxt = pcxt;
	leader->nparticipants = pcxt->nworkers_launched;
	if (leaderparticipates)
		leader->nparticipants++;
	leader->shared   = shared;
	leader->snapshot = snapshot;
	leader->area     = area;

	/* If no workers were successfully launched, back out (do serial build) */
	if (pcxt->nworkers_launched == 0)
	{
		acorn_end_parallel(leader);
		return;
	}

	/* Log participants */
	ereport(DEBUG1, (errmsg("acorn_hnsw build using %d parallel workers",
							pcxt->nworkers_launched)));

	/* Save leader state now that it's clear build will be parallel */
	bs->leader = leader;

	/* Join heap scan ourselves */
	if (leaderparticipates)
		acorn_parallel_scan_and_insert(bs->heap, bs->index, shared, area,
									   true);

	/* Wait for all launched workers */
	WaitForParallelWorkersToAttach(pcxt);
}

/*
 * Compute parallel workers (pgvector ComputeParallelWorkers): respects
 * plan_create_index_workers' safety checks, the table's parallel_workers
 * storage parameter, and max_parallel_maintenance_workers.
 */
static int
acorn_plan_parallel_workers(Relation heap, Relation index)
{
	int			parallel_workers;

	/* Make sure it's safe to use parallel workers */
	parallel_workers = plan_create_index_workers(RelationGetRelid(heap),
												 RelationGetRelid(index));
	if (parallel_workers == 0)
		return 0;

	/* Use parallel_workers storage parameter on table if set */
	parallel_workers = RelationGetParallelWorkers(heap, -1);
	if (parallel_workers != -1)
		return Min(parallel_workers, max_parallel_maintenance_workers);

	return max_parallel_maintenance_workers;
}

/* -----------------------------------------------------------------------
 * ambuild / ambuildempty
 * ----------------------------------------------------------------------- */

static void
acorn_build_internal(Relation heap, Relation index, IndexInfo *indexInfo,
					 ForkNumber forkNum, double *heap_tuples, double *index_tuples)
{
	int				m_eff = acorn_m_eff(index);
	int				efc   = acorn_opt_ef_construction(index);
	int				dims  = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;
	int				m_req = acorn_opt_m(index);
	int				gamma = acorn_opt_gamma(index);
	bool			inline_vectors = acorn_opt_inline_vectors(index);
	AcornBuildState bs;
	int				parallel_workers = 0;

	if (dims < 0)
		dims = 0;

	if (inline_vectors && dims <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("acorn_inline_vectors requires a vector column with fixed dimensions"),
				 errhint("Declare the column as vector(N).")));

	/*
	 * ACORN-gamma page-budget guard (paper §10/§12.3).  pgvector's neighbor
	 * tuple must fit one 8KB page: (level+2) * m_eff * sizeof(ItemPointerData).
	 * Levels are already capped at ACORN_MAX_LEVEL, so the page never
	 * overflows; the surprising effect a user actually hits is m_eff being
	 * silently clamped to HNSW_MAX_M when m*gamma exceeds it — meaning the
	 * requested gamma is not fully applied.  Warn so the user can lower gamma
	 * or m rather than wonder why recall/latency does not change with gamma.
	 */
	if (m_req * gamma > HNSW_MAX_M)
		ereport(WARNING,
				(errmsg("acorn_gamma=%d with m=%d requests %d neighbors per node, "
						"clamped to m_eff=%d (HNSW page limit %d)",
						gamma, m_req, m_req * gamma, m_eff, HNSW_MAX_M),
				 errhint("Lower acorn_gamma or m so that m*gamma <= %d to apply gamma fully.",
						 HNSW_MAX_M)));

	acorn_create_meta_page(index, forkNum, m_eff, efc, dims,
						   inline_vectors ? ACORN_T2_META_INLINE_VECTORS : 0);

	acorn_init_build_state(&bs, heap, index, indexInfo, forkNum, 0);

	/* Attempt a parallel build when the planner grants workers */
	if (heap != NULL && forkNum == MAIN_FORKNUM)
		parallel_workers = acorn_plan_parallel_workers(heap, index);
	if (parallel_workers > 0)
		acorn_begin_parallel(&bs, indexInfo->ii_Concurrent, parallel_workers);

	if (heap != NULL)
	{
		if (bs.leader)
			bs.reltuples = acorn_parallel_heapscan(&bs);
		else
			bs.reltuples = table_index_build_scan(heap, index, indexInfo,
												  true, true,
												  acorn_build_callback,
												  (void *) &bs, NULL);
	}

	if (bs.leader)
	{
		/* Adopt the shared graph for the leader-side flush */
		acorn_attach_shared(&bs, bs.leader->shared, bs.leader->area);
		LWLockAcquire(&bs.shared->flushLock, LW_EXCLUSIVE);
		if (!bs.shared->flushed)
		{
			acorn_mem_flush(bs.mb, index, forkNum, m_eff);
			bs.shared->flushed = true;
		}
		LWLockRelease(&bs.shared->flushLock);
		acorn_end_parallel(bs.leader);
	}
	else if (!bs.flushed)
		acorn_mem_flush(bs.mb, index, forkNum, m_eff);

	/*
	 * The build wrote pages without WAL (including any spilled on-disk
	 * inserts); log everything at once (pgvector BuildIndex pattern).
	 */
	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0,
						  RelationGetNumberOfBlocksInFork(index, forkNum),
						  true);

	if (heap_tuples)
		*heap_tuples  = bs.reltuples;
	if (index_tuples)
		*index_tuples = bs.indtuples;

	if (bs.graph_ctx)
		MemoryContextDelete(bs.graph_ctx);
	MemoryContextDelete(bs.tmp_ctx);
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
	{
		int64 filter_val = 0;

		if (indexInfo->ii_NumIndexAttrs > 1 && !isnull[1])
			filter_val = (int64) values[1];
		acorn_insert_element(index, MAIN_FORKNUM, value, filter_val, heap_tid,
							 rand_state, RelationNeedsWAL(index));
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(insertCtx);

	return false;
}
