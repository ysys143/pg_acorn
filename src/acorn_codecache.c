/*
 * acorn_codecache.c — per-index shared-memory SQ8 code cache (M1 read + M2 write)
 *
 * Registry: one named DSM segment ("pg_acorn_cc", PG17 dsm_registry) holds a
 * directory of per-index slots keyed by (dboid, relfilenumber).  Each slot
 * owns a DSA area containing a flat per-block directory: a dsa_pointer array
 * indexed by block number, each pointing to a fixed-stride array of entries
 * indexed by (offno - 1).  An entry holds a seqlock version, heaptid, nbrtid,
 * level, flags, filter_val and the SQ8 scale/offset/code.
 *
 * The flat layout replaces the dshash the design doc sketched: a lookup is
 * two dependent loads with NO partition lock and NO hashing, which matters
 * because the scan does one lookup per discovered neighbor (measured at
 * 60K/dim=128: dshash_find + per-entry dsa pointer chase cost ~0.9 us per
 * discovery and put cache-mode at ~2.5x inline-mode latency; the flat
 * directory is the same access pattern as the inline path's in-page read).
 *
 * Loading is lazy and non-blocking: the first scan that finds a slot EMPTY
 * CASes it to LOADING and walks the index main fork, quantizing every
 * element tuple's fp32 vector with acorn_sq8_encode (the same encoder the
 * inline build uses, so codes are bit-identical to inline codes).  Other
 * scans never wait: while LOADING and on any miss they use the element-page
 * fallback (design G4 — correctness never depends on cache state).  If the
 * pg_acorn.code_cache_size budget is exceeded mid-load the slot becomes
 * PARTIAL: present blocks serve, misses fall back.
 *
 * Locking contract:
 *   - dir->lock (exclusive) protects slot claiming and handle publication,
 *     (shared) handle reads.
 *   - Slot state transitions go through the atomic only: EMPTY -> LOADING by
 *     CAS (single loader), LOADING -> READY|PARTIAL by the loader after a
 *     write barrier.
 *   - slot->wlock (a per-slot LWLock) serializes M2 writers (insert/vacuum)
 *     against each other and against directory/page-array growth.  Readers
 *     take NO lock: they snapshot the block directory and copy each entry
 *     out under its per-entry seqlock (acorn_cc_entry_version), retrying once
 *     then falling back.  Growth publishes a fully-built replacement array
 *     and swaps the owning dsa_pointer atomically, so a lock-free reader
 *     observes either the old or the new array — both internally consistent.
 *
 * M2 caveats (M3):
 *   - Growth (new block index or a page array that must hold a larger offno)
 *     leaks the superseded array within the slot's own DSA until M3 eviction
 *     reclaims the whole slot.  Bounded by the number of growth events, not
 *     by query volume.
 *   - No eviction / LRU / stats SRF yet.  pg_acorn.scan_code_cache still
 *     defaults OFF; M3 flips it after acceptance.
 */

#include "postgres.h"

#include "miscadmin.h"
#include "port/atomics.h"
#include "storage/bufmgr.h"
#include "storage/dsm_registry.h"
#include "storage/itemid.h"
#include "storage/lwlock.h"
#include "utils/dsa.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pg_acorn.h"
#include "hnsw_compat.h"
#include "acorn_t2_page.h"
#include "acorn_dist.h"
#include "acorn_codecache.h"

/* -----------------------------------------------------------------------
 * Shared directory (named DSM segment)
 * ----------------------------------------------------------------------- */

#define ACORN_CC_SEGMENT_NAME	"pg_acorn_cc"
#define ACORN_CC_NSLOTS			64

/* slot states */
#define ACORN_CC_STATE_EMPTY	0
#define ACORN_CC_STATE_LOADING	1
#define ACORN_CC_STATE_READY	2
#define ACORN_CC_STATE_PARTIAL	3

/* fixed entry stride: keeps filter_val 8-aligned in the per-page array */
#define ACORN_CC_ENTRY_STRIDE(dim) \
	MAXALIGN(offsetof(AcornCodeCacheEntry, code) + (dim))

/* per-page entry array: 8-byte header, then nslots entries of fixed stride */
typedef struct AcornCCPageHdr
{
	uint16		nslots;			/* entries indexed by offno-1, < nslots+1 */
	uint16		reserved;
	uint32		reserved2;
} AcornCCPageHdr;

#define ACORN_CC_PAGE_SIZE(nslots, stride) \
	(sizeof(AcornCCPageHdr) + (Size) (nslots) * (stride))
#define ACORN_CC_PAGE_ENTRY(hdr, offno, stride) \
	((AcornCodeCacheEntry *) ((char *) (hdr) + sizeof(AcornCCPageHdr) + \
							  (Size) ((offno) - 1) * (stride)))

/*
 * Grow page arrays with headroom so a run of inserts onto the same heap
 * block does not realloc the cache page array on every tuple.  pgvector HNSW
 * packs few element tuples per index page (one per page at dim=128), so this
 * is small in absolute terms but avoids O(n^2) churn on pathological pages.
 */
#define ACORN_CC_PAGE_GROW(need)	((need) + 8)

/* Grow the block directory with headroom (same rationale, per-block). */
#define ACORN_CC_DIR_GROW(need)		((need) + 64)

/* -----------------------------------------------------------------------
 * Per-entry seqlock: the writer (insert/vacuum, holding slot->wlock) makes
 * the version odd before the first field write and even (== old + 2) after
 * the last, with a write barrier on each side.  A reader copies the entry
 * out between two even reads of the same version; an odd or changed version
 * means a concurrent write, so the reader retries once then falls back.
 * ----------------------------------------------------------------------- */

static inline void
acorn_cc_write_begin(AcornCodeCacheEntry *e)
{
	pg_atomic_write_u32(&e->version, pg_atomic_read_u32(&e->version) | 1);
	pg_write_barrier();
}

static inline void
acorn_cc_write_end(AcornCodeCacheEntry *e)
{
	pg_write_barrier();
	pg_atomic_write_u32(&e->version,
						(pg_atomic_read_u32(&e->version) & ~(uint32) 1) + 2);
}

typedef struct AcornCodeCacheSlot
{
	Oid				dboid;			/* InvalidOid = slot unclaimed */
	RelFileNumber	relnumber;
	pg_atomic_uint32 state;			/* ACORN_CC_STATE_* */
	uint32			generation;		/* bumped per (re)load */
	uint32			nelems;
	uint64			bytes;			/* accounted bytes */
	int				dim;			/* code length; stride = 40 + dim aligned */
	pg_atomic_uint32 nblocks;		/* length of the block directory (grows) */
	LWLock			wlock;			/* serializes M2 writers + growth */
	dsa_handle		area_handle;
	pg_atomic_uint64 blocks;		/* dsa_pointer[nblocks] page arrays (grows) */
} AcornCodeCacheSlot;

typedef struct AcornCodeCacheDirectory
{
	LWLock			lock;
	int				lock_tranche;
	int				dsa_tranche;
	int				wlock_tranche;
	pg_atomic_uint64 total_bytes;	/* global accounted bytes, all slots */
	AcornCodeCacheSlot slots[ACORN_CC_NSLOTS];
} AcornCodeCacheDirectory;

/* -----------------------------------------------------------------------
 * Backend-local state
 * ----------------------------------------------------------------------- */

struct AcornCodeCacheScan
{
	dsa_area	   *area;
	AcornCodeCacheSlot *slot;		/* live slot (for growth-aware lookup) */
	dsa_pointer		blocks_dp;		/* snapshot of slot->blocks at attach */
	dsa_pointer    *blocks;			/* mapped block directory snapshot */
	uint32			nblocks;		/* snapshot of slot->nblocks at attach */
	Size			stride;
	int				dim;			/* code length */
};

typedef struct AcornCCAttachKey
{
	Oid				dboid;
	RelFileNumber	relnumber;
} AcornCCAttachKey;

typedef struct AcornCCAttachEntry
{
	AcornCCAttachKey key;			/* must be first (HASH_BLOBS) */
	AcornCodeCacheScan scan;
} AcornCCAttachEntry;

static AcornCodeCacheDirectory *acorn_cc_dir = NULL;
static HTAB *acorn_cc_attached = NULL;

/* -----------------------------------------------------------------------
 * Directory attach
 * ----------------------------------------------------------------------- */

static void
acorn_cc_dir_init(void *ptr)
{
	AcornCodeCacheDirectory *dir = (AcornCodeCacheDirectory *) ptr;

	memset(dir, 0, sizeof(AcornCodeCacheDirectory));
	dir->lock_tranche = LWLockNewTrancheId();
	dir->dsa_tranche = LWLockNewTrancheId();
	dir->wlock_tranche = LWLockNewTrancheId();
	LWLockInitialize(&dir->lock, dir->lock_tranche);
	pg_atomic_init_u64(&dir->total_bytes, 0);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		pg_atomic_init_u32(&dir->slots[i].state, ACORN_CC_STATE_EMPTY);
		pg_atomic_init_u32(&dir->slots[i].nblocks, 0);
		pg_atomic_init_u64(&dir->slots[i].blocks, InvalidDsaPointer);
		LWLockInitialize(&dir->slots[i].wlock, dir->wlock_tranche);
	}
}

static AcornCodeCacheDirectory *
acorn_cc_get_dir(void)
{
	if (acorn_cc_dir == NULL)
	{
		bool		found;

		acorn_cc_dir = (AcornCodeCacheDirectory *)
			GetNamedDSMSegment(ACORN_CC_SEGMENT_NAME,
							   sizeof(AcornCodeCacheDirectory),
							   acorn_cc_dir_init, &found);
		LWLockRegisterTranche(acorn_cc_dir->lock_tranche, "pg_acorn_cc_dir");
		LWLockRegisterTranche(acorn_cc_dir->dsa_tranche, "pg_acorn_cc_dsa");
		LWLockRegisterTranche(acorn_cc_dir->wlock_tranche, "pg_acorn_cc_wlock");
	}
	return acorn_cc_dir;
}

/*
 * Find the slot for (dboid, relnumber), claiming a free one if absent.
 * Returns NULL when the directory is full (no cache for this index).
 */
static AcornCodeCacheSlot *
acorn_cc_slot_lookup(AcornCodeCacheDirectory *dir,
					 Oid dboid, RelFileNumber relnumber)
{
	AcornCodeCacheSlot *slot = NULL;
	AcornCodeCacheSlot *freeslot = NULL;

	LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		AcornCodeCacheSlot *s = &dir->slots[i];

		if (s->dboid == dboid && s->relnumber == relnumber)
		{
			slot = s;
			break;
		}
		if (freeslot == NULL && s->dboid == InvalidOid)
			freeslot = s;
	}
	if (slot == NULL && freeslot != NULL)
	{
		freeslot->dboid = dboid;
		freeslot->relnumber = relnumber;
		slot = freeslot;
	}
	LWLockRelease(&dir->lock);
	return slot;
}

/* -----------------------------------------------------------------------
 * Backend-local attachment cache
 * ----------------------------------------------------------------------- */

static void
acorn_cc_attach_init(void)
{
	HASHCTL		info;

	if (acorn_cc_attached != NULL)
		return;
	memset(&info, 0, sizeof(info));
	info.keysize = sizeof(AcornCCAttachKey);
	info.entrysize = sizeof(AcornCCAttachEntry);
	info.hcxt = TopMemoryContext;
	acorn_cc_attached = hash_create("acorn_codecache attachments", 8,
									&info,
									HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static AcornCCAttachEntry *
acorn_cc_attach_find(Oid dboid, RelFileNumber relnumber, bool *found)
{
	AcornCCAttachKey key;

	acorn_cc_attach_init();
	memset(&key, 0, sizeof(key));
	key.dboid = dboid;
	key.relnumber = relnumber;
	return (AcornCCAttachEntry *)
		hash_search(acorn_cc_attached, &key, HASH_FIND, found);
}

static AcornCCAttachEntry *
acorn_cc_attach_store(Oid dboid, RelFileNumber relnumber,
					  dsa_area *area, AcornCodeCacheSlot *slot,
					  dsa_pointer blocks_dp, dsa_pointer *blocks,
					  uint32 nblocks, Size stride, int dim)
{
	AcornCCAttachKey key;
	AcornCCAttachEntry *e;
	bool		found;

	acorn_cc_attach_init();
	memset(&key, 0, sizeof(key));
	key.dboid = dboid;
	key.relnumber = relnumber;
	e = (AcornCCAttachEntry *)
		hash_search(acorn_cc_attached, &key, HASH_ENTER, &found);
	e->scan.area = area;
	e->scan.slot = slot;
	e->scan.blocks_dp = blocks_dp;
	e->scan.blocks = blocks;
	e->scan.nblocks = nblocks;
	e->scan.stride = stride;
	e->scan.dim = dim;
	return e;
}

/*
 * Attach this backend to a READY/PARTIAL slot's DSA area and map the block
 * directory.  Attachments are pinned for the backend's lifetime
 * (dsa_pin_mapping) and cached, so each backend pays the attach cost once
 * per index.
 */
static AcornCodeCacheScan *
acorn_cc_attach(AcornCodeCacheDirectory *dir, AcornCodeCacheSlot *slot,
				Oid dboid, RelFileNumber relnumber)
{
	AcornCCAttachEntry *e;
	bool		found;
	dsa_handle	area_handle;
	dsa_pointer blocks_dp;
	uint32		nblocks;
	int			dim;
	dsa_area   *area;
	dsa_pointer *blocks;
	MemoryContext old;

	e = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
		return &e->scan;

	LWLockAcquire(&dir->lock, LW_SHARED);
	area_handle = slot->area_handle;
	dim = slot->dim;
	LWLockRelease(&dir->lock);

	/* blocks/nblocks grow lock-free; read them via their atomics */
	blocks_dp = (dsa_pointer) pg_atomic_read_u64(&slot->blocks);
	nblocks = pg_atomic_read_u32(&slot->nblocks);

	old = MemoryContextSwitchTo(TopMemoryContext);
	area = dsa_attach(area_handle);
	dsa_pin_mapping(area);
	MemoryContextSwitchTo(old);
	blocks = (dsa_pointer *) dsa_get_address(area, blocks_dp);

	e = acorn_cc_attach_store(dboid, relnumber, area, slot, blocks_dp,
							  blocks, nblocks, ACORN_CC_ENTRY_STRIDE(dim), dim);
	return &e->scan;
}

/* -----------------------------------------------------------------------
 * Loader
 * ----------------------------------------------------------------------- */

/*
 * Load all element tuples of `index` into the slot's block directory.
 * Caller owns the LOADING claim (won the EMPTY -> LOADING CAS).  On return
 * the slot is READY, or PARTIAL when the budget ran out; on error the slot
 * is published PARTIAL with whatever loaded (a block's entry array is fully
 * built before its directory pointer is set, so partial content is always
 * servable) and the error is re-thrown.
 */
static void
acorn_cc_load(AcornCodeCacheDirectory *dir, AcornCodeCacheSlot *slot,
			  Relation index, int dim)
{
	dsa_area   *area;
	MemoryContext old;
	uint64		budget = (uint64) acorn_code_cache_size_mb * 1024 * 1024;
	uint64		other_bytes;
	Size		stride = ACORN_CC_ENTRY_STRIDE(dim);
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	dsa_pointer blocks_dp;
	dsa_pointer *blocks;
	volatile uint64 bytes = 0;
	volatile uint32 nelems = 0;
	volatile uint32 final_state = ACORN_CC_STATE_READY;

	/* backend-local control structs must outlive the transaction */
	old = MemoryContextSwitchTo(TopMemoryContext);
	area = dsa_create(dir->dsa_tranche);
	dsa_pin(area);
	dsa_pin_mapping(area);
	MemoryContextSwitchTo(old);

	/*
	 * Over-allocate the block directory by the index's current page count
	 * plus headroom, so the common insert-onto-a-new-block case sets a slot
	 * in place without growing the directory.
	 */
	{
		uint32	dir_blocks = ACORN_CC_DIR_GROW(nblocks);

		blocks_dp = dsa_allocate0(area,
								  (Size) dir_blocks * sizeof(dsa_pointer));
		blocks = (dsa_pointer *) dsa_get_address(area, blocks_dp);
		bytes += (Size) dir_blocks * sizeof(dsa_pointer);

		/* publish handles before the state can leave LOADING */
		LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
		slot->area_handle = dsa_get_handle(area);
		slot->dim = dim;
		slot->generation++;
		LWLockRelease(&dir->lock);

		pg_atomic_write_u64(&slot->blocks, (uint64) blocks_dp);
		pg_atomic_write_u32(&slot->nblocks, dir_blocks);

		/* the loading backend is attached by construction */
		acorn_cc_attach_store(index->rd_locator.dbOid,
							  index->rd_locator.relNumber, area, slot,
							  blocks_dp, blocks, dir_blocks, stride, dim);
	}

	other_bytes = pg_atomic_read_u64(&dir->total_bytes);

	PG_TRY();
	{
		for (BlockNumber blkno = HNSW_METAPAGE_BLKNO + 1;
			 blkno < nblocks && final_state == ACORN_CC_STATE_READY;
			 blkno++)
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoff;
			dsa_pointer page_dp = InvalidDsaPointer;
			AcornCCPageHdr *ph = NULL;
			Size		page_sz = 0;
			uint32		page_elems = 0;

			CHECK_FOR_INTERRUPTS();

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoff = PageGetMaxOffsetNumber(page);

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
				 offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				AcornT2ElementTuple etup;
				AcornPgVector *vec;
				AcornCodeCacheEntry *e;

				if (!ItemIdIsUsed(iid) || !ItemIdHasStorage(iid))
					continue;
				etup = (AcornT2ElementTuple) PageGetItem(page, iid);
				if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
					continue;	/* skip meta/neighbor/inline-cont tuples */
				vec = (AcornPgVector *) AcornT2ElementTupleGetVector(etup);
				if ((int) vec->dim != dim)
					continue;	/* defensive: never cache a malformed code */

				if (ph == NULL)
				{
					/* lazy per-page array, sized for this page's offsets */
					page_sz = ACORN_CC_PAGE_SIZE(maxoff, stride);
					if (other_bytes + bytes + page_sz > budget)
					{
						final_state = ACORN_CC_STATE_PARTIAL;
						break;
					}
					page_dp = dsa_allocate_extended(area, page_sz,
													DSA_ALLOC_NO_OOM |
													DSA_ALLOC_ZERO);
					if (!DsaPointerIsValid(page_dp))
					{
						final_state = ACORN_CC_STATE_PARTIAL;
						break;
					}
					ph = (AcornCCPageHdr *) dsa_get_address(area, page_dp);
					ph->nslots = maxoff;
					bytes += page_sz;
				}

				e = ACORN_CC_PAGE_ENTRY(ph, offno, stride);
				/*
				 * Loader builds entries before the block is reachable, so no
				 * seqlock is needed here — but initialise version to an even
				 * value so the M2 writer's odd/even toggling starts correctly.
				 */
				pg_atomic_init_u32(&e->version, 0);
				e->heaptid = etup->heaptids[0];
				e->nbrtid = etup->neighbortid;
				e->level = etup->level;
				e->flags = ACORN_CC_PRESENT |
					((etup->deleted != 0) ? ACORN_CC_DELETED : 0);
				e->filter_val = etup->filter_val;
				acorn_sq8_encode(dim, vec->x, e->code, &e->scale, &e->offset);
				page_elems++;
			}

			UnlockReleaseBuffer(buf);

			/*
			 * Publish the block only after its entries are complete, so a
			 * PARTIAL slot (budget stop or error) never exposes a torn
			 * page array.  The array is allocated lazily on the page's
			 * first element tuple, so page_elems > 0 whenever one exists.
			 */
			if (page_elems > 0)
			{
				blocks[blkno] = page_dp;
				nelems += page_elems;
			}
		}
	}
	PG_CATCH();
	{
		/*
		 * Publish what loaded so far as PARTIAL: every published block is
		 * complete, and misses fall back (G4).  The slot never returns to
		 * EMPTY in M1, so the create-vs-reuse branch does not exist.
		 */
		slot->nelems = nelems;
		slot->bytes = bytes;
		pg_atomic_fetch_add_u64(&dir->total_bytes, bytes);
		pg_write_barrier();
		pg_atomic_write_u32(&slot->state, ACORN_CC_STATE_PARTIAL);
		PG_RE_THROW();
	}
	PG_END_TRY();

	slot->nelems = nelems;
	slot->bytes = bytes;
	pg_atomic_fetch_add_u64(&dir->total_bytes, bytes);
	pg_write_barrier();
	pg_atomic_write_u32(&slot->state, final_state);
}

/* -----------------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------------- */

AcornCodeCacheScan *
acorn_codecache_begin_scan(Relation index, int dim)
{
	AcornCodeCacheDirectory *dir;
	AcornCodeCacheSlot *slot;
	Oid			dboid = index->rd_locator.dbOid;
	RelFileNumber relnumber = index->rd_locator.relNumber;
	uint32		state;
	AcornCCAttachEntry *att;
	bool		found;

	if (!acorn_scan_code_cache || acorn_code_cache_size_mb <= 0 || dim <= 0)
		return NULL;

	/*
	 * Steady state: an existing attachment stays valid for the backend's
	 * lifetime in M1 (no eviction, slot states never regress, REINDEX
	 * changes the relfilenumber key), so repeat scans skip the directory
	 * lock entirely.
	 */
	att = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
		return &att->scan;

	dir = acorn_cc_get_dir();
	slot = acorn_cc_slot_lookup(dir, dboid, relnumber);
	if (slot == NULL)
		return NULL;			/* directory full: run at non-inline speed */

	state = pg_atomic_read_u32(&slot->state);
	if (state == ACORN_CC_STATE_EMPTY)
	{
		if (pg_atomic_compare_exchange_u32(&slot->state, &state,
										   ACORN_CC_STATE_LOADING))
		{
			acorn_cc_load(dir, slot, index, dim);
			state = pg_atomic_read_u32(&slot->state);
		}
		/* lost the CAS: `state` now holds the winner's value */
	}

	/* Readers never wait: a slot LOADING elsewhere means no cache this scan */
	if (state != ACORN_CC_STATE_READY && state != ACORN_CC_STATE_PARTIAL)
		return NULL;
	pg_read_barrier();

	if (slot->dim != dim)
		return NULL;			/* defensive: stale slot from a prior life */

	return acorn_cc_attach(dir, slot, dboid, relnumber);
}

/*
 * Resolve the block directory for `blkno`, honoring lock-free growth.
 *
 * The backend-local snapshot (cc->blocks/cc->nblocks) is the fast path; it
 * covers every block present at attach time.  An insert may have grown the
 * directory since — if the snapshot is out of range or shows no page yet,
 * re-read the slot's live (atomic) directory and refresh the snapshot.  This
 * keeps long-lived backends serving newly inserted elements without forcing
 * a slot lookup on the hot per-discovery path.
 */
static dsa_pointer *
acorn_cc_resolve_blocks(AcornCodeCacheScan *cc, BlockNumber blkno)
{
	dsa_pointer	live_dp;
	uint32		live_nblocks;

	if (blkno < cc->nblocks && DsaPointerIsValid(cc->blocks[blkno]))
		return cc->blocks;

	/* Snapshot miss: consult the live directory (may have grown). */
	live_dp = (dsa_pointer) pg_atomic_read_u64(&cc->slot->blocks);
	live_nblocks = pg_atomic_read_u32(&cc->slot->nblocks);
	pg_read_barrier();

	if (live_dp != cc->blocks_dp)
	{
		cc->blocks_dp = live_dp;
		cc->blocks = (dsa_pointer *) dsa_get_address(cc->area, live_dp);
	}
	cc->nblocks = live_nblocks;

	if (blkno >= cc->nblocks)
		return NULL;
	return cc->blocks;
}

bool
acorn_codecache_lookup(AcornCodeCacheScan *cc,
					   BlockNumber blkno, OffsetNumber offno,
					   AcornCodeCacheHit *out)
{
	dsa_pointer *blocks;
	dsa_pointer page_dp;
	AcornCCPageHdr *ph;
	AcornCodeCacheEntry *e;
	int			dim = cc->dim;

	if (dim <= 0 || dim > ACORN_CC_MAX_DIM)
		return false;

	blocks = acorn_cc_resolve_blocks(cc, blkno);
	if (blocks == NULL)
		return false;
	page_dp = blocks[blkno];
	if (!DsaPointerIsValid(page_dp))
		return false;
	ph = (AcornCCPageHdr *) dsa_get_address(cc->area, page_dp);
	if (offno < 1 || offno > ph->nslots)
		return false;
	e = ACORN_CC_PAGE_ENTRY(ph, offno, cc->stride);

	/*
	 * Seqlock copy-out: read an even version, copy the fixed header + code,
	 * then re-read; an odd or changed version means a concurrent
	 * insert/vacuum touched this entry, so fall back (G4).  One retry covers
	 * the common case where the snapshot landed mid-write; persistent churn
	 * (never in M2's single-writer-per-slot path) falls back too.
	 */
	for (int attempt = 0; attempt < 2; attempt++)
	{
		uint32	v0 = pg_atomic_read_u32(&e->version);
		uint32	v1;

		if (v0 & 1)
			continue;			/* writer in progress */
		pg_read_barrier();

		if ((e->flags & ACORN_CC_PRESENT) == 0 ||
			(e->flags & ACORN_CC_DELETED) != 0)
		{
			/* re-check version: a concurrent insert may be turning a
			 * vacuumed slot back into a live one */
			pg_read_barrier();
			if (pg_atomic_read_u32(&e->version) != v0)
				continue;
			return false;
		}

		out->heaptid = e->heaptid;
		out->nbrtid = e->nbrtid;
		out->level = e->level;
		out->flags = e->flags;
		out->filter_val = e->filter_val;
		out->scale = e->scale;
		out->offset = e->offset;
		out->dim = dim;
		memcpy(out->code, e->code, dim);

		pg_read_barrier();
		v1 = pg_atomic_read_u32(&e->version);
		if (v1 == v0)
			return true;		/* stable snapshot */
		/* changed under us: retry once, then fall back */
	}
	return false;
}

/* -----------------------------------------------------------------------
 * Write path (M2): aminsert upsert + ambulkdelete invalidate
 *
 * Both run in the backend that just modified the index, after the index
 * tuples are durably written.  They are best-effort cache maintenance:
 * never load, never error on a full directory or unwarmed slot — those
 * simply leave the cache as-is, and the affected lookups fall back to the
 * element-page read (G4).
 * ----------------------------------------------------------------------- */

/*
 * Find an existing READY/PARTIAL slot for (dboid, relnumber) and ensure this
 * backend is attached as a writer (area mapped, slot ref cached).  Returns
 * the backend-local scan handle, or NULL when there is no warm slot to
 * maintain (no cache exists yet — the next scan will lazily load a complete
 * snapshot, so skipping the write is correct, not a lost update).
 */
static AcornCodeCacheScan *
acorn_cc_writer_attach(Relation index)
{
	AcornCodeCacheDirectory *dir;
	AcornCodeCacheSlot *slot = NULL;
	Oid			dboid = index->rd_locator.dbOid;
	RelFileNumber relnumber = index->rd_locator.relNumber;
	AcornCCAttachEntry *att;
	bool		found;
	uint32		state;

	if (acorn_code_cache_size_mb <= 0)
		return NULL;

	att = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
		return &att->scan;

	if (acorn_cc_dir == NULL)
		return NULL;			/* directory never created => no warm slot */
	dir = acorn_cc_dir;

	LWLockAcquire(&dir->lock, LW_SHARED);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		AcornCodeCacheSlot *s = &dir->slots[i];

		if (s->dboid == dboid && s->relnumber == relnumber)
		{
			slot = s;
			break;
		}
	}
	LWLockRelease(&dir->lock);
	if (slot == NULL)
		return NULL;

	state = pg_atomic_read_u32(&slot->state);
	if (state != ACORN_CC_STATE_READY && state != ACORN_CC_STATE_PARTIAL)
		return NULL;			/* EMPTY/LOADING: let the loader build it */

	return acorn_cc_attach(dir, slot, dboid, relnumber);
}

/*
 * Grow the slot's block directory to cover at least `need` blocks.  Caller
 * holds slot->wlock.  Publishes a fully-populated replacement array and
 * swaps slot->blocks atomically, so a lock-free reader sees either the old
 * (smaller) or the new (larger) array, both internally consistent.  The old
 * array leaks within the slot's DSA until M3 eviction.
 */
static void
acorn_cc_grow_dir(AcornCodeCacheDirectory *dir, AcornCodeCacheScan *cc,
				  uint32 need)
{
	AcornCodeCacheSlot *slot = cc->slot;
	uint32		old_n = pg_atomic_read_u32(&slot->nblocks);
	uint32		new_n;
	dsa_pointer old_dp;
	dsa_pointer new_dp;
	dsa_pointer *old_blocks;
	dsa_pointer *new_blocks;

	if (need <= old_n)
		return;
	new_n = ACORN_CC_DIR_GROW(need);
	old_dp = (dsa_pointer) pg_atomic_read_u64(&slot->blocks);
	old_blocks = (dsa_pointer *) dsa_get_address(cc->area, old_dp);

	new_dp = dsa_allocate0(cc->area, (Size) new_n * sizeof(dsa_pointer));
	new_blocks = (dsa_pointer *) dsa_get_address(cc->area, new_dp);
	memcpy(new_blocks, old_blocks, (Size) old_n * sizeof(dsa_pointer));

	pg_write_barrier();
	pg_atomic_write_u64(&slot->blocks, (uint64) new_dp);
	pg_atomic_write_u32(&slot->nblocks, new_n);
	slot->bytes += (Size) new_n * sizeof(dsa_pointer);
	pg_atomic_fetch_add_u64(&dir->total_bytes,
							(uint64) new_n * sizeof(dsa_pointer));

	/* refresh this backend's snapshot */
	cc->blocks_dp = new_dp;
	cc->blocks = new_blocks;
	cc->nblocks = new_n;
}

/*
 * Ensure the page array for `blkno` exists and can index `offno`.  Caller
 * holds slot->wlock.  Returns the (possibly newly allocated/grown) page
 * header, or NULL if the cache budget is exhausted (caller skips the write;
 * the slot stays usable for every other element).  A grown page array is
 * published by atomic dsa_pointer swap; the old array leaks (M3).
 */
static AcornCCPageHdr *
acorn_cc_ensure_page(AcornCodeCacheDirectory *dir, AcornCodeCacheScan *cc,
					 BlockNumber blkno, OffsetNumber offno)
{
	AcornCodeCacheSlot *slot = cc->slot;
	uint64		budget = (uint64) acorn_code_cache_size_mb * 1024 * 1024;
	dsa_pointer live_dp;
	dsa_pointer page_dp;
	AcornCCPageHdr *ph;

	/*
	 * Refresh this backend's directory snapshot from the live slot: another
	 * backend may have grown it since we attached, leaving cc->blocks
	 * pointing at a superseded (still-valid but stale) array.  We hold
	 * slot->wlock, so no concurrent grower can move it out from under us.
	 */
	live_dp = (dsa_pointer) pg_atomic_read_u64(&slot->blocks);
	if (live_dp != cc->blocks_dp)
	{
		cc->blocks_dp = live_dp;
		cc->blocks = (dsa_pointer *) dsa_get_address(cc->area, live_dp);
	}
	cc->nblocks = pg_atomic_read_u32(&slot->nblocks);

	if (blkno >= cc->nblocks)
		acorn_cc_grow_dir(dir, cc, blkno + 1);

	page_dp = cc->blocks[blkno];
	if (DsaPointerIsValid(page_dp))
	{
		ph = (AcornCCPageHdr *) dsa_get_address(cc->area, page_dp);
		if (offno <= ph->nslots)
			return ph;			/* fits the existing array */

		/* grow the page array to hold offno */
		{
			uint16	new_nslots = ACORN_CC_PAGE_GROW(offno);
			Size	new_sz = ACORN_CC_PAGE_SIZE(new_nslots, cc->stride);
			Size	old_sz = ACORN_CC_PAGE_SIZE(ph->nslots, cc->stride);
			dsa_pointer new_dp;
			AcornCCPageHdr *new_ph;

			if (pg_atomic_read_u64(&dir->total_bytes) + (new_sz - old_sz) > budget)
				return NULL;
			new_dp = dsa_allocate_extended(cc->area, new_sz,
										   DSA_ALLOC_NO_OOM | DSA_ALLOC_ZERO);
			if (!DsaPointerIsValid(new_dp))
				return NULL;
			new_ph = (AcornCCPageHdr *) dsa_get_address(cc->area, new_dp);
			memcpy(new_ph, ph, old_sz);
			new_ph->nslots = new_nslots;

			pg_write_barrier();
			cc->blocks[blkno] = new_dp;	/* atomic pointer-width store */
			slot->bytes += (new_sz - old_sz);
			pg_atomic_fetch_add_u64(&dir->total_bytes, new_sz - old_sz);
			return new_ph;
		}
	}

	/* no page array yet: allocate one sized for offno */
	{
		uint16	nslots = ACORN_CC_PAGE_GROW(offno);
		Size	page_sz = ACORN_CC_PAGE_SIZE(nslots, cc->stride);

		if (pg_atomic_read_u64(&dir->total_bytes) + page_sz > budget)
			return NULL;
		page_dp = dsa_allocate_extended(cc->area, page_sz,
										DSA_ALLOC_NO_OOM | DSA_ALLOC_ZERO);
		if (!DsaPointerIsValid(page_dp))
			return NULL;
		ph = (AcornCCPageHdr *) dsa_get_address(cc->area, page_dp);
		ph->nslots = nslots;

		pg_write_barrier();
		cc->blocks[blkno] = page_dp;	/* publish the new page array */
		slot->bytes += page_sz;
		pg_atomic_fetch_add_u64(&dir->total_bytes, page_sz);
		return ph;
	}
}

void
acorn_codecache_insert(Relation index,
					   BlockNumber blkno, OffsetNumber offno,
					   const float *vec, int dim,
					   ItemPointer heaptid, ItemPointer nbrtid,
					   uint8 level, int64 filter_val)
{
	AcornCodeCacheScan *cc;
	AcornCodeCacheDirectory *dir = acorn_cc_dir;
	AcornCCPageHdr *ph;
	AcornCodeCacheEntry *e;

	cc = acorn_cc_writer_attach(index);
	if (cc == NULL || cc->dim != dim)
		return;					/* no warm slot / dim mismatch: skip (G4) */

	LWLockAcquire(&cc->slot->wlock, LW_EXCLUSIVE);

	ph = acorn_cc_ensure_page(dir, cc, blkno, offno);
	if (ph == NULL)
	{
		LWLockRelease(&cc->slot->wlock);
		return;					/* budget exhausted: element stays a miss */
	}

	e = ACORN_CC_PAGE_ENTRY(ph, offno, cc->stride);

	/*
	 * Seqlock write: bump version odd, write every field (TID reuse fully
	 * overwrites a previously vacuumed slot's code+meta), set PRESENT (and
	 * clear DELETED) implicitly via flags, then bump version even.  PRESENT
	 * becomes visible only at write_end's barrier, so a concurrent reader
	 * either sees the whole old entry, the whole new entry, or an odd
	 * version (-> retry/fallback).  Never a torn mix.
	 */
	acorn_cc_write_begin(e);
	e->heaptid = *heaptid;
	e->nbrtid = *nbrtid;
	e->level = level;
	e->filter_val = filter_val;
	acorn_sq8_encode(dim, vec, e->code, &e->scale, &e->offset);
	e->flags = ACORN_CC_PRESENT;	/* clears any prior DELETED */
	acorn_cc_write_end(e);

	cc->slot->nelems++;			/* advisory; not budget-load-bearing */

	LWLockRelease(&cc->slot->wlock);
}

void
acorn_codecache_invalidate(Relation index,
						   BlockNumber blkno, OffsetNumber offno)
{
	AcornCodeCacheScan *cc;
	dsa_pointer *blocks;
	dsa_pointer page_dp;
	AcornCCPageHdr *ph;
	AcornCodeCacheEntry *e;

	cc = acorn_cc_writer_attach(index);
	if (cc == NULL)
		return;

	LWLockAcquire(&cc->slot->wlock, LW_EXCLUSIVE);

	/* resolve the page (no growth — a removed TID always predates growth) */
	blocks = acorn_cc_resolve_blocks(cc, blkno);
	if (blocks == NULL || !DsaPointerIsValid(blocks[blkno]))
	{
		LWLockRelease(&cc->slot->wlock);
		return;
	}
	page_dp = blocks[blkno];
	ph = (AcornCCPageHdr *) dsa_get_address(cc->area, page_dp);
	if (offno < 1 || offno > ph->nslots)
	{
		LWLockRelease(&cc->slot->wlock);
		return;
	}
	e = ACORN_CC_PAGE_ENTRY(ph, offno, cc->stride);

	/*
	 * Clear PRESENT and set DELETED under the seqlock.  A later insert that
	 * reuses this (blkno,offno) overwrites the whole entry and re-sets
	 * PRESENT (acorn_codecache_insert), so TID reuse can never serve the
	 * stale code.
	 */
	acorn_cc_write_begin(e);
	e->flags = (e->flags & ~ACORN_CC_PRESENT) | ACORN_CC_DELETED;
	acorn_cc_write_end(e);

	LWLockRelease(&cc->slot->wlock);
}
