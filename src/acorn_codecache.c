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

#include "access/relation.h"
#include "access/xact.h"
#include "catalog/pg_authid.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "port/atomics.h"
#include "storage/bufmgr.h"
#include "storage/dsm_registry.h"
#include "storage/itemid.h"
#include "storage/lwlock.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/dsa.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/relfilenumbermap.h"

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

/*
 * Per-element overhead charged in the admission projection: amortized page
 * header + page-array growth headroom.  Only used to over-estimate a table's
 * footprint for LRU admission; the resident accounting (slot->bytes) is exact.
 */
#define ACORN_CC_PER_ELEM_OVERHEAD	8

/*
 * Retired-array list: each block-directory or page array superseded by a
 * growth is pushed here (still reachable by readers that snapshotted the old
 * pointer) and freed only once the slot has no active scan — the grace
 * period that proves no reader can touch freed memory (see acorn_cc_drain_
 * retired / the reclaim safety argument in the file header).
 */
#define ACORN_CC_RETIRE_MAX		256

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
	uint32			generation;		/* bumped per (re)load and per evict */
	uint32			nelems;
	uint64			bytes;			/* accounted bytes */
	int				dim;			/* code length; stride = 40 + dim aligned */
	pg_atomic_uint32 nblocks;		/* length of the block directory (grows) */
	pg_atomic_uint32 active_scans;	/* readers currently scanning this slot */
	pg_atomic_uint64 lastused;		/* logical clock at last begin_scan (LRU) */
	LWLock			wlock;			/* serializes M2 writers + growth + retire */
	dsa_handle		area_handle;
	/*
	 * Creator-pin ownership for the current area incarnation.  Set true under
	 * dir->lock when the loader creates+pins the area; set false under
	 * dir->lock by the SINGLE evictor that drops the pin.  Guards BUG-3: a
	 * concurrent or later evictor observing area_pinned == false never
	 * dsa_unpin()s again (dsa_unpin is not idempotent — a second call
	 * elog(ERROR)s "dsa_area not pinned").
	 */
	bool			area_pinned;
	pg_atomic_uint64 blocks;		/* dsa_pointer[nblocks] page arrays (grows) */
	/* retire list (under wlock): arrays superseded by growth, awaiting drain */
	uint32			n_retired;
	dsa_pointer		retired[ACORN_CC_RETIRE_MAX];
} AcornCodeCacheSlot;

typedef struct AcornCodeCacheDirectory
{
	LWLock			lock;
	int				lock_tranche;
	int				dsa_tranche;
	int				wlock_tranche;
	pg_atomic_uint64 total_bytes;	/* global accounted bytes, all slots */
	pg_atomic_uint64 clock;			/* logical LRU clock (monotonic) */
	AcornCodeCacheSlot slots[ACORN_CC_NSLOTS];
} AcornCodeCacheDirectory;

/* -----------------------------------------------------------------------
 * Backend-local state
 * ----------------------------------------------------------------------- */

struct AcornCodeCacheScan
{
	dsa_area	   *area;
	AcornCodeCacheSlot *slot;		/* live slot (for growth-aware lookup) */
	Oid				dboid;			/* identity this attach was resolved for */
	RelFileNumber	relnumber;
	dsa_pointer		blocks_dp;		/* snapshot of slot->blocks at attach */
	dsa_pointer    *blocks;			/* mapped block directory snapshot */
	uint32			nblocks;		/* snapshot of slot->nblocks at attach */
	Size			stride;
	int				dim;			/* code length */
	uint32			generation;		/* slot generation at attach (evict guard) */
	int				scan_refs;		/* active-scan refs this backend holds on the
									 * slot via THIS handle.  A COUNT, not a bool:
									 * the handle is shared per (dboid,relnumber),
									 * so two overlapping scans on the same index
									 * in one backend (e.g. a self-join) each take
									 * a ref and must each be balanced.  Each ref
									 * is one slot->active_scans increment. */
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

/* forward decls (eviction needs the backend-local attach cache helpers) */
static AcornCCAttachEntry *acorn_cc_attach_find(Oid dboid,
											   RelFileNumber relnumber,
											   bool *found);
static void acorn_cc_xact_callback(XactEvent event, void *arg);
static void acorn_cc_subxact_callback(SubXactEvent event,
									  SubTransactionId mySubid,
									  SubTransactionId parentSubid, void *arg);

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
	pg_atomic_init_u64(&dir->clock, 1);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		pg_atomic_init_u32(&dir->slots[i].state, ACORN_CC_STATE_EMPTY);
		pg_atomic_init_u32(&dir->slots[i].nblocks, 0);
		pg_atomic_init_u32(&dir->slots[i].active_scans, 0);
		pg_atomic_init_u64(&dir->slots[i].lastused, 0);
		pg_atomic_init_u64(&dir->slots[i].blocks, InvalidDsaPointer);
		LWLockInitialize(&dir->slots[i].wlock, dir->wlock_tranche);
	}
}

/* Monotonic logical clock tick for LRU lastused stamps. */
static inline uint64
acorn_cc_tick(AcornCodeCacheDirectory *dir)
{
	return pg_atomic_fetch_add_u64(&dir->clock, 1);
}

/* -----------------------------------------------------------------------
 * Retire + drain (reclaim of superseded growth arrays)
 *
 * RECLAIM SAFETY ARGUMENT (why no reader can dereference freed memory):
 *
 *   A block-directory or page array A is retired by a grower holding wlock,
 *   AFTER slot->blocks (or the page slot in the directory) no longer points
 *   to A.  A is freed only by acorn_cc_drain_retired, which runs under wlock
 *   and only when slot->active_scans == 0.
 *
 *   A reader brackets each scan with an active_scans ref: begin_scan does
 *     active_scans++  ;  (full barrier)  ;  read slot->blocks
 *   and end_scan does active_scans--.  A grower/draining writer does
 *     publish new array (swap pointer)  ;  (full barrier)  ;  read active_scans
 *   The two barriers pair (store-load on active_scans vs blocks):
 *     - If the drainer observes active_scans == 0, then every reader that
 *       could have observed A==slot->blocks has already done active_scans--,
 *       which is sequenced after its last dereference of A.  So no live
 *       reader holds a pointer into A: freeing it is safe.
 *     - If a reader's active_scans++ is NOT yet visible to the drainer, the
 *       drainer sees active_scans > 0 and DEFERS the free; that reader will
 *       be drained on a later pass once it has left.
 *     - A reader whose active_scans++ happens-before the drainer's read of 0
 *       cannot exist (that is the == 0 case).  A reader that increments after
 *       the drainer read 0 performs its slot->blocks load after the swap (by
 *       the barrier pairing) and therefore reads the NEW array, never A.
 *
 *   Mid-scan, a reader only ever moves FORWARD to a newer array (resolve_
 *   blocks / ensure_page refresh from the live slot); it never reverts to a
 *   retired one.  Combined with "freed only at active_scans == 0", a reader
 *   touches only arrays published no earlier than its own begin_scan, none of
 *   which are freed while it scans.  On any doubt the free is deferred, never
 *   forced — the G4 bias.
 *
 * Bounded retire list: if it fills before a drain, growth stops retiring and
 * the array leaks until whole-slot eviction (rare; a soak that never
 * quiesces).  Documented; not a correctness issue.
 * ----------------------------------------------------------------------- */

/*
 * Push a superseded array onto the slot's retire list, freeing it inline if
 * no scan is active (the common single-backend insert case never accumulates
 * a retire list).  Caller holds wlock.  Budget is decremented by `sz` here:
 * a retired array is logically released the instant it is superseded, so the
 * cap reflects only live arrays even before the physical free.
 */
static void
acorn_cc_retire(AcornCodeCacheDirectory *dir, dsa_area *area,
				AcornCodeCacheSlot *slot, dsa_pointer dp, Size sz)
{
	if (DsaPointerIsValid(dp))
	{
		/*
		 * StoreLoad fence: the caller already swapped slot->blocks (a store);
		 * we are about to load slot->active_scans.  This must be a FULL
		 * barrier, not a read (acquire) barrier — it pairs with the reader's
		 * "active_scans++ ; pg_memory_barrier() ; read slot->blocks" in
		 * acorn_cc_scan_acquire.  With both full fences, a reader whose
		 * increment is NOT visible to our load here is guaranteed to perform
		 * its slot->blocks load AFTER our swap, so it reads the NEW array,
		 * never `dp`.  A pg_read_barrier() would not order the prior store
		 * before this load and the reader could observe the old array while
		 * we free it.
		 */
		pg_memory_barrier();
		if (pg_atomic_read_u32(&slot->active_scans) == 0)
			dsa_free(area, dp);			/* no reader can hold it: free now */
		else if (slot->n_retired < ACORN_CC_RETIRE_MAX)
			slot->retired[slot->n_retired++] = dp;	/* defer to drain */
		/* else: list full -> leak until eviction (rare; never quiescing) */
	}
	if (sz > 0)
	{
		slot->bytes -= sz;
		pg_atomic_fetch_sub_u64(&dir->total_bytes, sz);
	}
}

/*
 * Free every deferred retired array iff no scan is active on the slot.
 * Caller holds wlock.  Called opportunistically before a growth allocation
 * and from end_scan when the last active scan leaves.
 */
static void
acorn_cc_drain_retired(dsa_area *area, AcornCodeCacheSlot *slot)
{
	if (slot->n_retired == 0)
		return;

	/*
	 * Full StoreLoad fence: the retired arrays were superseded by an earlier
	 * slot->blocks swap (store); we now load active_scans.  As in
	 * acorn_cc_retire, this must be a full barrier so a reader whose
	 * active_scans++ is invisible here is guaranteed to have read the new
	 * blocks pointer (post-swap), never a retired array.
	 */
	pg_memory_barrier();
	if (pg_atomic_read_u32(&slot->active_scans) != 0)
		return;					/* a reader may still hold a retired pointer */

	for (uint32 i = 0; i < slot->n_retired; i++)
	{
		if (DsaPointerIsValid(slot->retired[i]))
			dsa_free(area, slot->retired[i]);
		slot->retired[i] = InvalidDsaPointer;
	}
	slot->n_retired = 0;
}

/* -----------------------------------------------------------------------
 * Whole-slot LRU eviction
 *
 * Evicting a slot releases its ENTIRE DSA area — every block array, page
 * array and retired array at once — by dropping the creator's pin and
 * detaching this backend's mapping; the OS reclaims the segment once the
 * last backend that mapped it detaches (each does so when it notices the
 * generation bump at begin_scan).  The slot is cleared to EMPTY and its
 * generation bumped so any backend still holding a stale attach re-resolves.
 *
 * A slot is evictable only when state is READY/PARTIAL and active_scans == 0
 * (no in-flight reader): a slot mid-LOADING or being scanned is never
 * evicted.  Caller holds dir->lock EXCLUSIVE.
 * ----------------------------------------------------------------------- */

/* Forget this backend's cached attachment + mapping for a slot, if any. */
static void
acorn_cc_forget_attach(Oid dboid, RelFileNumber relnumber)
{
	AcornCCAttachEntry *att;
	bool		found;

	if (acorn_cc_attached == NULL)
		return;
	att = acorn_cc_attach_find(dboid, relnumber, &found);
	if (!found)
		return;
	if (att->scan.area != NULL)
		dsa_detach(att->scan.area);
	hash_search(acorn_cc_attached, &att->key, HASH_REMOVE, NULL);
}

/*
 * Evict one slot.  Caller holds dir->lock EXCLUSIVE.  Returns the bytes freed.
 *
 * Ordering (the evictor half of the BUG-2 hazard discipline): mark the slot
 * EMPTY and bump generation FIRST, then a full StoreLoad barrier, then read
 * active_scans.  This pairs with acorn_cc_scan_acquire's "active_scans++ ;
 * barrier ; reload state/generation":
 *   - If a racing reader's increment is visible here (active_scans > 0), we
 *     do NOT release the DSA area's creator pin — that reader may already
 *     hold the area pointer.  The slot is still left EMPTY (it will reload
 *     into a fresh area), and the old area's creator pin lingers until the
 *     postmaster recycles the segment.  A bounded, rare leak (the racing
 *     window is increment-then-reload), never a use-after-free.
 *   - If active_scans == 0 here, every reader either has not yet incremented
 *     (it will then observe EMPTY/gen++ after its own barrier and back out)
 *     or has already left.  No live reader holds a pointer into this area:
 *     dropping the creator pin and detaching is safe.
 */
static uint64
acorn_cc_evict_slot(AcornCodeCacheDirectory *dir, AcornCodeCacheSlot *slot)
{
	uint64		freed = slot->bytes;
	dsa_handle	handle = slot->area_handle;
	Oid			dboid = slot->dboid;
	RelFileNumber relnumber = slot->relnumber;
	dsa_area   *area;

	/*
	 * Serialize against M2 writers: insert/invalidate take slot->wlock and
	 * then deref this backend's area mapping.  Acquiring wlock here (we
	 * already hold dir->lock EXCLUSIVE; writers never hold dir->lock while
	 * taking wlock, so no ordering inversion) guarantees no writer is
	 * mid-write when we clear the slot.  After we bump generation under
	 * wlock, a writer that next takes wlock observes the mismatch and skips.
	 */
	LWLockAcquire(&slot->wlock, LW_EXCLUSIVE);

	/*
	 * Mark EMPTY first (under dir->lock) so no new begin_scan can attach to
	 * the area we are about to release; bump generation so a backend holding
	 * a stale attach (or mid-acquire) detaches / backs out.
	 */
	pg_atomic_write_u32(&slot->state, ACORN_CC_STATE_EMPTY);
	slot->generation++;
	slot->dboid = InvalidOid;
	slot->relnumber = InvalidRelFileNumber;
	slot->nelems = 0;
	slot->bytes = 0;
	slot->dim = 0;
	slot->n_retired = 0;
	for (uint32 i = 0; i < ACORN_CC_RETIRE_MAX; i++)
		slot->retired[i] = InvalidDsaPointer;
	pg_atomic_write_u32(&slot->nblocks, 0);
	pg_atomic_write_u64(&slot->blocks, InvalidDsaPointer);
	pg_atomic_fetch_sub_u64(&dir->total_bytes, freed);
	LWLockRelease(&slot->wlock);

	/* StoreLoad fence: publish EMPTY/gen++ before reading active_scans */
	pg_memory_barrier();
	if (pg_atomic_read_u32(&slot->active_scans) != 0)
	{
		/*
		 * A reader incremented in the race window.  It will see EMPTY/gen++
		 * and back out, but it may already hold the area pointer — do not
		 * free.  Forget only our own mapping; leave the creator pin
		 * (area_pinned stays true; a later evictor reclaims once quiescent).
		 */
		acorn_cc_forget_attach(dboid, relnumber);
		return freed;
	}

	/*
	 * BUG-3 fix — drop the creator pin EXACTLY ONCE per incarnation.
	 *
	 * Claim pin ownership under dir->lock: only the evictor that flips
	 * area_pinned true->false is allowed to dsa_unpin().  A concurrent or
	 * later evictor (e.g. reset() scanning all slots, or a racing
	 * evict_to_fit) sees area_pinned == false and never unpins again —
	 * dsa_unpin is not idempotent and would elog(ERROR) "dsa_area not
	 * pinned".  We always unpin through a FRESH transient attach of the
	 * handle, never through a cached dsa_pin_mapping mapping: detaching a
	 * cached mapping would pull it out from under the owning backend's later
	 * reads, and reusing it risks acting on a different incarnation after the
	 * handle is recycled.  Because we still hold dir->lock and area_pinned was
	 * true, the handle names THIS live incarnation; the transient attach +
	 * unpin + detach drops the creator pin and releases our transient
	 * mapping.  The segment's memory is reclaimed once every other backend
	 * detaches its own mapping (each does so at its next begin_scan
	 * generation check via acorn_cc_forget_attach).
	 */
	if (!slot->area_pinned)
	{
		/* another evictor already dropped this incarnation's pin */
		acorn_cc_forget_attach(dboid, relnumber);
		return freed;
	}
	slot->area_pinned = false;

	/* forget our own cached mapping first (it is pinned for our lifetime) */
	acorn_cc_forget_attach(dboid, relnumber);

	/* transient attach solely to drop the creator pin, then detach it */
	area = dsa_attach(handle);
	dsa_unpin(area);
	dsa_detach(area);

	return freed;
}

/*
 * Evict least-recently-used evictable slots until `need` bytes fit under the
 * budget, or no evictable slot remains.  Caller holds dir->lock EXCLUSIVE.
 * Returns true if enough room was made.
 */
static bool
acorn_cc_evict_to_fit(AcornCodeCacheDirectory *dir, uint64 need, uint64 budget)
{
	for (;;)
	{
		AcornCodeCacheSlot *victim = NULL;
		uint64		victim_lru = UINT64_MAX;

		if (pg_atomic_read_u64(&dir->total_bytes) + need <= budget)
			return true;

		for (int i = 0; i < ACORN_CC_NSLOTS; i++)
		{
			AcornCodeCacheSlot *s = &dir->slots[i];
			uint32		st = pg_atomic_read_u32(&s->state);
			uint64		lu;

			if (st != ACORN_CC_STATE_READY && st != ACORN_CC_STATE_PARTIAL)
				continue;
			if (pg_atomic_read_u32(&s->active_scans) != 0)
				continue;		/* in-flight reader: never evict */
			lu = pg_atomic_read_u64(&s->lastused);
			if (lu < victim_lru)
			{
				victim_lru = lu;
				victim = s;
			}
		}

		if (victim == NULL)
			return false;		/* nothing evictable; caller runs PARTIAL */

		acorn_cc_evict_slot(dir, victim);
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

	/*
	 * Balance any active-scan ref on transaction/subtransaction abort: an
	 * aborting query tears the executor down without running amendscan, so
	 * without this the slot's active_scans would leak and the slot could never
	 * be evicted.  Registered once, lazily, the first time this backend
	 * touches the cache (paired with the TopMemoryContext-lived attach cache).
	 */
	RegisterXactCallback(acorn_cc_xact_callback, NULL);
	RegisterSubXactCallback(acorn_cc_subxact_callback, NULL);
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
					  uint32 nblocks, Size stride, int dim, uint32 generation)
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
	e->scan.dboid = dboid;
	e->scan.relnumber = relnumber;
	e->scan.blocks_dp = blocks_dp;
	e->scan.blocks = blocks;
	e->scan.nblocks = nblocks;
	e->scan.stride = stride;
	e->scan.dim = dim;
	e->scan.generation = generation;
	e->scan.scan_refs = 0;
	return e;
}

/*
 * Attach this backend to a READY/PARTIAL slot's DSA area and map the block
 * directory.  Attachments are pinned for the backend's lifetime
 * (dsa_pin_mapping) and cached, so each backend pays the attach cost once
 * per index.  Returns NULL if the slot was evicted out from under us before
 * we could pin a mapping.
 *
 * BUG-3 corollary: dsa_attach() must run while dir->lock is held SHARED.
 * Eviction takes dir->lock EXCLUSIVE before it drops the creator pin, so
 * holding SHARED across dsa_attach + dsa_pin_mapping guarantees the area is
 * still alive when we attach, and our mapping pin then keeps it alive after
 * we release the lock — even once a later evictor drops the creator pin.
 * Without the lock, the handle could name an already-freed (or recycled)
 * segment, yielding "could not attach to dynamic shared area" or, worse, a
 * mapping of a different index's recycled area.
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
	uint32		generation;
	uint32		st;
	int			dim;
	dsa_area   *area;
	dsa_pointer *blocks;
	MemoryContext old;

	e = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
		return &e->scan;

	LWLockAcquire(&dir->lock, LW_SHARED);
	/* re-validate under the lock: the slot may have been evicted/reused */
	st = pg_atomic_read_u32(&slot->state);
	if ((st != ACORN_CC_STATE_READY && st != ACORN_CC_STATE_PARTIAL) ||
		slot->dboid != dboid || slot->relnumber != relnumber)
	{
		LWLockRelease(&dir->lock);
		return NULL;
	}
	area_handle = slot->area_handle;
	dim = slot->dim;
	generation = slot->generation;
	blocks_dp = (dsa_pointer) pg_atomic_read_u64(&slot->blocks);
	nblocks = pg_atomic_read_u32(&slot->nblocks);

	/*
	 * Attach + pin the mapping while still holding dir->lock SHARED so the
	 * area cannot be freed by a concurrent evictor mid-attach.  Once pinned,
	 * the mapping keeps the segment alive for our backend's lifetime.
	 */
	old = MemoryContextSwitchTo(TopMemoryContext);
	area = dsa_attach(area_handle);
	dsa_pin_mapping(area);
	MemoryContextSwitchTo(old);
	LWLockRelease(&dir->lock);

	blocks = (dsa_pointer *) dsa_get_address(area, blocks_dp);

	e = acorn_cc_attach_store(dboid, relnumber, area, slot, blocks_dp,
							  blocks, nblocks, ACORN_CC_ENTRY_STRIDE(dim), dim,
							  generation);
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
	uint64		projected;
	dsa_pointer blocks_dp;
	dsa_pointer *blocks;
	volatile uint64 bytes = 0;
	volatile uint32 nelems = 0;
	volatile uint32 final_state = ACORN_CC_STATE_READY;

	/*
	 * Whole-index LRU admission: before loading, evict least-recently-used
	 * READY/PARTIAL slots until this table's projected footprint fits beside
	 * the resident ones (whose bytes are tracked exactly in dir->total_bytes).
	 *
	 * The projection must be on the SAME scale as total_bytes (actual cache
	 * bytes), not the index's on-disk size — element tuples (80B header +
	 * vector) pack only a handful per page, so the cache table is a fraction
	 * of the index.  We bound the element count by nblocks * the max element
	 * tuples that fit on a page, then multiply by the per-entry stride plus
	 * the directory/page-header overhead.  This over-estimates (it assumes
	 * every page is packed with the smallest possible element tuple), so
	 * admission evicts slightly eagerly — never under-reserves.  If nothing
	 * is evictable the load proceeds and stops at PARTIAL at the real cap.
	 */
	{
		Size		min_etup = MAXALIGN(sizeof(AcornT2ElementTupleData) +
										offsetof(AcornPgVector, x) +
										(Size) dim * sizeof(float));
		uint64		max_elems = (uint64) nblocks *
			(BLCKSZ / Max(min_etup, 1));

		projected = max_elems * (stride + ACORN_CC_PER_ELEM_OVERHEAD) +
			(uint64) ACORN_CC_DIR_GROW(nblocks) * sizeof(dsa_pointer);
	}
	if (projected < budget)
	{
		LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
		acorn_cc_evict_to_fit(dir, projected, budget);
		LWLockRelease(&dir->lock);
	}

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
		slot->area_pinned = true;	/* this incarnation holds the creator pin */
		slot->dim = dim;
		slot->generation++;
		LWLockRelease(&dir->lock);

		pg_atomic_write_u64(&slot->blocks, (uint64) blocks_dp);
		pg_atomic_write_u32(&slot->nblocks, dir_blocks);

		/* the loading backend is attached by construction */
		acorn_cc_attach_store(index->rd_locator.dbOid,
							  index->rd_locator.relNumber, area, slot,
							  blocks_dp, blocks, dir_blocks, stride, dim,
							  slot->generation);
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

/*
 * Acquire an active-scan reference and confirm the slot is still live.
 *
 * Hazard-pointer discipline (the BUG-2 fix): publish the refcount FIRST, full
 * barrier, THEN revalidate the slot.  `expect_gen` is the generation observed
 * when this backend attached / validated the slot; if eviction has bumped it
 * (or flipped the slot away from READY/PARTIAL, or re-keyed it to another
 * index) the ref we just took is on a dead slot — we back it out and fail.
 *
 * This is what makes eviction's "active_scans == 0 => safe to free" sound:
 * the evictor sets state=EMPTY / gen++ BEFORE its own full barrier and read of
 * active_scans.  Either our increment is visible to the evictor (it sees
 * active_scans > 0 and does not free), OR our post-increment reload here sees
 * the EMPTY/gen bump and we back out before dereferencing slot->blocks.  No
 * reader that proceeds can hold a pointer into a freed area.
 *
 * Returns true if the ref is held and the slot is live; false if backed out.
 */
static bool
acorn_cc_scan_acquire(AcornCodeCacheDirectory *dir, AcornCodeCacheScan *cc,
					  uint32 expect_gen)
{
	AcornCodeCacheSlot *slot = cc->slot;
	uint32		st;

	pg_atomic_fetch_add_u32(&slot->active_scans, 1);
	pg_memory_barrier();		/* publish ref before revalidating (and before
								 * reading slot->blocks) */

	st = pg_atomic_read_u32(&slot->state);
	if (slot->generation != expect_gen ||
		(st != ACORN_CC_STATE_READY && st != ACORN_CC_STATE_PARTIAL) ||
		cc->dboid != slot->dboid || cc->relnumber != slot->relnumber)
	{
		/*
		 * Slot died under us (evicted/reloaded): drop only the ref we just
		 * added.  scan_refs counts SUCCESSFUL acquires, so a back-out must not
		 * touch it — another overlapping scan on this shared handle may legitly
		 * hold refs we must not clobber.
		 */
		pg_atomic_fetch_sub_u32(&slot->active_scans, 1);
		return false;
	}

	/* successful acquire: one more balanced ref on this handle */
	cc->scan_refs++;
	pg_atomic_write_u64(&slot->lastused, acorn_cc_tick(dir));

	/* refresh snapshot AFTER the ref is visible and the slot is confirmed live */
	cc->blocks_dp = (dsa_pointer) pg_atomic_read_u64(&slot->blocks);
	cc->nblocks = pg_atomic_read_u32(&slot->nblocks);
	cc->blocks = (dsa_pointer *) dsa_get_address(cc->area, cc->blocks_dp);
	return true;
}

AcornCodeCacheScan *
acorn_codecache_begin_scan(Relation index, int dim)
{
	AcornCodeCacheDirectory *dir;
	AcornCodeCacheSlot *slot;
	AcornCodeCacheScan *cc;
	Oid			dboid = index->rd_locator.dbOid;
	RelFileNumber relnumber = index->rd_locator.relNumber;
	uint32		state;
	AcornCCAttachEntry *att;
	bool		found;

	if (!acorn_scan_code_cache || acorn_code_cache_size_mb <= 0 || dim <= 0)
		return NULL;

	dir = acorn_cc_get_dir();

	/*
	 * Steady state: reuse a cached attachment, but REVALIDATE it — M3
	 * eviction can free the area and bump the slot generation under us.  A
	 * generation mismatch (or the slot now holding a different index, or no
	 * longer READY/PARTIAL) means our mapping is stale: detach and re-resolve.
	 */
	att = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
	{
		AcornCodeCacheSlot *s = att->scan.slot;
		uint32		st = pg_atomic_read_u32(&s->state);

		if (att->scan.generation == s->generation &&
			s->dboid == dboid && s->relnumber == relnumber &&
			(st == ACORN_CC_STATE_READY || st == ACORN_CC_STATE_PARTIAL))
		{
			/*
			 * Provisional pass; scan_acquire publishes the ref then RE-checks
			 * generation/state (BUG-2 hazard discipline).  On a back-out the
			 * slot was evicted between here and the increment: forget the
			 * stale mapping and fall through to a fresh resolve.
			 */
			if (acorn_cc_scan_acquire(dir, &att->scan, att->scan.generation))
				return &att->scan;
			acorn_cc_forget_attach(dboid, relnumber);
		}
		else
		{
			/* stale: drop the mapping and fall through to a fresh resolve */
			acorn_cc_forget_attach(dboid, relnumber);
		}
	}

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

	cc = acorn_cc_attach(dir, slot, dboid, relnumber);
	if (cc == NULL)
		return NULL;			/* slot evicted mid-attach: fall back (G4) */
	/*
	 * cc->generation was captured inside acorn_cc_attach (under dir->lock).
	 * If eviction bumps it before our ref is published, scan_acquire backs
	 * out and we fall back to the element-page path for this scan (G4).
	 */
	if (!acorn_cc_scan_acquire(dir, cc, cc->generation))
		return NULL;
	return cc;
}

/*
 * Drop ONE held active-scan ref on this handle (the per-begin_scan match for
 * the normal endscan path), decrementing slot->active_scans.  If this drops
 * the slot's last in-flight reader, drain its retire list (the reclaim grace
 * point).  NULL-safe and balances exactly one begin_scan: the handle is shared
 * per (dboid,relnumber), so it can hold more than one ref (overlapping scans);
 * each end drops one.  A call with no held ref is a no-op.
 */
static void
acorn_cc_release_one_ref(AcornCodeCacheScan *cc)
{
	AcornCodeCacheSlot *slot;

	if (cc == NULL || cc->scan_refs <= 0)
		return;
	slot = cc->slot;
	cc->scan_refs--;

	/* publish all our reads before dropping the ref */
	pg_memory_barrier();
	if (pg_atomic_sub_fetch_u32(&slot->active_scans, 1) == 0 &&
		slot->n_retired > 0)
	{
		/*
		 * Last reader out: try to reclaim deferred arrays.  Take wlock to
		 * serialize with growers and other drainers; recheck active_scans
		 * under the lock (another scan may have started).
		 */
		LWLockAcquire(&slot->wlock, LW_EXCLUSIVE);
		acorn_cc_drain_retired(cc->area, slot);
		LWLockRelease(&slot->wlock);
	}
}

/*
 * Drop EVERY held active-scan ref on this handle.  Used by abort cleanup,
 * where any number of overlapping scans on this index were torn down at once
 * without their endscans running.  Equivalent to calling release_one_ref until
 * the handle holds no ref.
 */
static void
acorn_cc_release_all_refs(AcornCodeCacheScan *cc)
{
	while (cc != NULL && cc->scan_refs > 0)
		acorn_cc_release_one_ref(cc);
}

/*
 * Release the active-scan reference taken by begin_scan.  Drains the slot's
 * retire list if this was the last active scan (the reclaim grace point).
 * Idempotent and NULL-safe so the AM can call it unconditionally at endscan.
 */
void
acorn_codecache_end_scan(AcornCodeCacheScan *cc)
{
	acorn_cc_release_one_ref(cc);
}

/*
 * Transaction/subtransaction ABORT cleanup: balance any active-scan ref this
 * backend still holds.
 *
 * The AM calls end_scan at amendscan/amrescan, but an aborting query or
 * transaction tears the executor down WITHOUT running amendscan (PostgreSQL
 * reclaims executor state via memory-context resets and resource owners, not
 * index_endscan).  The cc handle lives in TopMemoryContext (the attach cache),
 * so its held scan refs would otherwise survive the abort and leave
 * slot->active_scans permanently elevated — a slot that eviction and
 * pg_acorn_code_cache_reset() (both gated on active_scans == 0) can never
 * reclaim.  Walking the backend's attach cache here and dropping every held
 * ref restores the invariant: active_scans returns to 0 when a scan ends for
 * ANY reason.
 */
static void
acorn_cc_release_all_scan_refs(void)
{
	HASH_SEQ_STATUS seq;
	AcornCCAttachEntry *att;

	if (acorn_cc_attached == NULL)
		return;

	hash_seq_init(&seq, acorn_cc_attached);
	while ((att = (AcornCCAttachEntry *) hash_seq_search(&seq)) != NULL)
		acorn_cc_release_all_refs(&att->scan);
}

static void
acorn_cc_xact_callback(XactEvent event, void *arg)
{
	if (event == XACT_EVENT_ABORT || event == XACT_EVENT_PARALLEL_ABORT)
		acorn_cc_release_all_scan_refs();
}

static void
acorn_cc_subxact_callback(SubXactEvent event, SubTransactionId mySubid,
						  SubTransactionId parentSubid, void *arg)
{
	if (event == SUBXACT_EVENT_ABORT_SUB)
		acorn_cc_release_all_scan_refs();
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

	/*
	 * Snapshot miss: the live directory may have GROWN within this same
	 * incarnation (insert), so refresh from it.  But the snapshot must only
	 * ever be refreshed against cc->area — the incarnation we attached and
	 * pinned.  Eviction+reload while we scan (active_scans > 0 stops the area
	 * being FREED, not the slot being re-loaded) bumps the slot generation
	 * and re-points slot->blocks into a DIFFERENT area.  Dereferencing that
	 * new dsa_pointer against our old cc->area would read garbage / another
	 * index's data (the observed cross-read).  Growth never bumps generation,
	 * so a generation mismatch means reincarnation: stop trusting the cache
	 * for the rest of this scan and fall back (G4).
	 */
	if (cc->slot->generation != cc->generation)
		return NULL;

	live_dp = (dsa_pointer) pg_atomic_read_u64(&cc->slot->blocks);
	live_nblocks = pg_atomic_read_u32(&cc->slot->nblocks);
	pg_read_barrier();

	/* re-check generation after reading blocks: the read must belong to our
	 * incarnation (growth-only refresh) */
	if (cc->slot->generation != cc->generation)
		return NULL;

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

const AcornCodeCacheEntry *
acorn_codecache_resolve(AcornCodeCacheScan *cc,
						BlockNumber blkno, OffsetNumber offno)
{
	dsa_pointer *blocks;
	dsa_pointer page_dp;
	AcornCCPageHdr *ph;
	int			dim = cc->dim;

	if (dim <= 0 || dim > ACORN_CC_MAX_DIM)
		return NULL;

	blocks = acorn_cc_resolve_blocks(cc, blkno);
	if (blocks == NULL)
		return NULL;
	page_dp = blocks[blkno];
	if (!DsaPointerIsValid(page_dp))
		return NULL;
	ph = (AcornCCPageHdr *) dsa_get_address(cc->area, page_dp);
	if (offno < 1 || offno > ph->nslots)
		return NULL;
	return ACORN_CC_PAGE_ENTRY(ph, offno, cc->stride);
}

bool
acorn_codecache_read(AcornCodeCacheScan *cc,
					 const AcornCodeCacheEntry *e,
					 AcornCodeCacheHit *out)
{
	int			dim = cc->dim;

	if (e == NULL || dim <= 0 || dim > ACORN_CC_MAX_DIM)
		return false;

	/*
	 * Seqlock copy-out: read an even version, copy the fixed header + code,
	 * then re-read; an odd or changed version means a concurrent
	 * insert/vacuum touched this entry, so fall back (G4).  One retry covers
	 * the common case where the snapshot landed mid-write; persistent churn
	 * (never in M2's single-writer-per-slot path) falls back too.  `e` is the
	 * resolved entry pointer; the version field is mutated in place by the
	 * writer (a const cast for the atomic read is sound — pg_atomic_read does
	 * not write).
	 */
	for (int attempt = 0; attempt < 2; attempt++)
	{
		pg_atomic_uint32 *ver = (pg_atomic_uint32 *) &e->version;
		uint32	v0 = pg_atomic_read_u32(ver);
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
			if (pg_atomic_read_u32(ver) != v0)
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
		v1 = pg_atomic_read_u32(ver);
		if (v1 == v0)
			return true;		/* stable snapshot */
		/* changed under us: retry once, then fall back */
	}
	return false;
}

bool
acorn_codecache_lookup(AcornCodeCacheScan *cc,
					   BlockNumber blkno, OffsetNumber offno,
					   AcornCodeCacheHit *out)
{
	const AcornCodeCacheEntry *e = acorn_codecache_resolve(cc, blkno, offno);

	return acorn_codecache_read(cc, e, out);
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
	{
		AcornCodeCacheSlot *s = att->scan.slot;
		uint32		st = pg_atomic_read_u32(&s->state);

		/*
		 * Revalidate the cached attach: M3 eviction can have freed this
		 * incarnation's area and reloaded a new one (bumped generation).
		 * A match means the mapping is live; a mismatch means stale — forget
		 * it and re-resolve so we never write through a detached/foreign
		 * area mapping.
		 */
		if (att->scan.generation == s->generation &&
			s->dboid == dboid && s->relnumber == relnumber &&
			(st == ACORN_CC_STATE_READY || st == ACORN_CC_STATE_PARTIAL))
			return &att->scan;
		acorn_cc_forget_attach(dboid, relnumber);
	}

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

	/* opportunistic reclaim of earlier retirements before we allocate more */
	acorn_cc_drain_retired(cc->area, slot);

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

	/* retire the old directory (frees now if quiescent, else defers) */
	acorn_cc_retire(dir, cc->area, slot, old_dp,
					(Size) old_n * sizeof(dsa_pointer));

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

			if (pg_atomic_read_u64(&dir->total_bytes) + new_sz > budget)
				return NULL;
			acorn_cc_drain_retired(cc->area, slot);
			new_dp = dsa_allocate_extended(cc->area, new_sz,
										   DSA_ALLOC_NO_OOM | DSA_ALLOC_ZERO);
			if (!DsaPointerIsValid(new_dp))
				return NULL;
			new_ph = (AcornCCPageHdr *) dsa_get_address(cc->area, new_dp);
			memcpy(new_ph, ph, old_sz);
			new_ph->nslots = new_nslots;

			pg_write_barrier();
			cc->blocks[blkno] = new_dp;	/* atomic pointer-width store */
			slot->bytes += new_sz;
			pg_atomic_fetch_add_u64(&dir->total_bytes, new_sz);

			/* retire the old page array (frees now if quiescent) */
			acorn_cc_retire(dir, cc->area, slot, page_dp, old_sz);
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

	/*
	 * Re-validate the incarnation under wlock: eviction (which does not take
	 * wlock) may have flipped this slot EMPTY / bumped generation between
	 * writer_attach and here.  Writing then would deref our stale area against
	 * the new incarnation's (Invalid) block directory.  On a mismatch, skip
	 * the upsert — the element is simply a future cache miss (G4).
	 */
	if (cc->generation != cc->slot->generation ||
		cc->slot->dboid != index->rd_locator.dbOid ||
		cc->slot->relnumber != index->rd_locator.relNumber)
	{
		LWLockRelease(&cc->slot->wlock);
		return;
	}

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

/* -----------------------------------------------------------------------
 * Observability (M3): stats SRF + admin evict function
 * ----------------------------------------------------------------------- */

PG_FUNCTION_INFO_V1(pg_acorn_code_cache_stats);
PG_FUNCTION_INFO_V1(pg_acorn_code_cache_summary);
PG_FUNCTION_INFO_V1(pg_acorn_code_cache_evict);
PG_FUNCTION_INFO_V1(pg_acorn_code_cache_reset);

/*
 * pg_acorn_code_cache_stats() -> SETOF record, one row per occupied slot.
 * Columns: dboid oid, relfilenumber int8, indexrelid regclass,
 *          state text, nelems int8, bytes int8, generation int8,
 *          lastused int8, blocks_retired_pending int4.
 *
 * Read-only: snapshots each occupied slot under the directory lock into a
 * tuplestore, then emits.  indexrelid is resolved best-effort from the
 * relfilenumber (NULL if not in this database / not resolvable).
 */
Datum
pg_acorn_code_cache_stats(PG_FUNCTION_ARGS)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	AcornCodeCacheDirectory *dir;

	InitMaterializedSRF(fcinfo, 0);

	if (acorn_cc_dir == NULL)
		acorn_cc_dir = (AcornCodeCacheDirectory *)
			GetNamedDSMSegment(ACORN_CC_SEGMENT_NAME,
							   sizeof(AcornCodeCacheDirectory),
							   acorn_cc_dir_init, &(bool){false});
	dir = acorn_cc_dir;

	LWLockAcquire(&dir->lock, LW_SHARED);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		AcornCodeCacheSlot *s = &dir->slots[i];
		Datum		values[9];
		bool		nulls[9];
		uint32		st;
		const char *st_name;

		if (s->dboid == InvalidOid)
			continue;

		st = pg_atomic_read_u32(&s->state);
		switch (st)
		{
			case ACORN_CC_STATE_EMPTY:	 st_name = "empty";	  break;
			case ACORN_CC_STATE_LOADING: st_name = "loading"; break;
			case ACORN_CC_STATE_READY:	 st_name = "ready";	  break;
			case ACORN_CC_STATE_PARTIAL: st_name = "partial"; break;
			default:					 st_name = "unknown"; break;
		}

		memset(nulls, 0, sizeof(nulls));
		values[0] = ObjectIdGetDatum(s->dboid);
		values[1] = Int64GetDatum((int64) s->relnumber);

		/* indexrelid: best-effort resolution from relfilenumber */
		if (s->dboid == MyDatabaseId)
		{
			Oid		relid = RelidByRelfilenumber(InvalidOid, s->relnumber);

			if (OidIsValid(relid))
				values[2] = ObjectIdGetDatum(relid);
			else
				nulls[2] = true;
		}
		else
			nulls[2] = true;

		values[3] = CStringGetTextDatum(st_name);
		values[4] = Int64GetDatum((int64) s->nelems);
		values[5] = Int64GetDatum((int64) s->bytes);
		values[6] = Int64GetDatum((int64) s->generation);
		values[7] = Int64GetDatum((int64) pg_atomic_read_u64(&s->lastused));
		values[8] = Int32GetDatum((int32) s->n_retired);

		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc, values, nulls);
	}
	LWLockRelease(&dir->lock);

	return (Datum) 0;
}

/*
 * pg_acorn_code_cache_summary() -> (total_bytes int8, budget_bytes int8,
 *                                   n_slots int4, n_occupied int4).
 */
Datum
pg_acorn_code_cache_summary(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[4];
	bool		nulls[4] = {false, false, false, false};
	AcornCodeCacheDirectory *dir;
	int			occupied = 0;
	HeapTuple	tuple;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	tupdesc = BlessTupleDesc(tupdesc);

	if (acorn_cc_dir == NULL)
		acorn_cc_dir = (AcornCodeCacheDirectory *)
			GetNamedDSMSegment(ACORN_CC_SEGMENT_NAME,
							   sizeof(AcornCodeCacheDirectory),
							   acorn_cc_dir_init, &(bool){false});
	dir = acorn_cc_dir;

	LWLockAcquire(&dir->lock, LW_SHARED);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
		if (dir->slots[i].dboid != InvalidOid)
			occupied++;
	values[0] = Int64GetDatum((int64) pg_atomic_read_u64(&dir->total_bytes));
	LWLockRelease(&dir->lock);

	values[1] = Int64GetDatum((int64) acorn_code_cache_size_mb * 1024 * 1024);
	values[2] = Int32GetDatum(ACORN_CC_NSLOTS);
	values[3] = Int32GetDatum(occupied);

	tuple = heap_form_tuple(tupdesc, values, nulls);
	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

/*
 * pg_acorn_code_cache_evict(regclass) -> bool.  Force-evict one index's
 * slot.  Returns true if a slot was present and evicted, false if absent or
 * not evictable (mid-LOADING, or has an active scan).  Restricted to
 * superusers and members of pg_maintain (force-dropping shared cache state
 * is an administrative action).
 */
Datum
pg_acorn_code_cache_evict(PG_FUNCTION_ARGS)
{
	Oid			indexrelid = PG_GETARG_OID(0);
	Relation	index;
	Oid			dboid;
	RelFileNumber relnumber;
	AcornCodeCacheDirectory *dir;
	bool		evicted = false;

	if (!superuser() &&
		!has_privs_of_role(GetUserId(), ROLE_PG_MAINTAIN))
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser or have privileges of pg_maintain "
						"to evict the acorn code cache")));

	index = relation_open(indexrelid, AccessShareLock);
	dboid = index->rd_locator.dbOid;
	relnumber = index->rd_locator.relNumber;
	relation_close(index, AccessShareLock);

	if (acorn_cc_dir == NULL)
		PG_RETURN_BOOL(false);
	dir = acorn_cc_dir;

	LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		AcornCodeCacheSlot *s = &dir->slots[i];
		uint32		st;

		if (s->dboid != dboid || s->relnumber != relnumber)
			continue;
		st = pg_atomic_read_u32(&s->state);
		if ((st == ACORN_CC_STATE_READY || st == ACORN_CC_STATE_PARTIAL) &&
			pg_atomic_read_u32(&s->active_scans) == 0)
		{
			acorn_cc_evict_slot(dir, s);
			evicted = true;
		}
		break;
	}
	LWLockRelease(&dir->lock);

	PG_RETURN_BOOL(evicted);
}

/*
 * pg_acorn_code_cache_reset() -> int.  Force-evict every evictable slot
 * (READY/PARTIAL with no active scan) across the whole directory, returning
 * the count evicted.  Slots mid-LOADING or with an active scan are left
 * untouched (evict_slot self-guards and defers a raced free).  This is the
 * only way to clear ORPHAN slots left by dropped indexes — those have no
 * surviving regclass, so pg_acorn_code_cache_evict() cannot name them — and
 * is the clean-slate primitive for ops and deterministic tests.  Restricted
 * to superusers and members of pg_maintain.
 */
Datum
pg_acorn_code_cache_reset(PG_FUNCTION_ARGS)
{
	AcornCodeCacheDirectory *dir;
	int			n = 0;

	if (!superuser() &&
		!has_privs_of_role(GetUserId(), ROLE_PG_MAINTAIN))
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser or have privileges of pg_maintain "
						"to reset the acorn code cache")));

	/*
	 * Attach the directory even if this backend never used the cache: the
	 * orphan/foreign slots we must clear were created by other backends and
	 * live in the shared DSM segment.
	 */
	dir = acorn_cc_get_dir();

	LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
	{
		AcornCodeCacheSlot *s = &dir->slots[i];
		uint32		st = pg_atomic_read_u32(&s->state);

		if ((st == ACORN_CC_STATE_READY || st == ACORN_CC_STATE_PARTIAL) &&
			pg_atomic_read_u32(&s->active_scans) == 0)
		{
			acorn_cc_evict_slot(dir, s);
			n++;
		}
	}
	LWLockRelease(&dir->lock);

	PG_RETURN_INT32(n);
}
