/*
 * acorn_codecache.c — per-index shared-memory SQ8 code cache (M1: read path)
 *
 * Registry: one named DSM segment ("pg_acorn_cc", PG17 dsm_registry) holds a
 * directory of per-index slots keyed by (dboid, relfilenumber).  Each slot
 * owns a DSA area containing a dshash keyed by the element's index TID
 * packed into a uint64, mapping to the dsa_pointer of an immutable entry
 * (heaptid, nbrtid, level, flags, filter_val, SQ8 scale/offset/code).
 *
 * Loading is lazy and non-blocking: the first scan that finds a slot EMPTY
 * CASes it to LOADING and walks the index main fork, quantizing every
 * element tuple's fp32 vector with acorn_sq8_encode (the same encoder the
 * inline build uses, so codes are bit-identical to inline codes).  Other
 * scans never wait: while LOADING and on any miss they use the element-page
 * fallback (design G4 — correctness never depends on cache state).  If the
 * pg_acorn.code_cache_size budget is exceeded mid-load the slot becomes
 * PARTIAL: present entries serve, misses fall back.
 *
 * M1 caveats (fixed in M2 — do not rely on the cache for correctness):
 *   - no insert upsert: elements inserted after the load are misses, which
 *     fall back to the element-page read (correct, just slower);
 *   - no vacuum invalidation: an index TID reused after VACUUM could serve
 *     a stale entry.  Acceptable while pg_acorn.scan_code_cache defaults
 *     off; M2 adds ambulkdelete invalidation + entry versioning.  Stale
 *     deleted flags / heaptids are partially masked by the exact re-rank,
 *     which re-reads the element page before emission.
 */

#include "postgres.h"

#include "lib/dshash.h"
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

/*
 * Estimated per-entry cost beyond the payload: dshash item header + dsa
 * chunk overhead.  Used only for budget accounting (admission is advisory;
 * exceeding the estimate degrades to PARTIAL, never to an error).
 */
#define ACORN_CC_PER_ENTRY_OVERHEAD	48

/* pack an element index TID into the dshash key */
#define ACORN_CC_KEY(blkno, offno) \
	(((uint64) (blkno) << 16) | (uint16) (offno))

typedef struct AcornCodeCacheSlot
{
	Oid				dboid;			/* InvalidOid = slot unclaimed */
	RelFileNumber	relnumber;
	pg_atomic_uint32 state;			/* ACORN_CC_STATE_* */
	uint32			generation;		/* bumped per (re)load */
	uint32			nelems;
	uint64			bytes;			/* accounted bytes (payload + overhead) */
	int				dim;			/* code length; entries are 32 + dim B */
	dsa_handle		area_handle;
	dshash_table_handle table_handle;
} AcornCodeCacheSlot;

typedef struct AcornCodeCacheDirectory
{
	/*
	 * Locking contract: dir->lock (exclusive) protects slot claiming
	 * (dboid/relnumber assignment) and handle publication; (shared) protects
	 * handle reads.  Slot state transitions go through the atomic only:
	 * EMPTY -> LOADING by CAS (single loader), LOADING -> READY|PARTIAL by
	 * the loader after a write barrier.  Handles are published under the
	 * lock BEFORE the state leaves LOADING, so any backend that observes
	 * READY/PARTIAL and then takes the lock reads valid handles.
	 */
	LWLock			lock;
	int				lock_tranche;
	int				dsa_tranche;
	int				dshash_tranche;
	pg_atomic_uint64 total_bytes;	/* global accounted bytes, all slots */
	AcornCodeCacheSlot slots[ACORN_CC_NSLOTS];
} AcornCodeCacheDirectory;

/* dshash entry: packed element TID -> dsa_pointer of AcornCodeCacheEntry */
typedef struct AcornCCMapEntry
{
	uint64			key;
	dsa_pointer		entry;
} AcornCCMapEntry;

/* -----------------------------------------------------------------------
 * Backend-local state
 * ----------------------------------------------------------------------- */

struct AcornCodeCacheScan
{
	dsa_area	   *area;
	dshash_table   *table;
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

static const dshash_parameters acorn_cc_hash_params = {
	sizeof(uint64),
	sizeof(AcornCCMapEntry),
	dshash_memcmp,
	dshash_memhash,
	dshash_memcpy,
	0							/* tranche_id filled in at runtime */
};

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
	dir->dshash_tranche = LWLockNewTrancheId();
	LWLockInitialize(&dir->lock, dir->lock_tranche);
	pg_atomic_init_u64(&dir->total_bytes, 0);
	for (int i = 0; i < ACORN_CC_NSLOTS; i++)
		pg_atomic_init_u32(&dir->slots[i].state, ACORN_CC_STATE_EMPTY);
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
		LWLockRegisterTranche(acorn_cc_dir->dshash_tranche, "pg_acorn_cc_hash");
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
					  dsa_area *area, dshash_table *table)
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
	e->scan.table = table;
	return e;
}

/*
 * Attach this backend to a READY/PARTIAL slot's DSA area and dshash table.
 * Attachments are pinned for the backend's lifetime (dsa_pin_mapping) and
 * cached, so each backend pays the attach cost once per index.
 */
static AcornCodeCacheScan *
acorn_cc_attach(AcornCodeCacheDirectory *dir, AcornCodeCacheSlot *slot,
				Oid dboid, RelFileNumber relnumber)
{
	AcornCCAttachEntry *e;
	bool		found;
	dsa_handle	area_handle;
	dshash_table_handle table_handle;
	dsa_area   *area;
	dshash_table *table;
	dshash_parameters params = acorn_cc_hash_params;
	MemoryContext old;

	e = acorn_cc_attach_find(dboid, relnumber, &found);
	if (found)
		return &e->scan;

	LWLockAcquire(&dir->lock, LW_SHARED);
	area_handle = slot->area_handle;
	table_handle = slot->table_handle;
	params.tranche_id = dir->dshash_tranche;
	LWLockRelease(&dir->lock);

	old = MemoryContextSwitchTo(TopMemoryContext);
	area = dsa_attach(area_handle);
	dsa_pin_mapping(area);
	table = dshash_attach(area, &params, table_handle, NULL);
	MemoryContextSwitchTo(old);

	e = acorn_cc_attach_store(dboid, relnumber, area, table);
	return &e->scan;
}

/* -----------------------------------------------------------------------
 * Loader
 * ----------------------------------------------------------------------- */

/*
 * Load all element tuples of `index` into the slot's table.  Caller owns
 * the LOADING claim (won the EMPTY -> LOADING CAS).  On return the slot is
 * READY, or PARTIAL when the budget ran out; on error the slot is published
 * PARTIAL with whatever loaded (entries are fully built before they become
 * reachable, so partial content is always servable) and the error is
 * re-thrown.
 */
static void
acorn_cc_load(AcornCodeCacheDirectory *dir, AcornCodeCacheSlot *slot,
			  Relation index, int dim)
{
	dsa_area   *area;
	dshash_table *table;
	dshash_parameters params = acorn_cc_hash_params;
	MemoryContext old;
	uint64		budget = (uint64) acorn_code_cache_size_mb * 1024 * 1024;
	uint64		other_bytes;
	Size		esize = offsetof(AcornCodeCacheEntry, code) + dim;
	Size		ecost = MAXALIGN(esize) + ACORN_CC_PER_ENTRY_OVERHEAD;
	volatile uint64 bytes = 0;
	volatile uint32 nelems = 0;
	volatile uint32 final_state = ACORN_CC_STATE_READY;

	params.tranche_id = dir->dshash_tranche;

	/* backend-local control structs must outlive the transaction */
	old = MemoryContextSwitchTo(TopMemoryContext);
	area = dsa_create(dir->dsa_tranche);
	dsa_pin(area);
	dsa_pin_mapping(area);
	table = dshash_create(area, &params, NULL);
	MemoryContextSwitchTo(old);

	/* publish handles before the state can leave LOADING */
	LWLockAcquire(&dir->lock, LW_EXCLUSIVE);
	slot->area_handle = dsa_get_handle(area);
	slot->table_handle = dshash_get_hash_table_handle(table);
	slot->dim = dim;
	slot->generation++;
	LWLockRelease(&dir->lock);

	/* the loading backend is attached by construction */
	acorn_cc_attach_init();
	acorn_cc_attach_store(index->rd_locator.dbOid,
						  index->rd_locator.relNumber, area, table);

	other_bytes = pg_atomic_read_u64(&dir->total_bytes);

	PG_TRY();
	{
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);

		for (BlockNumber blkno = HNSW_METAPAGE_BLKNO + 1;
			 blkno < nblocks && final_state == ACORN_CC_STATE_READY;
			 blkno++)
		{
			Buffer		buf;
			Page		page;
			OffsetNumber maxoff;

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
				AcornCCMapEntry *m;
				dsa_pointer p;
				uint64		key;
				bool		found;

				if (!ItemIdIsUsed(iid) || !ItemIdHasStorage(iid))
					continue;
				etup = (AcornT2ElementTuple) PageGetItem(page, iid);
				if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
					continue;	/* skip meta/neighbor/inline-cont tuples */
				vec = (AcornPgVector *) AcornT2ElementTupleGetVector(etup);
				if ((int) vec->dim != dim)
					continue;	/* defensive: never cache a malformed code */

				if (other_bytes + bytes + ecost > budget)
				{
					final_state = ACORN_CC_STATE_PARTIAL;
					break;
				}

				p = dsa_allocate_extended(area, esize, DSA_ALLOC_NO_OOM);
				if (!DsaPointerIsValid(p))
				{
					final_state = ACORN_CC_STATE_PARTIAL;
					break;
				}

				/*
				 * Build the entry completely BEFORE it becomes reachable
				 * through the dshash, so concurrent readers (the slot may
				 * be observed PARTIAL after an error) never see a torn
				 * entry.
				 */
				e = (AcornCodeCacheEntry *) dsa_get_address(area, p);
				e->heaptid = etup->heaptids[0];
				e->nbrtid = etup->neighbortid;
				e->level = etup->level;
				e->flags = (etup->deleted != 0) ? ACORN_CC_DELETED : 0;
				e->filter_val = etup->filter_val;
				acorn_sq8_encode(dim, vec->x, e->code, &e->scale, &e->offset);

				key = ACORN_CC_KEY(blkno, offno);
				m = (AcornCCMapEntry *)
					dshash_find_or_insert(table, &key, &found);
				m->entry = p;
				dshash_release_lock(table, m);

				bytes += ecost;
				nelems++;
			}

			UnlockReleaseBuffer(buf);
		}
	}
	PG_CATCH();
	{
		/*
		 * Publish what loaded so far as PARTIAL: every inserted entry is
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

	if (!acorn_scan_code_cache || acorn_code_cache_size_mb <= 0 || dim <= 0)
		return NULL;

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

const AcornCodeCacheEntry *
acorn_codecache_lookup(AcornCodeCacheScan *cc,
					   BlockNumber blkno, OffsetNumber offno)
{
	uint64		key = ACORN_CC_KEY(blkno, offno);
	AcornCCMapEntry *m;
	const AcornCodeCacheEntry *e;

	m = (AcornCCMapEntry *) dshash_find(cc->table, &key, false);
	if (m == NULL)
		return NULL;
	e = (const AcornCodeCacheEntry *) dsa_get_address(cc->area, m->entry);
	dshash_release_lock(cc->table, m);

	/*
	 * Safe to dereference after releasing the partition lock: M1 entries
	 * are immutable and never freed (no eviction, no vacuum invalidation).
	 * M2 entry versioning revisits this contract.
	 */
	return e;
}
