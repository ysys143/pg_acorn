/*
 * acorn_codecache.h — per-index shared-memory SQ8 code cache (Tier 2)
 *
 * DiskANN-style split for non-inline acorn_hnsw indexes: SQ8 codes for all
 * elements live once in shared memory; index pages keep the TID-only layout.
 * The Tier 2 scan consults the cache at neighbor discovery instead of
 * pinning the neighbor's element page (~20 us/access measured).
 *
 * M1 = read path (load + lock-free lookup).  M2 = write path: aminsert
 * upserts the new element's entry, ambulkdelete invalidates removed TIDs,
 * and a per-entry seqlock makes those mutations safe against the lock-free
 * reader.  The cache is a hint, never an authority: any miss, torn read,
 * stale flag, growth-in-progress, full directory, LOADING slot, or disabled
 * GUC degrades to the element-page read — never to a wrong result (G4).
 */

#ifndef ACORN_CODECACHE_H
#define ACORN_CODECACHE_H

#include "postgres.h"

#include "storage/block.h"
#include "storage/itemptr.h"
#include "storage/off.h"
#include "utils/relcache.h"

/* entry flags */
#define ACORN_CC_DELETED		0x01
#define ACORN_CC_PRESENT		0x02	/* slot holds a loaded entry */

/*
 * Cached per-element entry — mirrors AcornT2InlineEntry minus the per-edge
 * indextid (the entry's position in the per-block directory identifies the
 * element instead).  Codes are produced by acorn_sq8_encode, so they are
 * bit-identical to the codes the inline build co-locates.
 *
 * `version` is a seqlock (even = stable, odd = a writer is mid-update).  M2
 * mutates entries in place on insert/vacuum; readers copy the fixed header +
 * code out under the seqlock and retry once on an odd-or-changed version,
 * then fall back to the element-page read.  The writer bumps version to odd
 * before the first field write and back to even (a write barrier between)
 * after the last, so a reader never observes a half-written entry.
 */
typedef struct AcornCodeCacheEntry
{
	pg_atomic_uint32 version;	/* seqlock: even = stable, odd = writing */
	ItemPointerData heaptid;	/* element's primary heap TID (heaptids[0]) */
	ItemPointerData nbrtid;		/* element's own neighbor-tuple TID */
	uint8			level;		/* element's highest layer */
	uint8			flags;		/* ACORN_CC_* */
	/* 2 bytes implicit padding (int64 below is 8-aligned at offset 24) */
	int64			filter_val;	/* inline filter value */
	float			scale;		/* SQ8: x[i] ~ offset + scale * code[i] */
	float			offset;
	uint8			code[FLEXIBLE_ARRAY_MEMBER];	/* dim bytes */
} AcornCodeCacheEntry;

StaticAssertDecl(offsetof(AcornCodeCacheEntry, code) == 40,
				 "AcornCodeCacheEntry codes must start at byte 40");

/*
 * Caller-side copy target for a lookup hit.  The reader copies the fixed
 * fields plus up to ACORN_CC_MAX_DIM code bytes out of shared memory under
 * the entry seqlock, so the returned data is a stable snapshot the scan can
 * use after any lock is dropped.
 */
#define ACORN_CC_MAX_DIM	2000		/* pgvector vector hard cap */

typedef struct AcornCodeCacheHit
{
	ItemPointerData heaptid;
	ItemPointerData nbrtid;
	uint8			level;
	uint8			flags;
	int64			filter_val;
	float			scale;
	float			offset;
	int				dim;
	uint8			code[ACORN_CC_MAX_DIM];
} AcornCodeCacheHit;

/*
 * Opaque per-backend handle for one index's cache table.  Returned by
 * acorn_codecache_begin_scan; lives in TopMemoryContext for the backend's
 * lifetime, so scans need no cleanup call.
 */
typedef struct AcornCodeCacheScan AcornCodeCacheScan;

/*
 * Resolve (and lazily load) the cache for a non-inline acorn_hnsw index.
 * Returns NULL whenever the cache cannot serve this scan — the caller must
 * use the element-page path.  Never blocks on another backend's load.
 */
extern AcornCodeCacheScan *acorn_codecache_begin_scan(Relation index, int dim);

/*
 * Look up one element by its index TID, copying a stable snapshot into *out.
 * Returns true on a present hit, false on miss / not-present / torn read
 * (caller falls back to the element-page read).
 *
 * This is the one-shot form (resolve + read); the scan's hot loop instead
 * software-pipelines the two halves via acorn_codecache_resolve /
 * acorn_codecache_read so the entry's DRAM line is prefetched ahead of the
 * read (the ~40 MB code table overflows CPU cache, so each entry read is a
 * cold miss otherwise).
 */
extern bool acorn_codecache_lookup(AcornCodeCacheScan *cc,
								   BlockNumber blkno, OffsetNumber offno,
								   AcornCodeCacheHit *out);

/*
 * Resolve one element's cache entry to a stable in-DSA pointer WITHOUT
 * copying it out.  This is the cheap half of a lookup: directory snapshot
 * refresh + page-array dereference + offno bound check.  Returns the entry
 * pointer (suitable for __builtin_prefetch), or NULL when the cache cannot
 * serve this (blkno,offno) — directory miss, no page array, out-of-range
 * offno, or a reincarnation guard trip.  The returned pointer is valid for
 * the rest of the scan: the area is pinned by the active-scan ref and any
 * growth only moves FORWARD to a newer array (resolve re-snapshots), so a
 * resolved pointer never dangles into freed memory while the scan runs.
 *
 * No seqlock copy-out happens here, so the pointer alone proves nothing about
 * the entry's contents; acorn_codecache_read performs the version-checked copy
 * and all the G4 safety checks.  A prefetch of a pointer that later turns out
 * evicted/torn simply wastes a prefetch — the read still falls back correctly.
 */
extern const AcornCodeCacheEntry *acorn_codecache_resolve(AcornCodeCacheScan *cc,
														  BlockNumber blkno,
														  OffsetNumber offno);

/*
 * Copy a resolved entry out under its seqlock into *out.  `e` must come from
 * acorn_codecache_resolve on the same scan.  Returns true on a present, stable
 * hit (sets *out); false on not-present / torn read / NULL entry, in which
 * case the caller falls back to the element-page read (G4).
 */
extern bool acorn_codecache_read(AcornCodeCacheScan *cc,
								 const AcornCodeCacheEntry *e,
								 AcornCodeCacheHit *out);

/*
 * Release the active-scan reference taken by begin_scan (M3).  The AM must
 * call this once per begin_scan at endscan; it is NULL-safe and idempotent.
 * Dropping the last active scan on a slot reclaims its retired growth arrays.
 */
extern void acorn_codecache_end_scan(AcornCodeCacheScan *cc);

/*
 * Write path (M2).  Both are no-ops unless a READY/PARTIAL slot for the
 * index already exists in this postmaster — they never trigger a load, so
 * an unwarmed index keeps the M1 "load lazily on first scan" behavior.
 */

/*
 * Upsert one element's entry after aminsert has durably written its tuples.
 * `vec`/`scale`-less: the encoder is applied internally so the cached code
 * is bit-identical to the loader's and the inline build's.
 */
extern void acorn_codecache_insert(Relation index,
								   BlockNumber blkno, OffsetNumber offno,
								   const float *vec, int dim,
								   ItemPointer heaptid, ItemPointer nbrtid,
								   uint8 level, int64 filter_val);

/* Invalidate one element's entry (ambulkdelete): clears PRESENT + DELETED. */
extern void acorn_codecache_invalidate(Relation index,
									   BlockNumber blkno, OffsetNumber offno);

#endif							/* ACORN_CODECACHE_H */
