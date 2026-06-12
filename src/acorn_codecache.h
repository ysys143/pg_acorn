/*
 * acorn_codecache.h — per-index shared-memory SQ8 code cache (Tier 2)
 *
 * DiskANN-style split for non-inline acorn_hnsw indexes: SQ8 codes for all
 * elements live once in shared memory; index pages keep the TID-only layout.
 * The Tier 2 scan consults the cache at neighbor discovery instead of
 * pinning the neighbor's element page (~20 us/access measured).
 *
 * M1 = read path only (docs/code-cache-design.md section 8).  The cache is
 * a hint, never an authority: any miss, a full directory, a LOADING slot in
 * another backend, or a disabled GUC degrades to the existing element-page
 * read — never to a wrong result.
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

/*
 * Cached per-element entry — mirrors AcornT2InlineEntry minus the per-edge
 * indextid (the dshash key identifies the element instead).  Codes are
 * produced by acorn_sq8_encode, so they are bit-identical to the codes the
 * inline build co-locates.
 */
typedef struct AcornCodeCacheEntry
{
	ItemPointerData heaptid;	/* element's primary heap TID (heaptids[0]) */
	ItemPointerData nbrtid;		/* element's own neighbor-tuple TID */
	uint8			level;		/* element's highest layer */
	uint8			flags;		/* ACORN_CC_* */
	/* 2 bytes implicit padding (int64 below is 8-aligned at offset 16) */
	int64			filter_val;	/* inline filter value */
	float			scale;		/* SQ8: x[i] ~ offset + scale * code[i] */
	float			offset;
	uint8			code[FLEXIBLE_ARRAY_MEMBER];	/* dim bytes */
} AcornCodeCacheEntry;

StaticAssertDecl(offsetof(AcornCodeCacheEntry, code) == 32,
				 "AcornCodeCacheEntry codes must start at byte 32");

/*
 * Opaque per-backend handle for one index's cache table.  Returned by
 * acorn_codecache_begin_scan; lives in TopMemoryContext for the backend's
 * lifetime (M1 has no eviction), so scans need no cleanup call.
 */
typedef struct AcornCodeCacheScan AcornCodeCacheScan;

/*
 * Resolve (and lazily load) the cache for a non-inline acorn_hnsw index.
 * Returns NULL whenever the cache cannot serve this scan — the caller must
 * use the element-page path.  Never blocks on another backend's load.
 */
extern AcornCodeCacheScan *acorn_codecache_begin_scan(Relation index, int dim);

/*
 * Look up one element by its index TID.  Returns NULL on miss (caller falls
 * back to the element-page read).  The returned entry is immutable and
 * never freed in M1, so it remains valid after the call.
 */
extern const AcornCodeCacheEntry *acorn_codecache_lookup(AcornCodeCacheScan *cc,
														 BlockNumber blkno,
														 OffsetNumber offno);

#endif							/* ACORN_CODECACHE_H */
