/*
 * pg_acorn.c — extension entry point
 *
 * Registers GUCs, installs the set_rel_pathlist_hook (Tier 1), and exposes
 * the acorn_hnsw index AM handler (Tier 2).
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/guc.h"

#include "pg_acorn.h"
#include "acorn_hook.h"
#include "acorn_am.h"

PG_MODULE_MAGIC;

/* GUC: toggle Tier 1 hook without restarting */
bool acorn_enable_hook = true;

/* GUC: default ACORN-gamma value for new acorn_hnsw indexes */
int acorn_default_gamma = ACORN_DEFAULT_GAMMA;

/* GUC: enable 2-hop expansion for in-filter scans (experimental) */
bool acorn_enable_2hop = false;

/* GUC: runtime ef_search cap for acorn_hnsw (Tier 2) streaming scans */
int acorn_ef_search = ACORN_DEFAULT_EF_SEARCH;

/*
 * GUC: member-first expansion for Tier 2 in-filter scans.  When on, the
 * expansion budget (ef_search) is spent on filter-PASSING candidates first;
 * failing candidates are only expanded when no passing candidate is queued.
 * Designed to pair with indexes built with acorn_payload_edges=true, whose
 * same-partition edges keep the predicate subgraph connected (Qdrant-style
 * filtered traversal).  Emission ordering safety is unchanged: a result is
 * only emitted when no unexpanded candidate of either kind is closer.
 */
bool acorn_member_first = false;

/*
 * GUC: buffered emission for Tier 2 in-filter scans.  When on (default), the
 * scan runs the expansion phase to completion FIRST — until the ef_search
 * expansion budget is exhausted or the candidate frontier empties — and only
 * then starts emitting, in exact-distance order.  The legacy eager mode
 * (off) emits the greedy local minimum as soon as no unexpanded candidate is
 * closer than the nearest discovered result, which violates the ORDER BY
 * contract: a node with a smaller exact distance can still be UNDISCOVERED
 * when the first row is emitted (measured: rank-0 emitted d=0.654 while an
 * undiscovered d=0.000 node existed).  Buffering trades the LIMIT-k early
 * exit (full budget ~2-14ms at inline ef=200-800) for zero inversions.
 */
bool acorn_buffered_emission = true;

/* GUC: Tier 2 scan fast-path toggle — direct C distance kernel (fmgr bypass) */
bool acorn_scan_direct_dist = true;

/*
 * GUC: build-path toggle — direct C distance kernel (fmgr bypass) for index
 * build and incremental insert.  Every distance during construction otherwise
 * goes through fmgr (133ns vs 28ns SIMD floor measured by W3); the in-memory
 * bulk build is distance-dominated, so this is the build-time lever.
 * Off forces the fmgr path (debug/benchmark; results numerically identical).
 */
bool acorn_build_direct_dist = true;

/*
 * GUC: deterministic build seed.  -1 (default) keeps the legacy PID-derived
 * level-RNG seeding (every rebuild produces a different graph).  >= 0 seeds
 * the level RNG deterministically: identical data + identical seed produce an
 * identical index, enabling reproducible benchmarks and prebuilt index pools.
 *
 * SERIAL BUILDS ONLY: a parallel build (max_parallel_maintenance_workers > 0
 * and the planner grants workers) interleaves tuples nondeterministically
 * across participants, so the graph varies run to run even at a fixed seed
 * and worker count.  Set max_parallel_maintenance_workers = 0 when a
 * reproducible graph is required.
 */
int acorn_build_seed = -1;

/*
 * GUC: Tier 2 scan fast-path toggle — prefetch neighbor pages per expansion.
 * Default OFF: measured -10..-13% QPS on warm shared_buffers (every
 * PrefetchBuffer is a redundant buffer-table lookup when the page is already
 * resident).  Enable for IO-bound deployments where the index exceeds
 * shared_buffers and reads hit the OS/disk.
 */
bool acorn_scan_prefetch = false;

/*
 * GUC: Tier 2 scan fast-path toggle — carry neighbor-tuple location + level
 * in the candidate queue so expansion does not re-read the element page.
 */
bool acorn_scan_single_read = true;

/*
 * GUC: Tier 2 scan fast-path toggle — single hash probe for the visited set
 * (HASH_ENTER with found flag instead of HASH_FIND + HASH_ENTER).
 */
bool acorn_scan_visited_oneprobe = true;

/*
 * GUC: Tier 2 scan fast-path toggle — direct int4 comparison for known
 * btree int4 operator predicates in the inline filter (fmgr bypass).
 */
bool acorn_scan_direct_filter = true;

/*
 * GUC: Tier 2 scan toggle — use co-located neighbor data (SQ8 vectors +
 * metadata in the layer-0 neighbor lists) on indexes built with
 * acorn_inline_vectors=true.  Off forces the classic per-neighbor
 * element-page path on the same index (debug/benchmark; the TID-slot
 * layout is unchanged, so both paths read inline indexes correctly).
 */
bool acorn_scan_inline_vectors = true;

/*
 * GUC: Tier 2 scan toggle — serve neighbor discovery on NON-inline indexes
 * from the per-index shared-memory SQ8 code cache (acorn_codecache.c).
 * Default off (M1: read path only; flip after the design gates pass).
 * Misses and disabled/LOADING slots fall back to the element-page read, so
 * results never depend on cache state.
 */
bool acorn_scan_code_cache = false;

/*
 * GUC: total shared-memory budget for the SQ8 code cache, in MB.  0 disables
 * the cache entirely.  A load that would exceed the budget stops and leaves
 * the slot PARTIAL (present entries serve, misses fall back).
 */
int acorn_code_cache_size_mb = 512;

void _PG_init(void);
void _PG_fini(void);

void
_PG_init(void)
{
	/* GUC: pg_acorn.enable_hook */
	DefineCustomBoolVariable(
		"pg_acorn.enable_hook",
		"Enable the ACORN-1 set_rel_pathlist_hook (Tier 1).",
		NULL,
		&acorn_enable_hook,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.default_gamma */
	DefineCustomIntVariable(
		"pg_acorn.default_gamma",
		"Default ACORN gamma for acorn_hnsw indexes (1 = ACORN-1, 2+ = ACORN-gamma).",
		NULL,
		&acorn_default_gamma,
		1,      /* default */
		1,      /* min */
		8,      /* max */
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.enable_2hop */
	DefineCustomBoolVariable(
		"pg_acorn.enable_2hop",
		"Enable 2-hop NaviX-Directed expansion in acorn_hnsw in-filter scans (experimental).",
		NULL,
		&acorn_enable_2hop,
		false,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.ef_search */
	DefineCustomIntVariable(
		"pg_acorn.ef_search",
		"Candidate list size (ef) for acorn_hnsw Tier 2 streaming scans. "
		"Higher values raise recall at the cost of more page reads.",
		NULL,
		&acorn_ef_search,
		ACORN_DEFAULT_EF_SEARCH,	/* default */
		1,							/* min */
		4000,						/* max (expansion budget; allows deep sweeps) */
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.member_first */
	DefineCustomBoolVariable(
		"pg_acorn.member_first",
		"Spend the Tier 2 ef_search budget on filter-passing candidates first "
		"(pair with acorn_payload_edges=true indexes).",
		NULL,
		&acorn_member_first,
		false,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.buffered_emission */
	DefineCustomBoolVariable(
		"pg_acorn.buffered_emission",
		"Run the Tier 2 expansion phase to completion (full ef_search budget) "
		"before emitting, so emitted distances are exactly ordered.  Off "
		"restores the legacy eager emission (greedy local minimum; may emit "
		"out of order before the budget is spent).",
		NULL,
		&acorn_buffered_emission,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_direct_dist */
	DefineCustomBoolVariable(
		"pg_acorn.scan_direct_dist",
		"Use direct C distance kernels (fmgr bypass) in acorn_hnsw scans "
		"for known pgvector distance functions.  Off forces the fmgr path "
		"(debug/benchmark; results are numerically identical).",
		NULL,
		&acorn_scan_direct_dist,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_prefetch */
	DefineCustomBoolVariable(
		"pg_acorn.scan_prefetch",
		"Prefetch distinct unvisited neighbor pages before the per-neighbor "
		"distance loop in acorn_hnsw scan expansions.  Off by default: pure "
		"overhead when the index is resident in shared_buffers; enable for "
		"IO-bound scans.",
		NULL,
		&acorn_scan_prefetch,
		false,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_single_read */
	DefineCustomBoolVariable(
		"pg_acorn.scan_single_read",
		"Capture neighbor-tuple location at node discovery so acorn_hnsw "
		"scan expansion does not re-read the element page (debug/benchmark; "
		"results identical).",
		NULL,
		&acorn_scan_single_read,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_visited_oneprobe */
	DefineCustomBoolVariable(
		"pg_acorn.scan_visited_oneprobe",
		"Use a single hash probe (HASH_ENTER) for the acorn_hnsw scan "
		"visited set instead of find-then-enter (debug/benchmark; results "
		"identical).",
		NULL,
		&acorn_scan_visited_oneprobe,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.build_direct_dist */
	DefineCustomBoolVariable(
		"pg_acorn.build_direct_dist",
		"Use direct C distance kernels (fmgr bypass) in acorn_hnsw index "
		"builds and inserts for known pgvector distance functions.  Off "
		"forces the fmgr path (debug/benchmark; results are numerically "
		"identical).",
		NULL,
		&acorn_build_direct_dist,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.build_seed */
	DefineCustomIntVariable(
		"pg_acorn.build_seed",
		"Seed for the acorn_hnsw level RNG during index build. "
		"-1 = legacy PID-derived seeding (each rebuild differs); "
		">= 0 = deterministic SERIAL builds for reproducible benchmarks "
		"(parallel builds are never bit-reproducible; set "
		"max_parallel_maintenance_workers = 0 for graph identity).",
		NULL,
		&acorn_build_seed,
		-1,			/* default: legacy PID-derived behavior */
		-1,			/* min */
		INT_MAX,	/* max */
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_direct_filter */
	DefineCustomBoolVariable(
		"pg_acorn.scan_direct_filter",
		"Evaluate known int4 comparison predicates on the inline filter "
		"value with direct C compares instead of fmgr (debug/benchmark; "
		"results identical).",
		NULL,
		&acorn_scan_direct_filter,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_inline_vectors */
	DefineCustomBoolVariable(
		"pg_acorn.scan_inline_vectors",
		"Use co-located neighbor data (SQ8 vectors + metadata in layer-0 "
		"neighbor lists) on acorn_inline_vectors indexes.  Off forces the "
		"classic per-neighbor element-page path (debug/benchmark).",
		NULL,
		&acorn_scan_inline_vectors,
		true,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.scan_code_cache */
	DefineCustomBoolVariable(
		"pg_acorn.scan_code_cache",
		"Serve acorn_hnsw neighbor discovery on non-inline indexes from the "
		"shared-memory SQ8 code cache.  Misses fall back to the classic "
		"element-page read; inline-vector indexes never consult the cache.",
		NULL,
		&acorn_scan_code_cache,
		false,
		PGC_USERSET,
		0,
		NULL, NULL, NULL
	);

	/* GUC: pg_acorn.code_cache_size */
	DefineCustomIntVariable(
		"pg_acorn.code_cache_size",
		"Total shared-memory budget for the acorn_hnsw SQ8 code cache. "
		"0 disables the cache; a load exceeding the budget stops and serves "
		"partially (misses fall back to element-page reads).",
		NULL,
		&acorn_code_cache_size_mb,
		512,		/* default 512 MB */
		0,			/* min: 0 = disabled */
		1048576,	/* max: 1 TB */
		PGC_USERSET,
		GUC_UNIT_MB,
		NULL, NULL, NULL
	);

	/* Register acorn_hnsw reloptions (m, ef_construction, acorn_gamma) */
	acorn_am_init();

	acorn_hook_init();
}

void
_PG_fini(void)
{
	acorn_hook_fini();
}
