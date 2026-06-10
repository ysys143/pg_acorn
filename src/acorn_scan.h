#ifndef ACORN_SCAN_H
#define ACORN_SCAN_H

#include "postgres.h"
#include "access/skey.h"
#include "nodes/execnodes.h"
#include "utils/relcache.h"

#include "acorn_dist.h"

/*
 * Resolve a direct C distance kernel (fmgr bypass) for the index's opclass
 * support function 1.  Returns NULL for unknown opclasses — callers must keep
 * the fmgr fallback.  Defined in acorn_scan.c; shared with the build path.
 */
AcornDistFn acorn_resolve_direct_dist(Relation index);

/*
 * ACORN-1 predicate subgraph traversal.
 *
 * Filter-failing candidates are excluded from the result set W but kept in
 * the traversal queue C, preserving graph connectivity.  This guarantees
 * top-k at any selectivity — the core ACORN invariant.
 *
 * Used by both Tier 1 (CustomScan executor) and Tier 2 (index AM scan).
 */

typedef struct AcornScanState
{
	int			ef_search;		/* candidate list size */
	int			k;				/* requested results */
	ExprState  *predicate;		/* NULL = unfiltered */
	ExprContext *econtext;		/* for predicate evaluation */
} AcornScanState;

/*
 * Execute ACORN-1 traversal.
 * result_tids_out must be caller-allocated with at least state->k slots.
 * Returns actual count (may be < k if graph has fewer matching nodes).
 */
int acorn_scan_execute(AcornScanState *state,
					   Relation index,
					   Relation heap,
					   Datum query_vec,
					   Snapshot snapshot,
					   ItemPointerData *result_tids_out);

/*
 * Resumable (streaming) scan — Tier 2 only.
 *
 * Unlike acorn_scan_execute (which rebuilds the whole traversal on every call),
 * this keeps a persistent frontier and emits heap TIDs one at a time in
 * approximate nearest-first order, expanding the graph lazily.  Each graph node
 * is expanded and emitted at most once, so pulling more results never re-runs
 * the traversal from the entry point — eliminating the O(ef) re-traversal cost
 * the ef-doubling batch loop paid at low selectivity.  The executor post-filters
 * and keeps pulling until its LIMIT is satisfied or the graph is exhausted.
 *
 * All state lives in `mcxt`; resetting/deleting that context frees the scan.
 */
typedef struct AcornStreamScan AcornStreamScan;

AcornStreamScan *acorn_stream_begin(Relation index,
									Datum query_vec,
									Snapshot snapshot,
									MemoryContext mcxt);

bool acorn_stream_next(AcornStreamScan *stream, ItemPointerData *heaptid_out);

/*
 * Tier 2 in-filter streaming scan.
 *
 * Extends the resumable frontier with ScanKey filter evaluation against the
 * inline filter_val stored in each AcornT2 element tuple.  Filter-failing
 * nodes are kept in the candidate set C (connectivity) but not added to R
 * (results), so no heap fetch is needed for scalar predicates.
 *
 * keys / nkeys point directly into the IndexScanDesc's keyData array;
 * caller must ensure they remain valid for the lifetime of the scan.
 */
typedef struct AcornT2StreamScan AcornT2StreamScan;

AcornT2StreamScan *acorn_t2_stream_begin(Relation index,
										  Datum query_vec,
										  ScanKey keys,
										  int nkeys,
										  int ef_search,
										  Snapshot snapshot,
										  MemoryContext mcxt);

bool acorn_t2_stream_next(AcornT2StreamScan *stream,
						   ItemPointerData *heaptid_out);

#endif /* ACORN_SCAN_H */
