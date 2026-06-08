#ifndef ACORN_SCAN_H
#define ACORN_SCAN_H

#include "postgres.h"
#include "nodes/execnodes.h"
#include "utils/relcache.h"

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

#endif /* ACORN_SCAN_H */
