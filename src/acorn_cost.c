/*
 * acorn_cost.c — amcostestimate for acorn_hnsw (Tier 2)
 *
 * Step 3 placeholder.  The real implementation uses clauselist_selectivity()
 * to price filtered scans cheaper than unfiltered ones, so the planner
 * automatically prefers acorn_hnsw over seq scan when filters are present.
 */

#include "postgres.h"
#include "nodes/pathnodes.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "utils/selfuncs.h"

#include "acorn_cost.h"

void
acorn_hnsw_costestimate(PlannerInfo *root,
						IndexPath *path,
						double loop_count,
						Cost *indexStartupCost,
						Cost *indexTotalCost,
						Selectivity *indexSelectivity,
						double *indexCorrelation,
						double *indexPages)
{
	/*
	 * Step 3 will call clauselist_selectivity() on path->indexquals and
	 * produce a cost proportional to n * selectivity * log(n).
	 * For now, return a placeholder that won't crash the planner.
	 */
	*indexStartupCost  = 0;
	*indexTotalCost    = 0;
	*indexSelectivity  = 1.0;
	*indexCorrelation  = 0.0;
	*indexPages        = 0;
}
