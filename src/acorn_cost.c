/*
 * acorn_cost.c — amcostestimate for acorn_hnsw (Tier 2)
 *
 * Mirrors pgvector's hnswcostestimate shape: an ORDER-BY-only index whose cost
 * is a small fraction of a full scan (proportional to the O(log N) graph
 * probes), so the planner prefers it over a seq scan when a vector ORDER BY is
 * present.  Returns +infinity when there is no ORDER BY (the index is useless
 * without it).
 */

#include "postgres.h"

#include <math.h>

#include "access/amapi.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "utils/float.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"

#include "acorn_am.h"
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
	GenericCosts costs;
	double		ratio;

	/* Never use the index without an ORDER BY (it only answers nearest-k) */
	if (path->indexorderbys == NULL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
		return;
	}

	MemSet(&costs, 0, sizeof(costs));
	genericcostestimate(root, path, loop_count, &costs);

	/*
	 * HNSW visits roughly O(log N) nodes, a small fraction of the index.  Use
	 * a conservative fraction so the index beats a seq scan but the planner
	 * still accounts for graph traversal work.
	 */
	if (path->indexinfo->tuples > 1)
		ratio = log(path->indexinfo->tuples) / path->indexinfo->tuples;
	else
		ratio = 1;
	if (ratio > 1)
		ratio = 1;

	costs.indexStartupCost = costs.indexTotalCost * ratio;

	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}
