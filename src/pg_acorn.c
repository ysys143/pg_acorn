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

	/* Register acorn_hnsw reloptions (m, ef_construction, acorn_gamma) */
	acorn_am_init();

	acorn_hook_init();
}

void
_PG_fini(void)
{
	acorn_hook_fini();
}
