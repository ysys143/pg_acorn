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

	/* Register acorn_hnsw reloptions (m, ef_construction, acorn_gamma) */
	acorn_am_init();

	acorn_hook_init();
}

void
_PG_fini(void)
{
	acorn_hook_fini();
}
