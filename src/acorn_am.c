/*
 * acorn_am.c — acorn_hnsw index Access Method handler (Tier 2)
 *
 * Step 3 placeholder.  Registers the handler function so the extension
 * loads cleanly; the full IndexAmRoutine will be wired in Step 3.
 */

#include "postgres.h"
#include "fmgr.h"
#include "access/amapi.h"

#include "acorn_am.h"

PG_FUNCTION_INFO_V1(acorn_hnsw_handler);

Datum
acorn_hnsw_handler(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("acorn_hnsw index AM not yet implemented"),
			 errhint("Step 3 will implement the full acorn_hnsw AM.")));
	PG_RETURN_POINTER(NULL);
}
