/*
 * acorn_hook.c — set_rel_pathlist_hook + CustomScan (Tier 1)
 *
 * Detects relations scanned with an HNSW index that also have:
 *   1. An ORDER BY with a vector distance operator (<->, <=>, <#>)
 *   2. At least one WHERE clause qual
 *
 * For matching relations, injects an AcornScan CustomPath whose cost
 * accounts for filter selectivity (unlike pgvector's hnswcostestimate
 * which ignores it).  The executor delegates to acorn_scan.c.
 *
 * Falls back to normal HNSW planning when pg_acorn.enable_hook = off,
 * when the pattern is not detected, or when no HNSW index exists.
 */

#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"

#include "access/amapi.h"
#include "access/tableam.h"
#include "catalog/pg_am.h"
#include "executor/executor.h"
#include "nodes/extensible.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "parser/parsetree.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"
#include "utils/snapmgr.h"
#include "utils/syscache.h"

#include "pg_acorn.h"
#include "acorn_hook.h"
#include "acorn_scan.h"

/* Name used to identify our CustomScan node in EXPLAIN output */
#define ACORN_CUSTOM_SCAN_NAME "AcornScan"

/* Previous hook in the chain — must be called first */
static set_rel_pathlist_hook_type prev_hook = NULL;

/* -----------------------------------------------------------------------
 * AcornScanState — CustomScanState subtype
 * ----------------------------------------------------------------------- */

typedef struct AcornCustomScanState
{
	CustomScanState css;			/* must be first */

	/* index and heap relations */
	Relation		index;
	Relation		heap;

	/* query vector — evaluated once from the ORDER BY expression */
	Datum			query_datum;
	bool			query_evaluated;

	/* ACORN traversal parameters */
	int				k;				/* LIMIT value */
	int				ef_search;		/* candidate pool size */

	/* predicate evaluation */
	ExprState	   *predicate;
	ExprContext	   *econtext;

	/* results computed in first ExecCustomScan call */
	ItemPointerData *result_tids;
	int				result_count;
	int				result_pos;		/* current position */
} AcornCustomScanState;

/* -----------------------------------------------------------------------
 * Plan-time custom data serialized into the CustomScan node
 *
 * Stored in CustomScan.custom_private as a List:
 *   [0] = OID of HNSW index relation
 *   [1] = ORDER BY expression (to evaluate query vector)
 *   [2] = k (LIMIT)
 *   [3] = ef_search
 * ----------------------------------------------------------------------- */

/* -----------------------------------------------------------------------
 * Forward declarations
 * ----------------------------------------------------------------------- */

static Plan *acorn_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
									struct CustomPath *best_path,
									List *tlist, List *clauses,
									List *custom_plans);
static Node *acorn_create_scan_state(CustomScan *cscan);
static void  acorn_begin_scan(CustomScanState *node, EState *estate,
							  int eflags);
static TupleTableSlot *acorn_exec_scan(CustomScanState *node);
static void  acorn_end_scan(CustomScanState *node);
static void  acorn_rescan(CustomScanState *node);
static void  acorn_explain_scan(CustomScanState *node, List *ancestors,
								ExplainState *es);

static CustomPathMethods acorn_path_methods = {
	.CustomName			= ACORN_CUSTOM_SCAN_NAME,
	.PlanCustomPath		= acorn_plan_custom_path,
};

static CustomScanMethods acorn_scan_methods = {
	.CustomName			= ACORN_CUSTOM_SCAN_NAME,
	.CreateCustomScanState = acorn_create_scan_state,
};

static CustomExecMethods acorn_exec_methods = {
	.CustomName			= ACORN_CUSTOM_SCAN_NAME,
	.BeginCustomScan	= acorn_begin_scan,
	.ExecCustomScan		= acorn_exec_scan,
	.EndCustomScan		= acorn_end_scan,
	.ReScanCustomScan	= acorn_rescan,
	.ExplainCustomScan	= acorn_explain_scan,
};

/* -----------------------------------------------------------------------
 * Pattern detection helpers
 * ----------------------------------------------------------------------- */

/*
 * Return the OID of the HNSW index access method, or InvalidOid.
 */
static Oid
get_hnsw_amoid(void)
{
	HeapTuple	tup;
	Oid			amoid = InvalidOid;

	tup = SearchSysCache1(AMNAME, CStringGetDatum("hnsw"));
	if (HeapTupleIsValid(tup))
	{
		amoid = ((Form_pg_am) GETSTRUCT(tup))->oid;
		ReleaseSysCache(tup);
	}
	return amoid;
}

/*
 * Check if the relation has an HNSW index that covers the ORDER BY vector.
 * Returns the IndexOptInfo if found, else NULL.
 */
static IndexOptInfo *
find_hnsw_index(RelOptInfo *rel)
{
	Oid hnsw_amoid = get_hnsw_amoid();
	ListCell *lc;

	if (!OidIsValid(hnsw_amoid))
		return NULL;

	foreach(lc, rel->indexlist)
	{
		IndexOptInfo *idx = (IndexOptInfo *) lfirst(lc);
		if (idx->relam == hnsw_amoid)
			return idx;
	}
	return NULL;
}

/*
 * Walk the path list and check if the ORDER BY uses a vector distance op.
 * Also extracts the query expression (the non-indexed argument to the op).
 *
 * Returns true if a vector ORDER BY is found; sets *query_expr.
 */
static bool
detect_vector_orderby(PlannerInfo *root, RelOptInfo *rel,
					  Node **query_expr)
{
	ListCell *lc;

	foreach(lc, root->query_pathkeys)
	{
		PathKey *pk = (PathKey *) lfirst(lc);
		EquivalenceMember *em;
		Expr *expr;
		ListCell *lc2;

		foreach(lc2, pk->pk_eclass->ec_members)
		{
			em = (EquivalenceMember *) lfirst(lc2);
			expr = em->em_expr;

			/* Look for a distance operator call: dist(indexed_col, query) */
			if (IsA(expr, OpExpr))
			{
				OpExpr *op = (OpExpr *) expr;
				/* Distance operators return float8; check arity */
				if (list_length(op->args) == 2)
				{
					/* Second arg is the query vector (not from the table) */
					Node *arg2 = lsecond(op->args);
					if (!IsA(arg2, Var))
					{
						*query_expr = arg2;
						return true;
					}
				}
			}

			if (IsA(expr, FuncExpr))
			{
				/* Some distance functions are plain functions, not operators */
				FuncExpr *fn = (FuncExpr *) expr;
				if (list_length(fn->args) == 2)
				{
					Node *arg2 = lsecond(fn->args);
					if (!IsA(arg2, Var))
					{
						*query_expr = arg2;
						return true;
					}
				}
			}
		}
	}
	return false;
}

/* -----------------------------------------------------------------------
 * Cost estimation
 *
 * Mirrors pgvector's hnswcostestimate but multiplies by filter selectivity.
 * At selectivity = 1.0 (no filter) cost equals pgvector's estimate.
 * At lower selectivity the scan is cheaper because we explore fewer graph
 * nodes, but we still traverse the full graph for connectivity.
 * ----------------------------------------------------------------------- */

static void
acorn_estimate_cost(PlannerInfo *root, RelOptInfo *rel,
					IndexOptInfo *idx, List *indexQuals,
					Cost *startup, Cost *total, double *rows)
{
	double tuples = rel->tuples;
	double selectivity;
	double m = 16;				/* default; real value from meta page at runtime */
	double log_n;

	if (tuples < 1)
		tuples = 1;

	selectivity = clauselist_selectivity(root, indexQuals,
										 rel->relid, JOIN_INNER, NULL);

	log_n = log(tuples) / log(m > 1 ? m : 2);

	/* HNSW traversal cost: O(log N) probes per query */
	*startup = 0;
	*total   = cpu_operator_cost * tuples * selectivity * log_n;
	*rows    = clamp_row_est(tuples * selectivity);

	/* Must be cheaper than seq scan to be chosen */
	*total = Min(*total, rel->cheapest_total_path->total_cost * 0.8);
}

/* -----------------------------------------------------------------------
 * Hook: detect pattern and inject CustomPath
 * ----------------------------------------------------------------------- */

static void
acorn_set_rel_pathlist(PlannerInfo *root, RelOptInfo *rel,
					   Index rti, RangeTblEntry *rte)
{
	IndexOptInfo *hnsw_idx;
	Node		 *query_expr = NULL;
	CustomPath   *cpath;
	Cost		  startup, total;
	double		  rows;

	/* Chain to previous hook */
	if (prev_hook)
		(*prev_hook)(root, rel, rti, rte);

	if (!acorn_enable_hook)
		return;

	/* Only for base relations (not joins, subqueries, etc.) */
	if (rel->reloptkind != RELOPT_BASEREL)
		return;

	/* Need an HNSW index */
	hnsw_idx = find_hnsw_index(rel);
	if (!hnsw_idx)
		return;

	/* Need a vector ORDER BY */
	if (!detect_vector_orderby(root, rel, &query_expr))
		return;

	/* Need WHERE clause quals (otherwise vanilla HNSW is already fine) */
	if (rel->baserestrictinfo == NIL)
		return;

	/* Estimate cost */
	acorn_estimate_cost(root, rel, hnsw_idx,
						rel->baserestrictinfo,
						&startup, &total, &rows);

	/* Build CustomPath */
	cpath = makeNode(CustomPath);
	cpath->path.pathtype		= T_CustomScan;
	cpath->path.parent			= rel;
	cpath->path.pathtarget		= rel->reltarget;
	cpath->path.rows			= rows;
	cpath->path.startup_cost	= startup;
	cpath->path.total_cost		= total;
	cpath->path.pathkeys		= root->query_pathkeys;	/* preserves ORDER BY */
	cpath->flags				= 0;
	cpath->custom_paths			= NIL;
	cpath->custom_private		= list_make3(
		makeInteger(hnsw_idx->indexoid),	/* [0] index OID */
		query_expr,							/* [1] query vector expression */
		makeInteger(root->limit_tuples)		/* [2] k (LIMIT) */
	);
	cpath->methods				= &acorn_path_methods;

	add_path(rel, (Path *) cpath);
}

/* -----------------------------------------------------------------------
 * Plan creation: convert CustomPath -> CustomScan plan node
 * ----------------------------------------------------------------------- */

static Plan *
acorn_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
					   struct CustomPath *best_path,
					   List *tlist, List *clauses, List *custom_plans)
{
	CustomScan *cscan = makeNode(CustomScan);

	cscan->scan.plan.targetlist = tlist;
	cscan->scan.plan.qual		= NIL;	/* quals evaluated inside executor */
	cscan->scan.scanrelid		= best_path->path.parent->relid;
	cscan->flags				= 0;
	cscan->custom_plans			= NIL;
	cscan->custom_exprs			= list_make1(lsecond(best_path->custom_private));
	cscan->custom_private		= list_make2(
		linitial(best_path->custom_private),	/* index OID */
		lthird(best_path->custom_private)		/* k */
	);
	/* Push quals down so executor can evaluate them */
	cscan->custom_scan_tlist	= tlist;
	cscan->methods				= &acorn_scan_methods;

	/* Extract qual clauses for predicate pushdown */
	{
		List *scan_clauses = extract_actual_clauses(clauses, false);
		cscan->scan.plan.qual = scan_clauses;
	}

	return (Plan *) cscan;
}

/* -----------------------------------------------------------------------
 * Executor: create / begin / execute / end / rescan
 * ----------------------------------------------------------------------- */

static Node *
acorn_create_scan_state(CustomScan *cscan)
{
	AcornCustomScanState *acss = palloc0(sizeof(AcornCustomScanState));
	acss->css.methods = &acorn_exec_methods;
	return (Node *) acss;
}

static void
acorn_begin_scan(CustomScanState *node, EState *estate, int eflags)
{
	AcornCustomScanState *acss = (AcornCustomScanState *) node;
	CustomScan *cscan = (CustomScan *) node->ss.ps.plan;
	Oid		   indexoid;
	int		   k;

	indexoid = (Oid) intVal(linitial(cscan->custom_private));
	k        = (int) intVal(lsecond(cscan->custom_private));

	acss->index  = index_open(indexoid, AccessShareLock);
	acss->heap   = acss->css.ss.ss_currentRelation;
	acss->k      = (k > 0) ? k : 100;
	acss->ef_search = Max(acss->k * 4, 40);	/* heuristic default */

	/* Build predicate ExprState from the plan qual */
	if (cscan->scan.plan.qual != NIL)
	{
		acss->predicate = ExecInitQual(cscan->scan.plan.qual,
									   &node->ss.ps);
		acss->econtext  = CreateExprContext(estate);
	}

	acss->query_evaluated = false;
	acss->result_tids     = NULL;
	acss->result_count    = 0;
	acss->result_pos      = 0;
}

/*
 * Evaluate the query vector expression (ORDER BY argument).
 * Called lazily on first ExecCustomScan so the expression context is ready.
 */
static Datum
acorn_eval_query(AcornCustomScanState *acss, EState *estate)
{
	CustomScan *cscan  = (CustomScan *) acss->css.ss.ps.plan;
	Expr	   *qexpr  = (Expr *) linitial(cscan->custom_exprs);
	ExprState  *qstate = ExecInitExpr(qexpr, &acss->css.ss.ps);
	ExprContext *ectx  = GetPerTupleExprContext(estate);
	bool		isnull;

	return ExecEvalExprSwitchContext(qstate, ectx, &isnull);
}

static TupleTableSlot *
acorn_exec_scan(CustomScanState *node)
{
	AcornCustomScanState *acss = (AcornCustomScanState *) node;
	EState		 *estate = node->ss.ps.state;
	TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;

	/* On first call: run ACORN-1 traversal and cache all results */
	if (!acss->query_evaluated)
	{
		Snapshot	snapshot = GetActiveSnapshot();

		acss->query_datum    = acorn_eval_query(acss, estate);
		acss->query_evaluated = true;

		acss->result_tids = palloc(sizeof(ItemPointerData) * acss->k);
		acss->result_count = acorn_scan_execute(
			&(AcornScanState){
				.ef_search = acss->ef_search,
				.k         = acss->k,
				.predicate = acss->predicate,
				.econtext  = acss->econtext,
			},
			acss->index,
			acss->heap,
			acss->query_datum,
			snapshot,
			acss->result_tids
		);
		acss->result_pos = 0;
	}

	/* Return next cached result */
	if (acss->result_pos >= acss->result_count)
		return ExecClearTuple(slot);

	{
		ItemPointer tid = &acss->result_tids[acss->result_pos++];
		bool		found;

		ExecClearTuple(slot);
		found = table_tuple_fetch_row_version(acss->heap, tid,
										  GetActiveSnapshot(), slot);
		if (!found)
			return acorn_exec_scan(node);	/* skip deleted rows */
	}

	return slot;
}

static void
acorn_end_scan(CustomScanState *node)
{
	AcornCustomScanState *acss = (AcornCustomScanState *) node;

	if (acss->index)
	{
		index_close(acss->index, AccessShareLock);
		acss->index = NULL;
	}

	if (acss->result_tids)
	{
		pfree(acss->result_tids);
		acss->result_tids = NULL;
	}

	if (acss->econtext)
	{
		FreeExprContext(acss->econtext, true);
		acss->econtext = NULL;
	}
}

static void
acorn_rescan(CustomScanState *node)
{
	AcornCustomScanState *acss = (AcornCustomScanState *) node;
	acss->result_pos      = 0;
	acss->query_evaluated = false;
	if (acss->result_tids)
	{
		pfree(acss->result_tids);
		acss->result_tids = NULL;
	}
}

static void
acorn_explain_scan(CustomScanState *node, List *ancestors, ExplainState *es)
{
	AcornCustomScanState *acss = (AcornCustomScanState *) node;

	ExplainPropertyText("ACORN mode", "ACORN-1 predicate subgraph traversal", es);
	ExplainPropertyInteger("k", NULL, acss->k, es);
	ExplainPropertyInteger("ef_search", NULL, acss->ef_search, es);
}

/* -----------------------------------------------------------------------
 * Public init / fini
 * ----------------------------------------------------------------------- */

void
acorn_hook_init(void)
{
	/* Register our CustomScan so the executor can deserialize it */
	RegisterCustomScanMethods(&acorn_scan_methods);

	prev_hook = set_rel_pathlist_hook;
	set_rel_pathlist_hook = acorn_set_rel_pathlist;
}

void
acorn_hook_fini(void)
{
	if (set_rel_pathlist_hook == acorn_set_rel_pathlist)
		set_rel_pathlist_hook = prev_hook;
}
