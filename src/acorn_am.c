/*
 * acorn_am.c — acorn_hnsw index Access Method handler (Tier 2)
 *
 * Registers the full IndexAmRoutine.  Build/insert live in acorn_build.c;
 * cost estimation in acorn_cost.c.  The scan uses the resumable streaming
 * traversal (acorn_stream_*): a pure index AM does not see the WHERE filter, so
 * it emits heap TIDs in approximate nearest-first order and the executor
 * post-filters on the Index Scan, pulling more as needed.
 */

#include "postgres.h"

#include <float.h>

#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "access/reloptions.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"

#include "pg_acorn.h"
#include "acorn_am.h"
#include "acorn_cost.h"
#include "acorn_scan.h"
#include "acorn_t2_page.h"

PG_FUNCTION_INFO_V1(acorn_hnsw_handler);

static relopt_kind acorn_relopt_kind;

/* -----------------------------------------------------------------------
 * Reloptions
 * ----------------------------------------------------------------------- */

void
acorn_am_init(void)
{
	acorn_relopt_kind = add_reloption_kind();
	add_int_reloption(acorn_relopt_kind, "m",
					  "Max number of connections per node",
					  ACORN_DEFAULT_M, ACORN_MIN_M, ACORN_MAX_M,
					  AccessExclusiveLock);
	add_int_reloption(acorn_relopt_kind, "ef_construction",
					  "Size of the dynamic candidate list for construction",
					  ACORN_DEFAULT_EF_CONSTRUCTION,
					  ACORN_MIN_EF_CONSTRUCTION, ACORN_MAX_EF_CONSTRUCTION,
					  AccessExclusiveLock);
	add_int_reloption(acorn_relopt_kind, "acorn_gamma",
					  "ACORN gamma: store m*gamma neighbors per node",
					  ACORN_DEFAULT_GAMMA, ACORN_MIN_GAMMA, ACORN_MAX_GAMMA,
					  AccessExclusiveLock);
	add_bool_reloption(acorn_relopt_kind, "acorn_payload_edges",
					   "Split layer-0 neighbor slots: half global nearest, "
					   "half nearest within the same payload partition",
					   false,
					   AccessExclusiveLock);
	/*
	 * Default OFF: the 50K 10-seed audit (bench/results_graph_audit.json)
	 * showed the heuristic statistically indistinguishable from nearest-only
	 * selection on the correlated fixture (unf_rr50 0.402 vs 0.408) at
	 * 1.4-2x build cost; the thesis-band raw ceilings proved to be bound by
	 * the t2 stream's emission rule, not neighbor selection.  Kept available
	 * for clustered workloads where it measurably reconnects layer 0
	 * (tier2_diversify.sql island fixture: unfiltered 0.700 -> 1.000).
	 */
	add_bool_reloption(acorn_relopt_kind, "acorn_diversify",
					   "Apply the HNSW diversity heuristic (Malkov Alg. 4 with "
					   "keepPrunedConnections) in neighbor selection",
					   false,
					   AccessExclusiveLock);
}

static bytea *
acorn_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"m", RELOPT_TYPE_INT, offsetof(AcornOptions, m)},
		{"ef_construction", RELOPT_TYPE_INT, offsetof(AcornOptions, efConstruction)},
		{"acorn_gamma", RELOPT_TYPE_INT, offsetof(AcornOptions, gamma)},
		{"acorn_payload_edges", RELOPT_TYPE_BOOL, offsetof(AcornOptions, payloadEdges)},
		{"acorn_diversify", RELOPT_TYPE_BOOL, offsetof(AcornOptions, diversify)},
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  acorn_relopt_kind,
									  sizeof(AcornOptions),
									  tab, lengthof(tab));
}

static bool
acorn_validate(Oid opclassoid)
{
	return true;
}

/* -----------------------------------------------------------------------
 * Scan
 * ----------------------------------------------------------------------- */

/*
 * Tier 2 streams results from a persistent ACORN frontier (acorn_scan.c
 * acorn_stream_*): one heap TID per amgettuple in approximate nearest-first
 * order.  Each graph node is expanded/emitted at most once, so the executor
 * pulling past the initial candidates never re-runs the traversal.
 */
typedef struct AcornScanOpaqueData
{
	bool				first;
	Datum				query;		/* detoasted query vector (lives in tmpCtx) */
	AcornT2StreamScan  *stream;		/* persistent T2 in-filter frontier */
	MemoryContext		tmpCtx;
} AcornScanOpaqueData;
typedef AcornScanOpaqueData *AcornScanOpaque;

static IndexScanDesc
acorn_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc	scan = RelationGetIndexScan(index, nkeys, norderbys);
	AcornScanOpaque so   = palloc0(sizeof(AcornScanOpaqueData));

	so->first  = true;
	so->query  = (Datum) 0;
	so->stream = NULL;
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "acorn scan", ALLOCSET_DEFAULT_SIZES);

	scan->opaque = so;
	return scan;
}

static void
acorn_rescan(IndexScanDesc scan, ScanKey keys, int nkeys,
			 ScanKey orderbys, int norderbys)
{
	AcornScanOpaque so = (AcornScanOpaque) scan->opaque;

	so->first  = true;
	so->query  = (Datum) 0;
	so->stream = NULL;
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys,
				scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * acorn_gettuple — bounded ACORN streaming scan.
 *
 * On the first call, acorn_t2_stream_begin sets up the frontier and the first
 * acorn_t2_stream_next runs a single bounded best-first traversal capped at
 * pg_acorn.ef_search (the GUC acorn_ef_search), materializing the ef_search
 * closest filter-passing nodes.  Subsequent calls emit those heap TIDs one at
 * a time in nearest-first order until the executor's LIMIT is met.
 *
 * ef_search is the recall/latency knob (mirrors hnsw.ef_search): raise it to
 * explore more of the graph at low selectivity, lower it for fewer page reads.
 */
static bool
acorn_gettuple(IndexScanDesc scan, ScanDirection dir)
{
	AcornScanOpaque so = (AcornScanOpaque) scan->opaque;

	Assert(ScanDirectionIsForward(dir));

	if (scan->orderByData == NULL)
		elog(ERROR, "cannot scan acorn_hnsw index without ORDER BY");
	if (!IsMVCCSnapshot(scan->xs_snapshot))
		elog(ERROR, "non-MVCC snapshots are not supported with acorn_hnsw");

	/* NULL order-by value yields no rows */
	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		so->first = false;
		return false;
	}

	/* First call: detoast query and start the streaming frontier */
	if (so->first)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

		so->query = PointerGetDatum(
			PG_DETOAST_DATUM(scan->orderByData->sk_argument));
		so->stream = acorn_t2_stream_begin(scan->indexRelation, so->query,
										   scan->keyData, scan->numberOfKeys,
										   acorn_ef_search,
										   scan->xs_snapshot, so->tmpCtx);
		so->first = false;

		MemoryContextSwitchTo(oldCtx);
	}

	{
		ItemPointerData tid;

		if (!acorn_t2_stream_next(so->stream, &tid))
			return false;

		scan->xs_heaptid        = tid;
		scan->xs_recheck        = false;
		scan->xs_recheckorderby = false;
		return true;
	}
}

static void
acorn_endscan(IndexScanDesc scan)
{
	AcornScanOpaque so = (AcornScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);
	pfree(so);
	scan->opaque = NULL;
}

/* -----------------------------------------------------------------------
 * Vacuum (minimal — no dead-tuple reclamation; sufficient for fresh indexes)
 * ----------------------------------------------------------------------- */

static IndexBulkDeleteResult *
acorn_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				 IndexBulkDeleteCallback callback, void *callback_state)
{
	BlockNumber nblocks;
	BlockNumber blkno;

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

	nblocks = RelationGetNumberOfBlocks(info->index);

	/* Block 0 is the metapage — skip it */
	for (blkno = 1; blkno < nblocks; blkno++)
	{
		Buffer			  buf;
		Page			  page;
		OffsetNumber	  off, maxoff;
		bool			  page_modified = false;
		bool			  needs_wal = RelationNeedsWAL(info->index);
		GenericXLogState *state = NULL;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBufferExtended(info->index, MAIN_FORKNUM, blkno,
								 RBM_NORMAL, info->strategy);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (needs_wal)
		{
			state = GenericXLogStart(info->index);
			page  = GenericXLogRegisterBuffer(state, buf, GENERIC_XLOG_FULL_IMAGE);
		}
		else
			page = BufferGetPage(buf);

		if (!HnswPageIsValid(page))
		{
			if (state)
				GenericXLogAbort(state);
			UnlockReleaseBuffer(buf);
			continue;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId				iid = PageGetItemId(page, off);
			AcornT2ElementTuple etup;
			int					i;
			bool				any_live;

			if (!ItemIdIsNormal(iid))
				continue;

			etup = (AcornT2ElementTuple) PageGetItem(page, iid);

			if (etup->type != HNSW_ELEMENT_TUPLE_TYPE)
				continue;

			if (etup->deleted)
				continue;

			/* Element is dead when every valid heaptid passes the callback */
			any_live = false;
			for (i = 0; i < HNSW_HEAPTIDS; i++)
			{
				if (!ItemPointerIsValid(&etup->heaptids[i]))
					break;
				if (!callback(&etup->heaptids[i], callback_state))
				{
					any_live = true;
					break;
				}
			}

			if (!any_live)
			{
				etup->deleted  = 1;
				page_modified  = true;
				stats->tuples_removed++;
			}
		}

		if (page_modified)
		{
			if (state)
				GenericXLogFinish(state);
			else
				MarkBufferDirty(buf);
		}
		else if (state)
			GenericXLogAbort(state);

		UnlockReleaseBuffer(buf);
	}

	return stats;
}

static IndexBulkDeleteResult *
acorn_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	if (info->analyze_only)
		return stats;
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	stats->num_pages = RelationGetNumberOfBlocks(info->index);
	return stats;
}

/* -----------------------------------------------------------------------
 * Handler
 * ----------------------------------------------------------------------- */

Datum
acorn_hnsw_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 3;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = true;	/* (vector ORDER BY, scalar filter) */
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = false;
#endif
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	/* Interface functions */
	amroutine->ambuild = acorn_build;
	amroutine->ambuildempty = acorn_buildempty;
	amroutine->aminsert = acorn_insert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = acorn_bulkdelete;
	amroutine->amvacuumcleanup = acorn_vacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = acorn_hnsw_costestimate;
	amroutine->amoptions = acorn_options;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = acorn_validate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = acorn_beginscan;
	amroutine->amrescan = acorn_rescan;
	amroutine->amgettuple = acorn_gettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = acorn_endscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;

	PG_RETURN_POINTER(amroutine);
}
