#ifndef ACORN_AM_H
#define ACORN_AM_H

/*
 * Tier 2: acorn_hnsw index Access Method
 *
 * Full IndexAmRoutine registration.  acorn_hnsw writes index pages in
 * pgvector 0.8.0's on-disk format (see hnsw_compat.h) so the shared traversal
 * in acorn_scan.c reads them unchanged.  ACORN-gamma is implemented by storing
 * m_eff = m * gamma neighbors per node and recording m_eff as meta->m.
 *
 * The handler function (acorn_hnsw_handler) is declared via PG_FUNCTION_INFO_V1
 * in acorn_am.c and registered through SQL.
 */

#include "postgres.h"
#include "access/genam.h"
#include "access/relscan.h"
#include "fmgr.h"
#include "nodes/execnodes.h"
#include "utils/relcache.h"

/* reloption defaults / bounds */
#define ACORN_DEFAULT_M              16
#define ACORN_MIN_M                  2
#define ACORN_MAX_M                  100
#define ACORN_DEFAULT_EF_CONSTRUCTION 64
#define ACORN_MIN_EF_CONSTRUCTION    4
#define ACORN_MAX_EF_CONSTRUCTION    1000
#define ACORN_DEFAULT_GAMMA           1
#define ACORN_MIN_GAMMA               1
#define ACORN_MAX_GAMMA               8

/* ef_search heuristic for index-AM scans */
#define ACORN_DEFAULT_EF_SEARCH      40

/*
 * Parsed reloptions for an acorn_hnsw index.  Layout must start with int32
 * vl_len_ for build_reloptions().
 */
typedef struct AcornOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly) */
	int			m;				/* base connections per node */
	int			efConstruction;	/* candidate list size at build time */
	int			gamma;			/* ACORN-gamma multiplier (m_eff = m*gamma) */
	bool		payloadEdges;	/* split layer-0 slots: half global / half same-partition */
	bool		diversify;		/* HNSW diversity heuristic in neighbor selection */
} AcornOptions;

/* -----------------------------------------------------------------------
 * Build / insert (acorn_build.c)
 * ----------------------------------------------------------------------- */

IndexBuildResult *acorn_build(Relation heap, Relation index,
							  IndexInfo *indexInfo);
void acorn_buildempty(Relation index);
bool acorn_insert(Relation index, Datum *values, bool *isnull,
				  ItemPointer heap_tid, Relation heap,
				  IndexUniqueCheck checkUnique,
				  bool indexUnchanged, IndexInfo *indexInfo);

/* Shared helpers exposed for acorn_am.c / acorn_build.c */
int  acorn_index_m(Relation index);		/* meta->m (= m_eff), 0 if empty */

/* Register reloptions; called from _PG_init (acorn_am.c) */
void acorn_am_init(void);

#endif /* ACORN_AM_H */
