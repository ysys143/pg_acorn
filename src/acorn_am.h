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
/*
 * acorn_payload_m: independent absolute size of the layer-0 payload half
 * (Qdrant-style).  0 = sentinel "symmetric": payload half = global half =
 * m_eff, i.e. the legacy 2*m_eff layer-0 layout.  Max is HNSW_MAX_NEIGHBORS
 * (=200) so global_m + payload_m can still be clamped to fit a page.
 */
#define ACORN_DEFAULT_PAYLOAD_M       0
#define ACORN_MIN_PAYLOAD_M           0
#define ACORN_MAX_PAYLOAD_M           200

/* ef_search heuristic for index-AM scans */
#define ACORN_DEFAULT_EF_SEARCH      40

/*
 * Auto-ef (pg_acorn.target_recall): when target_recall > 0, derive ef from the
 * estimated filter selectivity instead of using the manual ef_search.  Coarse,
 * MONOTONE heuristic — a convenience that removes per-selectivity ef tuning,
 * NOT a recall guarantee.  Anchored to the reference fixture at top-~10 KNN:
 *
 *   ef = clamp( max(EF_MIN, EF_PER_SEL * sel) * recall_factor, EF_MIN, EF_MAX )
 *   recall_factor = clamp( sqrt((1-ANCHOR) / (1-target_recall)), RF_MIN, RF_MAX )
 *
 * At target_recall == ANCHOR (0.95) recall_factor == 1, matching the gamma
 * sweep money cells (sel 1%/10%/20% -> ef ~100/400/800).
 */
#define ACORN_DEFAULT_TARGET_RECALL  0.0	/* 0 = off (manual ef_search) */
#define ACORN_AUTOEF_EF_MIN          100
#define ACORN_AUTOEF_EF_PER_SEL      4000
#define ACORN_AUTOEF_EF_MAX          4000
#define ACORN_AUTOEF_RECALL_ANCHOR   0.95
#define ACORN_AUTOEF_RF_MIN          0.5
#define ACORN_AUTOEF_RF_MAX          4.0

/*
 * acorn_payload_min_cardinality (Priority 4, Qdrant full_scan_threshold analog):
 * at build time, a node whose filter-value partition has fewer than this many
 * members gets NO payload edges -- those slots are filled with global links
 * instead (the existing graceful-degradation fill).  Such tiny filtered sets are
 * served by the exact prefilter anyway (see acorn_cost.c), so payload
 * connectivity for them is wasted.  0 = off (current behaviour: payload edges
 * for every partition, with graceful degradation only when a half cannot fill).
 * int4 filter columns only (uses the build histogram's partition counts).
 */
#define ACORN_DEFAULT_PAYLOAD_MIN_CARD  0
#define ACORN_MIN_PAYLOAD_MIN_CARD      0
#define ACORN_MAX_PAYLOAD_MIN_CARD      1000000

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
	int			payloadM;		/* layer-0 payload-half width (0 = symmetric = global_m) */
	bool		payloadEdges;	/* split layer-0 slots: half global / half same-partition */
	bool		diversify;		/* HNSW diversity heuristic in neighbor selection */
	bool		inlineVectors;	/* co-locate SQ8 vectors + metadata in neighbor lists */
	int			payloadMinCard;	/* P4: skip payload edges for partitions smaller than this (0 = off) */
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
