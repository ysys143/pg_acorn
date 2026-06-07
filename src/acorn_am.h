#ifndef ACORN_AM_H
#define ACORN_AM_H

/*
 * Tier 2: acorn_hnsw index Access Method
 *
 * Full IndexAmRoutine registration.  Build stores M*gamma neighbors per node
 * (controlled by the acorn_gamma reloption).  Scan delegates to acorn_scan.c.
 *
 * The handler function (acorn_hnsw_handler) is declared via PG_FUNCTION_INFO_V1
 * in acorn_am.c and registered through SQL only — no C-level declaration needed.
 */

/* reloption defaults */
#define ACORN_DEFAULT_M              16
#define ACORN_DEFAULT_EF_CONSTRUCTION 64
#define ACORN_DEFAULT_GAMMA           1

#endif /* ACORN_AM_H */
