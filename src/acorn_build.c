/*
 * acorn_build.c — ACORN-gamma index build + incremental insert (Tier 2)
 *
 * Step 3 placeholder.  Implements:
 *   1. M*gamma neighbor storage at build time (acorn_gamma reloption)
 *   2. Fixed-slot retry: replaces furthest neighbor when slots are full,
 *      fixing pgvector's "TODO Retry updating connections if not" bug.
 *
 * Unit test logic lives in test/unit/test_acorn_build.c and compiles
 * without PostgreSQL headers.  This file wraps that logic for the AM.
 */

#include "postgres.h"

#include "acorn_am.h"

/*
 * Placeholder — Step 3 will implement acorn_build_index() and
 * acorn_insert() using the retry logic unit-tested in test_acorn_build.c.
 */
