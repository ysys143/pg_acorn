-- tier2_code_cache.sql: shared-memory SQ8 code cache (M1 read path).
--
-- A NON-inline acorn_hnsw index keeps the TID-only layout on disk; with
-- pg_acorn.scan_code_cache=on the first scan loads every element's SQ8 code
-- + metadata into a per-index shared-memory table, and neighbor discovery
-- reads the cache instead of pinning element pages.  Emitted results are
-- re-ranked with exact distances, so at an expansion budget that exhausts
-- the frontier the result list is byte-identical to the cache-off scan.
--
-- Tests: GUC defaults (off / 512MB); identical ordered id lists for
-- cache=off vs cache=on cold (load happens in this scan) vs cache=on warm
-- (READY slot, pure hits); the same list matches the exact seqscan truth;
-- filter correctness + k results at a small ef; code_cache_size=0 disables
-- the cache (identical to off).
--
-- Determinism: setseed for data, pg_acorn.build_seed + serial build for the
-- graph, distinct random vectors so there are no distance ties; assertions
-- output booleans only.

\set ON_ERROR_STOP on
\set q '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'

CREATE SCHEMA test_tier2_cc;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_acorn;
SET search_path = test_tier2_cc, public;

-- Defaults: cache off, 512MB budget.
SHOW pg_acorn.scan_code_cache;
SHOW pg_acorn.code_cache_size;

-- 600 rows, 10 buckets -> bucket < 3 = ~30% selectivity (180 rows).
SELECT setseed(0.42);
CREATE TABLE items (
    id        serial PRIMARY KEY,
    bucket    int,
    embedding vector(8)
);

INSERT INTO items (bucket, embedding) SELECT
    (i % 10),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(1, 600) i;

-- Deterministic NON-inline index (the cache only serves non-inline layouts).
SET pg_acorn.build_seed = 42;
SET max_parallel_maintenance_workers = 0;
CREATE INDEX items_cc_idx ON items
    USING acorn_hnsw (embedding vector_l2_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2);
RESET pg_acorn.build_seed;

-- Ordered top-10 id list through the index scan at the session's ef_search.
CREATE FUNCTION topk_ids() RETURNS int[] AS $$
DECLARE
    r int[];
BEGIN
    SET enable_seqscan = off;
    SELECT array_agg(id) INTO r FROM (
        SELECT id FROM items WHERE bucket < 3
        ORDER BY embedding <-> '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'::vector
        LIMIT 10) t;
    RESET enable_seqscan;
    RETURN r;
END;
$$ LANGUAGE plpgsql;

-- ef_search 4000 > 600 nodes: the frontier empties before the budget, so
-- both modes expand the whole graph and the exact re-rank fixes the order —
-- identity (not just recall) is guaranteed for a tie-free fixture.
SET pg_acorn.ef_search = 4000;

SET pg_acorn.scan_code_cache = off;
CREATE TABLE run_off AS SELECT topk_ids() AS ids;

-- Cold: this scan finds the slot EMPTY, loads it, then serves from READY.
SET pg_acorn.scan_code_cache = on;
CREATE TABLE run_cold AS SELECT topk_ids() AS ids;

-- Warm: slot is READY, pure dshash hits.
CREATE TABLE run_warm AS SELECT topk_ids() AS ids;

SELECT (SELECT ids FROM run_off) = (SELECT ids FROM run_cold) AS cold_identical,
       (SELECT ids FROM run_off) = (SELECT ids FROM run_warm) AS warm_identical;

-- The exhaustive-ef list must equal the exact seqscan truth.
SET pg_acorn.scan_code_cache = off;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
SELECT (SELECT ids FROM run_off) = (
    SELECT array_agg(id) FROM (
        SELECT id FROM items WHERE bucket < 3
        ORDER BY embedding <-> :'q'::vector LIMIT 10) t
) AS matches_exact_truth;
RESET enable_indexscan;
RESET enable_bitmapscan;

-- Small-ef sanity with the cache on: the predicate must hold on every row
-- and the scan must still produce k rows (cache hits carry filter_val).
SET pg_acorn.scan_code_cache = on;
SET pg_acorn.ef_search = 200;
SET enable_seqscan = off;
SELECT bool_and(bucket < 3) AS filter_correct,
       count(*) = 10        AS returns_k
FROM (SELECT bucket FROM items WHERE bucket < 3
      ORDER BY embedding <-> :'q'::vector LIMIT 10) r;
RESET enable_seqscan;

-- code_cache_size = 0 disables the cache even with scan_code_cache = on.
SET pg_acorn.code_cache_size = 0;
SET pg_acorn.ef_search = 4000;
CREATE TABLE run_disabled AS SELECT topk_ids() AS ids;
SELECT (SELECT ids FROM run_off) = (SELECT ids FROM run_disabled)
    AS size0_disables_identical;
RESET pg_acorn.code_cache_size;
RESET pg_acorn.scan_code_cache;
RESET pg_acorn.ef_search;

DROP SCHEMA test_tier2_cc CASCADE;
