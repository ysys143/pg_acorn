-- tier2_payload_m.sql: additive acorn_payload_m reloption.
--
-- acorn_payload_m (int, default 0) sizes the layer-0 PAYLOAD half independently
-- of the global half (Qdrant payload_m style).  global_m = m*gamma stays the
-- global/upper-layer width; layer-0 total = global_m + payload_m.  Sentinel
-- 0 = symmetric (payload half = global half = m*gamma = the legacy 2*m_eff
-- layout), so existing indexes are unaffected.
--
-- Fixture: 2000 distinct random vectors (tie-free so the identity assertion is
-- exact), bucket = i % 50 -> bucket=0 is a 2% sparse predicate subgraph where
-- extra payload connectivity matters.  Identical data across the indexes;
-- pg_acorn.build_seed fixes the graph so symmetric builds are reproducible.

\set ON_ERROR_STOP on
\set q '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'

CREATE SCHEMA test_t2_pm;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_acorn;
SET search_path = test_t2_pm, public;

SELECT setseed(0.42);
CREATE TABLE items (
    id        serial PRIMARY KEY,
    bucket    int,
    embedding vector(8)
);
INSERT INTO items (bucket, embedding)
SELECT (i % 50),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(1, 2000) i;

CREATE TABLE items_sym  AS SELECT * FROM items;   -- default (no acorn_payload_m)
CREATE TABLE items_sym0 AS SELECT * FROM items;   -- explicit acorn_payload_m=0
CREATE TABLE items_big  AS SELECT * FROM items;   -- larger payload half

-- Deterministic, identical graphs for the symmetric pair.
SET pg_acorn.build_seed = 7;
SET max_parallel_maintenance_workers = 0;
CREATE INDEX items_sym_idx ON items_sym
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_diversify = false);
CREATE INDEX items_sym0_idx ON items_sym0
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_diversify = false,
          acorn_payload_m = 0);
CREATE INDEX items_big_idx ON items_big
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_diversify = false,
          acorn_payload_m = 64);
RESET pg_acorn.build_seed;

-- (a) reloption is parsed and stored.
SELECT reloptions FROM pg_class WHERE relname = 'items_big_idx';

-- (b) NON-REGRESSION / reduction guard: acorn_payload_m=0 is byte-equivalent to
-- the default (no option) — identical ordered id lists at every ef.  Proves the
-- reserved==0 sentinel + the generalized formulas collapse to the legacy layout.
SET enable_seqscan = off;
SET pg_acorn.ef_search = 100;
SELECT (SELECT array_agg(id ORDER BY ord) FROM (
            SELECT id, row_number() OVER (ORDER BY embedding <=> :'q'::vector) ord
            FROM items_sym WHERE bucket < 3
            ORDER BY embedding <=> :'q'::vector LIMIT 10) a)
     = (SELECT array_agg(id ORDER BY ord) FROM (
            SELECT id, row_number() OVER (ORDER BY embedding <=> :'q'::vector) ord
            FROM items_sym0 WHERE bucket < 3
            ORDER BY embedding <=> :'q'::vector LIMIT 10) b)
    AS payload_m0_identical_to_default;
RESET enable_seqscan;

-- Exact ground truth (seqscan) for the sparse equality and range quals.
SET enable_indexscan = off;
SET enable_bitmapscan = off;
CREATE TABLE truth_eq AS
    SELECT id FROM items WHERE bucket = 0
    ORDER BY embedding <=> :'q'::vector LIMIT 10;
CREATE TABLE truth_range AS
    SELECT id FROM items WHERE bucket < 3
    ORDER BY embedding <=> :'q'::vector LIMIT 10;
RESET enable_indexscan;
RESET enable_bitmapscan;

CREATE OR REPLACE FUNCTION pm_recall(tbl text, qual text, truth_tbl text, ef int)
RETURNS numeric AS $$
DECLARE matched int;
BEGIN
    EXECUTE format('SET pg_acorn.ef_search = %s', ef);
    SET enable_seqscan = off;
    EXECUTE format(
        'WITH r AS (SELECT id FROM %I WHERE %s
                    ORDER BY embedding <=> ''[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]''::vector
                    LIMIT 10)
         SELECT count(*) FROM r JOIN %I t USING (id)', tbl, qual, truth_tbl)
        INTO matched;
    RESET enable_seqscan;
    RETURN matched / 10.0;
END;
$$ LANGUAGE plpgsql;

-- (c) more payload edges never lose recall vs symmetric at equal ef, and the
-- sparse-subgraph recall holds an absolute floor.
SELECT pm_recall('items_big', 'bucket = 0', 'truth_eq', 100)
    >= pm_recall('items_sym', 'bucket = 0', 'truth_eq', 100) AS eq_big_ge_sym_ef100;
SELECT pm_recall('items_big', 'bucket < 3', 'truth_range', 100)
    >= pm_recall('items_sym', 'bucket < 3', 'truth_range', 100) AS range_big_ge_sym_ef100;
SELECT pm_recall('items_big', 'bucket = 0', 'truth_eq', 100) >= 0.5 AS big_eq_floor;

-- (d) filter correctness with a larger payload half.
SET enable_seqscan = off;
SET pg_acorn.ef_search = 100;
SELECT bool_and(bucket = 0) AS big_eq_filter_correct, count(*) = 10 AS big_eq_full_k
FROM (SELECT bucket FROM items_big WHERE bucket = 0
      ORDER BY embedding <=> :'q'::vector LIMIT 10) r;
RESET enable_seqscan;

-- (e) insert path: aminsert reads payload_m from the meta page and keeps the
-- layer-0 boundary consistent.
INSERT INTO items_big (bucket, embedding)
SELECT (i % 50),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(1, 100) i;
SET enable_seqscan = off;
SET pg_acorn.ef_search = 100;
SELECT bool_and(bucket = 0) AS insert_filter_correct, count(*) = 10 AS insert_full_k
FROM (SELECT bucket FROM items_big WHERE bucket = 0
      ORDER BY embedding <=> :'q'::vector LIMIT 10) r;
RESET enable_seqscan;
RESET pg_acorn.ef_search;

-- (f) reloption bounds: a value beyond the registered max is rejected at parse.
\set ON_ERROR_STOP off
CREATE INDEX items_bad_idx ON items_sym
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_payload_m = 999);
\set ON_ERROR_STOP on

DROP SCHEMA test_t2_pm CASCADE;
