-- tier2_build_two_pass.sql: B3 two-pass payload-edge build.
--
-- GUC pg_acorn.build_payload_two_pass: build the base HNSW graph first, then add
-- payload (partition) edges in a SECOND pass over the finished graph.  This must
-- be recall-EQUIVALENT to the legacy interleaved build on identical data.
--
-- SERIAL builds (max_parallel_maintenance_workers = 0) + build_seed = 42 make the
-- base graph deterministic, isolating the two-pass change.  Two-pass selects
-- payload neighbors from the COMPLETE partition (vs the partially-built partition
-- the interleaved path sees mid-insert), so recall is expected equal-or-better; a
-- small tolerance absorbs graceful-degradation backfill differences.
--
-- Fixture: 2000 uniform random vectors, bucket = i % 50 (bucket = 0 -> 2% sparse
-- partition; bucket < 5 -> 10%).  Identical data in items_off / items_on isolates
-- the GUC effect.  acorn_diversify pinned off (mirrors tier2_payload_edges).

\set ON_ERROR_STOP on
\set q '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'

CREATE SCHEMA test_t2_2pass;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_acorn;
SET search_path = test_t2_2pass, public;

SELECT setseed(0.42);
CREATE TABLE items_off (id serial PRIMARY KEY, bucket int, embedding vector(8));
INSERT INTO items_off (bucket, embedding)
SELECT (i % 50),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(1, 2000) i;
CREATE TABLE items_on AS SELECT * FROM items_off;

SHOW pg_acorn.build_payload_two_pass;	-- default off

SET max_parallel_maintenance_workers = 0;
SET pg_acorn.build_seed = 42;

-- interleaved (legacy) build
SET pg_acorn.build_payload_two_pass = off;
CREATE INDEX items_off_idx ON items_off
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_payload_m = 64, acorn_diversify = false);

-- two-pass build (same data, same seed, serial)
SET pg_acorn.build_payload_two_pass = on;
CREATE INDEX items_on_idx ON items_on
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_payload_m = 64, acorn_diversify = false);
RESET pg_acorn.build_payload_two_pass;
RESET pg_acorn.build_seed;
RESET max_parallel_maintenance_workers;

-- filter correctness + returns-k on the two-pass index
SET enable_seqscan = off;
SET pg_acorn.ef_search = 100;
SELECT bool_and(bucket = 0) AS tp_eq_filter_correct, count(*) = 10 AS tp_eq_full_k
FROM (SELECT bucket FROM items_on WHERE bucket = 0
      ORDER BY embedding <=> :'q'::vector LIMIT 10) r;
SELECT bool_and(bucket < 5) AS tp_range_filter_correct, count(*) = 10 AS tp_range_full_k
FROM (SELECT bucket FROM items_on WHERE bucket < 5
      ORDER BY embedding <=> :'q'::vector LIMIT 10) r;
RESET enable_seqscan;

-- exact ground truth via seqscan
SET enable_indexscan = off;
SET enable_bitmapscan = off;
CREATE TABLE truth_eq AS SELECT id FROM items_off WHERE bucket = 0
    ORDER BY embedding <=> :'q'::vector LIMIT 10;
CREATE TABLE truth_range AS SELECT id FROM items_off WHERE bucket < 5
    ORDER BY embedding <=> :'q'::vector LIMIT 10;
RESET enable_indexscan;
RESET enable_bitmapscan;

CREATE OR REPLACE FUNCTION acorn_recall(tbl text, qual text, truth_tbl text, ef int)
RETURNS numeric AS $$
DECLARE
    matched int;
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

-- recall equivalence: two-pass >= interleaved - 0.02 (expected equal/better)
SELECT acorn_recall('items_on', 'bucket = 0', 'truth_eq', 100)
     >= acorn_recall('items_off', 'bucket = 0', 'truth_eq', 100) - 0.02 AS tp_eq_ge_ef100;
SELECT acorn_recall('items_on', 'bucket = 0', 'truth_eq', 400)
     >= acorn_recall('items_off', 'bucket = 0', 'truth_eq', 400) - 0.02 AS tp_eq_ge_ef400;
SELECT acorn_recall('items_on', 'bucket < 5', 'truth_range', 100)
     >= acorn_recall('items_off', 'bucket < 5', 'truth_range', 100) - 0.02 AS tp_range_ge_ef100;
SELECT acorn_recall('items_on', 'bucket < 5', 'truth_range', 400)
     >= acorn_recall('items_off', 'bucket < 5', 'truth_range', 400) - 0.02 AS tp_range_ge_ef400;
RESET pg_acorn.ef_search;

DROP SCHEMA test_t2_2pass CASCADE;
