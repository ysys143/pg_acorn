-- tier2_payload_gate.sql: P4 payload gating by partition cardinality
-- (acorn_payload_min_cardinality).
--
-- At build time a node whose filter-value partition has fewer than
-- acorn_payload_min_cardinality members gets NO payload (same-partition) edges;
-- those layer-0 slots are filled with global links instead (the existing
-- graceful-degradation fill).  Slot COUNT is unchanged, so the on-disk format
-- and the scan are untouched -- this is purely a build-time neighbor-selection
-- change, applied in the in-memory build path (the common case).
--
-- High-cardinality filter (bucket = i % 64) so every partition has exactly 16
-- members and payload_m = 16 fills entirely with same-partition edges when
-- ungated.  Two tables hold the SAME data (same setseed) and SAME graph seed;
-- one index gates (min_card = 20 > 16), one does not (min_card = 0).  Asserts:
-- gating preserves exhaustive-ef correctness (== exact truth), AND gating
-- actually changes the graph -- a single-partition filtered query (maximally
-- sensitive to payload edges) returns a different top-10 from the ungated build.
-- Booleans only.

\set ON_ERROR_STOP on
\set q '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'
\set p '[0.9,0.1,0.2,0.8,0.3,0.7,0.4,0.6]'

CREATE SCHEMA test_tier2_gate;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_acorn;
SET search_path = test_tier2_gate, public;

-- 1024 rows, bucket = i % 64: every partition (bucket & 255 = bucket) has 16 members.
CREATE TABLE items_g0   (id serial PRIMARY KEY, bucket int, embedding vector(8));
CREATE TABLE items_gate (id serial PRIMARY KEY, bucket int, embedding vector(8));

SELECT setseed(0.42);
INSERT INTO items_g0 (bucket, embedding) SELECT (i % 64),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(0, 1023) i;

SELECT setseed(0.42);	-- identical data in the second table
INSERT INTO items_gate (bucket, embedding) SELECT (i % 64),
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(0, 1023) i;

SET pg_acorn.build_seed = 42;
SET max_parallel_maintenance_workers = 0;
CREATE INDEX g0_idx ON items_g0
    USING acorn_hnsw (embedding vector_l2_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_payload_edges = true,
          acorn_payload_m = 16, acorn_payload_min_cardinality = 0);
CREATE INDEX gate_idx ON items_gate
    USING acorn_hnsw (embedding vector_l2_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_payload_edges = true,
          acorn_payload_m = 16, acorn_payload_min_cardinality = 20);
RESET pg_acorn.build_seed;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;

-- Exact truth for a filtered query (bucket < 8 = 8 partitions, 128 rows).
-- Force a real Seq Scan: enable_seqscan is OFF globally (to force the index for
-- the measured queries), and enable_indexscan=off alone is only a cost penalty,
-- so seqscan must be turned ON here or the planner would answer "truth" with the
-- acorn index at the default ef (approximate, not exact).
SET enable_seqscan = on;
SET enable_indexscan = off;
SET enable_bitmapscan = off;
CREATE TABLE truth AS SELECT array_agg(id) AS ids FROM (
    SELECT id FROM items_g0 WHERE bucket < 8
    ORDER BY embedding <-> :'q'::vector LIMIT 10) t;
SET enable_seqscan = off;
RESET enable_indexscan;
RESET enable_bitmapscan;

-- Correctness: at an exhaustive ef both the gated and ungated indexes return the
-- exact truth (gating never breaks recall; freed slots keep the graph connected
-- through global links).
SET pg_acorn.ef_search = 4000;
CREATE TABLE r_g0_exh AS SELECT array_agg(id) AS ids FROM (
    SELECT id FROM items_g0 WHERE bucket < 8
    ORDER BY embedding <-> :'q'::vector LIMIT 10) t;
CREATE TABLE r_gate_exh AS SELECT array_agg(id) AS ids FROM (
    SELECT id FROM items_gate WHERE bucket < 8
    ORDER BY embedding <-> :'q'::vector LIMIT 10) t;

SELECT (SELECT ids FROM r_g0_exh)   = (SELECT ids FROM truth) AS g0_exact,
       (SELECT ids FROM r_gate_exh) = (SELECT ids FROM truth) AS gate_exact;

-- Filter correctness + k rows under gating.
SELECT bool_and(bucket < 8) AS filter_correct, count(*) = 10 AS returns_k
FROM (SELECT bucket FROM items_gate WHERE bucket < 8
      ORDER BY embedding <-> :'q'::vector LIMIT 10) r;

-- Gating took effect: a single-partition filtered query (bucket = 0, the case
-- most sensitive to same-partition payload edges) returns a DIFFERENT top-10
-- from the ungated build at a mid ef -- proving gating replaced the partition's
-- payload edges with global links.
SET pg_acorn.ef_search = 16;
SELECT (SELECT array_agg(id) FROM (SELECT id FROM items_g0 WHERE bucket = 0
            ORDER BY embedding <-> :'p'::vector LIMIT 10) a)
       IS DISTINCT FROM
       (SELECT array_agg(id) FROM (SELECT id FROM items_gate WHERE bucket = 0
            ORDER BY embedding <-> :'p'::vector LIMIT 10) b)
       AS gating_changed_graph;

RESET pg_acorn.ef_search;
RESET enable_seqscan;

DROP SCHEMA test_tier2_gate CASCADE;
