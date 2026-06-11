-- One-off: bounded RSS point at maintenance_work_mem = 32MB on the existing
-- bo_items 60K fixture (run in a FRESH session so VmHWM isolates this build).
\set QUIET on
SELECT pg_backend_pid() AS pid \gset
SELECT (regexp_match(pg_read_file(format('/proc/%s/status', :'pid')),
        'VmHWM:\s+(\d+) kB'))[1] AS vmhwm_pre_kb \gset
\echo pre_kb :vmhwm_pre_kb
SET pg_acorn.enable_hook = off;
SET pg_acorn.build_direct_dist = on;
SET pg_acorn.build_seed = 42;
SET maintenance_work_mem = '32MB';
SET max_parallel_maintenance_workers = 0;
ALTER TABLE bo_items SET (parallel_workers = 0);
\timing on
CREATE INDEX bo_rss32_idx ON bo_items
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_inline_vectors = true);
\timing off
CHECKPOINT;
SELECT (regexp_match(pg_read_file(format('/proc/%s/status', :'pid')),
        'VmHWM:\s+(\d+) kB'))[1] AS vmhwm_post_kb \gset
\echo post_kb :vmhwm_post_kb
DROP INDEX bo_rss32_idx;
