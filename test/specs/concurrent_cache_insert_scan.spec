# concurrent_cache_insert_scan.spec
# Validates the M2 write path under concurrency: with pg_acorn.scan_code_cache
# ON, a scanner reads the shared-memory cache (lock-free, per-entry seqlock)
# while an inserter upserts new entries into the same slot.  The scan must
# complete without error or wrong results regardless of interleaving — the
# cache is a hint, any torn read or growth-in-progress falls back to the
# element-page read (G4).

setup
{
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_acorn;

    CREATE TABLE cc_items (
        id        serial PRIMARY KEY,
        bucket    int,
        embedding vector(4)
    );

    INSERT INTO cc_items (bucket, embedding) SELECT
        (i % 10),
        ('[' || (i*0.1)::text || ',0.0,0.0,0.0]')::vector
    FROM generate_series(1, 200) i;

    -- NON-inline index: the cache only serves this layout.
    SET pg_acorn.build_seed = 7;
    CREATE INDEX cc_acorn ON cc_items
        USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
        WITH (m = 8, ef_construction = 32, acorn_gamma = 2);
    RESET pg_acorn.build_seed;

    -- Warm the cache to READY so the scanner reads cached entries and the
    -- inserter upserts into a live slot (exercising the seqlock + growth).
    SET pg_acorn.scan_code_cache = on;
    SET enable_seqscan = off;
    SELECT count(*) FROM (
        SELECT id FROM cc_items WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]' LIMIT 10) s;
    RESET enable_seqscan;
    RESET pg_acorn.scan_code_cache;
}

teardown
{
    DROP TABLE cc_items CASCADE;
}

session "scanner"
setup           { SET pg_acorn.scan_code_cache = on; SET enable_seqscan = off; }
step "begin_scan"  { BEGIN; }
step "run_scan"    {
    SELECT count(*) FROM (
        SELECT id FROM cc_items
        WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]'
        LIMIT 10
    ) sub;
}
step "end_scan"    { COMMIT; }

session "inserter"
step "insert_rows" {
    INSERT INTO cc_items (bucket, embedding)
    SELECT 1, ('[' || (0.9 + i*0.001)::text || ',0.0,0.0,0.0]')::vector
    FROM generate_series(1, 40) i;
}

# Scans complete without error and return k under every interleaving with the
# concurrent cache-upserting insert.
permutation "begin_scan" "insert_rows" "run_scan" "end_scan"
permutation "insert_rows" "begin_scan" "run_scan" "end_scan"
permutation "begin_scan" "run_scan" "insert_rows" "end_scan"
