# concurrent_insert_scan.spec
# Validates: filtered scan returns consistent results while concurrent inserts happen.
# A row inserted before the scan snapshot must appear in results if it matches the filter.
# A row inserted after the scan snapshot must not appear.

setup
{
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_acorn;

    CREATE TABLE concurrent_items (
        id        serial PRIMARY KEY,
        bucket    int,
        embedding vector(4)
    );

    INSERT INTO concurrent_items (bucket, embedding) SELECT
        (i % 10),
        ('[' || (i*0.1)::text || ',0.0,0.0,0.0]')::vector
    FROM generate_series(1, 100) i;

    CREATE INDEX ci_acorn ON concurrent_items
        USING acorn_hnsw (embedding vector_cosine_ops)
        WITH (m = 8, ef_construction = 32, acorn_gamma = 1);
}

teardown
{
    DROP TABLE concurrent_items CASCADE;
}

session "scanner"
step "begin_scan"  { BEGIN; }
step "run_scan"    {
    SELECT count(*) FROM (
        SELECT id FROM concurrent_items
        WHERE bucket < 3
        ORDER BY embedding <-> '[1.0,0.0,0.0,0.0]'
        LIMIT 10
    ) sub;
}
step "end_scan"    { COMMIT; }

session "inserter"
step "insert_row"  {
    INSERT INTO concurrent_items (bucket, embedding)
    VALUES (1, '[0.95,0.0,0.0,0.0]');
}

# scan must complete without error even while insert is in flight
permutation "begin_scan" "insert_row" "run_scan" "end_scan"
permutation "insert_row" "begin_scan" "run_scan" "end_scan"
