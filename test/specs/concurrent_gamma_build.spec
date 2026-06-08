# concurrent_gamma_build.spec
# Validates: reads against acorn_hnsw do not error or return garbage while
# a concurrent session is performing incremental inserts (graph is being modified).

setup
{
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS pg_acorn;

    CREATE TABLE build_items (
        id        serial PRIMARY KEY,
        bucket    int,
        embedding vector(4)
    );

    INSERT INTO build_items (bucket, embedding) SELECT
        (i % 5),
        ('[' || (i*0.01)::text || ',0.0,0.0,0.0]')::vector
    FROM generate_series(1, 50) i;

    CREATE INDEX bi_acorn ON build_items
        USING acorn_hnsw (embedding vector_cosine_ops)
        WITH (m = 8, ef_construction = 32, acorn_gamma = 2);
}

teardown
{
    DROP TABLE build_items CASCADE;
}

session "reader"
step "read"  {
    SELECT count(*) FROM (
        SELECT id FROM build_items
        WHERE bucket < 2
        ORDER BY embedding <-> '[0.5,0.0,0.0,0.0]'
        LIMIT 5
    ) sub;
}

session "writer"
step "write" {
    INSERT INTO build_items (bucket, embedding)
    VALUES (1, '[0.51,0.0,0.0,0.0]'),
           (2, '[0.52,0.0,0.0,0.0]'),
           (3, '[0.53,0.0,0.0,0.0]');
}

permutation "read" "write"
permutation "write" "read"
permutation "read" "write" "read"
