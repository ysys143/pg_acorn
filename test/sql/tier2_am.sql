-- tier2_am.sql: acorn_hnsw index AM — creation, scan, planner selection
-- TDD: golden file starts empty; populate after implementation passes

\set ON_ERROR_STOP on

CREATE SCHEMA test_tier2;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_acorn;
SET search_path = test_tier2, public;

CREATE TABLE items (
    id        serial PRIMARY KEY,
    category  text,
    embedding vector(4)
);

INSERT INTO items (category, embedding) SELECT
    CASE WHEN i % 5 = 0 THEN 'shoes' ELSE 'other' END,
    ('[' || (random())::text || ',' || (random())::text || ','
          || (random())::text || ',' || (random())::text || ']')::vector
FROM generate_series(1, 200) i;

-- Tier 2: acorn_hnsw index
CREATE INDEX items_acorn_idx ON items USING acorn_hnsw (embedding vector_cosine_ops)
  WITH (m = 8, ef_construction = 32, acorn_gamma = 1);

-- planner must use Index Scan on acorn_hnsw (not Seq Scan)
EXPLAIN (COSTS OFF)
SELECT id FROM items
WHERE category = 'shoes'
ORDER BY embedding <-> '[0.1,0.2,0.3,0.4]'
LIMIT 5;

-- basic correctness: results are all from requested category
SELECT bool_and(category = 'shoes') AS all_shoes
FROM (
    SELECT category FROM items
    WHERE category = 'shoes'
    ORDER BY embedding <-> '[0.1,0.2,0.3,0.4]'
    LIMIT 5
) r;

-- gamma=2: larger neighbor pool must build without error
CREATE INDEX items_acorn_g2 ON items USING acorn_hnsw (embedding vector_cosine_ops)
  WITH (m = 8, ef_construction = 32, acorn_gamma = 2);

SELECT indexname FROM pg_indexes
WHERE tablename = 'items' AND indexdef LIKE '%acorn_hnsw%'
ORDER BY indexname;

DROP SCHEMA test_tier2 CASCADE;
