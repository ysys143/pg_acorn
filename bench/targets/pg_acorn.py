"""pg_acorn target — Tier 1 (hook) and Tier 2 (acorn_hnsw AM)."""

import psycopg
import numpy as np


class PgAcornTarget:
    def __init__(self, dsn: str, tier: int = 2, gamma: int = 1):
        assert tier in (1, 2), "tier must be 1 or 2"
        self.tier = tier
        self.gamma = gamma
        self.name = f"pg_acorn_tier{tier}_g{gamma}"
        self.conn = psycopg.connect(dsn, autocommit=True)

    def setup(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        dim = vectors.shape[1]
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")
            cur.execute("DROP TABLE IF EXISTS bench_items CASCADE")
            cur.execute(f"""
                CREATE TABLE bench_items (
                    id      serial PRIMARY KEY,
                    bucket  int,
                    embedding vector({dim})
                )
            """)
            cur.executemany(
                "INSERT INTO bench_items (bucket, embedding) VALUES (%s, %s)",
                [(m["bucket"], v.tolist()) for v, m in zip(vectors, metadata)],
            )

            if self.tier == 1:
                # Tier 1: standard hnsw index, hook intercepts at query time
                cur.execute("""
                    CREATE INDEX ON bench_items
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
                cur.execute("SET pg_acorn.enable_hook = on")
            else:
                # Tier 2: acorn_hnsw AM with gamma
                cur.execute(f"""
                    CREATE INDEX ON bench_items
                    USING acorn_hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64, acorn_gamma = {self.gamma})
                """)

    def query_filtered(self, query: np.ndarray, bucket_threshold: int, k: int) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id FROM bench_items
                WHERE bucket < %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (bucket_threshold, query.tolist(), k),
            )
            return [row[0] for row in cur.fetchall()]

    def query_unfiltered(self, query: np.ndarray, k: int) -> list[int]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM bench_items ORDER BY embedding <=> %s::vector LIMIT %s",
                (query.tolist(), k),
            )
            return [row[0] for row in cur.fetchall()]

    def insert_batch(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO bench_items (bucket, embedding) VALUES (%s, %s)",
                [(m["bucket"], v.tolist()) for v, m in zip(vectors, metadata)],
            )

    def teardown(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS bench_items CASCADE")

    def close(self) -> None:
        self.conn.close()
