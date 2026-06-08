"""pgvector vanilla target — baseline (seq scan fallback under filter)."""

import psycopg
import numpy as np


class PgvectorTarget:
    name = "pgvector"

    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn, autocommit=True)

    def setup(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        dim = vectors.shape[1]
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
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
            cur.execute("""
                CREATE INDEX ON bench_items
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
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
