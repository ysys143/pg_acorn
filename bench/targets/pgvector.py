"""pgvector vanilla target — baseline (seq scan fallback under filter)."""

import psycopg
import numpy as np

from ._explain import explain_filtered as _explain_filtered
from ._bulk import copy_load as _copy_load


class PgvectorTarget:
    name = "pgvector"

    def __init__(self, dsn: str):
        self.conn = psycopg.connect(dsn, autocommit=True)

    def setup(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        dim = vectors.shape[1]
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # pg_acorn is in shared_preload_libraries, so its Tier-1
            # set_rel_pathlist_hook is active by default and would intercept
            # this baseline's filtered queries with an AcornScan CustomScan.
            # Disable it so "pgvector" measures *vanilla* HNSW + postfilter, not
            # ACORN-1. (GUC comes from the preloaded library; no extension needed.)
            cur.execute("SET pg_acorn.enable_hook = off")
            cur.execute("DROP TABLE IF EXISTS bench_items CASCADE")
            cur.execute(f"""
                CREATE TABLE bench_items (
                    id      serial PRIMARY KEY,
                    bucket  int,
                    embedding vector({dim})
                )
            """)
            _copy_load(cur, vectors, metadata)
            cur.execute("""
                CREATE INDEX ON bench_items
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
            # btree on the filter column so a free planner can choose a bitmap
            # prefilter (exact) over the HNSW postfilter at high selectivity.
            cur.execute("CREATE INDEX ON bench_items (bucket)")

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

    def explain_filtered(self, query: np.ndarray, bucket_threshold: int, k: int) -> dict:
        """Per-query page-access counts + scan node type for the filtered query."""
        return _explain_filtered(self.conn, query, bucket_threshold, k)

    def force_index_scan(self, on: bool) -> None:
        """Disable seq/bitmap scans so the planner uses the vector index — page
        counts and recall only compare meaningfully when the same plan is used."""
        with self.conn.cursor() as cur:
            if on:
                cur.execute("SET enable_seqscan = off")
                cur.execute("SET enable_bitmapscan = off")
            else:
                cur.execute("RESET enable_seqscan")
                cur.execute("RESET enable_bitmapscan")

    def set_ef_search(self, n: int) -> None:
        """Runtime recall/latency knob: HNSW dynamic candidate list size.
        SET is a utility statement and cannot bind parameters, so inline the
        (integer-validated) value. pgvector caps hnsw.ef_search at 1000, so the
        sweep's higher values saturate there (pgvector's frontier tops out)."""
        ef = min(int(n), 1000)
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {ef}")

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
