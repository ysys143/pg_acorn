"""pg_acorn target — Tier 1 (hook) and Tier 2 (acorn_hnsw AM)."""

import psycopg
import numpy as np

from ._explain import explain_filtered as _explain_filtered
from ._bulk import copy_load as _copy_load


class PgAcornTarget:
    def __init__(self, dsn: str, tier: int = 2, gamma: int = 1,
                 enable_2hop: bool = False):
        assert tier in (1, 2), "tier must be 1 or 2"
        self.tier = tier
        self.gamma = gamma
        self.enable_2hop = enable_2hop
        suffix = "_2hop" if enable_2hop else ""
        self.name = f"pg_acorn_tier{tier}_g{gamma}{suffix}"
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
            _copy_load(cur, vectors, metadata)

            if self.tier == 1:
                # Tier 1: standard hnsw index, hook intercepts at query time
                cur.execute("""
                    CREATE INDEX ON bench_items
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                """)
                cur.execute("SET pg_acorn.enable_hook = on")
            else:
                # Tier 2: acorn_hnsw AM with multicol index (embedding + bucket in-filter)
                cur.execute(f"""
                    CREATE INDEX ON bench_items
                    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
                    WITH (m = 16, ef_construction = 64, acorn_gamma = {self.gamma})
                """)
                cur.execute(
                    f"SET pg_acorn.enable_2hop = {'on' if self.enable_2hop else 'off'}"
                )

            # btree on the filter column so a free planner can choose a bitmap
            # prefilter (exact) instead of the acorn_hnsw scan at high selectivity.
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
        """Disable seq/bitmap scans so the planner uses the acorn_hnsw index —
        page counts and recall only compare meaningfully under the same plan."""
        with self.conn.cursor() as cur:
            if on:
                cur.execute("SET enable_seqscan = off")
                cur.execute("SET enable_bitmapscan = off")
            else:
                cur.execute("RESET enable_seqscan")
                cur.execute("RESET enable_bitmapscan")

    def set_ef_search(self, n: int) -> None:
        """Runtime recall/latency knob. Tier 2 reads pg_acorn.ef_search (bounded
        streaming scan); Tier 1 rides the standard HNSW hook via hnsw.ef_search.
        SET cannot bind parameters, so inline the integer-validated value."""
        ef = int(n)
        with self.conn.cursor() as cur:
            if self.tier == 2:
                cur.execute(f"SET pg_acorn.ef_search = {ef}")
            else:
                # Tier 1 rides pgvector's HNSW, whose ef_search caps at 1000.
                cur.execute(f"SET hnsw.ef_search = {min(ef, 1000)}")

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
