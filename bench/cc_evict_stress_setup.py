#!/usr/bin/env python3
"""Build the ccstress fixture (s1..s4) that cc_evict_stress.py scans.

Four NON-inline acorn_hnsw indexes (the cache only serves non-inline), each
sized so its SQ8 code table exceeds the 1MB churn budget the stress runs with
-> any single load evicts a peer mid-scan, exercising the lock-free reader vs
evictor hazard.  Matches the stress script's contract: tables s1..s4 with
(id, bucket, embedding vector(8)), indexes s1_idx..s4_idx, query
'[0.1..0.8]'::vector under L2, filter bucket < 3.

Run inside the throwaway container (server already up):
    python3 -u /workspace/bench/cc_evict_stress_setup.py
"""
import sys
import time

import psycopg

DSN_POSTGRES = "host=/var/run/postgresql dbname=postgres user=postgres"
DSN_CCSTRESS = "host=/var/run/postgresql dbname=ccstress user=postgres"
DIM = 8
N = 30000          # ~30K * 48B stride ~= 1.4MB cache table > 1MB budget
IDXS = ["s1", "s2", "s3", "s4"]


def main():
    # Create the ccstress DB on a maintenance connection.
    with psycopg.connect(DSN_POSTGRES, autocommit=True) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ccstress'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE ccstress")
            print("[setup] created database ccstress", flush=True)

    with psycopg.connect(DSN_CCSTRESS, autocommit=True) as c, c.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")
        cur.execute("SET max_parallel_maintenance_workers = 4")
        cur.execute("SET maintenance_work_mem = '1GB'")
        for k, t in enumerate(IDXS):
            cur.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
            cur.execute(f"CREATE TABLE {t} (id serial PRIMARY KEY, "
                        f"bucket int, embedding vector({DIM}))")
            # distinct seed per table so the four graphs differ
            cur.execute(f"SELECT setseed({0.11 * (k + 1):.3f})")
            t0 = time.perf_counter()
            cur.execute(
                f"INSERT INTO {t} (bucket, embedding) "
                "SELECT i %% 10, v.vec FROM generate_series(1, %s) i, LATERAL ("
                "  SELECT ('[' || string_agg(random()::text, ',') || ']')::vector"
                "  AS vec FROM generate_series(1, %s) d WHERE i = i) v"
                % (N, DIM))
            cur.execute("SET pg_acorn.build_seed = 42")
            cur.execute(
                f"CREATE INDEX {t}_idx ON {t} "
                "USING acorn_hnsw (embedding vector_l2_ops, bucket int4_acorn_ops) "
                "WITH (m = 16, ef_construction = 64, acorn_gamma = 2)")
            cur.execute(f"SELECT pg_relation_size('{t}_idx')")
            sz = cur.fetchone()[0]
            print(f"[setup] {t}: rows={N} idx_mb={sz / 1048576:.1f} "
                  f"build_s={time.perf_counter() - t0:.1f}", flush=True)
    print("[setup] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
