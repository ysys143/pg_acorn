"""Validate vector co-location (acorn_inline_vectors) against the classic layout.

Self-contained (modeled on validate_payload_edges.py / run_fastpath.sh): runs
inside its own container with its own named volume (bench/run_colocation.sh,
volume pg_acorn_coloc) — do NOT point this at the reserved pg_acorn_bench
compose stack.

Fixture: n=60K dim=128 CORRELATED (dominant 10-dim block + uniform spread so
bucket < k yields ~k% selectivity), fixed seed.  Two tables with identical
rows; both indexed gamma=2 payload_edges=on; one with acorn_inline_vectors=on.
Scan GUCs: enable_hook=off, member_first=on.

Sweep: sel in [1, 10, 40]%, ef_search in [100, 200, 400, 800, 1600].
Per cell and config: pages/query (EXPLAIN ANALYZE BUFFERS, plan-node
asserted), recall@10 vs numpy exact truth, median + per-query-min latency.

Hard asserts:
  - recall(on) >= recall(off) - 0.01 at every cell
  - pages(off) / pages(on) >= 10 for ef >= 200

W7 lesson applied: CHECKPOINT after builds, before any timing (post-build
checkpoint contention inflates medians 4-10x).
"""

import json
import os
import sys
import time

import numpy as np
import psycopg

DSN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PG_ACORN_DSN", "postgresql://postgres@localhost/postgres")

N, DIM, NQ, K = 60000, 128, 30, 10
GAMMA = 2
SELS = [1, 10, 40]                  # bucket < sel  =>  ~sel% selectivity
EFS = [100, 200, 400, 800, 1600]
REPEATS = 7
CONFIGS = [("off", "false"), ("on", "true")]   # acorn_inline_vectors
OUT = os.path.join(os.path.dirname(__file__), "results_colocation.json")

# --- correlated fixture (validate_payload_edges.py CORR=high) ---------------
rng = np.random.default_rng(0)
raw = rng.standard_normal((N, DIM)).astype(np.float32)
vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)

blocks = DIM // 10                                # 12 blocks for dim=128
block_norms = np.array([
    np.linalg.norm(vecs[:, i * 10:(i + 1) * 10], axis=1)
    for i in range(blocks)
]).T
dominant_block = np.argmax(block_norms, axis=1)   # 0..blocks-1
span = 100 // blocks                              # 8
buckets = dominant_block * span + rng.integers(0, span, size=N)
buckets = np.clip(buckets, 0, 99).astype(int)

queries = vecs[:NQ]


def vstr(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def exact_truth(q, sel):
    idx = np.where(buckets < sel)[0]
    sims = vecs[idx] @ q
    top = idx[np.argsort(-sims)[:K]]
    return set((top + 1).tolist())                # serial ids start at 1


def ensure_table(conn, name, inline):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT (SELECT count(*) FROM pg_class WHERE relname = %s), "
            "(SELECT count(*) FROM pg_class WHERE relname = %s)",
            (name, name + "_idx"))
        has_table, has_index = cur.fetchone()
        if has_table and has_index:
            cur.execute(f"SELECT count(*) FROM {name}")
            if cur.fetchone()[0] == N:
                print(f"reusing persisted {name} + index", flush=True)
                return
            print(f"{name}: row count mismatch -- rebuilding", flush=True)
        cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE")
        cur.execute(f"CREATE TABLE {name} "
                    f"(id serial PRIMARY KEY, bucket int, embedding vector({DIM}))")
        print(f"loading {N} rows into {name}...", flush=True)
        with cur.copy(f"COPY {name} (bucket, embedding) FROM STDIN") as cp:
            for i in range(N):
                cp.write_row((int(buckets[i]), vstr(vecs[i])))
        print(f"building {name}_idx (gamma={GAMMA}, payload_edges=on, "
              f"inline_vectors={inline})...", flush=True)
        t0 = time.perf_counter()
        cur.execute(f"""CREATE INDEX {name}_idx ON {name}
                        USING acorn_hnsw (embedding vector_cosine_ops,
                                          bucket int4_acorn_ops)
                        WITH (m = 16, ef_construction = 64,
                              acorn_gamma = {GAMMA},
                              acorn_payload_edges = true,
                              acorn_inline_vectors = {inline})""")
        print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)


def apply_gucs(cur, ef):
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET pg_acorn.enable_hook = off")
    cur.execute("SET pg_acorn.member_first = on")
    cur.execute(f"SET pg_acorn.ef_search = {ef}")


def run_cell(conn, sel, ef):
    """Measure one (sel, ef) cell for both configs, interleaved timing."""
    truth = [exact_truth(q, sel) for q in queries]
    qstrs = [vstr(q) for q in queries]
    out = {}

    with conn.cursor() as cur:
        for cname, _ in CONFIGS:
            tbl = f"coloc_{cname}"
            sql = (f"SELECT id FROM {tbl} WHERE bucket < %s "
                   f"ORDER BY embedding <=> %s::vector LIMIT %s")
            apply_gucs(cur, ef)

            pages = []
            for qs in qstrs:
                cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql,
                            (sel, qs, K))
                plan = cur.fetchone()[0][0]["Plan"]
                node = plan
                while node["Node Type"] in ("Limit", "Sort", "Gather"):
                    node = node["Plans"][0]
                assert (node["Node Type"] == "Index Scan"
                        and node.get("Index Name") == tbl + "_idx"), \
                    f"plan confound: {node['Node Type']} / {node.get('Index Name')}"
                pages.append(plan["Shared Hit Blocks"] + plan["Shared Read Blocks"])

            recalls = []
            for qs, t in zip(qstrs, truth):
                cur.execute(sql, (sel, qs, K))
                ids = [r[0] for r in cur.fetchall()]
                recalls.append(len(set(ids) & t) / K)

            out[cname] = {
                "pages_per_query": float(np.mean(pages)),
                "recall": float(np.mean(recalls)),
            }

        # interleaved timed passes (shared docker VM: min approaches the
        # uncontended floor; median reported alongside)
        lat = {cname: np.empty((REPEATS, NQ)) for cname, _ in CONFIGS}
        for cname, _ in CONFIGS:        # warmup
            tbl = f"coloc_{cname}"
            sql = (f"SELECT id FROM {tbl} WHERE bucket < %s "
                   f"ORDER BY embedding <=> %s::vector LIMIT %s")
            apply_gucs(cur, ef)
            for qs in qstrs:
                cur.execute(sql, (sel, qs, K))
                cur.fetchall()
        for r in range(REPEATS):
            for cname, _ in CONFIGS:
                tbl = f"coloc_{cname}"
                sql = (f"SELECT id FROM {tbl} WHERE bucket < %s "
                       f"ORDER BY embedding <=> %s::vector LIMIT %s")
                apply_gucs(cur, ef)
                L = lat[cname]
                for qi, qs in enumerate(qstrs):
                    t0 = time.perf_counter()
                    cur.execute(sql, (sel, qs, K))
                    cur.fetchall()
                    L[r, qi] = time.perf_counter() - t0

    for cname, _ in CONFIGS:
        med = np.median(lat[cname], axis=0)       # per-query median
        mn = lat[cname].min(axis=0)               # per-query min
        out[cname]["lat_ms_median"] = float(np.median(med) * 1e3)
        out[cname]["lat_ms_min_mean"] = float(mn.mean() * 1e3)
    return out


def main():
    conn = psycopg.connect(DSN, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")

    for cname, inline in CONFIGS:
        ensure_table(conn, f"coloc_{cname}", inline)

    with conn.cursor() as cur:
        # W7: post-build checkpoint contention inflates medians 4-10x
        cur.execute("CHECKPOINT")
        for cname, _ in CONFIGS:
            cur.execute(f"SELECT pg_relation_size('coloc_{cname}_idx')")
            sz = cur.fetchone()[0]
            print(f"coloc_{cname}_idx: {sz / 1024 / 1024:.0f} MB", flush=True)

    sel_actual = {s: float((buckets < s).mean()) for s in SELS}
    print(f"selectivities: {sel_actual}", flush=True)

    results = {"n": N, "dim": DIM, "nq": NQ, "k": K, "gamma": GAMMA,
               "payload_edges": True, "member_first": True,
               "repeats": REPEATS, "selectivity": sel_actual, "cells": {}}
    failures = []

    print(f"\n{'sel':>4} {'ef':>5} {'cfg':>4} {'pages/q':>9} {'recall':>7} "
          f"{'med ms':>8} {'min ms':>8}", flush=True)
    for sel in SELS:
        for ef in EFS:
            cell = run_cell(conn, sel, ef)
            results["cells"][f"{sel}_{ef}"] = cell
            for cname, _ in CONFIGS:
                c = cell[cname]
                print(f"{sel:>4} {ef:>5} {cname:>4} {c['pages_per_query']:>9.1f} "
                      f"{c['recall']:>7.3f} {c['lat_ms_median']:>8.2f} "
                      f"{c['lat_ms_min_mean']:>8.2f}", flush=True)

            d_recall = cell["off"]["recall"] - cell["on"]["recall"]
            ratio = (cell["off"]["pages_per_query"]
                     / max(cell["on"]["pages_per_query"], 1e-9))
            line = (f"  -> recall delta(off-on)={d_recall:+.3f}  "
                    f"pages ratio={ratio:.1f}x")
            ok = True
            if d_recall > 0.01:
                ok = False
                failures.append(f"sel={sel} ef={ef}: recall delta {d_recall:.3f} > 0.01")
            if ef >= 200 and ratio < 10.0:
                ok = False
                failures.append(f"sel={sel} ef={ef}: pages ratio {ratio:.1f}x < 10x")
            print(line + ("" if ok else "  [FAIL]"), flush=True)

    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved {OUT}", flush=True)

    if failures:
        print("VALIDATION FAILURES:", flush=True)
        for fl in failures:
            print("  " + fl, flush=True)
        sys.exit(1)
    print("ALL VALIDATION ASSERTS PASSED", flush=True)


if __name__ == "__main__":
    main()
