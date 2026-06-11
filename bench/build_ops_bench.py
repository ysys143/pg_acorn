#!/usr/bin/env python3
"""build_ops_bench.py — Z4 build-operations measurements.

Stages (pick with --stages, comma separated):

  rss        (a) maintenance_work_mem bounds backend memory: build n=60K
             inline index with mwm=64MB vs mwm=2GB in FRESH connections and
             record each backend's peak RSS (VmHWM from /proc/<pid>/status,
             read via pg_read_file as superuser).  Bounded vs unbounded.

  workers    (b) worker-count scaling at n=60K: build the acceptance config
             (m=16, efc=64, gamma=2, payload_edges, inline_vectors) with
             max_parallel_maintenance_workers/parallel_workers reloption set
             to 0/1/2/4; record wall time and the max number of parallel
             workers observed in pg_stat_activity during each build.

  recall     (b) recall parity at n=60K: REAL recall (exact truth tables on
             40 queries, sel=10% bucket<10) of a parallel-built (2 workers)
             vs serial-built index at equal ef.  Gate: |delta| <= 0.02.

  confirm250k  one 250K build with the best worker setting; compare with the
             4968s single-threaded acceptance baseline.

Hygiene: pg_acorn.enable_hook=off, fixed data seed + build_seed=42,
CHECKPOINT after builds before any timing of subsequent stages.
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
import psycopg

K = 10
NQ = 40
DATA_SEED = 42
BUILD_SEED = 42


def qstr(v) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"


def make_fixture(n: int, dim: int):
    """Dominant-block correlated fixture (same as build_bench / acceptance)."""
    rng = np.random.default_rng(DATA_SEED)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    blocks = dim // 10
    block_norms = np.array([
        np.linalg.norm(vecs[:, i * 10:(i + 1) * 10], axis=1)
        for i in range(blocks)
    ]).T
    dominant = np.argmax(block_norms, axis=1)
    span = 100 // blocks
    buckets = np.clip(dominant * span + rng.integers(0, span, size=n),
                      0, 99).astype(int)
    qraw = rng.standard_normal((NQ, dim)).astype(np.float32)
    queries = qraw / np.linalg.norm(qraw, axis=1, keepdims=True)
    return vecs, buckets, queries


def load_table(conn, vecs, buckets, dim, table="bo_items"):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"CREATE TABLE {table} (id serial PRIMARY KEY, "
                    f"bucket int, embedding vector({dim}))")
        t0 = time.perf_counter()
        with cur.copy(f"COPY {table} (bucket, embedding) FROM STDIN") as cp:
            for i in range(len(vecs)):
                cp.write_row((int(buckets[i]), qstr(vecs[i])))
        cur.execute(f"ANALYZE {table}")
        print(f"[load] {len(vecs)} rows -> {table} in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)


BUILD_SQL = """CREATE INDEX {name} ON {table}
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_inline_vectors = true)"""


def build_index(cur, *, table="bo_items", name="bo_idx", mwm="2GB",
                workers=0, drop=True):
    if drop:
        cur.execute(f"DROP INDEX IF EXISTS {name}")
    cur.execute("SET pg_acorn.enable_hook = off")
    cur.execute("SET pg_acorn.build_direct_dist = on")
    cur.execute(f"SET pg_acorn.build_seed = {BUILD_SEED}")
    cur.execute(f"SET maintenance_work_mem = '{mwm}'")
    cur.execute(f"SET max_parallel_maintenance_workers = {workers}")
    cur.execute(f"ALTER TABLE {table} SET (parallel_workers = {workers})")
    t0 = time.perf_counter()
    cur.execute(BUILD_SQL.format(name=name, table=table))
    dt = time.perf_counter() - t0
    cur.execute("CHECKPOINT")
    return dt


def backend_vmhwm_kb(cur) -> int:
    cur.execute("SELECT pg_read_file(format('/proc/%s/status', "
                "pg_backend_pid()))")
    for line in cur.fetchone()[0].splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return -1


def sample_rss_anon(pid, stop_evt, out):
    """Max RssAnon of the building backend, sampled from /proc (this script
    runs inside the container).  RssAnon excludes shared_buffers pages, which
    dominate VmHWM (the backend touches every index page it writes), so it
    isolates the build's PRIVATE allocations — the thing maintenance_work_mem
    is supposed to bound."""
    peak = 0
    while not stop_evt.is_set():
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        peak = max(peak, int(line.split()[1]))
                        break
        except OSError:
            pass
        time.sleep(1.0)
    out["peak_rss_anon_kb"] = peak


def watch_parallel_workers(dsn, stop_evt, out):
    """Poll pg_stat_activity for parallel workers on CREATE INDEX."""
    conn = psycopg.connect(dsn, autocommit=True)
    peak = 0
    with conn.cursor() as cur:
        while not stop_evt.is_set():
            cur.execute(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE backend_type = 'parallel worker'")
            peak = max(peak, cur.fetchone()[0])
            time.sleep(0.5)
    conn.close()
    out["peak_parallel_workers"] = peak


def topk_ids(cur, queries, table, ef, sel=10):
    cur.execute("SET pg_acorn.enable_hook = off")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SET pg_acorn.ef_search = {ef}")
    out = []
    for q in queries:
        cur.execute(f"SELECT id FROM {table} WHERE bucket < {sel} "
                    f"ORDER BY embedding <=> %s::vector LIMIT {K}",
                    (qstr(q),))
        out.append([r[0] for r in cur.fetchall()])
    cur.execute("RESET enable_seqscan")
    return out


def truth_ids(cur, queries, table, sel=10):
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET enable_seqscan = on")
    out = []
    for q in queries:
        cur.execute(f"SELECT id FROM {table} WHERE bucket < {sel} "
                    f"ORDER BY embedding <=> %s::vector LIMIT {K}",
                    (qstr(q),))
        out.append([r[0] for r in cur.fetchall()])
    cur.execute("RESET enable_indexscan")
    cur.execute("RESET enable_bitmapscan")
    return out


def recall(ids, truth):
    hits = sum(len(set(a) & set(b)) for a, b in zip(ids, truth))
    return hits / (len(truth) * K)


def stage_rss(dsn, vecs, buckets, res):
    """Fresh connection per mwm setting; peak RssAnon sampled during the
    build isolates the backend's private memory (graph + transients).

    32MB guarantees a spill at n=60K dim=128 (~62MB of graph): the bounded
    build's private peak stays near the budget while the 2GB build holds
    the whole graph in memory."""
    out = {}
    for mwm in ("32MB", "2GB"):
        conn = psycopg.connect(dsn, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            pid = cur.fetchone()[0]
            watch = {}
            stop = threading.Event()
            th = threading.Thread(target=sample_rss_anon,
                                  args=(pid, stop, watch))
            th.start()
            dt = build_index(cur, name="bo_rss_idx", mwm=mwm, workers=0)
            stop.set()
            th.join()
            cur.execute("DROP INDEX bo_rss_idx")
        conn.close()
        out[mwm] = {"pid": pid,
                    "peak_rss_anon_kb": watch.get("peak_rss_anon_kb", -1),
                    "build_s": round(dt, 1)}
        print(f"[rss] mwm={mwm}: peak RssAnon "
              f"{watch.get('peak_rss_anon_kb')} kB (build {dt:.1f}s)",
              flush=True)
    res["rss"] = out


def stage_workers(dsn, res, counts=(0, 1, 2, 4)):
    out = {}
    for w in counts:
        watch = {}
        stop = threading.Event()
        th = threading.Thread(target=watch_parallel_workers,
                              args=(dsn, stop, watch))
        th.start()
        conn = psycopg.connect(dsn, autocommit=True)
        with conn.cursor() as cur:
            dt = build_index(cur, name="bo_w_idx", mwm="2GB", workers=w)
        conn.close()
        stop.set()
        th.join()
        out[str(w)] = {"build_s": round(dt, 1),
                       "peak_parallel_workers": watch.get(
                           "peak_parallel_workers", -1)}
        print(f"[workers] w={w}: {dt:.1f}s "
              f"(peak parallel workers seen: "
              f"{watch.get('peak_parallel_workers')})", flush=True)
    res["workers_60k"] = out


def stage_recall(dsn, queries, res, efs=(100, 200, 400)):
    conn = psycopg.connect(dsn, autocommit=True)
    out = {}
    with conn.cursor() as cur:
        truth = truth_ids(cur, queries, "bo_items")

        build_index(cur, name="bo_ser_idx", mwm="2GB", workers=0)
        ser = {ef: recall(topk_ids(cur, queries, "bo_items", ef), truth)
               for ef in efs}
        cur.execute("DROP INDEX bo_ser_idx")

        build_index(cur, name="bo_par_idx", mwm="2GB", workers=2)
        par = {ef: recall(topk_ids(cur, queries, "bo_items", ef), truth)
               for ef in efs}
        cur.execute("DROP INDEX bo_par_idx")

    conn.close()
    for ef in efs:
        d = par[ef] - ser[ef]
        out[str(ef)] = {"serial": round(ser[ef], 4),
                        "parallel": round(par[ef], 4),
                        "delta": round(d, 4),
                        "within_0.02": abs(d) <= 0.02}
        print(f"[recall] ef={ef}: serial={ser[ef]:.4f} "
              f"parallel={par[ef]:.4f} delta={d:+.4f}", flush=True)
    res["recall_60k"] = out


def stage_confirm250k(dsn, dim, res, workers=2):
    vecs, buckets, _ = make_fixture(250_000, dim)
    conn = psycopg.connect(dsn, autocommit=True)
    load_table(conn, vecs, buckets, dim, table="bo_items_250k")
    watch = {}
    stop = threading.Event()
    th = threading.Thread(target=watch_parallel_workers,
                          args=(dsn, stop, watch))
    th.start()
    with conn.cursor() as cur:
        dt = build_index(cur, table="bo_items_250k", name="bo_250k_idx",
                         mwm="2GB", workers=workers)
        cur.execute("SELECT pg_relation_size('bo_250k_idx') / 1024 / 1024")
        size_mb = cur.fetchone()[0]
    stop.set()
    th.join()
    conn.close()
    res["confirm_250k"] = {
        "workers": workers,
        "build_s": round(dt, 1),
        "size_mb": int(size_mb),
        "peak_parallel_workers": watch.get("peak_parallel_workers", -1),
        "baseline_single_thread_s": 4967.9,
        "speedup": round(4967.9 / dt, 2),
        "gate_2000s": dt <= 2000,
    }
    print(f"[250k] w={workers}: {dt:.1f}s (baseline 4967.9s, "
          f"speedup {4967.9/dt:.2f}x, gate<=2000s: {dt <= 2000})",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn",
                    default="postgresql://postgres:postgres@localhost:5433/bench")
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--stages", default="rss,workers,recall")
    ap.add_argument("--worker-counts", default="0,1,2,4",
                    help="comma-separated max_parallel_maintenance_workers "
                         "settings for the workers stage")
    ap.add_argument("--workers-250k", type=int, default=2)
    ap.add_argument("--out", default="bench/results_build_ops.json")
    args = ap.parse_args()
    stages = args.stages.split(",")

    res = {"meta": {"n": args.n, "dim": args.dim, "stages": stages,
                    "build_seed": BUILD_SEED, "data_seed": DATA_SEED,
                    "config": "m=16 efc=64 gamma=2 payload_edges inline",
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())}}

    if any(s in stages for s in ("rss", "workers", "recall")):
        vecs, buckets, queries = make_fixture(args.n, args.dim)
        conn = psycopg.connect(args.dsn, autocommit=True)
        load_table(conn, vecs, buckets, args.dim)
        conn.close()
    else:
        queries = make_fixture(1000, args.dim)[2]

    if "rss" in stages:
        stage_rss(args.dsn, vecs, buckets, res)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    if "workers" in stages:
        counts = tuple(int(c) for c in args.worker_counts.split(","))
        stage_workers(args.dsn, res, counts)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    if "recall" in stages:
        stage_recall(args.dsn, queries, res)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    if "confirm250k" in stages:
        stage_confirm250k(args.dsn, args.dim, res, args.workers_250k)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
