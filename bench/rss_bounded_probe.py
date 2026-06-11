#!/usr/bin/env python3
"""rss_bounded_probe.py — bounded-vs-unbounded private memory (Z4 part a).

Runs INSIDE the postgres container (reads /proc directly):

  1. mwm=32MB build on the existing bo_items 60K fixture with a 1s RssAnon
     sampler on the building backend.  The graph stops growing at the
     budget (spill WARNING ~3.5 min in); after a further grace period the
     query is cancelled (the on-disk tail uses bounded per-tuple memory and
     adds ~30 min of wall time without changing the peak).  Peak RssAnon =
     the bounded number.
  2. mwm=2GB build to completion with the same sampler.  Peak RssAnon =
     the unbounded number (the whole ~62MB graph resident) and a clean
     w=0 build-time reference.

RssAnon (not VmHWM) because the backend touches every shared_buffers page
it writes — shared pages dominate VmHWM and hide the private graph.
"""

import json
import threading
import time

import psycopg

DSN = "postgresql://postgres:postgres@localhost:5432/bench"
OUT = "bench/results_rss_bounded.json"
CANCEL_AFTER_S = 480     # past the ~210s spill point + grace
BUILD_SQL = """CREATE INDEX bo_rssprobe_idx ON bo_items
    USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
    WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
          acorn_payload_edges = true, acorn_inline_vectors = true)"""


def sample(pid, stop, out):
    peak = 0
    while not stop.is_set():
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


def run_build(mwm, cancel_after=None):
    conn = psycopg.connect(DSN, autocommit=True)
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid()")
    pid = cur.fetchone()[0]
    cur.execute("SET pg_acorn.enable_hook = off")
    cur.execute("SET pg_acorn.build_direct_dist = on")
    cur.execute("SET pg_acorn.build_seed = 42")
    cur.execute(f"SET maintenance_work_mem = '{mwm}'")
    cur.execute("SET max_parallel_maintenance_workers = 0")
    cur.execute("ALTER TABLE bo_items SET (parallel_workers = 0)")
    cur.execute("DROP INDEX IF EXISTS bo_rssprobe_idx")

    watch = {}
    stop = threading.Event()
    th = threading.Thread(target=sample, args=(pid, stop, watch))
    th.start()

    canceller = None
    if cancel_after:
        def cancel():
            time.sleep(cancel_after)
            try:
                c2 = psycopg.connect(DSN, autocommit=True)
                c2.execute("SELECT pg_cancel_backend(%s)", (pid,))
                c2.close()
            except Exception:
                pass
        canceller = threading.Thread(target=cancel)
        canceller.start()

    t0 = time.perf_counter()
    completed = True
    try:
        cur.execute(BUILD_SQL)
    except psycopg.errors.QueryCanceled:
        completed = False
    dt = time.perf_counter() - t0

    stop.set()
    th.join()
    if canceller:
        canceller.join()
    if completed:
        cur.execute("CHECKPOINT")
        cur.execute("DROP INDEX IF EXISTS bo_rssprobe_idx")
    conn.close()
    return {"mwm": mwm, "peak_rss_anon_kb": watch.get("peak_rss_anon_kb"),
            "build_s": round(dt, 1), "completed": completed,
            "note": ("cancelled after the spill point: the on-disk tail "
                     "uses bounded per-tuple memory" if not completed else
                     "full in-memory build")}


def main():
    res = {"n": 60000, "dim": 128,
           "config": "m=16 efc=64 gamma=2 payload_edges inline seed=42"}
    res["bounded_32MB"] = run_build("32MB", cancel_after=CANCEL_AFTER_S)
    print("[rss-probe] 32MB:", res["bounded_32MB"], flush=True)
    res["unbounded_2GB"] = run_build("2GB")
    print("[rss-probe] 2GB:", res["unbounded_2GB"], flush=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
