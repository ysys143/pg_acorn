#!/usr/bin/env python3
"""M3 eviction-under-concurrent-scan stress (BUG-2 proof, aarch64).

Forces continuous LRU-eviction churn (code_cache_size sized to hold far fewer
than the live index set) while N backends scan random indexes with the code
cache ON.  A load that evicts a slot another backend is mid-scanning is exactly
the hazard the M3 reader-revalidation + evictor deferred-free must survive.

Pass criteria: (1) zero crashes (the server stays up the whole run), and
(2) every cache-on scan returns the SAME ids as the cache-off truth — cache
state never changes a result (G4).

Run inside the bench postgres container against the ccstress DB:
  python3 -u /workspace/bench/cc_evict_stress.py --dsn 'host=/var/run/postgresql dbname=ccstress user=postgres' --secs 45 --threads 6
"""
import argparse
import random
import threading
import time

import psycopg

IDXS = ["s1", "s2", "s3", "s4"]
QV = "[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]"


def topk(cur, tbl, cache, ef=200):
    cur.execute(f"SET pg_acorn.scan_code_cache = {'on' if cache else 'off'}")
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SET pg_acorn.ef_search = {ef}")
    cur.execute(
        f"SELECT id FROM {tbl} WHERE bucket < 3 "
        f"ORDER BY embedding <-> '{QV}'::vector LIMIT 10")
    return tuple(r[0] for r in cur.fetchall())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn",
                    default="host=/var/run/postgresql dbname=ccstress user=postgres")
    ap.add_argument("--secs", type=float, default=45)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--budget_mb", type=int, default=1)
    args = ap.parse_args()

    # Ground truth (cache OFF) for each index.
    truth = {}
    with psycopg.connect(args.dsn, autocommit=True) as c, c.cursor() as cur:
        for t in IDXS:
            truth[t] = topk(cur, t, cache=False)
        print(f"[truth] {truth}", flush=True)

    stop = time.perf_counter() + args.secs
    stats = {"scans": 0, "mismatch": 0, "errors": 0}
    lock = threading.Lock()

    def worker(wid):
        local = 0
        try:
            with psycopg.connect(args.dsn, autocommit=True) as c, c.cursor() as cur:
                # Tiny budget on every backend -> each load churns eviction.
                cur.execute(f"SET pg_acorn.code_cache_size = '{args.budget_mb}MB'")
                while time.perf_counter() < stop:
                    t = random.choice(IDXS)
                    got = topk(cur, t, cache=True,
                               ef=random.choice([100, 200, 400]))
                    local += 1
                    if got != truth[t]:
                        with lock:
                            stats["mismatch"] += 1
                        print(f"[MISMATCH] w{wid} {t}: {got} != {truth[t]}",
                              flush=True)
                    # Occasionally force-evict / reset to add churn.
                    r = random.random()
                    if r < 0.05:
                        cur.execute(
                            f"SELECT pg_acorn_code_cache_evict('{t}_idx')")
                    elif r < 0.07:
                        cur.execute("SELECT pg_acorn_code_cache_reset()")
        except Exception as e:                       # a crash drops the conn
            with lock:
                stats["errors"] += 1
            print(f"[ERROR] w{wid}: {type(e).__name__}: {e}", flush=True)
        with lock:
            stats["scans"] += local

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(args.threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # Post-run liveness + identity recheck on a fresh connection.
    alive = False
    final_ok = True
    try:
        with psycopg.connect(args.dsn, autocommit=True) as c, c.cursor() as cur:
            alive = True
            for t in IDXS:
                if topk(cur, t, cache=True) != truth[t]:
                    final_ok = False
    except Exception as e:
        print(f"[POSTCHECK ERROR] {type(e).__name__}: {e}", flush=True)

    print(f"\n[done] scans={stats['scans']} mismatch={stats['mismatch']} "
          f"errors={stats['errors']} server_alive={alive} "
          f"final_identity_ok={final_ok}", flush=True)
    ok = (stats["mismatch"] == 0 and stats["errors"] == 0
          and alive and final_ok)
    print("[verdict] " + ("PASS" if ok else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
