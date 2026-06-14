"""3-way benchmark — PostgreSQL side (pgvector native + pg_acorn), psycopg from host.

Two modes:
  --mode latency   : for each config, isolate its KNN index INSIDE a txn
                     (BEGIN; DROP others; measure; ROLLBACK -- no rebuilds), sweep
                     ef, record recall@10 + median/p95/min latency per sel.
  --mode throughput: assumes the orchestrator has left ONLY the target config's KNN
                     index present (persistent drop); launches multiprocessing
                     clients (own connection each) that loop the query for a window;
                     aggregate QPS, concurrency swept.

Configs: acorn_g2p0 (tv_acorn_noinline), acorn_g2p64 (tv_acorn_g2p64), acorn_inline
(tv_acorn_idx), pgv_iterative (tv_pgv_hnsw + hnsw.iterative_scan), pgv_prefilter
(no hnsw -> bitmap on bucket + exact Sort).  Same fixture/truth/SQL as
thesis_validation (seed 0 correlated 250K, cosine, bucket < sel).

Run from the host (Postgres on localhost:5432):
  uv run --with numpy --with "psycopg[binary]" python3 bench/bench3way_pg.py --mode latency
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import psycopg

from thesis_validation import make_fixture, exact_truth, qstr, SQL, K, NQ

DSN = os.environ.get("PG_DSN",
                     "host=localhost port=5432 dbname=bench user=postgres password=postgres")
SELS = [1, 2, 5, 10, 20]
ACORN_EFS = [100, 200, 400, 800, 1600]
PGV_EFS = [40, 100, 200, 400, 800]
CONC = [1, 4, 8, 16]
DUR = 6.0

ALL_KNN = ["tv_acorn_noinline", "tv_acorn_idx", "tv_acorn_autoef",
           "tv_acorn_g2p64", "tv_pgv_hnsw"]
CONFIGS = {
    "acorn_g2p0":    {"keep": "tv_acorn_noinline", "kind": "acorn"},
    "acorn_g2p64":   {"keep": "tv_acorn_g2p64",    "kind": "acorn"},
    "acorn_inline":  {"keep": "tv_acorn_idx",      "kind": "acorn"},
    "pgv_iterative": {"keep": "tv_pgv_hnsw",       "kind": "pgv_iterative"},
    "pgv_prefilter": {"keep": None,                "kind": "pgv_prefilter"},
}


def apply_gucs(cur, kind, ef):
    if kind == "acorn":
        cur.execute("SET enable_seqscan=off")
        cur.execute("SET pg_acorn.member_first=on")
        cur.execute("SET pg_acorn.scan_code_cache=on")
        cur.execute("SET pg_acorn.scan_inline_vectors=on")
        if ef is not None:
            cur.execute(f"SET pg_acorn.ef_search={ef}")
    elif kind == "pgv_iterative":
        cur.execute("SET enable_seqscan=off")
        try:
            # strict_order = correct top-k by distance; max_scan_tuples raised
            # so the iterative scan is not cut off on the correlated filter
            # (default 20000 strangles recall) -- gives pgvector its best shot.
            cur.execute("SET hnsw.iterative_scan='strict_order'")
            cur.execute("SET hnsw.max_scan_tuples=40000")
        except Exception as e:
            print("  [warn] iterative_scan unavailable:", e, flush=True)
        if ef is not None:
            cur.execute(f"SET hnsw.ef_search={ef}")
    elif kind == "pgv_prefilter":
        cur.execute("SET enable_seqscan=on")    # bitmap/seq + Sort, exact
        cur.execute("SET enable_indexscan=on")


def plan_of(cur, sel, q):
    cur.execute("EXPLAIN (FORMAT JSON) " + SQL, (int(sel), qstr(q), K))
    import json as _j
    p = cur.fetchone()[0]
    return _j.dumps(p)[:400]


def latency_mode(conn, queries, truths, out, only=None):
    for name, cfg in CONFIGS.items():
        if only and name != only:
            continue
        cur = conn.cursor()
        cur.execute("BEGIN")
        for idx in ALL_KNN:
            if idx != cfg["keep"]:
                cur.execute(f"DROP INDEX IF EXISTS {idx}")
        kind = cfg["kind"]
        apply_gucs(cur, kind, None)
        efs = [None] if kind == "pgv_prefilter" else (
            PGV_EFS if kind == "pgv_iterative" else ACORN_EFS)
        print(f"[{name}] plan: {plan_of(cur, 10, queries[0])}", flush=True)
        out[name] = {}
        for sel in SELS:
            cells = []
            for ef in efs:
                apply_gucs(cur, kind, ef)
                cur.execute(SQL, (int(sel), qstr(queries[0]), K))
                cur.fetchall()
                lats, recs = [], []
                for rep in range(3):
                    for i in range(NQ):
                        t0 = time.perf_counter()
                        cur.execute(SQL, (int(sel), qstr(queries[i]), K))
                        ids = {r[0] for r in cur.fetchall()}
                        lats.append((time.perf_counter() - t0) * 1e3)
                        if rep == 0:
                            recs.append(len(ids & truths[sel][i]) / K)
                a = np.array(lats)
                cells.append({"ef": ef, "recall": round(float(np.mean(recs)), 4),
                              "med_ms": round(float(np.median(a)), 2),
                              "p95_ms": round(float(np.percentile(a, 95)), 2),
                              "min_ms": round(float(np.min(a)), 2)})
                print(f"  [{name} sel={sel}% ef={ef}] r={cells[-1]['recall']:.3f} "
                      f"med={cells[-1]['med_ms']}ms p95={cells[-1]['p95_ms']}ms", flush=True)
            out[name][str(sel)] = cells
        cur.execute("ROLLBACK")
    return out


def _tp_worker(args):
    kind, ef, sel, qstrs, end_t = args
    conn = psycopg.connect(DSN, autocommit=True)
    cur = conn.cursor()
    apply_gucs(cur, kind, ef)
    cur.execute(SQL, (int(sel), qstrs[0], K))
    cur.fetchall()
    n = 0
    qi = 0
    while time.time() < end_t:
        cur.execute(SQL, (int(sel), qstrs[qi % len(qstrs)], K))
        cur.fetchall()
        qi += 1
        n += 1
    conn.close()
    return n


def throughput_mode(name, sel, ef, queries):
    cfg = CONFIGS[name]
    qstrs = [qstr(q) for q in queries]
    # verify the right index serves (parent connection)
    conn = psycopg.connect(DSN, autocommit=True)
    cur = conn.cursor()
    apply_gucs(cur, cfg["kind"], ef)
    print(f"[{name} tp] plan: {plan_of(cur, sel, queries[0])}", flush=True)
    conn.close()
    res = {}
    for nproc in CONC:
        end_t = time.time() + 1.0 + DUR
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            counts = list(ex.map(_tp_worker,
                                 [(cfg["kind"], ef, sel, qstrs, end_t)] * nproc))
        wall = time.time() - t0
        qps = sum(counts) / wall
        res[str(nproc)] = {"qps": round(qps, 1), "total": sum(counts),
                           "wall_s": round(wall, 2)}
        print(f"  [{name} tp sel={sel}% ef={ef} conc={nproc}] qps={qps:.0f} "
              f"(n={sum(counts)} in {wall:.1f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["latency", "throughput"], required=True)
    ap.add_argument("--out", default="bench/results_3way_pg.json")
    ap.add_argument("--config")          # throughput: config name
    ap.add_argument("--sel", type=int)   # throughput
    ap.add_argument("--ef", type=int)    # throughput (None for prefilter)
    ap.add_argument("--only")            # latency: measure just this config (merge)
    args = ap.parse_args()

    print("[fixture] seed0 correlated ...", flush=True)
    vecs, buckets, queries = make_fixture()
    truths = {s: [exact_truth(vecs, buckets, queries[i], s) for i in range(NQ)]
              for s in SELS}

    if args.mode == "latency":
        conn = psycopg.connect(DSN, autocommit=True)
        out = {"meta": {"engine": "pg (pgvector native + acorn)", "fixture": "seed0 250K",
                        "k": K, "nq": NQ}, "latency": {}}
        if args.only and os.path.exists(args.out):
            out = json.load(open(args.out))          # merge into existing
            out.setdefault("latency", {})
        latency_mode(conn, queries, truths, out["latency"], only=args.only)
        conn.close()
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[done] -> {args.out}", flush=True)
    else:
        ef = args.ef if args.ef and args.ef > 0 else None
        res = throughput_mode(args.config, args.sel, ef, queries)
        # append/merge into out file
        out = {}
        if os.path.exists(args.out):
            out = json.load(open(args.out))
        out.setdefault("throughput", {})[args.config] = {
            "sel": args.sel, "ef": ef, "by_conc": res}
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
