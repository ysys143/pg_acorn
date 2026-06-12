#!/usr/bin/env python3
"""Phase C measurement 1: inline vs non-inline acorn index, same data, 250K.

Both indexes coexist on tv_items (g2 + payload_edges, build differs only in
acorn_inline_vectors).  Each mode is measured inside a transaction that DROPs
the competing index and ROLLBACKs afterwards, so the planner choice is forced
without destroying either index.

Protocol mirrors remeasure_money_cells.py (FAIR-1T, quiet host): same
fixture/queries/truths, unmeasured prewarm pass + NWARM, PASSES timed passes,
EXPLAIN plan checks, one EXPLAIN (ANALYZE, BUFFERS) sample per cell.

Run inside the bench postgres container:
  python3 -u /workspace/bench/noinline_ab.py
"""

import argparse
import json
import os
import re
import time

import numpy as np
import psycopg

from thesis_validation import (K, NWARM, SQL, make_fixture, exact_truth,
                               qstr)

DSN = os.environ.get("PG_DSN",
                     "host=/var/run/postgresql dbname=bench user=postgres")
OUT = os.path.join(os.path.dirname(__file__), "results_noinline_ab.json")
PASSES = 3
SELS = [1, 10, 20]
EFS = [100, 200, 400, 800, 1600]
# mode -> (index to keep, index to drop in-txn)
# "cache" = noinline index + pg_acorn.scan_code_cache=on (M1 read path)
MODES = {
    "inline": ("tv_acorn_idx", "tv_acorn_noinline"),
    "noinline": ("tv_acorn_noinline", "tv_acorn_idx"),
    "cache": ("tv_acorn_noinline", "tv_acorn_idx"),
}


def plan_text(cur, sel, q):
    cur.execute("EXPLAIN (FORMAT TEXT) " + SQL, (int(sel), qstr(q), K))
    return "\n".join(r[0] for r in cur.fetchall())


def buffers_sample(cur, sel, q):
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, TIMING OFF, FORMAT TEXT) " + SQL,
                (int(sel), qstr(q), K))
    txt = "\n".join(r[0] for r in cur.fetchall())
    m = re.search(r"Buffers: shared hit=(\d+)(?: read=(\d+))?", txt)
    hit = int(m.group(1)) if m else 0
    read = int(m.group(2)) if m and m.group(2) else 0
    return hit, read


def measure(cur, queries, truth, sel):
    for q in queries:                       # unmeasured prewarm pass
        cur.execute(SQL, (int(sel), qstr(q), K))
        cur.fetchall()
    for _ in range(NWARM):
        cur.execute(SQL, (int(sel), qstr(queries[0]), K))
        cur.fetchall()
    lats = np.empty((PASSES, len(queries)))
    recalls = []
    for p in range(PASSES):
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            cur.execute(SQL, (int(sel), qstr(q), K))
            ids = {r[0] for r in cur.fetchall()}
            lats[p, qi] = (time.perf_counter() - t0) * 1e3
            if p == 0:
                recalls.append(len(ids & truth[qi]) / K)
    return lats, float(np.mean(recalls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="inline,noinline",
                    help="comma list from: " + ",".join(MODES))
    ap.add_argument("--efs", default=",".join(str(e) for e in EFS),
                    help="comma list of ef_search values")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    run_modes = [m.strip() for m in args.modes.split(",")]
    assert all(m in MODES for m in run_modes), run_modes
    run_efs = [int(e) for e in args.efs.split(",")]

    print("[fixture] regenerating (deterministic) ...", flush=True)
    vecs, buckets, queries = make_fixture()
    truths = {s: [exact_truth(vecs, buckets, q, s) for q in queries]
              for s in SELS}

    conn = psycopg.connect(DSN, autocommit=True, prepare_threshold=0)
    out = {"meta": {"protocol": "FAIR-1T A/B "
                    f"modes={','.join(run_modes)}, "
                    f"passes={PASSES}, nq={len(queries)}",
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                 time.gmtime())},
           "index_size": {}, "ops": []}

    with conn.cursor() as cur:
        for idx in ("tv_acorn_idx", "tv_acorn_noinline"):
            cur.execute("SELECT pg_relation_size(%s::regclass)", (idx,))
            out["index_size"][idx] = cur.fetchone()[0]
            print(f"[size] {idx}: "
                  f"{out['index_size'][idx] / 1024 / 1024:.0f} MB", flush=True)

        cur.execute("SET enable_seqscan = off")
        cur.execute("SET enable_bitmapscan = off")
        cur.execute("SET enable_sort = off")
        cur.execute("SET pg_acorn.member_first = on")
        cur.execute("SET pg_acorn.scan_inline_vectors = on")
        cur.execute("SHOW pg_acorn.buffered_emission")
        assert cur.fetchone()[0] == "on"

        for mode in run_modes:
            keep, drop = MODES[mode]
            cur.execute("SET pg_acorn.scan_code_cache = "
                        + ("on" if mode == "cache" else "off"))
            cur.execute("BEGIN")
            cur.execute(f"DROP INDEX {drop}")
            for sel in SELS:
                for ef in run_efs:
                    cur.execute(f"SET pg_acorn.ef_search = {ef}")
                    plan = plan_text(cur, sel, queries[0])
                    assert (f"Index Scan using {keep}" in plan
                            and "Index Cond" in plan), \
                        f"plan confound ({mode}):\n{plan}"
                    lats, rec = measure(cur, queries, truths[sel], sel)
                    hit, read = buffers_sample(cur, sel, queries[0])
                    op = {"engine": f"acorn_g2pe_mf_{mode}", "sel": sel,
                          "ef": ef, "recall": rec,
                          "med_ms": float(np.median(lats)),
                          "p90_ms": float(np.percentile(lats, 90)),
                          "min_mean_ms": float(lats.min(axis=0).mean()),
                          "buf_hit": hit, "buf_read": read}
                    out["ops"].append(op)
                    print(f"[{mode}] sel={sel}% ef={ef} recall={rec:.3f} "
                          f"med={op['med_ms']:.2f}ms "
                          f"p90={op['p90_ms']:.2f}ms "
                          f"min_mean={op['min_mean_ms']:.2f}ms "
                          f"buf={hit}+{read}", flush=True)
                    with open(args.out, "w") as f:
                        json.dump(out, f, indent=1)
            cur.execute("ROLLBACK")

    out["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
