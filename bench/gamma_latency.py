"""Clean current-binary acorn latency on the correlated fixture vs Qdrant.

Closes the open Phase D question: recall is now known comparable to Qdrant
(see results_gamma_sweep.json); this measures LATENCY for acorn gamma=2/3/4
(non-inline + code cache ON + prefetch — acorn's shipping config) so the
acorn-vs-Qdrant verdict can be made at matched recall.

Same correlated fixture/queries/truth as the Qdrant rematch.  Latency is
host-sensitive: run with the co-tenant paused.  median + min over PASSES.

Run inside the bench postgres container:
  python3 -u /workspace/bench/gamma_latency.py
"""
import json
import os
import time

import numpy as np
import psycopg

from thesis_validation import K, make_fixture, exact_truth, qstr, SQL

DSN = os.environ.get("PG_DSN",
                     "host=/var/run/postgresql dbname=bench user=postgres")
OUT = os.path.join(os.path.dirname(__file__), "results_gamma_latency.json")
SELS = [1, 2, 5, 10, 20]
EFS = [100, 200, 400, 800, 1600]
PASSES = 3
GAMMAS = {2: "tv_acorn_noinline", 3: "tv_acorn_g3", 4: "tv_acorn_g4"}
ALL_ACORN = ["tv_acorn_idx", "tv_acorn_noinline", "tv_acorn_g3", "tv_acorn_g4"]


def measure(cur, queries, truths, sel):
    # prewarm pass (also triggers the code-cache load on the first scan)
    for q in queries:
        cur.execute(SQL, (int(sel), qstr(q), K))
        cur.fetchall()
    lats = np.empty((PASSES, len(queries)))
    recs = []
    for p in range(PASSES):
        for qi, q in enumerate(queries):
            t0 = time.perf_counter()
            cur.execute(SQL, (int(sel), qstr(q), K))
            ids = {r[0] for r in cur.fetchall()}
            lats[p, qi] = (time.perf_counter() - t0) * 1e3
            if p == 0:
                recs.append(len(ids & truths[sel][qi]) / K)
    return float(np.median(lats)), float(lats.min(axis=0).mean()), float(np.mean(recs))


def main():
    print("[fixture] correlated seed0 ...", flush=True)
    vecs, buckets, queries = make_fixture()
    truths = {s: [exact_truth(vecs, buckets, q, s) for q in queries] for s in SELS}

    conn = psycopg.connect(DSN, autocommit=True, prepare_threshold=0)
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_bitmapscan = off")
    cur.execute("SET enable_sort = off")
    cur.execute("SET pg_acorn.member_first = on")
    cur.execute("SET pg_acorn.scan_inline_vectors = on")
    cur.execute("SET pg_acorn.scan_code_cache = on")   # shipping config

    out = {"meta": {"fixture": "correlated seed0 250K", "k": K, "passes": PASSES,
                    "config": "non-inline + code cache ON + prefetch",
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "results": {}}
    for gamma, name in GAMMAS.items():
        out["results"][str(gamma)] = {}
        cur.execute("BEGIN")
        for idx in ALL_ACORN:
            if idx != name:
                cur.execute(f"DROP INDEX IF EXISTS {idx}")
        # plan check: this index must serve
        cur.execute("EXPLAIN (FORMAT TEXT) " + SQL, (10, qstr(queries[0]), K))
        plan = "\n".join(r[0] for r in cur.fetchall())
        assert f"using {name}" in plan and "Index Cond" in plan, \
            f"plan confound for {name}:\n{plan}"
        for sel in SELS:
            cells = []
            for ef in EFS:
                cur.execute(f"SET pg_acorn.ef_search = {ef}")
                med, mn, rec = measure(cur, queries, truths, sel)
                cells.append({"ef": ef, "recall": round(rec, 4),
                              "med_ms": round(med, 2), "min_ms": round(mn, 2)})
                print(f"[g{gamma} sel={sel}% ef={ef}] r={rec:.3f} "
                      f"med={med:.1f}ms min={mn:.1f}ms", flush=True)
            out["results"][str(gamma)][str(sel)] = cells
        cur.execute("ROLLBACK")
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    conn.close()
    print(f"\n[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
