"""payload_m success metric: does an independent payload half reach correlated
recall at lower latency than raising gamma?

Three non-inline acorn_hnsw indexes on the correlated tv_items fixture (same
build_seed), code cache ON (shipping config):
  g2p0  = gamma2, payload_m=0   -> L0 = 32 global + 32 payload  (symmetric baseline)
  g2p64 = gamma2, payload_m=64  -> L0 = 32 global + 64 payload  (the new lever)
  g4    = gamma4, payload_m=0   -> L0 = 64 global + 64 payload  (raise-gamma baseline)

Hypothesis: g2p64 has the SAME payload connectivity (64) as g4 but HALF the
global half (32 vs 64) and fewer total L0 neighbors (96 vs 128) -> cheaper per
expansion -> reaches the same correlated recall at LOWER latency.  Compare vs
g4 (results_gamma_latency.json) and Qdrant (results_qdrant_rematch.json).

Recall is deterministic; latency needs a quiet host (pause the co-tenant).
Run inside the bench postgres container:
  python3 -u /workspace/bench/payload_m_sweep.py
"""
import json
import os
import time

import numpy as np
import psycopg

from thesis_validation import K, make_fixture, exact_truth, qstr, SQL

DSN = os.environ.get("PG_DSN",
                     "host=/var/run/postgresql dbname=bench user=postgres")
OUT = os.path.join(os.path.dirname(__file__), "results_payload_m_sweep.json")
SELS = [1, 2, 5, 10, 20]
EFS = [100, 200, 400, 800, 1600]
PASSES = 3
# name -> (gamma, payload_m)
VARIANTS = {
    "g2p0":  (2, 0),
    "g2p64": (2, 64),
    "g4":    (4, 0),
}
NAMES = {k: f"tv_pm_{k}" for k in VARIANTS}
DROP_OTHERS = ["tv_acorn_idx", "tv_acorn_noinline"] + list(NAMES.values())


def build(cur, key):
    name = NAMES[key]
    gamma, pm = VARIANTS[key]
    cur.execute(f"SELECT count(*) FROM pg_class WHERE relname='{name}'")
    if cur.fetchone()[0]:
        print(f"[build] {name} exists, skip", flush=True)
        return
    print(f"[build] {name} gamma={gamma} payload_m={pm} ...", flush=True)
    cur.execute("SET maintenance_work_mem='2GB'")
    cur.execute("SET max_parallel_maintenance_workers=4")
    cur.execute("SET pg_acorn.build_seed=42")
    cur.execute("ALTER TABLE tv_items SET (parallel_workers=4)")
    cur.execute(
        f"CREATE INDEX {name} ON tv_items "
        f"USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops) "
        f"WITH (m=16, ef_construction=64, acorn_gamma={gamma}, "
        f"acorn_payload_edges=true, acorn_inline_vectors=false, "
        f"acorn_payload_m={pm})")
    cur.execute(f"SELECT pg_size_pretty(pg_relation_size('{name}'))")
    print(f"[build] {name} done size={cur.fetchone()[0]}", flush=True)


def measure(cur, queries, truths, sel):
    for q in queries:                       # prewarm (loads code cache)
        cur.execute(SQL, (int(sel), qstr(q), K)); cur.fetchall()
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
    for key in VARIANTS:
        build(cur, key)

    cur.execute("SET enable_seqscan=off")
    cur.execute("SET enable_bitmapscan=off")
    cur.execute("SET enable_sort=off")
    cur.execute("SET pg_acorn.member_first=on")
    cur.execute("SET pg_acorn.scan_inline_vectors=on")
    cur.execute("SET pg_acorn.scan_code_cache=on")

    out = {"meta": {"fixture": "correlated seed0 250K", "k": K, "passes": PASSES,
                    "variants": {k: {"gamma": v[0], "payload_m": v[1]}
                                 for k, v in VARIANTS.items()},
                    "config": "non-inline + code cache ON + prefetch"},
           "results": {}}
    for key in VARIANTS:
        keep = NAMES[key]
        out["results"][key] = {}
        cur.execute("BEGIN")
        for idx in DROP_OTHERS:
            if idx != keep:
                cur.execute(f"DROP INDEX IF EXISTS {idx}")
        cur.execute("EXPLAIN (FORMAT TEXT) " + SQL, (10, qstr(queries[0]), K))
        plan = "\n".join(r[0] for r in cur.fetchall())
        assert f"using {keep}" in plan and "Index Cond" in plan, f"{key}:\n{plan}"
        for sel in SELS:
            cells = []
            for ef in EFS:
                cur.execute(f"SET pg_acorn.ef_search={ef}")
                med, mn, rec = measure(cur, queries, truths, sel)
                cells.append({"ef": ef, "recall": round(rec, 4),
                              "med_ms": round(med, 2), "min_ms": round(mn, 2)})
                print(f"[{key} sel={sel}% ef={ef}] r={rec:.3f} "
                      f"med={med:.1f} min={mn:.1f}", flush=True)
            out["results"][key][str(sel)] = cells
        cur.execute("ROLLBACK")
        with open(OUT, "w") as f:
            json.dump(out, f, indent=1)
    conn.close()
    print(f"\n[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
