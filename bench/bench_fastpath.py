"""Fastpath micro-benchmark for the tier2 (acorn_hnsw AM) scan path.

Measures QPS, recall@10 and pages/query (EXPLAIN ANALYZE BUFFERS) at
mid-selectivity (~42%) for ef_search in [100, 400], attributing gains to each
constant-factor optimization (fmgr bypass, prefetch, single-read).

Measurement design (the host docker VM is shared with another bench stack, so
wall-clock between sessions is very noisy):
  - Every optimization is behind a pg_acorn.scan_* GUC, so ONE binary serves
    all configs.  Configs are measured INTERLEAVED within the same session
    (per repeat: config A queries, config B queries, ...), and the per-query
    metric is the MIN latency over repeats — contention bursts hit all
    configs alike and the min approaches the uncontended floor.
  - The table + index are built ONCE and persisted (PGDATA on a docker
    volume): acorn_build seeds levels from MyProcPid, so a rebuild would
    change the graph and invalidate cross-stage comparisons.
  - Identity: every config must return IDENTICAL ordered per-query id lists,
    both across configs and vs the saved baseline stage file (produced by the
    pre-optimization binary at commit a392fcc).

Usage: python3 bench/bench_fastpath.py --stage baseline [--dsn DSN]
"""

import argparse
import json
import os
import time

import numpy as np
import psycopg

N, DIM, NQ, K = 30000, 128, 50, 10
GAMMA = 2
SEL_BUCKET = 40            # bucket < 40 -> ~41.8% with high-correlation buckets
EFS = [100, 400]
REPEATS = 15               # interleaved timed passes; per-query min is the metric
TABLE = "fastpath_items"
BASELINE_FILE = os.path.join(os.path.dirname(__file__), "results_fastpath_baseline.json")

# Cumulative toggle order — each entry adds one optimization on top of the
# previous config.  Only toggles that exist in this binary are used.
# scan_prefetch goes LAST: it regresses the warm-cache path (default off),
# so it must not sit inside the chain and pollute later attributions.
TOGGLE_ORDER = ["scan_direct_dist", "scan_single_read",
                "scan_visited_oneprobe", "scan_direct_filter",
                "scan_prefetch"]


def make_data():
    """correlation='high' replica of bench/fixtures/synthetic.py (seed=42)."""
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((N, DIM)).astype(np.float32)
    vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    blocks = DIM // 10
    block_norms = np.array([
        np.linalg.norm(vectors[:, i * 10:(i + 1) * 10], axis=1)
        for i in range(blocks)
    ]).T
    dominant_block = np.argmax(block_norms, axis=1)
    buckets = (dominant_block * (100 // blocks)).astype(int)
    buckets = np.clip(buckets, 0, 99)
    return vectors, buckets


def vstr(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def ensure_table(conn, vectors, buckets):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")
        cur.execute(
            "SELECT (SELECT count(*) FROM pg_class WHERE relname = %s), "
            "(SELECT count(*) FROM pg_class WHERE relname = %s)",
            (TABLE, TABLE + "_idx"))
        has_table, has_index = cur.fetchone()
        if has_table and has_index:
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            if cur.fetchone()[0] == N:
                print("reusing persisted table + index", flush=True)
                return
            print("row count mismatch -- rebuilding", flush=True)
        cur.execute(f"DROP TABLE IF EXISTS {TABLE} CASCADE")
        cur.execute(f"CREATE TABLE {TABLE} "
                    f"(id serial PRIMARY KEY, bucket int, embedding vector({DIM}))")
        print(f"loading {N} rows...", flush=True)
        with cur.copy(f"COPY {TABLE} (bucket, embedding) FROM STDIN") as cp:
            for i in range(N):
                cp.write_row((int(buckets[i]), vstr(vectors[i])))
        print(f"building acorn_hnsw gamma={GAMMA}... (one-time; persisted)", flush=True)
        t0 = time.perf_counter()
        cur.execute(f"""CREATE INDEX {TABLE}_idx ON {TABLE}
                        USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
                        WITH (m = 16, ef_construction = 64, acorn_gamma = {GAMMA})""")
        print(f"index built in {time.perf_counter() - t0:.1f}s", flush=True)


def exact_truth(vectors, buckets, q):
    idx = np.where(buckets < SEL_BUCKET)[0]
    sims = vectors[idx] @ q
    top = idx[np.argsort(-sims)[:K]]
    return set((top + 1).tolist())            # serial ids start at 1


SQL = (f"SELECT id FROM {TABLE} WHERE bucket < %s "
       f"ORDER BY embedding <=> %s::vector LIMIT %s")

# --- Tier 1 leg: pgvector hnsw index + AcornScan CustomScan hook -----------

T1_TABLE = "fastpath_t1"
T1_SQL = (f"SELECT id FROM {T1_TABLE} WHERE bucket < %s "
          f"ORDER BY embedding <=> %s::vector LIMIT %s")


def ensure_t1(conn, vectors, buckets):
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM pg_class WHERE relname = %s",
                    (T1_TABLE + "_idx",))
        if cur.fetchone()[0]:
            cur.execute(f"SELECT count(*) FROM {T1_TABLE}")
            if cur.fetchone()[0] == N:
                print("reusing persisted t1 table + hnsw index", flush=True)
                return
        cur.execute(f"DROP TABLE IF EXISTS {T1_TABLE} CASCADE")
        cur.execute(f"CREATE TABLE {T1_TABLE} "
                    f"(id serial PRIMARY KEY, bucket int, embedding vector({DIM}))")
        print(f"loading {N} rows into {T1_TABLE}...", flush=True)
        with cur.copy(f"COPY {T1_TABLE} (bucket, embedding) FROM STDIN") as cp:
            for i in range(N):
                cp.write_row((int(buckets[i]), vstr(vectors[i])))
        print("building pgvector hnsw index... (one-time; persisted)", flush=True)
        cur.execute(f"""CREATE INDEX {T1_TABLE}_idx ON {T1_TABLE}
                        USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = 64)""")


def run_t1(conn, vectors, buckets, queries, configs):
    """Tier 1 (AcornScan CustomScan over pgvector hnsw) leg.  Only the
    direct-dist and single-read toggles touch this path."""
    truth = [exact_truth(vectors, buckets, q) for q in queries]
    qstrs = [vstr(q) for q in queries]
    out = {}

    def t1_config(cur, gucs):
        cur.execute("SET enable_seqscan = off")
        # The AcornScan custom path claims NIL pathkeys (planner adds a Sort),
        # so the ordered hnsw Index Scan path always wins under LIMIT unless
        # index scans are disabled.  This leg measures the t1 hook executor,
        # not plan choice.
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET pg_acorn.enable_hook = on")
        for t, val in gucs.items():
            cur.execute(f"SET pg_acorn.{t} = {'on' if val else 'off'}")

    with conn.cursor() as cur:
        for cname, gucs in configs:
            t1_config(cur, gucs)
            pages, provider = [], None
            for qs in qstrs:
                cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + T1_SQL,
                            (SEL_BUCKET, qs, K))
                plan = cur.fetchone()[0][0]["Plan"]
                node = plan
                while node["Node Type"] in ("Limit", "Sort", "Incremental Sort",
                                            "Gather", "Result"):
                    node = node["Plans"][0]
                provider = node.get("Custom Plan Provider")
                assert node["Node Type"] == "Custom Scan" and provider == "AcornScan", \
                    f"t1 plan confound: {node['Node Type']} / {provider}"
                pages.append(plan["Shared Hit Blocks"] + plan["Shared Read Blocks"])

            ids_per_query, recalls = [], []
            for qs, t in zip(qstrs, truth):
                cur.execute(T1_SQL, (SEL_BUCKET, qs, K))
                ids = [r[0] for r in cur.fetchall()]
                ids_per_query.append(ids)
                recalls.append(len(set(ids) & t) / K)
            out[cname] = {
                "gucs": gucs,
                "pages_per_query": float(np.mean(pages)),
                "recall": float(np.mean(recalls)),
                "ids_per_query": ids_per_query,
            }

        lat = {cname: np.empty((REPEATS, NQ)) for cname, _ in configs}
        for cname, gucs in configs:        # warmup
            t1_config(cur, gucs)
            for qs in qstrs:
                cur.execute(T1_SQL, (SEL_BUCKET, qs, K))
                cur.fetchall()
        for r in range(REPEATS):
            for cname, gucs in configs:
                t1_config(cur, gucs)
                L = lat[cname]
                for qi, qs in enumerate(qstrs):
                    t0 = time.perf_counter()
                    cur.execute(T1_SQL, (SEL_BUCKET, qs, K))
                    cur.fetchall()
                    L[r, qi] = time.perf_counter() - t0

    for cname, _ in configs:
        min_lat = lat[cname].min(axis=0)
        med_lat = np.median(lat[cname], axis=0)
        out[cname]["qps_min"] = float(NQ / min_lat.sum())
        out[cname]["lat_ms_p50"] = float(np.median(med_lat) * 1e3)
        out[cname]["lat_ms_p90"] = float(np.quantile(med_lat, 0.9) * 1e3)

    ref = out[configs[0][0]]["ids_per_query"]
    for cname, _ in configs[1:]:
        assert out[cname]["ids_per_query"] == ref, \
            f"T1 TRAVERSAL CHANGED: config {cname} differs from {configs[0][0]}"
    return out


def detect_configs(cur):
    """Cumulative configs from the scan_* toggles present in this binary."""
    cur.execute("SELECT name FROM pg_settings WHERE name LIKE 'pg_acorn.scan%'")
    present = {r[0].split(".", 1)[1] for r in cur.fetchall()}
    avail = [t for t in TOGGLE_ORDER if t in present]
    configs = [("alloff", {t: False for t in avail})]
    state = {t: False for t in avail}
    for t in avail:
        state = dict(state)
        state[t] = True
        configs.append(("+" + t, dict(state)))
    return configs


def apply_config(cur, ef, gucs):
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET pg_acorn.enable_hook = off")
    cur.execute(f"SET pg_acorn.ef_search = {ef}")
    for t, val in gucs.items():
        cur.execute(f"SET pg_acorn.{t} = {'on' if val else 'off'}")


def run_ef(conn, vectors, buckets, queries, ef, configs):
    truth = [exact_truth(vectors, buckets, q) for q in queries]
    qstrs = [vstr(q) for q in queries]
    out = {}

    with conn.cursor() as cur:
        # --- per-config: plan + buffers, identity, recall ---
        for cname, gucs in configs:
            apply_config(cur, ef, gucs)
            pages, plan_node, index_name = [], None, None
            for qs in qstrs:
                cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + SQL,
                            (SEL_BUCKET, qs, K))
                plan = cur.fetchone()[0][0]["Plan"]
                node = plan
                while node["Node Type"] in ("Limit", "Sort", "Gather"):
                    node = node["Plans"][0]
                plan_node = node["Node Type"]
                index_name = node.get("Index Name")
                assert plan_node == "Index Scan" and index_name == TABLE + "_idx", \
                    f"plan confound: {plan_node} / {index_name}"
                pages.append(plan["Shared Hit Blocks"] + plan["Shared Read Blocks"])

            ids_per_query, recalls = [], []
            for qs, t in zip(qstrs, truth):
                cur.execute(SQL, (SEL_BUCKET, qs, K))
                ids = [r[0] for r in cur.fetchall()]
                ids_per_query.append(ids)
                recalls.append(len(set(ids) & t) / K)

            out[cname] = {
                "gucs": gucs,
                "pages_per_query": float(np.mean(pages)),
                "plan_node": plan_node,
                "recall": float(np.mean(recalls)),
                "ids_per_query": ids_per_query,
            }

        # --- interleaved timed passes ---
        lat = {cname: np.empty((REPEATS, NQ)) for cname, _ in configs}
        # warmup
        for cname, gucs in configs:
            apply_config(cur, ef, gucs)
            for qs in qstrs:
                cur.execute(SQL, (SEL_BUCKET, qs, K))
                cur.fetchall()
        for r in range(REPEATS):
            for cname, gucs in configs:
                apply_config(cur, ef, gucs)
                L = lat[cname]
                for qi, qs in enumerate(qstrs):
                    t0 = time.perf_counter()
                    cur.execute(SQL, (SEL_BUCKET, qs, K))
                    cur.fetchall()
                    L[r, qi] = time.perf_counter() - t0

    for cname, _ in configs:
        min_lat = lat[cname].min(axis=0)
        med_lat = np.median(lat[cname], axis=0)   # per-query median over repeats
        out[cname]["qps_min"] = float(NQ / min_lat.sum())
        out[cname]["lat_ms_min_mean"] = float(min_lat.mean() * 1e3)
        out[cname]["lat_ms_p50"] = float(np.median(med_lat) * 1e3)
        out[cname]["lat_ms_p90"] = float(np.quantile(med_lat, 0.9) * 1e3)
        out[cname]["qps_median"] = float(NQ / med_lat.sum())

    # identity across configs
    ref = out[configs[0][0]]["ids_per_query"]
    for cname, _ in configs[1:]:
        assert out[cname]["ids_per_query"] == ref, \
            f"TRAVERSAL CHANGED: config {cname} differs from {configs[0][0]} at ef={ef}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="baseline | fmgr_bypass | prefetch | single_read | ...")
    ap.add_argument("--dsn", default=os.environ.get(
        "DSN", "postgresql://postgres@localhost:5432/postgres"))
    args = ap.parse_args()

    vectors, buckets = make_data()
    actual_sel = float((buckets < SEL_BUCKET).mean())
    queries = vectors[:NQ]

    conn = psycopg.connect(args.dsn, autocommit=True)
    ensure_table(conn, vectors, buckets)

    with conn.cursor() as cur:
        configs = detect_configs(cur)
    print(f"configs: {[c for c, _ in configs]}", flush=True)

    results = {"stage": args.stage, "n": N, "dim": DIM, "nq": NQ, "k": K,
               "gamma": GAMMA, "selectivity": actual_sel, "repeats": REPEATS,
               "efs": {}}
    print(f"\n=== stage={args.stage}  sel={actual_sel:.3f}  ef={EFS} ===", flush=True)
    for ef in EFS:
        r = run_ef(conn, vectors, buckets, queries, ef, configs)
        results["efs"][str(ef)] = r
        for cname, _ in configs:
            c = r[cname]
            print(f"  ef={ef:>4} {cname:<22} qps_min={c['qps_min']:8.1f}  "
                  f"p50={c['lat_ms_p50']:7.3f}ms  p90={c['lat_ms_p90']:7.3f}ms  "
                  f"recall={c['recall']:.3f}  pages/q={c['pages_per_query']:8.1f}",
                  flush=True)

    # --- Tier 1 leg (only the toggles that touch the t1 path) ---
    t1_configs = [c for c in configs if c[0] in
                  ("alloff", "+scan_direct_dist", "+scan_single_read")]
    if len(t1_configs) > 1:
        ensure_t1(conn, vectors, buckets)
        t1 = run_t1(conn, vectors, buckets, queries, t1_configs)
        results["t1"] = t1
        for cname, _ in t1_configs:
            c = t1[cname]
            print(f"  t1      {cname:<22} qps_min={c['qps_min']:8.1f}  "
                  f"p50={c['lat_ms_p50']:7.3f}ms  p90={c['lat_ms_p90']:7.3f}ms  "
                  f"recall={c['recall']:.3f}  pages/q={c['pages_per_query']:8.1f}",
                  flush=True)

    # --- traversal identity vs saved baseline stage (pre-optimization binary) ---
    if args.stage != "baseline" and os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE) as f:
            base = json.load(f)
        for ef in EFS:
            b = base["efs"][str(ef)]
            bids = b.get("ids_per_query") or b["alloff"]["ids_per_query"]
            for cname, _ in configs:
                r = results["efs"][str(ef)][cname]
                assert r["ids_per_query"] == bids, \
                    f"TRAVERSAL CHANGED vs baseline binary: {cname} at ef={ef}"
        print("identity check vs baseline binary: PASS", flush=True)

    out = os.path.join(os.path.dirname(__file__),
                       f"results_fastpath_{args.stage}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
