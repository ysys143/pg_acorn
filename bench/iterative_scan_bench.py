"""pgvector iterative scan vs acorn: head-to-head at 250K.

Compares stock pgvector HNSW with iterative scan (relaxed/strict) against
acorn's buffered-emission results at selectivities 1/2/5/10/20%.

Run inside pgvec_iterative container:
  python3 bench/iterative_scan_bench.py \
    --dsn 'host=localhost dbname=bench user=postgres'
"""
import argparse
import json
import os
import time

import numpy as np
import psycopg

DSN = os.environ.get("PG_DSN", "host=localhost dbname=bench user=postgres")
N, DIM, NQ, K = 250_000, 128, 40, 10
SELS = [1, 2, 5, 10, 20]
EFS = [40, 100, 200, 400, 800, 1600]
RNG_DATA = np.random.default_rng(42)
RNG_QUERY = np.random.default_rng(99)
NWARM = 3


def make_data():
    vecs = RNG_DATA.random((N, DIM), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    buckets = RNG_DATA.integers(0, 100, size=N)  # bucket 0-99
    queries = RNG_QUERY.random((NQ, DIM), dtype=np.float32)
    qnorms = np.linalg.norm(queries, axis=1, keepdims=True)
    queries /= qnorms
    return vecs, buckets, queries


def load_table(conn, vecs, buckets):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS itertest CASCADE")
        cur.execute("""
            CREATE TABLE itertest (
                id serial PRIMARY KEY,
                embedding vector(%s),
                bucket int
            )
        """ % DIM)
        rows = [(vecs[i].tolist(), int(buckets[i])) for i in range(N)]
        with cur.copy("COPY itertest (embedding, bucket) FROM STDIN") as cp:
            for emb, b in rows:
                cp.write_row(("[" + ",".join(f"{x:.6f}" for x in emb) + "]", b))
    conn.commit()
    print(f"[load] {N} rows loaded", flush=True)


def build_hnsw(conn):
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS itertest_hnsw")
        t0 = time.perf_counter()
        cur.execute("""
            CREATE INDEX itertest_hnsw ON itertest
            USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=64)
        """)
    conn.commit()
    print(f"[build] HNSW done in {time.perf_counter()-t0:.1f}s", flush=True)


def exact_topk(cur, q, bucket_lo, bucket_hi):
    cur.execute("""
        SELECT id FROM itertest
        WHERE bucket >= %s AND bucket < %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (bucket_lo, bucket_hi, "[" + ",".join(f"{x:.6f}" for x in q) + "]", K))
    return {r[0] for r in cur.fetchall()}


def recall(got, truth):
    return len(got & truth) / len(truth) if truth else 1.0


def run_query(cur, q, bucket_lo, bucket_hi, ef, mode):
    qvec = "[" + ",".join(f"{x:.6f}" for x in q) + "]"
    cur.execute(f"SET hnsw.ef_search = {ef}")
    if mode == "postfilter":
        cur.execute("RESET hnsw.iterative_scan")
    elif mode == "relaxed":
        cur.execute("SET hnsw.iterative_scan = relaxed_order")
    elif mode == "strict":
        cur.execute("SET hnsw.iterative_scan = strict_order")

    cur.execute("""
        SELECT id FROM itertest
        WHERE bucket >= %s AND bucket < %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (bucket_lo, bucket_hi, qvec, K))
    return {r[0] for r in cur.fetchall()}


def bench_sel(conn, queries, sel, mode):
    bucket_lo = 0
    bucket_hi = sel  # sel% = buckets 0..(sel-1)
    results = []
    with conn.cursor() as cur:
        # ground truth
        truth_sets = [exact_topk(cur, queries[i], bucket_lo, bucket_hi)
                      for i in range(NQ)]
        # warmup
        for i in range(NWARM):
            run_query(cur, queries[i % NQ], bucket_lo, bucket_hi, 40, mode)

        for ef in EFS:
            lats = []
            recs = []
            for i in range(NQ):
                t0 = time.perf_counter()
                got = run_query(cur, queries[i], bucket_lo, bucket_hi, ef, mode)
                lats.append((time.perf_counter() - t0) * 1000)
                recs.append(recall(got, truth_sets[i]))
            med_lat = float(np.median(lats))
            med_rec = float(np.mean(recs))
            results.append({
                "ef": ef,
                "median_ms": round(med_lat, 2),
                "recall": round(med_rec, 4),
            })
            print(f"  [sel={sel}% {mode}] ef={ef}: "
                  f"{med_lat:.1f}ms recall={med_rec:.3f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--out", default="bench/results_iterative_scan.json")
    ap.add_argument("--skip-load", action="store_true")
    args = ap.parse_args()

    print("[data] generating...", flush=True)
    vecs, buckets, queries = make_data()

    conn = psycopg.connect(args.dsn, autocommit=True)

    if not args.skip_load:
        load_table(conn, vecs, buckets)
        build_hnsw(conn)

    res = {
        "meta": {
            "n": N, "dim": DIM, "nq": NQ, "k": K,
            "index": "hnsw m=16 efc=64 vector_cosine_ops",
            "modes": ["postfilter", "relaxed", "strict"],
        },
        "results": {}
    }

    for sel in SELS:
        print(f"\n[sel={sel}%]", flush=True)
        res["results"][str(sel)] = {}
        for mode in ["postfilter", "relaxed", "strict"]:
            res["results"][str(sel)][mode] = bench_sel(conn, queries, sel, mode)

        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)

    conn.close()
    print(f"\n[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
