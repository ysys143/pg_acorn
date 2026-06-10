"""Validate acorn_payload_edges against the gamma-only baseline.

Self-contained (modeled on diag_ef_ceiling.py): generates CORRELATED data
(the correlation='high' logic from bench/fixtures/synthetic.py, refined so
buckets spread uniformly over 0..99 within each spatial block), computes
exact ground truth with numpy, and sweeps pg_acorn.ef_search for three
index configurations:

    A. acorn_gamma=2, payload_edges=off   (current best baseline)
    B. acorn_gamma=1, payload_edges=on    (payload edges replace gamma)
    C. acorn_gamma=2, payload_edges=on    (combined)

at selectivity 1% (bucket < 1) and 40% (bucket < 40).

Target: payload_edges=on reaches recall >= 0.95 at materially lower ef /
higher QPS than off.

DSN comes from argv[1] or PG_ACORN_DSN (default: local socket, as inside the
self-contained test container).  Do NOT point this at the reserved
pg_acorn_bench compose stack.
"""

import os
import sys
import time

import numpy as np
import psycopg

DSN = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
    "PG_ACORN_DSN", "postgresql://postgres@localhost/postgres")

N, DIM, NQ, K = 20000, 128, 30, 10
EFS = [40, 100, 200, 400]
SELS = [1, 40]                       # bucket < sel  =>  ~sel% selectivity
CONFIGS = [
    ("g2_off", 2, "false"),
    ("g1_on",  1, "true"),
    ("g2_on",  2, "true"),
]

# --- correlated fixture (synthetic.py correlation='high', bucket spread 0..99) ---
rng = np.random.default_rng(0)
raw = rng.standard_normal((N, DIM)).astype(np.float32)
vecs = raw / np.linalg.norm(raw, axis=1, keepdims=True)

blocks = DIM // 10                                   # 12 blocks for dim=128
block_norms = np.array([
    np.linalg.norm(vecs[:, i * 10:(i + 1) * 10], axis=1)
    for i in range(blocks)
]).T                                                  # (n, blocks)
dominant_block = np.argmax(block_norms, axis=1)       # 0..blocks-1
span = 100 // blocks                                  # 8
# base bucket from the dominant block (spatial correlation) + uniform spread
# inside the block's range so bucket < k yields ~k% selectivity for any k.
buckets = dominant_block * span + rng.integers(0, span, size=N)
buckets = np.clip(buckets, 0, 99).astype(int)

queries = vecs[rng.choice(N, NQ, replace=False)]


def exact_truth(q, thresh, k):
    idx = np.where(buckets < thresh)[0]
    sims = vecs[idx] @ q                               # cosine (unit vectors)
    top = idx[np.argsort(-sims)[:k]]
    return set((top + 1).tolist())                     # serial ids start at 1


def qstr(q):
    return "[" + ",".join(f"{x:.6f}" for x in q) + "]"


conn = psycopg.connect(DSN, autocommit=True)
with conn.cursor() as cur:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_acorn")
    cur.execute("DROP TABLE IF EXISTS pe_items CASCADE")
    cur.execute(f"CREATE TABLE pe_items (id serial PRIMARY KEY, bucket int, "
                f"embedding vector({DIM}))")
    print(f"loading {N} rows (correlated buckets)...", flush=True)
    with cur.copy("COPY pe_items (bucket, embedding) FROM STDIN") as cp:
        for i in range(N):
            cp.write_row((int(buckets[i]), qstr(vecs[i])))

results = []
for name, gamma, payload in CONFIGS:
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS pe_items_idx")
        print(f"\nbuilding {name} (gamma={gamma}, payload_edges={payload})...",
              flush=True)
        t0 = time.perf_counter()
        cur.execute(f"""CREATE INDEX pe_items_idx ON pe_items
                        USING acorn_hnsw (embedding vector_cosine_ops,
                                          bucket int4_acorn_ops)
                        WITH (m = 16, ef_construction = 64,
                              acorn_gamma = {gamma},
                              acorn_payload_edges = {payload})""")
        print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    for sel in SELS:
        truths = [exact_truth(q, sel, K) for q in queries]
        for ef in EFS:
            recalls, t0 = [], time.perf_counter()
            with conn.cursor() as cur:
                cur.execute("SET enable_seqscan = off")
                cur.execute(f"SET pg_acorn.ef_search = {ef}")
                for q, truth in zip(queries, truths):
                    cur.execute(
                        "SELECT id FROM pe_items WHERE bucket < %s "
                        "ORDER BY embedding <=> %s::vector LIMIT %s",
                        (sel, qstr(q), K))
                    ids = {r[0] for r in cur.fetchall()}
                    recalls.append(len(ids & truth) / K)
            qps = NQ / (time.perf_counter() - t0)
            rec = float(np.mean(recalls))
            results.append((name, sel, ef, rec, qps))
            print(f"  {name:7s} sel={sel:>2}%  ef={ef:>4}  "
                  f"recall={rec:.3f}  qps={qps:.1f}", flush=True)

print("\n=== summary (recall / qps) ===")
hdr = "config   sel%  " + "  ".join(f"ef={e:<14}" for e in EFS)
print(hdr)
for name, _, _ in CONFIGS:
    for sel in SELS:
        cells = []
        for ef in EFS:
            row = next(r for r in results
                       if r[0] == name and r[1] == sel and r[2] == ef)
            cells.append(f"{row[3]:.3f}/{row[4]:7.1f}   ")
        print(f"{name:8s} {sel:>3}   " + "".join(cells))

with conn.cursor() as cur:
    cur.execute("DROP TABLE IF EXISTS pe_items CASCADE")
print("done", flush=True)
