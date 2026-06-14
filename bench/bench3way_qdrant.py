"""3-way benchmark — Qdrant side (httpx client from the host).

Measures Qdrant v1.16 forced-HNSW filtered KNN on the correlated 250K fixture:
recall@10 + single-client latency (median/p95/min) + REAL concurrent throughput
(multiprocessing clients -> aggregate QPS, concurrency swept).  Reuses the load
(setup), fixture, and per-query search from qdrant_rematch.py.

Run from the host (Qdrant on localhost:6333):
  uv run --with numpy --with httpx python3 bench/bench3way_qdrant.py \
    --reload --out bench/results_3way_qdrant.json

The collection persists (unlike qdrant_rematch, which deletes at the end).
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import httpx
import numpy as np

import qdrant_rematch as qr  # make_fixture, exact_truth, setup, search, COLL, N, NQ, K

SELS = [1, 2, 5, 10, 20]
EFS = [40, 100, 200, 400, 800, 1600]
# matched-recall (~0.95) money-cell ef per selectivity (from REPORT_qdrant_final)
MONEY_EF = {1: 100, 2: 100, 5: 200, 10: 400, 20: 800}
CONC = [1, 4, 8, 16]
DUR = 6.0          # throughput steady-state window (s)
WARMUP = 1.0


def latency_sweep(url, queries, truths):
    c = httpx.Client(base_url=url, timeout=600.0)
    out = {}
    for sel in SELS:
        cells = []
        for ef in EFS:
            # prewarm
            qr.search(c, queries[0], sel, ef)
            lats, recs = [], []
            for rep in range(3):
                for i in range(NQ := qr.NQ):
                    t0 = time.perf_counter()
                    got = qr.search(c, queries[i], sel, ef)
                    lats.append((time.perf_counter() - t0) * 1e3)
                    if rep == 0:
                        recs.append(len(got & truths[sel][i]) / qr.K)
            a = np.array(lats)
            cells.append({"ef": ef, "recall": round(float(np.mean(recs)), 4),
                          "med_ms": round(float(np.median(a)), 2),
                          "p95_ms": round(float(np.percentile(a, 95)), 2),
                          "min_ms": round(float(np.min(a)), 2)})
            print(f"  [qdrant lat sel={sel}% ef={ef}] r={cells[-1]['recall']:.3f} "
                  f"med={cells[-1]['med_ms']}ms p95={cells[-1]['p95_ms']}ms", flush=True)
        out[str(sel)] = cells
    c.close()
    return out


def _tp_worker(args):
    url, sel, ef, qlist, end_t = args
    c = httpx.Client(base_url=url, timeout=600.0)
    # warm
    qr.search(c, np.array(qlist[0], dtype=np.float32), sel, ef)
    n = 0
    qi = 0
    while time.time() < end_t:
        qr.search(c, np.array(qlist[qi % len(qlist)], dtype=np.float32), sel, ef)
        qi += 1
        n += 1
    c.close()
    return n


def throughput(url, queries, sel, ef):
    qlist = [q.tolist() for q in queries]
    res = {}
    for nproc in CONC:
        end_t = time.time() + WARMUP + DUR
        # account for warmup by measuring count only over DUR: workers loop till end_t,
        # we approximate QPS over the full (WARMUP+DUR) window then scale — instead,
        # start counting after WARMUP by giving each worker the same end and dividing
        # the total by the wall window actually spent looping.
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            counts = list(ex.map(_tp_worker,
                                 [(url, sel, ef, qlist, end_t)] * nproc))
        wall = time.time() - t0
        total = sum(counts)
        qps = total / wall
        res[str(nproc)] = {"qps": round(qps, 1), "total": total,
                           "wall_s": round(wall, 2)}
        print(f"  [qdrant tp sel={sel}% ef={ef} conc={nproc}] qps={qps:.0f} "
              f"(n={total} in {wall:.1f}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--out", default="bench/results_3way_qdrant.json")
    ap.add_argument("--tp-sels", default="1,10,20")
    args = ap.parse_args()

    print("[fixture] seed0 correlated ...", flush=True)
    vecs, buckets, queries = qr.make_fixture()
    truths = {s: [qr.exact_truth(vecs, buckets, queries[i], s) for i in range(qr.NQ)]
              for s in SELS}

    c = httpx.Client(base_url=args.url, timeout=600.0)
    if args.reload:
        print("[reload] rebuilding Qdrant collection (forced HNSW)...", flush=True)
        qr.setup(c, vecs, buckets)
    # HNSW-engaged gate
    probe = float(np.mean([len(qr.search(c, queries[i], 10, 40) & truths[10][i]) / qr.K
                           for i in range(qr.NQ)]))
    print(f"[GATE] sel=10% ef=40 recall={probe:.3f} "
          f"({'HNSW engaged' if probe < 0.999 else 'STILL EXACT'})", flush=True)
    c.close()

    out = {"meta": {"engine": "qdrant v1.16 HNSW m16 efc64 forced", "n": qr.N,
                    "k": qr.K, "nq": qr.NQ, "fixture": "seed0 correlated 250K",
                    "gate_recall_sel10_ef40": round(probe, 4),
                    "dur_s": DUR, "conc": CONC},
           "latency": latency_sweep(args.url, queries, truths),
           "throughput": {}}
    for sel in [int(x) for x in args.tp_sels.split(",")]:
        out["throughput"][str(sel)] = {"ef": MONEY_EF[sel],
                                       "by_conc": throughput(args.url, queries, sel,
                                                             MONEY_EF[sel])}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
