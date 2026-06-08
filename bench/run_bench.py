#!/usr/bin/env python3
"""Benchmark runner.

Usage:
    python bench/run_bench.py --scenario a --dsn "postgresql://..." [--qdrant http://localhost:6333]
    python bench/run_bench.py --scenario all --dsn "postgresql://..."

Scenarios:
    a  selectivity sweep
    b  post-filter recall degradation (pgvector only)
    c  incremental insert recall
    d  filter-query correlation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from targets.pgvector import PgvectorTarget
from targets.pg_acorn import PgAcornTarget
from targets.qdrant import QdrantTarget
import scenarios.a_selectivity_sweep as a_selectivity_sweep
import scenarios.b_postfilter_recall as b_postfilter_recall
import scenarios.c_incremental_recall as c_incremental_recall
import scenarios.d_correlation as d_correlation
from fixtures.sift1m_subset import load as load_sift
from fixtures.synthetic import load as load_synthetic


def build_ground_truth(pgvector: PgvectorTarget,
                       queries: np.ndarray,
                       selectivities: list[int],
                       k: int) -> dict:
    """Brute-force ground truth via seq scan (disable index scans)."""
    truth: dict = {}
    with pgvector.conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off")
        cur.execute("SET enable_bitmapscan = off")
        for sel in selectivities:
            for i, q in enumerate(queries):
                cur.execute(
                    "SELECT id FROM bench_items WHERE bucket < %s "
                    "ORDER BY embedding <=> %s::vector LIMIT %s",
                    (sel, q.tolist(), k),
                )
                truth[(sel, i)] = [r[0] for r in cur.fetchall()]
        cur.execute("RESET enable_indexscan")
        cur.execute("RESET enable_bitmapscan")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="a",
                        choices=["a", "b", "c", "d", "all"])
    parser.add_argument("--dsn", default="postgresql://localhost/postgres")
    parser.add_argument("--qdrant", default=None,
                        help="Qdrant base URL (omit to skip)")
    parser.add_argument("--output", default="bench/results.json")
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--n-vectors", type=int, default=50_000)
    parser.add_argument("--n-queries", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading fixtures (dim={args.dim}, n={args.n_vectors})...")
    vectors, metadata = load_sift(n=args.n_vectors, dim=args.dim)
    queries = vectors[:args.n_queries]

    targets = [
        PgvectorTarget(args.dsn),
        PgAcornTarget(args.dsn, tier=1, gamma=1),
        PgAcornTarget(args.dsn, tier=2, gamma=1),
        PgAcornTarget(args.dsn, tier=2, gamma=2),
    ]
    if args.qdrant:
        targets.append(QdrantTarget(args.qdrant))

    all_results: dict = {}
    run_scenarios = (
        ["a", "b", "c", "d"] if args.scenario == "all" else [args.scenario]
    )

    for target in targets:
        print(f"\n[{target.name}] setting up...")
        target.setup(vectors, metadata)

        pgvector_ref = PgvectorTarget(args.dsn)

        truth = build_ground_truth(
            pgvector_ref, queries,
            selectivities=a_selectivity_sweep.SELECTIVITIES,
            k=a_selectivity_sweep.K,
        )
        target_results: dict = {}

        if "a" in run_scenarios:
            print(f"[{target.name}] scenario A: selectivity sweep")
            target_results["a"] = a_selectivity_sweep.run(target, queries, truth)

        if "b" in run_scenarios and target.name == "pgvector":
            print(f"[{target.name}] scenario B: post-filter recall")
            target_results["b"] = b_postfilter_recall.run(
                pgvector_ref.conn, queries, truth
            )

        if "c" in run_scenarios and target.name != "qdrant":
            print(f"[{target.name}] scenario C: incremental insert recall")
            inc_vectors, inc_meta = load_sift(n=args.n_vectors // 5, dim=args.dim)

            def brute_truth(q, sel):
                with pgvector_ref.conn.cursor() as cur:
                    cur.execute("SET enable_indexscan=off; SET enable_bitmapscan=off")
                    cur.execute(
                        "SELECT id FROM bench_items WHERE bucket<%s "
                        "ORDER BY embedding<->%s LIMIT 10",
                        (sel, q.tolist()),
                    )
                    ids = [r[0] for r in cur.fetchall()]
                    cur.execute("RESET enable_indexscan; RESET enable_bitmapscan")
                    return ids

            target_results["c"] = c_incremental_recall.run(
                target, queries, inc_vectors, inc_meta, brute_truth
            )

        if "d" in run_scenarios:
            print(f"[{target.name}] scenario D: correlation")
            syn_low, syn_low_meta = load_synthetic(
                n=args.n_vectors, dim=args.dim, correlation="low"
            )
            syn_high, syn_high_meta = load_synthetic(
                n=args.n_vectors, dim=args.dim, correlation="high"
            )
            target.teardown()
            target.setup(syn_low, syn_low_meta)
            truth_low = build_ground_truth(
                pgvector_ref, queries[:d_correlation.N_QUERIES],
                [d_correlation.SELECTIVITY], d_correlation.K,
            )
            r_low = d_correlation.run(target, queries, truth_low, "low")

            target.teardown()
            target.setup(syn_high, syn_high_meta)
            truth_high = build_ground_truth(
                pgvector_ref, queries[:d_correlation.N_QUERIES],
                [d_correlation.SELECTIVITY], d_correlation.K,
            )
            r_high = d_correlation.run(target, queries, truth_high, "high")
            target_results["d"] = {"low": r_low, "high": r_high}

        all_results[target.name] = target_results
        target.teardown()
        target.close()
        pgvector_ref.close()

    Path(args.output).write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
