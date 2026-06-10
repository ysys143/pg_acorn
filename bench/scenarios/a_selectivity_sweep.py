"""Scenario A: filter selectivity sweep — 1% to 80%.

Measures recall@10 and QPS for each target across selectivity levels.
ACORN should maintain recall >= 0.9 at all levels; pgvector recall collapses
at high selectivity.
"""

from __future__ import annotations

import time
import numpy as np


SELECTIVITIES = [1, 5, 10, 40, 80]  # percent — bucket < N gives N% selectivity
EF_SEARCHES = [10, 40, 100, 200, 400, 800, 1600]  # recall/QPS knob swept per selectivity
K = 10
N_QUERIES = 100
N_EXPLAIN = 20  # queries sampled for page-I/O (EXPLAIN BUFFERS), outside timing


def compute_recall(result_ids: list[int], truth_ids: list[int]) -> float:
    return len(set(result_ids) & set(truth_ids)) / len(truth_ids)


def brute_force(target, query: np.ndarray, bucket_threshold: int, k: int) -> list[int]:
    """Ground truth via seq scan (caller must disable index scans externally)."""
    return target.query_filtered(query, bucket_threshold, k)


def run(target, queries: np.ndarray, ground_truth: dict, force_index: bool = True) -> dict:
    results = {}

    # force_index=True pins each target on its vector index (apples-to-apples
    # index comparison). force_index=False (--free-planner) lets the planner
    # choose freely — e.g. a bitmap prefilter on the filter column at high
    # selectivity — which is the realistic, production behavior and the regime
    # where the exact-fallback lever shows up. Ground truth uses a separate
    # connection and is unaffected.
    force = getattr(target, "force_index_scan", None) if force_index else None
    if force:
        force(True)
    try:
        return _run_inner(target, queries, ground_truth, results)
    finally:
        if force:
            force(False)


def _run_inner(target, queries: np.ndarray, ground_truth: dict, results: dict) -> dict:
    # ef_search is the recall/QPS knob. Targets that expose set_ef_search (pgvector,
    # pg_acorn tier2, qdrant) are swept; targets without it (none currently) yield a
    # single point under ef=None.
    set_ef = getattr(target, "set_ef_search", None)
    ef_values = EF_SEARCHES if set_ef else [None]

    for sel in SELECTIVITIES:
        results[sel] = {}
        for ef in ef_values:
            if set_ef and ef is not None:
                set_ef(ef)

            recalls, latencies = [], []
            for i, q in enumerate(queries[:N_QUERIES]):
                t0 = time.perf_counter()
                ids = target.query_filtered(q, sel, K)
                latencies.append(time.perf_counter() - t0)

                truth = ground_truth[(sel, i)]
                recalls.append(compute_recall(ids, truth))

            point = {
                "recall_mean": float(np.mean(recalls)),
                "recall_min":  float(np.min(recalls)),
                "qps":         N_QUERIES / sum(latencies),
                "p99_ms":      float(np.percentile(latencies, 99) * 1000),
            }
            # page-I/O: separate, non-timed sample pass (EXPLAIN ANALYZE executes
            # the query, so it must stay out of the wall-clock loop above).
            point.update(_measure_page_io(target, queries, sel))
            results[sel][ef] = point

    return results


def _measure_page_io(target, queries: np.ndarray, sel: int) -> dict:
    """Sample per-query logical page accesses (shared hit + read).

    Returns None-valued fields for targets without EXPLAIN support (e.g. Qdrant).
    """
    explain = getattr(target, "explain_filtered", None)
    if explain is None:
        return {"pages_total_mean": None, "pages_hit_mean": None,
                "pages_read_mean": None, "plan": None}

    io = [explain(q, sel, K) for q in queries[:min(N_EXPLAIN, N_QUERIES)]]
    io = [x for x in io if x]
    if not io:
        return {"pages_total_mean": None, "pages_hit_mean": None,
                "pages_read_mean": None, "plan": None}

    plans = sorted({x.get("plan", "?") for x in io})
    return {
        "pages_total_mean": float(np.mean([x["pages_total"] for x in io])),
        "pages_hit_mean":   float(np.mean([x["pages_hit"]   for x in io])),
        "pages_read_mean":  float(np.mean([x["pages_read"]  for x in io])),
        "plan": ",".join(plans),   # scan node(s) actually used (should be index)
    }
