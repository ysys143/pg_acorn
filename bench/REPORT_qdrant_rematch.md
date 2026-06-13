# Phase D — Qdrant rematch with HNSW forced (250K, correlated fixture)

Date: 2026-06-14. Harness: `bench/qdrant_rematch.py`. Result:
`bench/results_qdrant_rematch.json`. Qdrant v1.16.0 container.

## What this fixes

Prior Qdrant numbers (`QDRANT_CODEPATH.md`) measured exact brute-force — no
segment crossed the per-segment indexing threshold, so no HNSW graph existed
(recall 1.000 everywhere by construction). This run forces filtered HNSW:
`optimizers_config.indexing_threshold = 10` (KB, indexes every segment) and
`hnsw_config.full_scan_threshold = 10` (KB ~ 20 points, so filtered queries
use the graph, not a full scan), m=16 ef_construct=64 to match acorn/pgvector.

HNSW-engaged gate: **sel=10% ef=40 recall = 0.302** (« 1.0). The graph is
genuinely being searched — this is the first valid acorn-vs-Qdrant-HNSW
comparison. Same fixture as `thesis_validation.py` (seed 0, CORRELATED buckets
— the hard case where the filter correlates with vector position), same
queries, same exact ground truth, same `bucket < sel` semantics.

## Matched-recall comparison (recall >= ~0.95 money cell; median ms)

| sel | Qdrant HNSW | acorn g2 in-filter (results_thesis_250k) | verdict |
|----:|---|---|---|
| 1% | ef100 r0.98, 6.5 ms | ef100 r0.97, ~7 ms | ~tie |
| 2% | ef100 r0.99, 6.2 ms | ef200 r0.98, ~57 ms | Qdrant faster |
| 5% | ef200 r0.98, 20 ms | ef400 r0.95, ~148 ms | Qdrant faster |
| 10% | ef400 r0.98, 23 ms; ef800 r1.0, 65 ms | caps ~0.94 (never reaches 0.95) | Qdrant: higher recall AND faster |
| 20% | ef800 r0.97, 139 ms | caps ~0.85-0.91 (never reaches 0.95) | Qdrant: reaches recall acorn cannot |

## Verdict — honest

**Qdrant's filtered HNSW beats acorn on this fixture.** Two distinct gaps:

1. Recall ceiling (the solid finding — recall is deterministic): on the
   correlated fixture acorn's ACORN-gamma in-filter traversal (gamma=2, the
   max the HNSW page budget allows at m=16) caps at ~0.91-0.94 at sel 10/20%
   and cannot reach 0.95+ at any ef. Qdrant reaches 0.97-1.0. This is an
   algorithmic limitation of acorn's in-filter graph on filters that
   correlate with vector position.
2. Latency at matched recall: Qdrant is ~2-9x faster at sel 2-5%. (Caveat: the
   acorn latencies here are from a contention-noisy thesis run and predate the
   cache/prefetch work; the recall-ceiling gap is the cleaner, load-independent
   finding. A clean acorn re-run on the correlated fixture with the current
   optimized binary would tighten the latency numbers but cannot close the
   recall ceiling.)

## What this does and does not change

- Does NOT change the pgvector result: acorn still beats pgvector's prefilter,
  postfilter, and iterative scan IN POSTGRES. That is acorn's value
  proposition — best filtered ANN inside Postgres (transactional, SQL,
  joinable, on-disk), not "beats a dedicated engine."
- Confirms the expected category gap: Qdrant is a dedicated in-memory Rust
  vector engine with no transactional/MVCC/buffer-manager overhead. It being
  faster and higher-recall on pure filtered ANN is the baseline, not a
  surprise.
- Surfaces a genuine acorn limitation worth recording: the in-filter recall
  ceiling on correlated high-selectivity filters. Closing it (richer gamma,
  a different filtered-traversal heuristic like NaviX-Directed / FAVOR, or a
  filter-aware entry-point strategy) is exactly what roadmap Phase 2 targets.

## Caveats

- Qdrant ran from the host (in-memory, no PG layer); acorn runs in PostgreSQL.
  Different cost structures — the latency comparison is indicative, not a
  controlled same-substrate measurement.
- Single fixture (correlated, the hard case). On a uniform/independent filter,
  acorn reaches 0.99/0.96 at sel 10/20% (see the emission results) and the gap
  would narrow; a uniform-fixture Qdrant rematch was not run.
- 250K, dim=128, single dev host.
