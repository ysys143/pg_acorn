# pg_acorn Benchmark — Scenario A (Filter Selectivity Sweep)

Smoke-scale run validating the harness end-to-end and the two-tier architecture.

## Configuration

| Parameter | Value |
|-----------|-------|
| Fixture | synthetic unit vectors (`fixtures/sift1m_subset.py`) |
| Vectors | 2,000 |
| Dimensions | 64 |
| Queries | 20 |
| k | 10 |
| Index params | m=16, ef_construction=64 |
| Targets | pgvector, pg_acorn Tier 1 (g1), Tier 2 (g1), Tier 2 (g2) |
| Qdrant | skipped (service not started) |
| Wall time | ~40s |

> Small `n` is deliberate: the Tier 2 build is currently brute-force O(n²)
> (exact nearest neighbors), so large fixtures are infeasible without an
> approximate build. See "Limitations".

## Recall@10 vs filter selectivity

| Target | 1% | 5% | 10% | 40% | 80% |
|--------|----|----|----|----|----|
| pgvector (seq scan) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **pg_acorn Tier 1 (ACORN-1 hook)** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |
| pg_acorn Tier 2 (acorn_hnsw, g1) | 0.035 | 0.195 | 0.395 | 0.990 | 0.995 |
| pg_acorn Tier 2 (acorn_hnsw, g2) | 0.035 | 0.190 | 0.395 | 0.990 | 0.995 |

(Charts: `bench/report/scenario_a.png`. Raw data: `bench/results.json`.)

## Interpretation

This run cleanly validates the **reason the project has two tiers**:

- **Tier 1 (hook + CustomScan) holds recall = 1.0 at every selectivity**,
  including the adversarial 1% case. This is the ACORN-1 invariant working as
  designed: because the hook sees the `WHERE` predicate, it can keep
  filter-failing nodes in the traversal queue (preserving graph connectivity)
  while excluding them from results — so top-k is guaranteed regardless of
  selectivity.

- **Tier 2 (pure index AM) recall collapses at low selectivity** (1% → 0.035)
  and recovers as selectivity rises (40%+ → 0.99). This is expected and
  important: a standard PostgreSQL index AM does **not** receive an arbitrary
  `WHERE` filter on another column — the executor applies it as a *post-filter*
  on the Index Scan. The AM returns the `ef_search` (=40) nearest unfiltered
  candidates; at 1% selectivity only ~0.4 of them survive the post-filter, so
  recall is near zero. This is exactly the post-filter failure mode that
  motivates ACORN, reproduced here for our own Tier 2.

- **gamma=2 ≈ gamma=1 here** because the bottleneck at low selectivity is the
  post-filter, not graph connectivity; a denser graph cannot help once the
  candidate set is filtered away after retrieval.

## Limitations / future work

1. **Tier 2 brute-force build is O(n²).** Replace with a proper greedy
   multi-layer HNSW construction to scale to 10K+ vectors and run the full
   SIFT/Cohere benchmarks in the plan.
2. **Tier 2 needs iterative scan** to be competitive under selective filters:
   keep returning candidates (expanding `ef`) until the executor has k
   post-filter matches (pgvector 0.8 added this for `hnsw`). Without it Tier 2
   is only useful at high selectivity.
3. **Synthetic unit-vector fixture.** Queries use `<=>` (cosine); for unit
   vectors this matches the L2 ground-truth ordering. Swap in real SIFT/Cohere
   vectors for headline numbers.
4. **Qdrant + scenarios B/C/D** not run here (harness supports them).

## Reproduce

```sh
# inside the pg_acorn_test image with Postgres started + extensions created:
python3 bench/run_bench.py --scenario a \
    --dsn "postgresql://postgres@localhost/postgres" \
    --n-vectors 2000 --dim 64 --n-queries 20 --output bench/results.json
python3 bench/report.py --input bench/results.json --output bench/report
```
