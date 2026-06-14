# 3-way filtered-KNN benchmark — pgvector native vs Qdrant vs pg_acorn

Date: 2026-06-15. Harnesses: `bench/bench3way_pg.py` (psycopg from host),
`bench/bench3way_qdrant.py` (httpx from host), `bench/bench3way_report.py`
(extraction). Results: `bench/results_3way_pg.json`, `bench/results_3way_qdrant.json`.

Same correlated 250K fixture for all three engines (thesis_validation seed 0,
dim 128, cosine, bucket 0-99, filter `bucket < sel`, top-k=10, exact truth).
pgvector + pg_acorn share the IDENTICAL `tv_items` table in one Postgres
container; Qdrant holds byte-identical data (point id = i+1). m=16,
ef_construct=64 everywhere.

## Engines / configs

- **pgvector native** (stock `hnsw` on tv_items): two filtered paths —
  PREFILTER (bitmap on `tv_bucket_btree` + exact Sort, recall 1.0) and
  ITERATIVE_SCAN (`hnsw.iterative_scan=strict_order`, `max_scan_tuples` raised).
- **pg_acorn**: g2p0 (gamma2, symmetric), g2p64 (gamma2 + payload_m=64, the
  correlated sweet spot), inline (vector co-location). Shipping scan config
  (code cache, member_first, prefetch).
- **Qdrant** v1.16 forced HNSW (`indexing_threshold`/`full_scan_threshold`≈0).

## Methodology + two pitfalls that had to be fixed

Clients run from the HOST over TCP for all engines (psycopg / httpx),
multiprocessing for throughput (no GIL cap). Latency single-client; throughput a
concurrency sweep {1,4,8,16}. Index isolation: a Postgres txn-drop for latency
(`BEGIN; DROP others; measure; ROLLBACK`), persistent drops for throughput;
EXPLAIN-verified the right index served each run.

- **Qdrant must SETTLE before measuring.** Right after load, the forced-HNSW
  optimizer churns (114% CPU) — first measurement gave p95 of 1100 ms and 12 QPS
  (garbage). Fix: wait for `indexed_vectors_count=250000` + status `green` +
  CPU idle, THEN measure. (Raising `indexing_threshold` to disarm the optimizer
  is WRONG — it de-indexes the segments back to exact search, recall 1.0.)
- **pgvector iterative_scan needs strict_order + a raised scan budget**, else
  `relaxed_order` + default `max_scan_tuples` collapses recall.

## Recall + latency (matched-recall ~0.95; min_ms = host-robust floor)

The shared 8-core macOS Docker VM has heavy scheduling jitter, so MEDIAN and p95
are inflated 2-7x; the MIN (uncontended floor) is the cleaner cross-engine signal
and aligns with the quiet-host `REPORT_payload_m`/`REPORT_qdrant_final` numbers.

| sel | acorn g2p64 | acorn inline | acorn g2p0 | pgv prefilter | qdrant HNSW | pgv iterative |
|----:|---|---|---|---|---|---|
| 1%  | 2.75 (r1.00) | 2.06 (r0.98) | 2.62 (r0.98) | 3.26 (r1.00) | 2.62 (r0.97) | FAIL r0.50 |
| 2%  | 3.44 (r0.99) | 2.81 (r0.98) | 3.82 (r0.98) | 6.40 (r1.00) | 3.34 (r0.97) | FAIL r0.39 |
| 5%  | 4.99 (r0.99) | 4.73 (r0.99) | 7.61 (r0.99) | 13.2 (r1.00) | 5.11 (r0.94) | FAIL r0.22 |
| 10% | 9.36 (r0.99) | 5.23 (r0.94) | 6.98 (r0.94) | 22.7 (r1.00) | 7.46 (r0.96) | FAIL r0.26 |
| 20% | 18.9 (r1.00) | 14.9 (r0.96) | 15.3 (r0.96) | 37.4 (r1.00) | 12.1 (r0.98) | FAIL r0.41 |

(values = min_ms at the lowest ef reaching recall>=0.94; "FAIL" = never reaches
0.94 at any ef. Full med/p95 in results_3way_pg.json.)

## Real throughput — peak QPS (concurrency-swept, multiprocessing)

NOISY on the shared VM (QPS-vs-concurrency is erratic; peak is usually @conc=4
before 8-core + client saturation). Indicative, not absolute.

| engine | sel 1% peak | sel 10% peak |
|---|---:|---:|
| pgv prefilter | **179** (@4) | 25 (@4) |
| qdrant HNSW | 93 (@4) | 24 (@1) |
| acorn g2p64 | 79 (@4) | **44** (@4) |

(qdrant also: sel5% 64, sel20% 8.)

## Index size + build time (250K x 128)

| index | size | build (approx, partly contended) |
|---|---:|---:|
| pgvector hnsw | 199 MB | 5m 42s |
| acorn g2p0 | 249 MB | (prior) |
| acorn g2p64 | 301 MB | 18m 09s |
| acorn inline | 4057 MB | (prior; inline = full vectors) |

acorn builds are ~3x slower than pgvector hnsw and larger; the inline layout is
4 GB (co-located vectors) and is a latency play, not a footprint play.

## Verdict

1. **pg_acorn beats pgvector-native at filtered KNN — decisively.** pgvector's
   iterative filtered HNSW FAILS on the correlated fixture: recall caps at
   0.22-0.50 at any ef / scan budget (strict_order, raised max_scan_tuples). Its
   only reliable path is the exact PREFILTER (recall 1.0) — which is 1.3-2x
   slower than acorn at the latency floor and scales worse with selectivity
   (sort grows). This is exactly acorn's thesis: correlated filtered ANN is where
   stock pgvector breaks and ACORN does not.
2. **pg_acorn is competitive with Qdrant inside PostgreSQL.** Recall ties (both
   0.94-1.0). At the host-robust latency floor acorn and Qdrant are within ~2x
   across all selectivities (e.g. sel10% acorn-inline 5.2 ms vs Qdrant 7.5 ms;
   sel20% Qdrant 12.1 ms vs acorn 14.9-18.9 ms). Qdrant edges ahead at the
   highest selectivity; acorn matches or beats it at low-mid selectivity — while
   being transactional, joinable, SQL-native, and on-disk.
3. **Throughput (this VM):** modest and noisy for all. At low selectivity
   pgvector prefilter wins QPS (cheap exact sort over few rows, parallelizes
   well); at mid selectivity acorn leads (its in-filter graph beats prefilter's
   large sort and Qdrant's heavier per-query search). Absolute QPS is depressed
   by the shared VM + Python clients + Qdrant's HTTP/JSON path.

## Caveats (honest)

- Shared 8-core macOS Docker VM with a co-tenant: median/p95/throughput are
  jitter-inflated (2-7x); MIN latency is the comparable floor. Quiet-host
  absolutes for acorn/qdrant are in REPORT_payload_m / REPORT_qdrant_final.
- Transport differs: Postgres libpq (binary) vs Qdrant HTTP/JSON; both via their
  native client + multiprocessing — "native-client end-to-end," not engine-core.
- in-PostgreSQL (MVCC, buffer manager, txn) vs Qdrant in-memory: different
  substrates; acorn matching a dedicated engine within ~2x is the headline, not a
  loss.
- Single correlated fixture (the hard case), 250K, dim 128, one host. On a
  uniform filter pgvector iterative would fare better and the gaps would narrow.
- Throughput QPS is indicative (noisy concurrency scaling); peak@conc reported.
