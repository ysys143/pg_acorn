# Which code path did our Qdrant numbers measure?

Verdict from qdrant v1.16.0 source (tag `v1.16.0`, github.com/qdrant/qdrant) plus an
empirical probe against our running `qdrant/qdrant:v1.16.0` container
(`bench/qdrant_codepath_probe.py`, 2026-06-10).

## Verdict

All prior Qdrant numbers at n=30K and n=50K (`results_30k_high_free.json`,
`results_50k_low.json`, and n=100K up to sel=40% in
`results_100k_low_qdrantonly.json`) were **exact brute-force scans, not HNSW**.
No segment ever crossed the optimizer's per-segment indexing threshold, so no
HNSW graph was ever built; every search ran through `PlainVectorIndex`
(exhaustive scoring of all filter-passing points). Recall 1.000 at every
selectivity and hnsw_ef-flat latency are by construction, not a property of
Qdrant's filtered HNSW.

The `qtest_threshold.py` experiment (n=50K, `full_scan_threshold: 10`) also
measured exact scans in BOTH arms: `full_scan_threshold` only routes searches
*within an HNSW-indexed segment*, and at 50K there were no indexed segments.

## Threshold mechanics (source citations, tag v1.16.0)

### 1. `indexing_threshold` — per-segment, in kilobytes of vector storage

- Default constant: `lib/collection/src/optimizers_builder.rs:21`
  `pub const DEFAULT_INDEXING_THRESHOLD_KB: usize = 10_000;`
  Field doc (`optimizers_builder.rs:67-76`): "Maximum size (in kilobytes) of
  vectors allowed for plain index, exceeding this threshold will enable vector
  indexing"; `#[serde(alias = "indexing_threshold_kb")]`. `None` falls back to
  the default, `Some(0)` maps to `usize::MAX` = indexing disabled
  (`optimizers_builder.rs:118-122`). Shipped `config/config.yaml:129`:
  `indexing_threshold_kb: 10000`.
- Trigger check is **per segment**, in bytes:
  `lib/collection/src/collection_manager/optimizers/indexing_optimizer.rs:117-138`
  `let is_big_for_index = storage_size_bytes >= indexing_threshold_bytes;`
  `let optimize_for_index = is_big_for_index && !is_indexed;`
- Even when an optimization runs, the rebuilt segment only gets HNSW if it is
  big enough: `lib/collection/src/collection_manager/optimizers/segment_optimizer.rs:239-260`
  `let threshold_is_indexed = maximal_vector_store_size_bytes >= thresholds.indexing_threshold_kb.saturating_mul(BYTES_IN_KB);`
  ... `if threshold_is_indexed { ... config.index = Indexes::Hnsw(vector_hnsw); }`
  Otherwise segments keep the default `Indexes::Plain {}`
  (`lib/segment/src/types.rs:1308-1311`).
- Segment count: `optimizers_builder.rs:104-113` — with
  `default_segment_number = 0` (default), Qdrant uses
  `(num_cpus / 2).clamp(2, 8)` appendable segments.

Arithmetic for our bench (dim=128 f32 = 512 B/vector; container sees 8 CPUs ->
4 segments):

| n | total vectors | per segment (4) | vs 10,000 KiB threshold | indexed? |
|------|------------|-----------------|--------------------------|----------|
| 30K | 15,000 KiB | 3,750 KiB | 0.37x | no |
| 50K | 25,000 KiB | 6,250 KiB | 0.61x | no |
| 100K | 50,000 KiB | 12,500 KiB | 1.22x | yes |

So 30K/50K collections never index regardless of how long you wait; the
threshold is per segment, not per collection.

### 2. `full_scan_threshold` — in kilobytes, converted to a point count

- Definition: `lib/segment/src/types.rs:641-648` — "Minimal size threshold (in
  KiloBytes) below which full-scan is preferred over HNSW search",
  `#[serde(alias = "full_scan_threshold_kb")]`. Default
  `lib/segment/src/types.rs:1723`:
  `pub const DEFAULT_FULL_SCAN_THRESHOLD: usize = 10_000;`
- At index build/open it is converted KB -> points:
  `lib/segment/src/index/hnsw_index/hnsw.rs:244-253`
  `full_scan_threshold.saturating_mul(BYTES_IN_KB).checked_div(avg_vector_size)`.
  For 512 B vectors: default 10,000 KB -> **20,000 points**; our
  `full_scan_threshold: 10` test -> **20 points**.
- It only takes effect inside `HNSWIndex::search`
  (`hnsw.rs:1293-1443`), i.e. only on segments that already have a graph.

### 3. Filtered-search planner (inside an HNSW-indexed segment)

`lib/segment/src/index/hnsw_index/hnsw.rs:1356-1441`:

- Cardinality from the payload index: `hnsw.rs:1387-1393`
  `payload_index.estimate_cardinality(query_filter, ...)` returns
  `{min, exp, max}` estimations.
- `query_cardinality.max < full_scan_threshold` -> **plain exact scan** of the
  filter-passing ids (`hnsw.rs:1395-1406`, telemetry
  `filtered_small_cardinality`; implementation `search_vectors_plain`,
  `hnsw.rs:1199-1224`, scores every id from `payload_index.query_points`).
- `query_cardinality.min > full_scan_threshold` -> **graph traversal with
  FilteredScorer** (`hnsw.rs:1408-1419`, telemetry
  `filtered_large_cardinality`); `ef = params.hnsw_ef.unwrap_or(self.config.ef)`
  where the stored `config.ef = ef_construct` (`hnsw.rs:957-959`,
  `lib/segment/src/index/hnsw_index/config.rs:47`) — so "server default ef" in
  our bench was 64, not 128.
- Ambiguous range -> point sampling (`hnsw.rs:1425-1440`,
  `lib/segment/src/index/sample_estimation.rs::sample_check_cardinality`,
  Agresti-Coull interval over up to 1000 sampled points).
- Payload-aware subgraph: built at index-build time, not search time.
  `payload_m` defaults to `m` (`hnsw.rs:456-463`:
  `config.payload_m.unwrap_or(config.m)`), and for every indexed payload field,
  blocks with cardinality >= the point-converted `full_scan_threshold` get
  extra HNSW links merged into the main graph (`hnsw.rs:546-601`).

### 4. `exact: true` and observability

- `SearchParams` (`lib/segment/src/types.rs:566-575`): `hnsw_ef: Option<usize>`,
  `exact: bool` — "Search without approximation. If set to true, search may run
  long but with exact results."
- `hnsw.rs:1306, 1362-1378`: `if exact || is_hnsw_disabled` ->
  `search_vectors_plain` (telemetry `filtered_exact`). So `exact:true` forces
  brute force even on indexed segments.
- `GET /collections/{name}` -> `CollectionInfo`
  (`lib/collection/src/operations/types.rs:188-195`):
  `indexed_vectors_count` ("Approximate number of indexed vectors ... stored in
  a specialized vector index") vs `points_count`. Plain segments contribute 0
  (`lib/segment/src/index/plain_vector_index.rs:222-224`:
  `fn indexed_vector_count(&self) -> usize { 0 }`).
  **`indexed_vectors_count == 0` is the smoking gun for "no HNSW".**
- `GET /telemetry?details_level=10` exposes per-segment
  `vector_index_searches` counters: `unfiltered_plain / filtered_plain`
  (PlainVectorIndex), `unfiltered_hnsw`, `filtered_small_cardinality`,
  `filtered_large_cardinality`, `filtered_exact`, `unfiltered_exact`
  (`hnsw.rs:1445-1459`). Zero-count keys are omitted from the JSON.

### 5. Unindexed segments are exact by construction

`lib/segment/src/index/plain_vector_index.rs:84-140`: filtered search fetches
`filtered_ids_vec = payload_index.query_points(filter, ...)` (line 114) and
scores ALL of them with `FilteredScorer.peek_top_iter`. No approximation
anywhere; `hnsw_ef` is ignored. Recall vs exact ground truth = 1.0 always.

## Empirical probe results (bench/qdrant_codepath_probe.py, 30 queries, K=10)

### Phase A — n=30K, exact bench config (defaults)

`indexed_vectors_count` stayed **0** (status green, optimizer ok, 4 segments).
Every search hit `filtered_plain` only:

| mode | sel=1% recall / p50 | sel=40% recall / p50 |
|------|---------------------|----------------------|
| default | 1.000 / 0.89 ms | 1.000 / 31.59 ms |
| exact=true | 1.000 / 0.95 ms | 1.000 / 13.65 ms |
| hnsw_ef=10 | 1.000 / 0.95 ms | 1.000 / 18.26 ms |
| hnsw_ef=100 | 1.000 / 0.95 ms | 1.000 / 23.03 ms |
| hnsw_ef=400 | 1.000 / 0.91 ms | 1.000 / 18.84 ms |

default == exact, flat in ef, telemetry 100% plain: the bench measured exact
brute force.

### Phase B — n=60K, `default_segment_number: 2` (30K pts = 15,000 KiB/segment, crosses threshold)

`indexed_vectors_count`: 0 -> 40,000 (t=16s) -> 60,000 (t=22s), green at t=44s.
(With the default 4 segments even 60K would NOT cross: 7,500 KiB/segment.)

| query | path (telemetry) | recall | p50 |
|-------|------------------|--------|-----|
| sel=1% ef=10/400/default | filtered_small_cardinality (exact) | 1.000 | ~1 ms |
| sel=40% ef=10 | filtered_large_cardinality (graph) | 0.167 | 10.6 ms |
| sel=40% default (ef=64) | graph | 0.413 | 23.3 ms |
| sel=40% ef=400 | graph | 0.747 | 41.4 ms |
| sel=40% exact=true | filtered_exact | 1.000 | 26.2 ms |
| sel=80% ef=10 / default / ef=400 | graph | 0.513 / 0.707 / 0.893 | 22.8 / 17.9 / 24.4 ms |
| unfiltered ef=10 / ef=400 | unfiltered_hnsw | 0.703 / 0.883 | 21.1 / 16.5 ms |

Even on a genuinely indexed collection, sel=1% (600 pts << 20,000-point
full-scan threshold) is still answered exactly — recall 1.0, ef-flat. The graph
only engages when per-segment filter cardinality exceeds ~20,000 points, and
then recall is strongly ef-dependent (0.167 -> 0.747 at sel=40%). This matches
our own `results_100k_low_qdrantonly.json`: recall 1.0 flat for sel<=40%
(<=10K pts/segment), but ef-dependent 0.43..0.915 at sel=80% (20K pts/segment).

### Phase C — n=60K indexed, `full_scan_threshold: 10` (-> 20 points; graph forced for all filters)

| hnsw_ef | sel=1% recall / p50 | sel=40% recall / p50 |
|---------|---------------------|----------------------|
| 10 | 0.737 / 2.17 ms | 0.290 / 31.8 ms |
| 50 | 0.997 / 2.02 ms | 0.503 / 16.5 ms |
| 100 | 1.000 / 3.88 ms | 0.687 / 28.0 ms |
| 200 | 1.000 / 4.56 ms | 0.823 / 45.8 ms |
| 400 | 1.000 / 2.90 ms | 0.960 / 42.3 ms |
| exact=true | 1.000 / 1.73 ms | 1.000 / 24.7 ms |

On a real graph with the filter forced, recall at sel=1% does drop below 1.0 at
low ef (0.737 at ef=10) but saturates by ef=100 — the payload-aware subgraph
(payload_m defaults to m=16; every bucket value block of ~600 pts >= 20-point
block threshold gets extra links) keeps low-selectivity filtered search
well-connected. sel=40% behaves like ordinary filtered HNSW (0.29 -> 0.96).

## Consequences for our comparisons

1. Every "Qdrant recall=1.0, latency flat in ef" row at n=30K/50K is a plain
   exact-scan baseline. It is comparable to our `prefilter` exact target, not
   to pg_acorn/pgvector HNSW traversal.
2. `qtest_threshold.py`'s conclusion is void: `full_scan_threshold` was never
   exercised because no segment had a graph at n=50K.
3. The flat-but-below-1.0 qdrant recall in `results_50k_low.json` (0.866 at
   sel=1%, flat across ef) cannot be graph behavior (no graph existed);
   ef-flatness plus exact scan implies a harness ground-truth artifact
   (tie-breaking/precision divergence between qdrant cosine and pg `<=>` on
   that clustered dataset) and needs a separate check.
4. For honest engine comparisons at bench scale, either (a) report Qdrant as
   "exact prefilter" at these sizes, or (b) force indexing
   (`indexing_threshold_kb` low or bigger segments) AND force the graph
   (`full_scan_threshold` low) and record
   `indexed_vectors_count`/telemetry path counters alongside, as
   `qdrant_codepath_probe.py` does.
