# Designs to borrow from Qdrant's filtered HNSW

Date: 2026-06-14. Grounded in `bench/QDRANT_CODEPATH.md` (Qdrant v1.16 source
citations) + the acorn-vs-Qdrant measurements (`REPORT_qdrant_final.md`). Each
item: Qdrant mechanism, what pg_acorn does today, the borrowable design,
feasibility under the in-Postgres constraints (8KB page -> HNSW_MAX_M=100, the
index AM API, MVCC), and expected impact.

Context: pg_acorn ALREADY borrowed several Qdrant ideas — per-value payload
sub-HNSW (`acorn_payload_edges`, code comment "mirroring Qdrant's per-value
sub-HNSW"), inline quantized vectors for paged reads (`acorn_inline_vectors` +
the shared-memory code cache), and split layer-0 density (2*m_eff at L0 vs
m_eff above). The list below is what remains.

## Priority 1 — independent `payload_m` (split payload density from gamma) — DONE (merged 306a31d)

SHIPPED as the `acorn_payload_m` reloption (additive: L0 = global_m + payload_m,
backward-compatible via the reserved=0 symmetric sentinel). Bench
(`bench/REPORT_payload_m.md`): payload_m=64 at gamma=2 matches gamma=4 recall at
1.7-2.8x lower latency and a smaller index; beats Qdrant at sel 1-5%, narrows
the gap to ~2.5x at sel 10-20%. Default unchanged (payload_m=0 symmetric).
Original analysis below.


- Qdrant: separate `m` (global links) and `payload_m` (per-payload-value
  links), tuned independently (`hnsw.rs:456-463`).
- pg_acorn today: one `gamma` knob. Layer-0 = 2*m_eff slots split m_eff global
  + m_eff same-partition, m_eff = m*gamma. gamma scales BOTH halves together,
  uniformly for every node.
- Borrow: an `acorn_payload_m` reloption so the payload half can be densified
  WITHOUT inflating the global half. Today, raising correlated-filter recall
  forces a higher gamma, which also doubles global expansion cost.
- Feasibility: HIGH. Build-time reloption + change the L0 slot-split ratio;
  record the split in the meta page. No scan change, no on-disk format break.
- Impact: HIGH — directly targets the measured high-selectivity latency gap
  (gamma=4 lifted correlated recall but raised latency BECAUSE it densified
  global too; an independent payload_m gets the recall without that cost).
- Evidence: gamma sweep — sel=20% reaching ~0.95 needs ef~700 at gamma=2 vs
  ef~400 at gamma=4, and g4's extra density is half-wasted on the global side.

## Priority 2 — cardinality-aware path (exact vs graph) — INVESTIGATED, no change

DIAGNOSED (2026-06-14, `bench/REPORT_plan_choice.md` + `bench/plan_choice_probe.py`):
the plan-choice safety net shows the current selectivity-aware cost model already
routes CORRECTLY. Across sel 1-50% the planner picks acorn (estimated cost 28-436x
below the bitmap prefilter, which must sort thousands of passing rows) and acorn
is empirically far faster. The tiny-cardinality extreme is already handled by
`n_expand = Min(40/sel, N)` saturating to N as sel -> 0, switching the planner to
the exact prefilter below ~40/N selectivity (= Qdrant's small-cardinality branch,
the Postgres way). P2's hypothesized mis-routing did not reproduce; the only
residual is a sub-1% band (both plans < ~5 ms, below the fixture granularity)
whose fix (ef/recall-aware cost) is exactly the change that previously
destabilized mid-band plan choice. Decision: no change — it adds the documented
regression risk for no measurable benefit. The probe is kept as a regression
guard. Original analysis below.

- Qdrant: at search time estimates filter cardinality (with up-to-1000-point
  sampling, Agresti-Coull) and routes small filtered sets to an exact plain
  scan, large to graph traversal, ambiguous to sampling (`hnsw.rs:1356-1441`).
- pg_acorn today: the AM always uses the graph; the analogous exact-vs-approx
  choice is the POSTGRES planner picking acorn-scan vs bitmap prefilter, via
  `acorn_hnsw_costestimate` — which is selectivity-aware but has "no recall
  signal" (acorn_cost.c).
- Borrow (Postgres-idiomatic): put a recall signal into amcostestimate so the
  planner reliably prefers a bitmap prefilter when the filtered set is tiny
  (exact, recall 1.0, cheap) — that IS Qdrant's small-cardinality branch, done
  the Postgres way. Optionally an in-AM exact fallback for extreme cases.
- Feasibility: MEDIUM. Cost-model work; the recall-vs-ef curve is now measured
  (gamma sweep) so the model can be calibrated.
- Impact: MEDIUM — fixes the extremes (very small / very large filtered sets)
  where a fixed graph path is wrong.

## Priority 3 — selectivity-aware default ef (auto recall targeting) — DONE

SHIPPED as `pg_acorn.target_recall` (default 0 = off / manual ef_search). When
> 0, a Tier-2 scan estimates the query's filter selectivity from a build-time
equi-depth histogram of the filter column (stored in the meta page, zero-bounds
back-compat sentinel like payload_m) and derives the expansion budget from a
coarse MONOTONE heuristic (acorn_am.h): `ef = clamp(max(EF_MIN, EF_PER_SEL*sel)
* sqrt((1-0.95)/(1-target)))`. Selectivity source = index-resident histogram
(decided with the user: self-contained, no ANALYZE dependency, Qdrant
payload-index parallel; v1 int4 filter columns, others fall back to manual ef).
Mapping = coarse heuristic (decided with the user: dataset-agnostic, no
single-fixture overfit; a convenience that removes per-selectivity ef tuning,
NOT a recall guarantee). Decoupled from the cost model (P2) — auto-ef only sets
the runtime ef, never the plan choice. Test `tier2_auto_ef.sql`. Original
analysis below.

- Qdrant: ef defaults to ef_construct and is applied automatically; users
  rarely hand-tune.
- pg_acorn today: `pg_acorn.ef_search` is a manual global knob; reaching a
  recall target requires the operator to know the right ef per selectivity.
- Borrow: derive a default ef from the estimated selectivity + a target-recall
  GUC (the recall-vs-(ef,sel) surface is now measured), so a query hits, say,
  recall>=0.95 without manual ef tuning.
- Feasibility: MEDIUM. A mapping table / formula from the gamma-sweep data.
- Impact: MEDIUM — usability; removes the per-workload ef-tuning burden.

## Priority 4 — gate payload links by partition cardinality — DONE

SHIPPED as `acorn_payload_min_cardinality` (reloption, default 0 = off): at build
time a node whose filter-value partition has fewer than this many members gets NO
payload (same-partition) edges; those layer-0 slots are filled with global links
instead. Slot COUNT is unchanged (global fills the payload region), so the
on-disk format and the scan are untouched — purely a build-time neighbor change.
Per-partition counts come from the P3 build pre-scan (exact, int4 filters);
applied in the in-memory build path (the common case; on-disk overflow + insert
keep graceful degradation). Test `tier2_payload_gate.sql`.

HONEST NOTE: the CORE of this item was ALREADY implemented before P4 — the build
has "graceful degradation" in both paths (acorn_build.c) that fills under-fillable
payload slots with global links. That already reclaims the wasted slots whenever
a partition is too small to fill the payload half. The `acorn_payload_min_card`
knob adds the EXPLICIT Qdrant-style threshold: it also skips payload edges for
partitions in the [payload_m, threshold) range (large enough to fill, but small
enough to go exact anyway), which graceful degradation alone does not. Original
analysis below.

- Qdrant: builds the payload sub-graph only for blocks with cardinality >=
  the (point-converted) full_scan_threshold; tiny blocks get none
  (`hnsw.rs:546-601`).
- pg_acorn today: every node gets m_eff/2 same-partition slots regardless of
  how many nodes share its partition (256 partitions, value mod 256).
- Borrow: skip the payload half for nodes in tiny partitions (those filtered
  sets would go exact anyway), freeing those slots for global links.
- Feasibility: MEDIUM (needs partition-size knowledge at build time).
- Impact: LOW-MEDIUM — negligible for low-cardinality filters (bucket 0-99);
  matters for sparse/high-cardinality payloads.

## Priority 5 — observability parity (per-scan path telemetry) — DONE (7a1d51d)

SHIPPED as `pg_acorn_scan_stats()` / `pg_acorn_scan_stats_reset()`: backend-local
cumulative Tier-2 scan counters (scans, expansions, code-cache hits, element-page
loads, exact re-rank reads, emits). `cc_hits/(cc_hits+loads)` is the code-cache
hit ratio. Always-on increments at the traversal sites (where the compiled-out
`ACORN_CC_DEBUG` counters already sat); test `tier2_scan_stats.sql`. Pulled ahead
of Priorities 2/3 deliberately — it is the low-risk measurement substrate that
makes a recall-aware cost model (P2) and auto-ef (P3) calibratable and
verifiable, and it closes the visibility gaps (stale-recall, cache-residency)
that cost debugging time during Phase C/D. Original analysis below.

- Qdrant: per-segment counters (filtered_plain / filtered_large_cardinality /
  filtered_exact / ...) via telemetry (`hnsw.rs:1445-1459`).
- pg_acorn today: `ACORN_CC_DEBUG` counters (compiled out) + the code-cache
  stats SRF.
- Borrow: a lightweight per-scan counter view (path taken, expansions, cache
  hits/misses, re-rank reads) exposed via a function or EXPLAIN.
- Feasibility: HIGH. Mostly wiring existing counters to a SRF.
- Impact: LOW-MEDIUM — ops/debugging; would have caught the stale-recall and
  cache-residency confusions earlier.

## Not borrowable (in-Postgres constraints)

- Arbitrary per-node neighbor counts: Qdrant (in-memory) has no page cap;
  pg_acorn is bounded by HNSW_MAX_M=100 (8KB neighbor tuple). payload_m split
  works WITHIN this cap, but a Qdrant-style very-dense graph does not.
- Segment architecture + async optimizer: not the Postgres AM model.

## Recommended sequence

1. Priority 1 (`acorn_payload_m`) — DONE. Highest impact; closed the
   high-selectivity latency gap Phase D surfaced.
2. Priority 5 (per-scan telemetry) — DONE. Reordered ahead of 2/3: it is the
   low-risk measurement substrate the other two need to be calibratable and
   verifiable (you cannot calibrate a recall-aware cost model or auto-ef without
   per-scan visibility into expansions / path / cache hits).
3. Priority 3 (auto-ef) — DONE. Index-resident histogram + coarse heuristic;
   target_recall GUC. Decoupled from the cost model.
4. Priority 2 (recall-aware cost) — INVESTIGATED, no change. The plan-choice
   safety net showed the current cost model already routes correctly (acorn 1-50%,
   tiny-cardinality -> prefilter via n_expand saturation); changing it adds the
   documented mid-band regression risk for no measurable gain
   (bench/REPORT_plan_choice.md).
5. Priority 4 (gate payload links by cardinality) — DONE
   (acorn_payload_min_cardinality). Core was already present (build graceful
   degradation); the knob adds the explicit Qdrant-style threshold.

All borrow-list items are now addressed (P1/P3/P4/P5 shipped, P2 investigated +
declined with evidence). Remaining acorn work is its own roadmap (high-selectivity
per-expansion overhead; >=1M crossover), not Qdrant borrows.
