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

## Priority 1 — independent `payload_m` (split payload density from gamma)

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

## Priority 2 — cardinality-aware path (exact vs graph)

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

## Priority 3 — selectivity-aware default ef (auto recall targeting)

- Qdrant: ef defaults to ef_construct and is applied automatically; users
  rarely hand-tune.
- pg_acorn today: `pg_acorn.ef_search` is a manual global knob; reaching a
  recall target requires the operator to know the right ef per selectivity.
- Borrow: derive a default ef from the estimated selectivity + a target-recall
  GUC (the recall-vs-(ef,sel) surface is now measured), so a query hits, say,
  recall>=0.95 without manual ef tuning.
- Feasibility: MEDIUM. A mapping table / formula from the gamma-sweep data.
- Impact: MEDIUM — usability; removes the per-workload ef-tuning burden.

## Priority 4 — gate payload links by partition cardinality

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

## Priority 5 — observability parity (per-scan path telemetry)

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

1. Priority 1 (`acorn_payload_m`) — highest impact, high feasibility, directly
   closes the high-selectivity latency gap that Phase D surfaced.
2. Priority 2 (recall-aware cost) — fixes the extremes.
3. Priorities 3/5 — usability + observability.
