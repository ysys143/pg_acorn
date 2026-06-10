# pg_acorn

Filterable HNSW for PostgreSQL via the ACORN algorithm.

pgvector's HNSW index cannot be used with WHERE clauses — the planner falls back to a
sequential scan, and even when the index survives, high filter selectivity disconnects the
graph. pg_acorn fixes this.

## The Problem

```sql
-- pgvector today: planner picks seq scan, index ignored
SELECT * FROM items WHERE category = 'shoes'
ORDER BY embedding <-> '[0.1, 0.2, ...]' LIMIT 10;

-- CTE workaround: no top-k guarantee, pays cost for 1000 rows to return 10
WITH candidates AS (
  SELECT * FROM items ORDER BY embedding <-> '[...]' LIMIT 1000
)
SELECT * FROM candidates WHERE category = 'shoes' LIMIT 10;
```

## The Solution: ACORN

ACORN (arxiv 2403.04871) preserves graph connectivity under filters by keeping
filter-failing nodes in the traversal candidate queue while excluding them from results.
top-k is guaranteed regardless of filter selectivity.

Qdrant 1.16, Elastic/Lucene, and Weaviate have shipped ACORN in production. pg_acorn
brings the same to PostgreSQL.

## Two-Tier Architecture

### Tier 1 — Zero migration (existing `USING hnsw` indexes)

Install the extension and existing indexes immediately gain ACORN-1 filtered search via
a `set_rel_pathlist_hook` + `CustomScan`. No index rebuild needed.

```sql
CREATE EXTENSION pg_acorn;

-- existing index unchanged
CREATE INDEX ON items USING hnsw (embedding vector_cosine_ops);

-- planner now selects AcornScan path automatically when WHERE is present
SELECT * FROM items WHERE category = 'shoes'
ORDER BY embedding <-> '[...]' LIMIT 10;
```

Supported pgvector versions: 0.8.x

### Tier 2 — Full ACORN-gamma (`acorn_hnsw` index AM)

A dedicated index access method with M*gamma neighbor storage at build time and correct
`amcostestimate` that accounts for filter selectivity. Independent of pgvector internals.

```sql
CREATE INDEX ON items USING acorn_hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64, acorn_gamma = 2);

-- planner selects acorn_hnsw automatically when WHERE is present
SELECT * FROM items WHERE category = 'shoes'
ORDER BY embedding <-> '[...]' LIMIT 10;
```

`acorn_gamma` values:
- `1` — ACORN-1: search-time neighbor expansion only, no build overhead
- `2+` — ACORN-gamma: M*gamma neighbors stored at build time, higher recall

`acorn_payload_edges` (bool, default `false`): when on, each node's layer-0
neighbor slots are split — half global nearest (standard HNSW), half nearest
among nodes sharing the same payload partition (`hash(filter_val) % 256`,
identity for small ints).  Same-value edges make the predicate subgraph
navigable on correlated/low-selectivity filters (Qdrant "Filterable HNSW"
style) without raising gamma:

```sql
CREATE INDEX ON items USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
  WITH (m = 16, ef_construction = 64, acorn_payload_edges = true);
SET pg_acorn.member_first = on;  -- spend ef_search on filter-passing nodes first
```

`pg_acorn.member_first` (bool, default `off`): scan-side companion to payload
edges.  The ACORN scan keeps filter-failing nodes expandable for
connectivity; with member_first the ef_search budget prefers passing
candidates, so once the traversal touches one partition member the
same-partition edges let it drain the predicate subgraph directly
(measured: recall 1.0 at ef_search=100, ~2 ms median on a 20k/128d 1%
filter, vs 0.94 at ef_search=400 and ~24 ms without).

`acorn_inline_vectors` (bool, default `false`): vector co-location.  Each
layer-0 neighbor slot additionally stores an SQ8-quantized copy of the
target's vector (1 byte/dim, per-vector scale/offset) plus its heap TID,
filter value, deleted flag, and neighbor-list pointer.  The scan then
discovers and filters neighbors from the neighbor list itself — one
expansion touches ~2 pages instead of ~1 page per discovered neighbor —
and re-ranks emitted results with exact distances (element-page read per
emitted/border candidate, 40-deep window).  Trade-offs: the index grows by
`2*m*gamma * (40 + dim)` bytes per node (~11x at dim=128, gamma=2), and
inserts re-read neighbor element pages to keep co-located entries in sync
(correct-first; a torn update degrades that edge to the classic read path,
never to a wrong result).  Requires a fixed-dimension vector column.  The
layout is recorded in the index meta page at build time; set
`pg_acorn.scan_inline_vectors = off` to force the classic scan path on an
inline index (A/B debugging).

```sql
CREATE INDEX ON items USING acorn_hnsw (embedding vector_cosine_ops, bucket int4_acorn_ops)
  WITH (m = 16, ef_construction = 64, acorn_gamma = 2,
        acorn_payload_edges = true, acorn_inline_vectors = true);
```

Note on drivers: bind small integer filter constants carefully — the
operator family includes cross-type (int2/int8) comparisons so quals like
`bucket < $1::smallint` (psycopg's default int binding) still push down as
index conditions.

## Installation

```bash
make
make install
psql -c "CREATE EXTENSION pg_acorn"
```

Requires: PostgreSQL 16+, pgvector 0.8.x (for Tier 1)

## Benchmarks

See `bench/` for reproducible benchmarks against pgvector vanilla and Qdrant 1.16.

Scenarios covered:
- A: Filter selectivity sweep (1% – 80%)
- B: Post-filter recall degradation (pgvector CTE workaround)
- C: Incremental insert recall stability
- D: Filter-query correlation adversarial case

## Development

```bash
make installcheck          # SQL regression tests
make isolation-installcheck # concurrent session tests
make unit                  # C unit tests
```

Tests use TDD: SQL files define expected behavior, golden files are populated after
implementation passes.

## Roadmap

- [x] Landscape survey (`docs/filterable-hnsw-landscape.md`)
- [ ] Tier 1: Hook + CustomScan (ACORN-1)
- [ ] Tier 2: `acorn_hnsw` index AM (ACORN-gamma)
- [ ] Benchmark harness (4 scenarios, 4 targets)
- [ ] pgvector upstream PR with benchmark evidence

## Reference

- ACORN paper: https://arxiv.org/abs/2403.04871
- Qdrant 1.16 release: https://qdrant.tech/blog/qdrant-1.16.x/
- Elastic filtered HNSW: https://www.elastic.co/search-labs/blog/filtered-hnsw-knn-search
