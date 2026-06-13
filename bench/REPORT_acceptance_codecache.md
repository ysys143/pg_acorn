# Acceptance: shared-memory SQ8 code cache (Phase C / C-1), n=250K

Date: 2026-06-13. Binary: merged main with M1+M2+M3 (`c327417`). Quiet host
(co-tenant ezis paused for the run). Data: `bench` DB, `tv_items` 250K dim=128
cosine, both `tv_acorn_idx` (inline, 4057 MB) and `tv_acorn_noinline` (249 MB)
present; cache forced via `pg_acorn.scan_code_cache=on` over the non-inline
index (inline index dropped in-txn). Latency reported as min-of-reps
(`min_mean`, ms) — the only host-contention-robust metric here.
Artifact: `bench/results_acceptance_codecache.json`.

## Gate results

| Gate | Target | Result | Verdict |
|---|---|---|---|
| G1 size | acorn footprint <= 1.3x stock | index 249 MB (1.26x stock 198 MB) + cache 40 MB shmem = 289 MB total (1.46x); vs inline 4057 MB (20.5x) | PASS (index), 1.46x with cache |
| G2 speed | money cells <= 1.3x inline | 1.49x-1.72x inline | **FAIL** |
| G3 recall | parity with inline | identical within noise (see below) | PASS |
| G4 correctness | results independent of cache state | 36,295-scan churn stress mismatch=0; docker-test 19+4; cache-on==cache-off | PASS |
| G5 memory | bounded by code_cache_size, no backend growth | 40 MB resident for 250K (« 512 MB default); bounded under eviction churn | PASS |

## Money cells (min_mean ms; recall>=0.95 operating point)

| cell | recall (inline/cache) | inline | cache | cache/inline | stock prefilter |
|---|---|---:|---:|---:|---:|
| sel=1% ef=200 | 1.000 / 1.000 | 2.46 | 4.23 | 1.72x | ~5.5 |
| sel=10% ef=800 | 0.990 / 0.993 | 44.19 | 70.14 | 1.59x | 147 |
| sel=20% ef=800 | 0.960 / 0.960 | 69.10 | 103.04 | 1.49x | 210 |

The cache penalty grows with ef (more discoveries -> more cache lookups): at
sel=1% ef=800 (over-resolved, not a money cell) it is 43.3 vs 7.5 ms = 5.7x.
At the money cells it is ~1.5-1.7x.

## Correction to the earlier (contended-host) read

`REPORT_index_size.md` / `results_codecache_ab_clean.json` reported the cache
"recovering inline-class latency, sometimes winning" (e.g. sel=1% ef=200
cache 4.19 vs inline 6.22). That inline bar was noise-inflated: on a genuinely
quiet host inline runs that cell in 2.46 ms, and the cache is consistently
**1.5-1.7x slower than inline**, not at parity. The honest picture:

- Inline keeps each neighbor's SQ8 code co-located in the layer-0 neighbor
  tuple — a page the scan has already pinned to read the TID array — so the
  code read is a hot, in-page access.
- The cache holds codes in a separate 40 MB DSA region. Each neighbor lookup
  is a lock-free two-load into that region; at 250K the 40 MB table overflows
  CPU cache, so lookups take DRAM misses that inline's co-located read avoids.
  This locality gap is the 1.5-1.7x.

## What the cache IS good for

It is a SIZE play, not a speed-match:

- 289 MB total footprint vs inline 4057 MB — **14x smaller** — while still
  BEATING stock pgvector prefilter at every money cell (4.2 vs 5.5, 70 vs 147,
  103 vs 210 ms) and recall parity with inline.
- vs the no-cache non-inline index (the size-optimal config that LOST to
  pgvector at 790 ms): the cache is ~5-10x faster — it converts the
  size-optimal layout from "loses to pgvector" into "beats pgvector".

So the deployment tradeoff at 250K is:
- inline: fastest, 4 GB (>shared_buffers at this scale).
- cache: 1.5-1.7x slower than inline, 289 MB, still beats pgvector.
- non-inline (no cache): smallest, but loses to pgvector.

## Recommendation (original, pre-optimization — SUPERSEDED below)

Do NOT flip the defaults. Keep `acorn_inline_vectors` default on and
`pg_acorn.scan_code_cache` default OFF. The cache ships as an opt-in
size-optimized mode: enable it when the inline index does not fit memory /
budget. Revisit the default at n>=1M, where the 16 GB-extrapolated inline
index thrashes shared_buffers and the cache's bounded 160 MB-extrapolated
footprint should turn the locality penalty into a win — that crossover is the
next thing worth measuring before any default change.

## Update 2026-06-14 — software prefetch + default graduated to ON

The 1.5-1.7x penalty was DRAM-miss latency on the separate SQ8 code region
(pointer-chase). A software-prefetch pipeline (lookup split into resolve +
seqlock-read; each neighbor's entry `__builtin_prefetch`ed K=6 ahead in the
discovery loop) hid it. Re-measured:

- 60K controlled before/after (same fixture, binary-only swap, min-of-15):
  1.17-2.12x cache speedup at the high-ef money cells; cache reaches <=1.3x
  inline.
- 250K within-run: cache 0.88-0.90x inline at the sel=10%/20% ef=800 money
  cells (cache now at/below inline at the expensive operating points); the one
  >1.3x cell is sel=1% ef=200, where both are <6 ms and there are too few
  discoveries to amortize a prefetch.

Why the cache can now match/beat inline: it touches ~half the pages inline
does (no per-neighbor element-page pin; 981 vs 1791 buffers at ef=800), and
prefetch removed the DRAM-miss tax that previously masked that advantage.

Defaults flipped: `pg_acorn.scan_code_cache` now defaults **ON**. (The
`acorn_inline_vectors` reloption was already default OFF, so the DEFAULT index
is the 249 MB non-inline layout — now cache-served, hence fast. inline stays
an explicit opt-in for lowest-latency / low-ef-dominated workloads.)

Flipping the default to ON surfaced a latent bug it then fixed (commit
c6148ce): `slot->active_scans` leaked on (a) aborted scans (amendscan skipped)
and (b) overlapping same-index scans (a self-join; the old bool ref-flag could
only track one). A leaked ref left a slot permanently non-evictable. Fixed
with an int refcount + transaction/subtransaction abort callbacks; a regression
guard (aborted-scan -> reset reclaims) was added to tier2_code_cache_evict.
docker-test 19+4 green; eviction-churn stress PASS.

Open item unchanged: the n>=1M crossover is still worth measuring — at that
scale inline thrashes shared_buffers and the cache should win outright, not
just match.
