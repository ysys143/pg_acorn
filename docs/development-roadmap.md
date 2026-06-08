# pg_acorn 발전 로드맵

> 작성: 2026-06-08
> 근거 문서: `docs/sigmod2026-fvs-postgresql-analysis.md` (SIGMOD 2026 FVS-in-PG 논문 분석)
> 목적: 현재 구현에서 출발해 논문이 지적한 격차를 어떤 순서로 메울지 정의한다.

---

## 0. 현재 상태 (baseline)

| 구성 요소 | 상태 |
|----------|------|
| **Tier 1** (hook + CustomScan) | 필터 인지 ACORN-γ 탐색. WHERE 술어를 직접 봄. 전 선택도 recall 1.0 |
| **Tier 2** (`acorn_hnsw` AM) | 멀티레이어 HNSW 빌드(이번 세션 완료) + ef 확장 반복 post-filter 스캔. recall@10 1%=1.0 |
| **공유 탐색** (`acorn_scan.c`) | 멀티레이어 greedy descent(`:302`) + layer-0 ACORN 탐색(`:363`). **1-hop**, 필터 실패 노드는 C 유지(`:471`) |
| **ACORN-γ** | `m_eff = m×gamma` 이웃을 빌드 시 저장(`acorn_build.c:80`), `meta->m`에 기록 |
| **벤치마크** | scenario A–D 하네스, 합성 unit 벡터, **page-I/O 미측정** |

### 논문 대비 격차 (멀티레이어 빌드 완료 후 갱신)

분석 문서 Section 11의 "최우선 격차(멀티레이어 HNSW)"는 **이미 해소됨**. 남은 격차:

| 격차 | 논문 근거 | 난이도 | 위험 | 적용 위치 |
|------|----------|--------|------|----------|
| page-I/O 미측정 | §1, §12.4 | 소 | 매우 낮음 | bench |
| 실데이터/Qdrant/시나리오 B–D 미실행 | §6, §8 | 중 | 낮음 | bench |
| ACORN-γ L_max 경고 없음 | §10, §12.3 | 극소 | 매우 낮음 | `acorn_build.c` |
| ef_search C 상한 (코드에 TODO) | §11 | 소 | 낮음 | `acorn_scan.c:515` |
| Tier 2 반복 스캔이 ef 2배마다 전체 재탐색 | (자체 측정) | 중 | 낮음 | `acorn_am.c:253` |
| Translation Map 없음 | §5, §12.1 | 대 | 중 | `acorn_build.c`+`acorn_scan.c` |
| 런타임 2-hop (ACORN-1) 없음 | §4, §6.1 | 대 | 중 | `acorn_scan.c` |
| NaviX-Directed 없음 | §6.1, §8, §12.2 | 중 | 중 | `acorn_scan.c` |
| 컬럼나/INCLUDE 컬럼 | §13 | 특대 | 높음 | 아키텍처 |

---

## 의존성 그래프 (로드맵 순서의 근거)

```
Phase 0 (측정·안전)  ──┬──> Phase 1 (TM) ──> Phase 2 (2-hop ─> NaviX-Directed)
                       │         ▲                    │
   page-I/O 계측 ───────┘         └── 효과 증명 의존 ───┘
   실데이터 하네스                                      │
   L_max 경고 / ef 상한 / 반복스캔 재개                 v
                                              Phase 3 (타 알고리즘·컬럼나)
```

**왜 이 순서인가**: 논문은 "distance 계산 수는 PG end-to-end의 신뢰할 수 없는 proxy"라고 단언한다(§1). 따라서 **page-I/O 측정 없이는 TM/2-hop이 실제로 페이지 접근을 줄였는지 증명할 수 없다.** TM은 2-hop을 PG에서 실행 가능하게 만드는 전제(2-hop은 스텝당 1+M+M² 페이지, §4)이며, NaviX-Directed는 2-hop 위에서만 의미가 있다.

---

## Phase 0 — 측정 기반 + 안전장치 (전제 조건)

목표: 이후 모든 최적화를 **증명 가능**하게 만들고, 저위험 정합성 항목을 닫는다.

> 진행: [x] page-I/O 계측 · [x] L_max(gamma-clamp) 경고 · [x] Tier 2 재개 스캔 ·
> [ ] ef_search C 상한 · [ ] 실데이터/Qdrant/B–D. (커밋: `9481116`·`7dc4390`·`55fe795`·`e7ee9a9`·`d710ae0`, 브랜치 `bench-page-io`)

1. **[완료] page-I/O 계측** (`bench/`, §12.4)
   - `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`로 `shared_hit + shared_read` 추출 → `pages_per_query` 지표.
   - `bench/results.json`·`RESULTS.md` 리포트에 `QPS`/`recall@10`/`p99`와 **나란히 필수 컬럼으로** 추가.
   - 산출물: 현재 ACORN-γ의 쿼리당 페이지 접근 baseline. 이후 TM/2-hop의 효과를 이 숫자로 측정.

2. **실데이터 하네스 보강** (`bench/fixtures/`, `bench/targets/qdrant.py`)
   - 합성 unit 벡터 → 실 SIFT/Cohere 임베딩. 음의 상관관계 시나리오 포함(§8.2 — 논문의 가장 가혹한 케이스).
   - Qdrant 타깃 활성화 + 시나리오 B(post-filter 저하)/C(증분 insert)/D(상관관계) 정식 실행.

3. **[완료] ACORN-γ gamma-clamp 경고** (`acorn_build.c`, §12.3) — 극소, 독립적
   - 레벨이 `ACORN_MAX_LEVEL`로 이미 캡되어 페이지 오버플로는 불가능 → 논문의 `l_max<10`은
     dead code. 실제 발화 가치가 있는 케이스로 조정: `m*gamma > HNSW_MAX_M`일 때 `m_eff`가
     조용히 100으로 clamp되어 gamma가 완전히 적용되지 않음을 `ereport(WARNING)` + `errhint`.
   - 검증: gamma=8(m=16)에서 발화, gamma=2에서 무발화. 회귀 7/7 통과.

4. **[보류] ef_search C 상한** (`acorn_scan.c:515` TODO) — 소, 저우선
   - 후보 C 상한은 ACORN 저선택도 연결성(recall)에 민감하고 원저자가 의도적으로 미룬 최적화.
     강행 시 recall 강검증 필요. 재개 스캔 도입으로 무한 탐색 위험은 이미 완화됨(streaming은
     executor LIMIT 충족 시 정지) → 우선순위 낮음.

5. **[완료] Tier 2 재개 스캔 (순수 스트리밍)** (`acorn_scan.c` `acorn_stream_*` + `acorn_am.c`) — 중
   - ef-doubling batch(매 확장마다 entry point부터 전체 재탐색)를 streaming frontier로 교체.
     각 노드 1회 확장·방출 → 재탐색 제거. Tier 1 batch 경로는 불변(격리).
   - 결과(RESULTS.md Run 4, 인덱스 강제): g1 @ 1% 63,000→**19,638** 페이지(3.2배↓), recall
     1.000→0.998. 전 구간 index-scan 페이지 2–4배↓. 회귀 7/7 + 격리 2/2 + 단위 10/10 통과.
   - 주: 한때 hybrid(batch+stream)를 시도했으나, 그 동기였던 "g2@1% 회귀"가 실은 플래너의
     seqscan 선택(측정 교란)이었음이 밝혀져 되돌림. [[bench-force-index-confound]] 참고.

6. **[신규] `acorn_cost.c` 비용 모델 재보정** — 중, 신규 (이번 분석에서 정당화)
   - 1%/10k에서 seqscan(371p·2.4ms)이 acorn 인덱스(~20k p·170ms)보다 압도적으로 싼데,
     플래너는 g1엔 인덱스(과소평가)·g2엔 seqscan을 골라 plan이 어긋남. cost 추정이 실제
     index-scan 페이지 접근을 반영하도록 보정 → 올바른 index-vs-seqscan crossover 확보.
   - 벤치 방법론도 함께 수정됨: scenario A가 인덱스 강제 + plan 노드 기록 (Run 4).

**완료 기준**: `pages_per_query`가 모든 타깃 리포트에 존재, 실데이터로 시나리오 A–D 통과, `make docker-test`/`docker-unit` 회귀 없음.

---

## Phase 1 — Translation Map (2-hop의 전제)

목표: `indextid → heaptid` 인메모리 해시맵으로 2-hop 술어 평가 시 인덱스 페이지 재조회 제거(§5).

- **빌드 시** (`acorn_build.c`, 인덱스 완성 후): 전체 인덱스 1회 스캔으로 `HTAB` 구성. 엔트리당 ~18B (1M=18MB, 35M=630MB).
  ```c
  typedef struct AcornTranslationMap {
      HTAB *indextid_to_heaptid;
      MemoryContext mcxt;
  } AcornTranslationMap;
  ```
- **쿼리 시** (`acorn_scan.c`): 필터 평가 경로에서 heaptid 해석을 TM 조회(~10ns)로 대체.
- **검증**: Phase 0의 `pages_per_query`로 heaptid fetch 비용 하락 확인(논문 §5: 60–75% → 8–17%).

> 주의: 현재 1-hop 경로는 노드를 distance 계산하며 element 튜플을 이미 읽어 heaptid가 공짜다. **TM의 실익은 2-hop이 생겨야 발생**하므로 Phase 2와 묶어 검증한다.

---

## Phase 2 — 2-hop ACORN-1 + NaviX-Directed (논문의 간판)

목표: 런타임 2-hop 확장으로, 저장 비용 없이(γ=1) effective density 확보. pgvector 페이지 천장(§10)에 막히는 초대규모·초저선택도 영역을 연다.

1. **런타임 2-hop 확장** (`acorn_scan.c`, §4·§6.1) — TM 위에서만 실행 가능.
2. **NaviX-Directed** (§12.2): 1-hop을 거리순 정렬 → 가까운 비통과 노드의 2-hop 먼저 탐색. 논문상 ACORN 대비 1.2–1.7x.
3. **(선택) Adaptive heuristic 전환**: Blind/Directed/Onehop-s를 탐색 상태(C/W 채움률·통과율)로 동적 선택(§6.1).

### 아키텍처 제약 — Tier 1 회귀 방지

이 변경들은 Tier 1이 공유하는 `acorn_scan.c`를 건드린다. **회귀 가드 필수**:
- 가법적으로: γ=1 경로는 런타임 2-hop을 쓰고, γ>1(현재 ACORN-γ) 경로는 기존 동작 보존.
- `no_regression.sql`(acorn==pgvector top-k) + `recall_filter` 통과, ≥40% 선택도 recall ≥0.9 유지를 머지 게이트로.

**전략적 가치**: TM + 2-hop + NaviX-Directed가 들어가면 pg_acorn은 **오픈소스 PostgreSQL 최초의 제대로 된 graph-based filtered vector search**가 된다(§7). NaviX·TM은 AlloyDB 구독으로도 못 쓰는 연구 프로토타입.

---

## Phase 3 — 타 알고리즘 / 장기 방향

- **ScaNN형 클러스터 인덱스** (§6.3): 음의 상관관계에서 유일한 승자. 별도 AM 규모의 큰 작업 — 채택 여부는 Phase 0 실데이터 결과(음상관 시나리오)를 보고 결정.
- **컬럼나 / INCLUDE 컬럼** (§13): 필터 컬럼 사본을 인덱스에 저장(PG 11+ `INCLUDE`)해 heap double-lookup을 근본 제거. 논문이 명시한 장기 방향. 설계 스파이크부터.

---

---

## 부록 — 신규 알고리즘 지형 (FAVOR / JAG / Curator / SeRF / KHI)

"이것들은 더 나중인가?"에 대한 답: **대부분 Phase 3 이후**. 단, 세 게이트로 *언제*가 갈린다.

- **G1 아키텍처 적합**: filter-agnostic이며 pgvector 페이지 포맷(`hnsw_compat.h`)을 재사용 가능한가?
- **G2 문제 동일성**: 속성 **동등** WHERE 술어 문제인가? (범위 필터/멀티테넌트는 다른 문제)
- **G3 PG 검증**: PG에서 우위가 입증됐는가? (Phase 0 측정 전엔 전부 미지)

| 알고리즘 | 계열 | G1 | G2 | 배치 | 비고 |
|---------|------|----|----|------|------|
| **FAVOR** ([2605.07770](https://arxiv.org/abs/2605.07770)) | filter-agnostic graph (HNSW) | O | O | **Phase 2.5 후보** | ACORN/NaviX의 사촌. selectivity-aware exclusion distance. NaviX-Directed의 대안 휴리스틱으로 평가 가능 |
| **Curator** ([2401.07119](https://arxiv.org/abs/2401.07119), [2601.01291](https://arxiv.org/abs/2601.01291)) | 파티션/트리 (per-label) | X | △ | Phase 3 | ScaNN 계열. 저선택도·멀티테넌트·음상관 보완. 새 AM 필요 |
| **JAG** ([2602.10258](https://arxiv.org/abs/2602.10258)) | attribute graph (filter-specific) | X | O | Phase 3+ | 속성을 그래프에 결합. 새 디스크 포맷/AM 필요 |
| **SeRF** ([SIGMOD'24](https://miaoqiao.github.io/paper/SIGMOD24_SeRF.pdf)) | range-filter segment graph | X | X | Phase 3+ | **범위 필터** 전용. 질의 클래스 확장 결정 필요 |
| **KHI** | 다속성 파티션 트리 + per-node HNSW | X | X | Phase 3+ | 다속성 범위 RFANNS. 질의 클래스 확장 + 새 AM |

**해석**
- **부류 A (FAVOR)**: 우리 filter-agnostic graph 계열. 유일하게 기존 Tier 1/2 구조에 끼울 수 있어 Phase 2와 가깝다.
- **부류 B (JAG/Curator/SeRF/KHI)**: 필터를 인덱스에 박아넣는 specialized index. pgvector 페이지 포맷 재사용 전제를 깨고 **새 AM**을 요구하거나, **다른 질의 클래스(범위 필터)** 지원 결정을 요구한다. 따라서 "채택"이 아니라 "방향 전환"이며 Phase 3 이후.
- 공통: 전부 standalone 라이브러리 벤치마크 결과뿐 → **PG 우위는 Phase 0 page-I/O 계측 이후에만 판단 가능**. 그 전까지는 research-watch 대상.

> 참고 서베이: RF-ANNS 분류 체계 [arXiv:2505.06501](https://arxiv.org/abs/2505.06501), FANNS 벤치마크 [arXiv:2507.21989](https://arxiv.org/html/2507.21989v1).

---

## 우선순위 요약

| 순위 | 항목 | 이유 |
|------|------|------|
| P0 | page-I/O 계측 | 다른 모든 작업의 증명 도구 |
| P0 | L_max 경고 / ef 상한 / 반복스캔 재개 | 저위험, 즉시 정합성·성능 이득 |
| P0 | 실데이터 + Qdrant + 시나리오 B–D | 의미 있는 측정 기반 |
| P1 | Translation Map | 2-hop 전제 |
| P2 | 2-hop ACORN-1 + NaviX-Directed | 논문 간판, 대규모·저선택도 개방 |
| P3 | ScaNN / 컬럼나 | 장기, 큰 아키텍처 결정 |
```
