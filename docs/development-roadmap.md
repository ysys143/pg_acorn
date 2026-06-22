# pg_acorn 발전 로드맵

> 최초 작성: 2026-06-08 (SIGMOD 2026 FVS-in-PG 분석 기반, 쿼리-side Phase 0→3)
> 현 시점 재정리: 2026-06-19 — 3-way 벤치·스케일링, 빌드-perf 캠페인(B1-B4/N1-N4),
> M-ACORN 음성 결론, extension-lock 라이브락 발견을 반영해 **트랙 구조로 재편**.
> 2026-06-19 (2차): **그랜드 플랜(North Star + 4-Phase) + Track S(안정화: 대용량·빌드병렬·
> 다중세션, 1.0 게이트)** 추가.
> 근거 문서: `docs/project-log.md`(마스터 원장), `bench/COMPETITIVE_VERDICT.md`(경쟁판정 SSOT),
> `docs/sigmod2026-fvs-postgresql-analysis.md`, `docs/macorn-penalty-findings.md`,
> `docs/build-perf-backlog.md`, `bench/OVERHEAD_LEDGER.md`,
> `docs/issue678-vdbbench-gap-analysis.md`(베이스라인 정직성 — [pgvector#678](https://github.com/pgvector/pgvector/issues/678)).

---

## 0. 갱신된 baseline — 무엇이 끝났나

| 영역 | 상태 |
|------|------|
| 멀티레이어 HNSW 빌드 / ACORN-γ / 공유 탐색 (Tier 1 hook + Tier 2 `acorn_hnsw` AM) | 완료 |
| **Phase 0 (측정·안전)**: page-I/O 계측, L_max(gamma-clamp) 경고, resume 스트리밍 스캔, 비용모델 재보정 | **완료** |
| 실데이터 + Qdrant + 3-way 스케일링 (100K/1M/10M) | **완료** (`REPORT_scale`/`REPORT_3way`) |
| Qdrant 차용 5종 (payload_m, auto-ef, payload-gate, telemetry, plan-choice) | 완료 |
| code-cache 스캔 최적화 | 완료 (default ON) |
| 빌드-perf 캠페인 (B1-B4, N1-N4) | 완료 — N4 CV-broadcast 버그 수정, two-pass/N1 음성 결론 |
| M-ACORN (빌드-시 predicate penalty) | **음성 결론** — ACORN scan-time gamma와 중복, 휴면 (`feat/macorn-penalty`) |

원래 로드맵의 **Phase 0는 사실상 닫혔다** (유일 잔여: `ef_search` C 상한 — 저우선 보류). 미해결 잔여:
`ef_search` C 상한(`acorn_scan.c:515` TODO, 저우선).

### 측정으로 확정된 격차/한계 (이 로드맵의 구동력)

| 한계 | 근거 | 영향 트랙 |
|------|------|----------|
| 높은 통과율(10-20%)+matched-recall에서 Qdrant latency 갭 (**배수 미해결/INDICATIVE**) | `REPORT_qdrant_final`(250K, cross-substrate)·`COMPETITIVE_VERDICT` | C |
| 대용량 빌드 **메모리 벽**: 그래프가 mwm 초과→on-disk 스필 | 2M@8GB·10M@32GB 실측 | B |
| 병렬 빌드 **extension-lock 라이브락** (스필+다워커) | 10M/8워커 분당 2블록 정체 | B |
| atomic allocator 회귀 (병렬 98 vs OLD 87분) | 빌드-perf 캠페인 | B |
| 직렬 빌드 꼬리 (WAL flush ~16분/82GB + P_NEW 한장씩) | 캠페인 + 라이브락 진단 | B |
| 쿼리 동시처리 **미검증** (throughput INDICATIVE) | 본 세션 | E |
| 단일 노드 / int4·256-파티션 필터 모델 | 코드 구조 | D |

---

## 그랜드 플랜 — North Star + 4-Phase 아크

**North Star:** *PostgreSQL 안에서 전용 벡터엔진(Qdrant/Pinecone)에 필적하는 필터드 ANN* —
트랜잭션·SQL·조인·on-disk를 유지한 채. pgvector 초월은 이미 달성. 다음은 전용엔진과의
**substrate(버퍼매니저) 갭**을 좁히는 것. 핵심 통찰(측정): 남은 갭은 거리계산이 아니라
**per-neighbor 버퍼 비용** — 스캔 exec의 ~45%가 neighbor 이중로드(`OVERHEAD_LEDGER.md`).

| Phase | 버전 | 목표 | 핵심 작업 | 졸업 기준 |
|-------|------|------|-----------|-----------|
| **I. 정직한 베이스라인 + quick win** | 0.1.x | 경쟁 숫자 확정 + 측정된 저위험 이득 | C0 재측정, **C0b/C0c 정합 베이스라인(work_mem·VDBBench·BQ-rerank·k100)**, **C1 이중로드 dedup(~1.5x)**, B1 벌크 사전확장, B3 atomic 게이트 | Qdrant 갭 단일 숫자 확정 + **iterative/pgvector 강베이스라인 정합 후 갭 확정** + C1 실측 + 10M 병렬 빌드 완주 |
| **II. substrate 갭 종결** (승부처) | 0.2.x | 버퍼 per-hop 비용 구조 제거 | shmem topology-colocated graph(~2.6x, ledger #4), D2 INCLUDE 컬럼나 | 고선택도서도 Qdrant 동급 latency |
| **III. 스케일 & 폭** | 0.3.x+ | 대용량·필터 폭 | B2 flush 병렬화, mwm/spill 견고화, D1 멀티/범위 필터, upstream PR | 10M+ 깨끗한 빌드/쿼리, 새 질의클래스 |
| **IV. 안정화 & 1.0** | 1.0 | 견고성·안정성 약속 | **Track S 전체(아래)**, API/포맷 안정화 | Track S exit 충족 + 논제 실증 |

> **투영치(~1.5x, ~2.6x)는 ledger(n=30K) 근거의 예측**이며 Phase I C0가 스케일에서 확정한다.

```
0.1.0 (now) ── 이김:pgvector / parity:recall / 갭:substrate-buffer
   │  Phase I  0.1.x ── C0 재측정 + C1 이중로드 dedup + B1 라이브락 제거
   │  Phase II 0.2.x ── colocated graph + INCLUDE → 전용엔진 필적 [승부처]
   │  Phase III 0.3.x ── 10M+ 빌드 / 범위·멀티필터 / upstream
   └  Phase IV 1.0  ── Track S(안정화) 완수 + 논제 실증
```

**관통 규율(이번 정리의 제도화):** ① 측정 우선(레버는 page-I/O로 증명 후 주장) ② 경쟁 숫자는
`COMPETITIVE_VERDICT.md` 단일 진실원, 단일 배수 quote 금지 ③ **반증 가설 재시도 금지**(two-pass·
M-ACORN·2-hop·N1 — `project-log.md` disproven) ④ 매 항해 후 `project-log.md`+memory `project-map` 갱신
⑤ **경쟁 베이스라인은 상대 최강 정합 구성으로**(iterative=work_mem 충분, pgvector=BQ-rerank 포함) — 약구성 상대 승리 인용 금지.

---

## 전략적 포지셔닝 / 가치 명제

**정밀화된 North Star**: "전용엔진에 필적"보다 정확히는 — *pgvector의 **구조적** 필터드 갭을,
pgvector-호환 형태로 메운다* → **standalone niche + upstream 후보(양방향 승리)**.

| 대상 | 그쪽의 필터드 검색 | pg_acorn의 자리 |
|------|-------------------|-----------------|
| **pgvector** | 보수적 scope. 0.8 iterative scan = 단순(인덱스 더 스캔). 우리 벤치서 **높은 통과율(10-20%)서 recall 0.22~0.50 붕괴** *(C0b/C0c 정합 재측정 전 잠정 — work_mem·BQ-rerank 정합 후 확정)* | pgvector가 의도적으로 단순히 푼 **구조적 갭** → 지속적. ACORN이 그 약점을 정조준 |
| **pgvectorscale** | StreamingDiskANN + **smallint 라벨(`&&`)만** (Filtered DiskANN). raw ANN 스케일/속도가 강점 | **다른 칸** — 일반-술어 ACORN은 비어 있음. 단 그들의 on-disk 스케일은 우리 약점 |

**생태계 landscape** (서베이: `docs/filterable-hnsw-landscape.md`): 그래프 기반 선두들은 **in-graph
filter-aware 순회로 수렴** — Qdrant 1.16(네이티브 ACORN), Weaviate(ACORN+적응 스위칭), Elastic/Lucene
(ACORN-1). prefilter/bitset 계열 = Milvus·MongoDB Atlas·Oracle 26. **pgvector만 `iterative_scan`
(discarded candidate에서 그래프 재탐색하되 traversal은 filter-agnostic — #678; "post-filter"는 부정확)**
— 즉 *필터를 *향해* 순회하는 ACORN-style은 pgvector가 유일하게 안 한 축*. pg_acorn은
그 빈 칸을 **pgvector 호환 형태**로 메운다. (단정 주의: filtered ANN 자체는 사실상 보편, "다들 ACORN"은
아님 — ACORN-style은 그래프 선두 3사에 한정.)

**upstream vs standalone (endgame)**: Tier 2(`acorn_hnsw`)는 pgvector HNSW 페이지 포맷 재사용 →
**DiskANN과 달리 pgvector가 "받아먹기" 가능**(아키텍처 표 "Upstream PR target = yes"). 따라서
가치는 양방향: (a) standalone "ACORN for Postgres", (b) pgvector upstream 기여. 둘 다 승리.

**단, 완성 조건 (Tier 1/2 split)**: 일반-술어 우위는 **Tier 1 hook**(임의 WHERE qual을 `ExecQual`로
평가, ACORN-1 search-time-only, upstream 불가)에서만 데모됨. 강한 인덱스(γ·payload·partition,
upstream 대상)인 **Tier 2는 필터 모델이 좁음**(단일 int4 + identity-mod-256). → "일반-술어 + 강한
인덱스"를 한 tier에 합치는 **Track D1이 niche 완성의 임계경로** (아래).

**pgvector에게의 가치 — 냉정한 평가**: pg_acorn은 pgvector를 *대체*하지 않는다(그 위에 얹힘). 따라서
"pgvector에게의 가치"는 경쟁가치가 아니라 **기여/영향 가치 = pgvector의 약점을 pgvector의 재료로
고쳐 보이는 것**이다. 이 축에서 가치는 *지속적·자연적합·축 정렬*이라 실재한다:

- **지속적**: pgvector가 필터드를 의도적으로 단순히(iterative scan) 풀었고 그게 높은 통과율서 붕괴
  (0.22~0.50 — *C0b/C0c 정합 재측정 전 잠정, work_mem·BQ-rerank 정합 후 확정*). 보수적이라 깊게 안 고침 → 갭이 안 닫힌다.
- **자연적합**: Tier 2가 HNSW 포맷 재사용 → DiskANN과 달리 받아먹기 가능(최고의 보완 형태).
- **축 정렬**: filtered ANN은 "벡터 in Postgres"의 존재이유. DiskANN(스케일/비용)은 어느 엔진이든
  쫓는 제네릭 축이자 자본·crowded 게임 — 솔로/PG 맥락에선 filtered가 방어가능한 올바른 베팅.

단 이 가치는 아직 **잠재태**다 — 세 게이트가 "잠재 → 실현"의 자물쇠:
1. **미완성** (Tier 1=일반술어이나 upstream불가; Tier 2=강하나 int4/256 좁음) → **D1**.
2. **증거 얇음** (단일호스트·INDICATIVE·합성) → **C0** 대규모 엄밀 재측정.
3. **메인테이너 수용성** (pgvector는 *이미* iterative scan을 택함 — ACORN 복잡도를 받을지는 lean
   철학상 거절 가능, 우리가 통제 못 하는 변수).

→ **netting**: 기여/영향 가치 = **중상(조건부)**, 셋 충족 시 **고**(upstream/niche 실현). standalone
프로덕트 가치 = **하~중**(Track S 미하드닝 + "iterative scan으로 족함" 관성). 즉 *경쟁자*가 아니라
**pgvector를 더 낫게 만드는 지속가능한 증명**으로서 냉정히도 가치는 실재하고 방어가능하다.

> **용어(selectivity)**: 본 문서는 **PostgreSQL 플래너 관례 = 통과 행 비율**을 쓴다(sel 20% = 20% 통과
> = "high"). 통상 어법("highly selective = few pass")과 *반대*라 혼동 위험 → 가능하면 **"통과율"**로
> 명시한다(예: "20% 통과율"). Qdrant 갭은 **높은 통과율(10-20%)**에서, ACORN 강점은 **낮은 통과율
> (0.1-1%, restrictive)**에서.

---

## 개발 축 (트랙) — 메인(B/C/D) + 안정화(S, 1.0 게이트) + 강등된 A

### Track A — 쿼리 알고리즘: 2-hop은 시도·폐기됨 (강등)

> **정정(2026-06-19)**: 원 로드맵(06-08)은 TM→2-hop→NaviX를 "연구 간판"으로 베팅했으나,
> 06-10에 **실제 구현·벤치된 뒤 폐기**됐다. 이 트랙은 미래작업이 아니라 *측정으로 반증된 가설*이다.

- **A1. Translation Map — 부분 실현됨**: TM의 핵심 이득(필터 평가 시 heaptid 재조회 제거)은
  `feb2d7c`의 query-local 노드 캐시(`acorn_t2_load_node` — element 페이지 단일 읽기로
  distance+metadata 결합, `acorn_scan.c`)로 **이미 실현돼 main에 살아있다**. 독립 HTAB TM은
  2-hop이 없으면 추가 실익이 불분명 → 보류.
- **A2/A3. 런타임 2-hop (ACORN-1) + NaviX-Directed — 구현 후 폐기 (근거 있음)**:
  `feb2d7c`가 NaviX-Directed 2-hop을 TDD로 구현(filter-실패 1-hop 노드를 거리순 min-heap D로
  모아 k_d=Max(m/2,4)개 확장, C 소진 시 drain). 벤치(`bench/RESULTS.md`)에서 **스케일 회귀로 패배**:
  - n=50K선 +0.017(도움) → **n=200K선 역전**: tier2_g1 @10% 선택도 2hop-**on 0.779 vs off 0.890 = −0.111**
  - "tier2_g2 (2hop off)가 이 스케일 최고 변종". 원인: D-injection이 가까운 *통과* 노드를 더 먼
    2-hop 이웃으로 밀어냄 → "보수적 drain 정책 필요"(RESULTS.md)
  - `283e10b`: 벤치 스윕에서 **2-hop 변종을 빼고 gamma=4로 교체** → **build-time ACORN-γ가 승자**
  - **0.1.0 정리**: 스캔 코드는 이미 삭제돼 있었고 `enable_2hop` GUC + `tier2_2hop` 테스트만
    inert 잔재로 남아 **제거**함. Phase 1 노드캐시는 유지.
- **남은 투기적 여지 (저신뢰)**: 보수적 D-drain 정책으로 2-hop 부활이 *가능할 수도* 있으나,
  측정상 γ-확장이 더 단순·강력해 우선순위 낮음. 재도전 시 **≥200K에서 γ=4 대비 A/B**가 게이트.

### Track B — 빌드 확장성/성능 (대용량 enabler · 본 세션 신규 트랙)
*왜*: 대용량의 실질 한계가 빌드 메모리 벽 + 직렬/라이브락임을 측정으로 확인. 상세는
`docs/build-perf-backlog.md`, 라이브락은 프로젝트 메모리 `build-extension-lock-livelock`.

- **B1. 벌크 사전확장 (최우선)** — `ReadBufferExtended(P_NEW)` 한장씩(`acorn_build.c:304`) →
  HNSW가 아는 최종 노드/페이지 수로 `smgrzeroextend` 일괄 확장 후 사전할당 블록에 기록.
  **병렬 빌드 extension-lock 라이브락 제거** + 직렬 빌드도 가속. flush 코드 한정, 중난이도.
- **B2. flush 병렬화** — 직렬 `log_newpage_range` flush(~16분/82GB at 10M)를 워커 분산.
- **B3. atomic allocator 회귀 정리** — B1/B2 lock-free 할당의 캐시라인 바운싱(98 vs 87분) 완화
  또는 GUC 게이트로 OLD 경로 선택 가능하게.
- **B4. spill 경로 견고화** — mwm 초과 시 graceful degrade (현재는 병렬 라이브락/직렬 초저속).

### Track C — 경쟁 갭: 높은 통과율(10-20%) latency (승부처)
*왜*: **recall은 Qdrant와 parity**(Z3 후 0.97-1.0). 남은 건 **고선택도+matched-recall의 latency
갭** — 단 **크기 미해결**(INDICATIVE, cross-substrate). 권위 판정은 `bench/COMPETITIVE_VERDICT.md`.
ledger(`OVERHEAD_LEDGER.md`)가 최대 수정가능 항목을 이미 짚음: `acorn_stream_expand`의
**neighbor 이중로드**(element 페이지를 distance·heaptid용으로 2회 핀) = scan exec의 ~45%.

- **C0. (전제) 깨끗한 재측정** — same-protocol·median-basis(prepared·literal·median+p99 분리),
  >=200K에서 Qdrant 갭 magnitude 확정. "1.6-4.4x"는 이게 끝나기 전엔 사실로 인용 금지.
  - **C0b. iterative_scan 정합 베이스라인 ([pgvector#678](https://github.com/pgvector/pgvector/issues/678) — 메인테이너 본인 설계)**: 현재 우리 측정은
    `strict_order` + `max_scan_tuples=40000`만(recall 0.22~0.50). **그러나 #678(ankane 2372612113/
    2390036999/2428287137)이 명시한 스캔 종료조건은 *튜플충족 | max_scan_tuples | work_mem* 중 최초 도달
    — 기본 work_mem(우리 fair-config 4MB)에선 후보힙이 수천에서 차므로 `max_scan_tuples=40000`이 절대
    binding이 안 되고 work_mem에서 끊긴다.** 즉 현 0.22~0.50엔 *work_mem 미상향 confound*가 섞일 수 있다
    (`bench/bench3way_pg.py:64-65`·`bench/scalebench.py:63-65` 모두 work_mem/scan_mem_multiplier 미설정).
    마저 할 것: **(i) `work_mem`(또는 `hnsw.search_mem_multiplier`)를 올려 max_scan_tuples가 실제 binding이
    되게, (ii) `relaxed_order` × max_scan_tuples 스윕, (iii) 현재 pgvector 버전**. **메커니즘 정정**:
    post-filter 아님 — discarded candidate를 entry로 **그래프 재탐색**(HnswSearchLayer)이되 *traversal이
    filter-agnostic*이라 상관필터 passing-cluster로 못 향함(이게 천장 원인). 거대 예산선 exact로 수렴함도 인정.
    **work_mem까지 충분히 준 정합 베이스라인 없이 "iterative collapse" 주장/이슈 게시 금지** (작성자에게
    메커니즘·설정 틀리면 신뢰 붕괴). 상세: `docs/issue678-vdbbench-gap-analysis.md` §3A.
  - **C0c. VDBBench 정합 + pgvector 강베이스라인 (비교가능성)**: 현 측정은 자체 하니스(합성 250K/
    Cohere 1M·10M)뿐이라 jkatz 공개 VDBBench 표(#678 2381458610, `Performance1536D5M`)·리더보드와 한 줄도
    대조 불가. 할 것: **(i) VDBBench Case 11(5M OpenAI 1536d)로 같은 좌표계 확보, (ii) pgvector `halfvec`/
    `binary_quantize`+rerank 베이스라인 추가**(jkatz 2392576311: BQ-rerank 2.34GiB @ r0.90 vs flat 38GiB —
    flat-만-상대 승리는 불충분), **(iii) k=100 축**(jkatz가 일부러 측정), **(iv) ankane이 #678에서 쓴
    qdrant ann-filtering-benchmark-datasets(arxiv 2.1M 384d)로 필터 주장 재현**. 상세: gap-analysis §3B-E.
- **C1. neighbor 이중로드 dedup** (ledger route #2) — element 페이지 1회 핀으로 distance+heaptid+
  deleted 동시 처리. 투영 ~1.5x(775→1170 QPS @ recall 0.953). C 변경, 저위험.
- **C2. 추가 처방** — 후보 C 상한(`ef_search` cap, 기존 TODO), 거리계산 SIMD/양자화,
  heap fetch 절감(→ Track D2 INCLUDE 연결). C0/C1 결과로 결정.

> 주: 한때 저선택도용으로 기대했던 2-hop은 폐기됐고(Track A), 고선택도 갭은 애초에 별도 문제다.

### Track D — 데이터모델/스코프 확장 (장기 · 큰 결정)
- **D1. Tier-2 술어 모델 확장 — niche 완성 임계경로** (포지셔닝 참조). 현재 Tier 2는 단일 **int4 +
  identity-mod-256**(`acorn_build.c:185`, "low-cardinality int4 가정: partition==value")만 강하게
  처리. 일반-술어 우위를 *강한 인덱스*에 합치려면 아래를 다뤄야 함.
  **핵심 렌즈 = 푸시다운 vs recheck**: 그래프 순회에 밀어넣을 술어 vs 인덱스 밖이라 post-filter
  recheck(heap+술어 재평가)할 술어의 경계 설계. (Tier 1=전부 recheck-during-traversal, Tier 2=인덱싱한
  것만 pushdown — 이 경계가 설계 축.)
  - **타입**: int4 외 int8/numeric/text/uuid/timestamp/bool/enum/array/jsonb. `filter_val` int64 가정 → 해시/인코딩 일반화(+ text collation).
  - **고카디널리티**: identity-mod-256 충돌(partition≠value) → payload 엣지 무의미. 해시 재설계/동적 파티션 수/per-value sub-index.
  - **연산자**: `=`(현재)뿐. `IN`(파티션 합집합)·`<>`/`NOT`(여집합)·`IS NULL`·`LIKE`·배열(`&&`,`@>`,`ANY`).
  - **범위(`BETWEEN`,`<`,`>`)**: 등식-파티션으로 표현 불가 → SeRF류 segment graph or 별도 질의클래스.
  - **합성**: `AND`(교집합, 쉬움) vs **`OR`(서브그래프 합집합, 어려움)**, 중첩; 다중 컬럼(단일 slot→복합키/다차원).
  - **표현식/함수**: `lower(c)=x`, `t::date=…` → expression-index or recheck.
  - **런타임/조인 값**: `=$1`(prepared, 런타임 entry 선택), 조인 유래 필터(상수 아님).
  - **분포/스큐/NULL**: 거대 단일 파티션, NULL-heavy, 비균등 저카디.
  - **비용추정**: 위 복합·범위·OR 술어의 선택도 추정 → acorn vs bitmap vs seqscan 라우팅(plan-choice 진단 확장).
  - **싼 interim (D1 전 main 반입 가능)**: max-card graceful-degrade(`payload_max_card`, N2가 build-perf
    브랜치에 존재) → 고카디 파티션은 global-only 폴백 = 정확성+무회귀 보장(차별성은 없지만 퇴화 안전).
- **D2. INCLUDE/컬럼나 필터 사본 (§13)** — 필터 컬럼 사본을 인덱스에 저장(PG11+ `INCLUDE`)해
  heap double-lookup 근본 제거. 논문 명시 장기 방향. 설계 스파이크부터.

### Track E — 검증/운영 (지속)
- **E1. 쿼리 동시처리 실측 (미검증)** — 멀티클라이언트 QPS 코어 스케일링, 락/공유버퍼 병목.
  코드상 동시 스캔 대응 흔적(overlapping scan ref, backend-local 카운터)은 있으나 부하 실측 없음.
- **E2. latency/throughput 정밀 측정** — 현재 INDICATIVE 단계 탈출(측정 환경 확정). **로드젠 분리(별도
  클라이언트 호스트)** 명시 — clients 동일호스트 localhost가 transport-bound 근본원인(#678 2444063432
  wahajali 미해결 질문 = 우리 약점).
- **E3. index-fits-in-RAM 특성화**, 단일노드 천장 문서화.
- **E4. 순서충실도(out-of-order율) 측정** — recall과 별개 1급 정합성 지표(#678 2379953370 선례:
  batch_size×selectivity 표). strict/relaxed 및 pg_acorn 자체 순서 보장 정량화. 현재 recall@10만 측정.

> E1/E2/E3는 아래 Track S(안정화)의 *검증* 입력이다 — S가 그것을 하드닝 작업으로 흡수한다.

### Track S — Stabilization / Hardening (**1.0 게이트**)
*왜*: 성능 트랙(B/C/D)과 별개로, 견고성 3축을 명시적으로 안정화해야 1.0을 부를 수 있다.
기반은 있으나(BUG-3 동시성 수정 + isolation 4종) **부하·스케일 하 검증·하드닝이 비어 있다.**

- **S1. 대용량 안정화** — (a) index > shared_buffers 랜덤-IO 절벽 특성화 → PG17 prefetch/ReadStream
  검토(ledger #7: 콜드일 때만 유효), (b) 10M+ 크래시복구·재시작 무결성 + 대규모 빌드 RSS 회귀 가드,
  (c) spill 경로 graceful degrade(B1과 묶음). [E3 흡수]
- **S2. 빌드 병렬 안정화** — (a) B1 후 **spill+병렬 stress 테스트 신설**(현 parity는 in-memory 위주),
  (b) atomic 회귀 GUC 게이트(B3) + 워커 스케일링 상한 문서화, (c) build_seed 결정성 회귀 가드.
- **S3. 다중세션 안전성 (최우선 보강)** — (a) **E1 부하 검증**: pgbench 16/32 동접 필터드-KNN QPS
  코어 스케일링 + LWLock 경합 계측(pgvector #766 류), (b) code-cache **지속 churn + 고동접** stress
  로 "ef=1600 크래시=환경적" 가설을 결정적으로 닫기, (c) 동시 insert/update/delete + scan + vacuum
  race + MVCC 가시성 stress(현 `concurrent_insert_scan` 너머).
  *exit*: 정의된 동접·스케일에서 0 크래시 · 0 잘못된결과 · QPS 선형 스케일 입증.

> **자산**: 병렬빌드 parity/spill/stress 회귀(`tier2_build_parallel`), code-cache 동시성(BUG-3
> dsa double-unpin 수정 + hazard ptr) + isolation 4종(`concurrent_insert_scan`/`cache_insert_scan`/
> `cache_evict_scan`/`gamma_build`). S는 이 위에 **부하·스케일 차원**을 더한다.

---

## 의존성 그래프 / 시퀀스

```
[지금]
 ├─ C0 (깨끗한 재측정) → C1 (이중로드 dedup) → C2     ← 갭 magnitude 미해결, 저위험 진입
 ├─ B1 (벌크 사전확장) ──→ B2/B4                        ← 대용량 빌드 잠금 해제, 자족적
 └─ A1 노드캐시 실현됨 · A2/A3 2-hop은 폐기(γ 승)        ← 트랙 강등, 부활은 투기적
                                  │
[E1/E2 측정]은 C 효과 증명의 전제 ┘
```

**왜 이 순서인가**: (1) C·B는 *측정으로 확정된 한계*를 직접 공략하고 자족적이라 즉시 가치.
(2) Track A의 핵심(2-hop)은 이미 시도→폐기됐으므로 신규 알고리즘 작업은 우선순위에서 내려간다 —
γ-확장이 현 시점 답. (3) SIGMOD §1: "distance 계산 수는 PG end-to-end의 신뢰 못 할 proxy" →
어떤 신규 쿼리 최적화든 page-I/O(Phase 0 완료)로 **≥200K에서 γ=4 대비 A/B 증명**이 게이트.

## 우선순위 요약

| 순위 | 항목 | Phase | 근거 |
|------|------|-------|------|
| **P0** | **C0b/C0c 정합 베이스라인** (work_mem·VDBBench·BQ-rerank·k100) | I | upstream 이슈/논문 게시 신뢰성 전제 — 성능 이득(C1)보다 먼저 닫을 리스크 |
| **P0** | C0 깨끗한 재측정 → C1 이중로드 dedup | I | 유일 잔여 성능 갭(magnitude 미해결), 진입 저위험, ~1.5x |
| **P0** | B1 벌크 사전확장 | I | 대용량 빌드 라이브락 제거 — 자족적, S1·S2 교집합 |
| **P1** | **S3 다중세션 안전성** (E1 부하검증부터) | I→IV | 가장 미검증·"기본기"의 핵심 — 프로덕션 신뢰 전제 |
| **P2** | S1 대용량 안정화 / S2 빌드병렬 안정화 / B2 flush | III·IV | B1 위에서 하드닝; 1.0 게이트 |
| **P2** | C2 추가 처방 (ef cap·SIMD·heap fetch) | II | C0/C1 결과로 결정 |
| **P3** | colocated graph(II) / D 데이터모델·컬럼나 | II·III | 승부처·큰 아키텍처 결정 |
| **보류** | A2 2-hop 부활(보수적 D-drain) | — | 이미 폐기됨(γ 승) — 투기적, 저신뢰 |

> **1.0 게이트 = Track S 전체 exit 충족.** 성능(C/colocated)이 1.0을 부르지 않는다 — 견고성이 부른다.

## 권장 다음 1~2 액션
1. **B1 (벌크 사전확장)** — 대용량 빌드 라이브락/저속을 정공법으로 제거. 자족적·즉시 가치, A/C와 독립.
2. **C0/C1 (Qdrant 갭)** — 깨끗한 median-basis 재측정으로 magnitude 확정 + ledger route #2(neighbor 이중로드 dedup, ~1.5x) 구현.

---

## 부록 — 신규 알고리즘 지형 (FAVOR / JAG / Curator / SeRF / KHI)

대부분 Track D(스코프 확장) 이후. 세 게이트로 *언제*가 갈린다.
- **G1 아키텍처 적합**: filter-agnostic이며 pgvector 페이지 포맷(`hnsw_compat.h`) 재사용 가능?
- **G2 문제 동일성**: 속성 *동등* WHERE 술어 문제? (범위/멀티테넌트는 다른 문제)
- **G3 PG 검증**: PG에서 우위 입증? (측정 *인프라*는 완비 — page-I/O 계측 + 실데이터 3-way
  하네스. 우리 ACORN-γ는 측정됨. 단 아래 신규 알고리즘들은 **PG 포팅 자체가 없어** PG 우위 미지.)

| 알고리즘 | 계열 | G1 | G2 | 배치 | 비고 |
|---------|------|----|----|------|------|
| **FAVOR** ([2605.07770](https://arxiv.org/abs/2605.07770)) | filter-agnostic graph (HNSW) | O | O | **투기적** (2-hop 폐기 후) | ACORN/NaviX의 사촌. selectivity-aware exclusion distance. 폐기된 2-hop의 "보수적 drain" 대안이 될 수 있으나 ≥200K에서 γ 대비 입증 필요 |
| **Curator** ([2401.07119](https://arxiv.org/abs/2401.07119)) | 파티션/트리 (per-label) | X | △ | Track D | ScaNN 계열. 저선택도·멀티테넌트·음상관 보완. 새 AM |
| **JAG** ([2602.10258](https://arxiv.org/abs/2602.10258)) | attribute graph | X | O | Track D+ | 속성을 그래프에 결합. 새 디스크 포맷/AM |
| **SeRF** ([SIGMOD'24](https://miaoqiao.github.io/paper/SIGMOD24_SeRF.pdf)) | range-filter segment graph | X | X | Track D+ | 범위 필터 전용. 질의 클래스 확장 결정 |
| **KHI** | 다속성 파티션 트리 + per-node HNSW | X | X | Track D+ | 다속성 범위 RFANNS. 새 AM |

- **부류 A (FAVOR)**: 우리 filter-agnostic graph 계열 → Track A에 가깝게 끼울 수 있음.
- **부류 B (JAG/Curator/SeRF/KHI)**: 필터를 인덱스에 박는 specialized index → 새 AM/다른 질의클래스.
  "채택"이 아니라 "방향 전환", Track D 이후. 전부 standalone 라이브러리 벤치 결과뿐 → 우리 하네스로
  측정하려면 **PG 포팅이 선행**돼야 함 (측정 인프라는 이미 있음).

> 서베이: RF-ANNS 분류 [arXiv:2505.06501](https://arxiv.org/abs/2505.06501),
> FANNS 벤치 [arXiv:2507.21989](https://arxiv.org/html/2507.21989v1).
