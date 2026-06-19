# pg_acorn 발전 로드맵

> 최초 작성: 2026-06-08 (SIGMOD 2026 FVS-in-PG 분석 기반, 쿼리-side Phase 0→3)
> 현 시점 재정리: 2026-06-19 — 3-way 벤치·스케일링, 빌드-perf 캠페인(B1-B4/N1-N4),
> M-ACORN 음성 결론, extension-lock 라이브락 발견을 반영해 **트랙 구조로 재편**.
> 근거 문서: `docs/sigmod2026-fvs-postgresql-analysis.md`, `docs/macorn-penalty-findings.md`,
> `docs/build-perf-backlog.md`, `bench/REPORT_scale.md`, `bench/REPORT_3way.md`.

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
| 고선택도에서 Qdrant 대비 **1.6~4.4배 느림** | 3-way 벤치 | C |
| 대용량 빌드 **메모리 벽**: 그래프가 mwm 초과→on-disk 스필 | 2M@8GB·10M@32GB 실측 | B |
| 병렬 빌드 **extension-lock 라이브락** (스필+다워커) | 10M/8워커 분당 2블록 정체 | B |
| atomic allocator 회귀 (병렬 98 vs OLD 87분) | 빌드-perf 캠페인 | B |
| 직렬 빌드 꼬리 (WAL flush ~16분/82GB + P_NEW 한장씩) | 캠페인 + 라이브락 진단 | B |
| 쿼리 동시처리 **미검증** (throughput INDICATIVE) | 본 세션 | E |
| 단일 노드 / int4·256-파티션 필터 모델 | 코드 구조 | D |

---

## 개발 축 — 3 메인 트랙 + 2 보조

### Track A — 쿼리 알고리즘 프런티어 (연구 간판 · 장기 차별화)
원 로드맵 Phase 1→2. *왜*: 오픈소스 PostgreSQL 최초의 제대로 된 graph-based filtered
vector search (SIGMOD §7). NaviX·TM은 AlloyDB 구독으로도 못 쓰는 연구 프로토타입.

- **A1. Translation Map** (`indextid → heaptid` 인메모리 `HTAB`, SIGMOD §5) — 2-hop의 전제.
  빌드 후 전체 1회 스캔으로 구성, 엔트리 ~18B (1M=18MB, 35M=630MB). 쿼리 시 필터 평가 경로의
  heaptid 해석을 ~10ns 조회로 대체. 중난이도.
  > 1-hop은 이미 element 튜플을 읽어 heaptid가 공짜이므로, TM의 실익은 A2가 생겨야 발생 → A2와 묶어 검증.
- **A2. 런타임 2-hop (ACORN-1, §4·§6.1)** — γ=1로 저장비용 없이 effective density 확보.
  A1 위에서만 실행 가능 (2-hop은 스텝당 1+M+M² 페이지). pgvector 페이지 천장(§10)에 막히는
  초대규모·초저선택도 영역을 연다. 대난이도.
- **A3. NaviX-Directed (§12.2)** — 1-hop을 거리순 정렬 → 가까운 비통과 노드의 2-hop 먼저.
  논문상 ACORN 대비 1.2~1.7x. A2 위에서만.
- **회귀 가드 (필수)**: 이 변경들은 Tier 1 공유 `acorn_scan.c`를 건드림. 가법적으로 γ=1만 2-hop,
  γ>1은 기존 동작 보존. 머지 게이트: `no_regression.sql`(acorn==pgvector top-k) + `recall_filter`,
  ≥40% 선택도 recall ≥0.9. 효과는 Phase 0 `pages_per_query`로 증명.

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

### Track C — 경쟁 갭: 고선택도 latency (승부처)
*왜*: 측정상 고선택도에서 Qdrant 1.6~4.4배 느림 — 유일하게 명확한 패배 지점.

- **C1. 고선택도 경로 프로파일링** — 시간 소비 분해(그래프 순회 vs heap fetch vs 거리계산).
  처방의 전제. 저위험 진입.
- **C2. 원인별 처방** — 후보 C 상한(`ef_search` cap, 기존 TODO), 거리계산 SIMD/양자화,
  heap fetch 절감(→ Track D2 INCLUDE 연결). C1 결과로 결정.

> Track A(2-hop)는 저선택도·초대규모를 열지만, 고선택도 갭은 별도 문제라 C로 분리.

### Track D — 데이터모델/스코프 확장 (장기 · 큰 결정)
- **D1. 필터 모델 확장** — 현재 단일 int4 + 256 파티션(`filter_val & 255`) 한계. 멀티컬럼/
  고카디널리티/범위 필터. 범위로 가면 SeRF/KHI 영역 = 새 AM/질의클래스 결정.
- **D2. INCLUDE/컬럼나 필터 사본 (§13)** — 필터 컬럼 사본을 인덱스에 저장(PG11+ `INCLUDE`)해
  heap double-lookup 근본 제거. 논문 명시 장기 방향. 설계 스파이크부터.

### Track E — 검증/운영 (지속)
- **E1. 쿼리 동시처리 실측 (미검증)** — 멀티클라이언트 QPS 코어 스케일링, 락/공유버퍼 병목.
  코드상 동시 스캔 대응 흔적(overlapping scan ref, backend-local 카운터)은 있으나 부하 실측 없음.
- **E2. latency/throughput 정밀 측정** — 현재 INDICATIVE 단계 탈출(측정 환경 확정).
- **E3. index-fits-in-RAM 특성화**, 단일노드 천장 문서화.

---

## 의존성 그래프 / 시퀀스

```
[지금]
 ├─ C1 (고선택도 프로파일링) ─────────→ C2 처방        ← 측정된 패배 지점, 저위험 진입
 ├─ B1 (벌크 사전확장) ──→ B2/B4                        ← 대용량 빌드 잠금 해제, 자족적
 └─ A1 (Translation Map) ─→ A2 (2-hop) ─→ A3 (NaviX)   ← 연구 간판, 길고 Tier1 회귀 위험
                                  │
[E1/E2 측정]은 A·C 효과 증명의 전제 ┘
```

**왜 이 순서인가**: (1) C·B는 *측정으로 확정된 한계*를 직접 공략하고 자족적이라 즉시 가치.
(2) A는 가장 큰 차별화지만 길고 Tier 1 회귀 위험이 있어, B1/C1로 빠른 가치를 확보한 뒤 본격 착수.
(3) SIGMOD §1: "distance 계산 수는 PG end-to-end의 신뢰 못 할 proxy" → page-I/O(Phase 0 완료) 없이는
A의 효과 증명 불가. A2는 A1 없이는 PG에서 실행 불가(2-hop 페이지 폭증), A3는 A2 위에서만 의미.

## 우선순위 요약

| 순위 | 항목 | 근거 |
|------|------|------|
| **P0** | C1 고선택도 프로파일링 | 측정된 유일 패배 지점, 진입 저위험 |
| **P0** | B1 벌크 사전확장 | 대용량 빌드 라이브락 제거 — 자족적, 영향 큼 |
| **P1** | A1 Translation Map → A2 2-hop | 연구 차별화, 저선택도·초대규모 개방 |
| **P1** | E1 동시처리 실측 | "기본기 OK?" 미검증분 종결 |
| **P2** | A3 NaviX / B2 flush / C2 처방 | 상위 의존 |
| **P3** | D 데이터모델 / 컬럼나 | 큰 아키텍처 결정, 실데이터 음상관 결과 보고 |

## 권장 다음 1~2 액션
1. **B1 (벌크 사전확장)** — 대용량 빌드 라이브락/저속을 정공법으로 제거. 자족적·즉시 가치, A/C와 독립.
2. **C1 (고선택도 프로파일링)** — Qdrant 갭 원인 분해. C2 처방의 전제.

---

## 부록 — 신규 알고리즘 지형 (FAVOR / JAG / Curator / SeRF / KHI)

대부분 Track D(스코프 확장) 이후. 세 게이트로 *언제*가 갈린다.
- **G1 아키텍처 적합**: filter-agnostic이며 pgvector 페이지 포맷(`hnsw_compat.h`) 재사용 가능?
- **G2 문제 동일성**: 속성 *동등* WHERE 술어 문제? (범위/멀티테넌트는 다른 문제)
- **G3 PG 검증**: PG에서 우위 입증? (Phase 0 측정 전엔 전부 미지)

| 알고리즘 | 계열 | G1 | G2 | 배치 | 비고 |
|---------|------|----|----|------|------|
| **FAVOR** ([2605.07770](https://arxiv.org/abs/2605.07770)) | filter-agnostic graph (HNSW) | O | O | **Track A 2.5 후보** | ACORN/NaviX의 사촌. selectivity-aware exclusion distance. NaviX-Directed 대안 휴리스틱 |
| **Curator** ([2401.07119](https://arxiv.org/abs/2401.07119)) | 파티션/트리 (per-label) | X | △ | Track D | ScaNN 계열. 저선택도·멀티테넌트·음상관 보완. 새 AM |
| **JAG** ([2602.10258](https://arxiv.org/abs/2602.10258)) | attribute graph | X | O | Track D+ | 속성을 그래프에 결합. 새 디스크 포맷/AM |
| **SeRF** ([SIGMOD'24](https://miaoqiao.github.io/paper/SIGMOD24_SeRF.pdf)) | range-filter segment graph | X | X | Track D+ | 범위 필터 전용. 질의 클래스 확장 결정 |
| **KHI** | 다속성 파티션 트리 + per-node HNSW | X | X | Track D+ | 다속성 범위 RFANNS. 새 AM |

- **부류 A (FAVOR)**: 우리 filter-agnostic graph 계열 → Track A에 가깝게 끼울 수 있음.
- **부류 B (JAG/Curator/SeRF/KHI)**: 필터를 인덱스에 박는 specialized index → 새 AM/다른 질의클래스.
  "채택"이 아니라 "방향 전환", Track D 이후. 전부 standalone 벤치 결과뿐 → PG 우위는 측정 후 판단.

> 서베이: RF-ANNS 분류 [arXiv:2505.06501](https://arxiv.org/abs/2505.06501),
> FANNS 벤치 [arXiv:2507.21989](https://arxiv.org/html/2507.21989v1).
