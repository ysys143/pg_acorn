# pgvector#678 (Iterative Index Scans) 정독 + VDBBench 방법론 갭 분석

> 출처: [pgvector/pgvector#678](https://github.com/pgvector/pgvector/issues/678) (댓글 44개 전체, 2024-09-22 ~ 2024-10-30).
> 지목 댓글: [comment 2381458610](https://github.com/pgvector/pgvector/issues/678#issuecomment-2381458610) (jkatz, VectorDBBench 결과).
> 작성: 2026-06-19. 목적: 이 스레드가 확립한 측정 방법론과 사실관계 대비, pg_acorn 벤치/주장이 놓친 점 정리.

---

## 0. 한 줄 결론

우리는 `iterative_scan`을 베이스라인에 포함은 했지만, 이 스레드가 사실상의 표준으로 굳힌 **VDBBench(VectorDBBench) 방법론**의 핵심 축 여러 개를 빠뜨렸다. 특히 **iterative 베이스라인에 `work_mem`/`scan_mem_multiplier`를 올리지 않아** "pgvector iterative는 상관필터에서 recall 0.2–0.5로 붕괴"라는 우리의 헤드라인 주장이 *misconfiguration 아니냐*는 반박에 취약하다.

---

## 1. 스레드의 정체 (프레이밍 교정)

#678은 **ACORN 스레드가 아니다.** pgvector 저자(ankane)가 **iterative index scan**을 제안·논의한, 즉 **pgvector 0.8.0의 `hnsw.iterative_scan` / `ivfflat.iterative_scan`의 기원 스레드**다. 여기서 다루는 "필터 문제"는 ACORN식 pre-filter가 아니라 **post-filter 과잉필터링(overfiltering)으로 LIMIT을 못 채우는 문제**다.

핵심 사실:

- **OP 미해결 문제**(본문): `SELECT * FROM items WHERE embedding <-> $1 < 0.1 ORDER BY embedding <-> $1 LIMIT 10;` — 거리 임계 필터에서 LIMIT보다 적은 행만 임계 내에 있으면 스캔이 조기 종료되지 않는다. ankane은 이 때문에 iterative를 기본값으로 켜기 어렵다고 함(comment 2380877893).
- **스캔 종료 조건**(ankane 2372612113 / 2390036999 / 2428287137): *(a) 충분한 튜플 확보, (b) `hnsw.ef_stream`(후의 `max_scan_tuples`, 기본 20,000), (c) `work_mem` 초과* 중 **먼저 도달하는 것**에서 종료.
- **strict vs relaxed**: `relaxed_order`는 recall↑이지만 **결과 순서가 깨짐**(ORDER BY SQL 시맨틱 위반). alanwli/jkatz/ankane의 서브스레드(2372509536–2379999605), out-of-order율 표(2379953370).
- **IVFFlat**: strict는 거의 효과 없음(첫 iteration 이후 대부분 discard), relaxed만 의미(2389665802).
- **최종 출시 GUC**(2406345995, 2436628408): `hnsw.iterative_search = strict_order | relaxed_order`, `ivfflat.iterative_search = relaxed_order`, `hnsw.max_scan_tuples`(기본 20,000), `hnsw.search_mem_multiplier`. → 우리 코드의 `hnsw.iterative_scan` 명칭은 0.8.0 릴리스와 일치(스레드 토론 중 `iterative_search`는 출시 전 가칭).

---

## 2. comment 2381458610 (jkatz)이 확립한 VDBBench 방법론 (골드 스탠다드)

| 축 | jkatz가 실제로 한 것 |
|---|---|
| 하니스 | **VectorDBBench** 표준 유틸 (자체 제작 아님) — 공개 리더보드/타 결과와 직접 대조 가능 |
| 데이터셋 | **Case 11 = `Performance1536D5M` (5M OpenAI 1536d)** + 10M Cohere 768d |
| k | **k=10 그리고 k=100 둘 다** (큰 k 거동 확인 목적 명시) |
| 측정 지표 | recall, **QPS(single client)·QPS(max=동시성)**, **single p99·max QPS p99**, **index size(GiB)** |
| 곡선 | ef_search **전 구간 사다리**(80→100→120→155→180→240→350→600) = recall↔QPS Pareto 전체 (단일점 아님) |
| 양자화 | **flat / halfvec / bq-rerank** 비교 (BQ-rerank: 2.34 GiB @ r0.90 vs flat 38 GiB) (2392576311) |
| 환경 | r7gd.16xlarge(64 vCPU/512 GB), `shared_buffers=128GB`, `work_mem=64MB`, `effective_cache_size=256GB`, **워크로드 전부 메모리 상주** |
| 미해결 질문 | wahajali(2444063432): "VDBBench가 DB와 같은 인스턴스냐 별도 VM이냐?" → 답 안 됨(#700로 이관) |

추가로 jkatz가 확인한 함정: **ef_search < LIMIT 일 때 recall이 인위적으로 부풀려진다**(iterative가 LIMIT 채우려 더 파기 때문). ef_search ≥ LIMIT가 되면 recall이 정상 정렬됨(2381458610, ankane와의 offlist 논의).

---

## 3. 우리가 놓친 것 (우선순위 순, 근거 포함)

### 🔴 A. [공정성·치명] iterative 베이스라인에 `work_mem`/`scan_mem_multiplier`를 안 올림

- 스레드의 종료 조건상 **work_mem이 cap의 하나**(2372612113). 그런데 우리 하니스는
  - `bench/bench3way_pg.py:64-65`
  - `bench/scalebench.py:63-65`

  둘 다 **`iterative_scan='strict_order'` + `max_scan_tuples=40000`만 설정하고 `work_mem`/`hnsw.scan_mem_multiplier`는 손대지 않는다**(grep: 쿼리타임 work_mem/scan_mem_multiplier 설정 "NONE FOUND").
- 우리 자신의 fair-config 주석(`bench/head2head_fair.py:4`)이 보여주듯 PG 기본 `work_mem`은 4MB 수준. dim 1024에서 4MB면 discarded-candidate 힙은 수천 개 수준에서 차므로 **`max_scan_tuples=40000`은 절대 binding이 안 되고 work_mem에서 먼저 끊긴다.**
- 결과: `bench/REPORT_3way.md:67,79`의 헤드라인 — "strict_order + max_scan_tuples=40000으로 공정하게 줘도 iterative는 0.22–0.50에 캡" — 이 **"work_mem을 안 올린 misconfig 때문 아니냐"**는 반박에 취약. pgvector가 정확히 이 용도로 문서화한 노브(`hnsw.search_mem_multiplier`, 2436628408)를 안 건드림.
- **조치**: iterative에 `work_mem`을 크게(예 2GB) 또는 `hnsw.scan_mem_multiplier`를 올려 `max_scan_tuples`가 실제 binding이 되게 재측정. 이게 닫혀야 헤드라인이 방어 가능.

### 🔴 B. [비교가능성] VDBBench 표준 케이스를 안 돌림 → 우리 숫자가 고립

- jkatz의 표(2381458610)와 공개 VDBBench 리더보드는 `Performance1536D5M`(5M OpenAI 1536d) 기준. 우리는 자체 `thesis_validation`(합성 250K dim128) + Cohere 1024d만 사용. **동일 케이스가 없어 jkatz 공개 숫자와 한 줄도 대조 불가.**
- VDBBench엔 pgvector 클라이언트가 이미 있고, pg_acorn을 클라이언트로 추가하면 리더보드 맥락에 우리 결과를 얹을 수 있는데 그 통합이 없음.

### 🟠 C. [축 누락] k=100을 한 번도 안 봄

- 전 하니스가 `K=10` 고정(`bench3way_pg.py`, `scalebench.py`, `iterative_scan_bench.py:19` 등). jkatz는 k=100을 **일부러** 측정(2381458610). 큰 k는 iterative/ACORN 모두를 다르게 압박.

### 🟠 D. [경쟁 베이스라인 누락] halfvec / `binary_quantize`+rerank 미비교

- bench 전체에 `halfvec`/`binary_quantize`/`bq` 흔적 0건. 그러나 pgvector의 실제 권장 고성능 경로는 **BQ-rerank**(jkatz 2392576311: 2.34 GiB 인덱스 @ r0.90, flat 38 GiB 대비). 우리는 **flat HNSW만 상대**로 이김 — pgvector의 진짜 강구성과는 미대결.
- README는 ">50M은 compression 써라"라고 말하면서 정작 pgvector+BQ를 베이스라인으로 측정한 적이 없음.

### 🟠 E. [데이터셋] ankane이 이 스레드에서 실제로 쓴 필터 벤치셋을 안 씀

- ankane은 필터 벤치를 **qdrant `ann-filtering-benchmark-datasets`(arxiv 2.1M, 384d)**로 돌림(2380359173, 2389665802). 이게 공개 표준 필터-ANN 데이터셋인데 우리는 자체 합성/Cohere로 대체. 표준셋을 쓰면 필터 주장이 즉시 재현·대조 가능.
- jkatz가 추천한 ANN-Benchmarks 표준셋(`dbpedia-openai-1000k-angular`, `sift-128-euclidean`, `gist-960-euclidean`)도 미사용.

### 🟡 F. [부하생성기 배치] wahajali의 미해결 질문이 곧 우리 약점

- 스레드 마지막 미해결 이슈(2444063432): "로드젠 동일호스트 vs 별도 VM". 우리는 **clients를 같은 호스트 localhost TCP로 실행**(`bench/REPORT_scale.md` "Clients run on the same host", caveat 4 자인). 우리 QPS는 transport/client-bound.
- VDBBench 모범관행은 **별도 클라이언트 VM**. 우리는 미적용 → throughput "INDICATIVE" 꼬리표의 근본 원인.

### 🟡 G. [정합성 지표] out-of-order율을 안 잼

- 스레드는 순서 깨짐을 recall과 별개의 1급 정합성 지표로 취급하고 batch_size×selectivity 표까지 만듦(2379953370). 우리는 recall@10만 측정하고 **결과 순서 정확도(out-of-order %)는 미측정.** pg_acorn 자체가 strict 순서를 보장하는지/얼마나 어긋나는지 정량화 없음.

### 🟡 H. [공정 프레이밍] iterative는 "결과 부족" 해결용이지 "상관필터 recall" 해결용이 아님

- jkatz/ankane 모두 iterative를 **overfiltering(LIMIT 미충족)** 대책으로 설계했다고 명시. ACORN이 노리는 "상관필터 하 recall" 문제와는 **다른 문제.** 우리가 "pgvector iterative 붕괴"라고 칠 때, 설계 목적 밖 시나리오로 공격하는 셈(틀린 건 아니나 프레이밍이 불공정으로 읽힐 수 있음).
- jkatz가 짚은 *ef<k에서 recall 인위적 범프*(2381458610)를 우리는 `PGV_EFS` 최소 40>k=10이라 회피했지만, 동시에 **iterative가 가장 빛나는 저-ef 구간도 안 봄.**

### ⚪ I. [기타]

- **IVFFlat iterative(relaxed)** 베이스라인 0건 — README는 50M+에 IVFFLAT을 권하면서 벤치 없음.
- **거리 임계 술어**(OP 핵심 미해결): pg_acorn도 `WHERE dist < x` 종료 문제를 풀지도, 언급하지도 않음 — 범위 술어가 scope 밖임을 명시할 필요.

---

## 4. 권장 조치 (영향 큰 순)

1. **iterative 베이스라인 재측정**: `work_mem`(또는 `hnsw.search_mem_multiplier`)를 올려 `max_scan_tuples`가 실제 binding이 되게 → A를 닫아야 헤드라인이 방어됨. (두 하니스의 `pgv_iterative` 분기만 수정.)
2. **VDBBench Case 11(5M OpenAI)로 측정** → jkatz 표와 같은 좌표계에 우리 숫자를 올리기 + **k=100 추가**.
3. **pgvector halfvec / BQ-rerank를 베이스라인에 추가** (pgvector 진짜 강구성과 대결).
4. **qdrant ann-filtering-benchmark-datasets**로 필터 주장 재현.
5. **out-of-order율** 측정 추가, **로드젠 분리** 또는 최소한 transport-bound 한계를 더 분명히.
6. README/ABOUT의 "pgvector는 post-filter라 SeqScan 폴백" 서사를 "0.8.0 iterative_scan을 충분한 메모리로 줘도 상관필터에서는…"로 정밀화(공정 프레이밍).

---

## 5. 코드 레퍼런스

- `bench/bench3way_pg.py:58-69` — pgv_iterative GUC 설정 (work_mem 누락)
- `bench/scalebench.py:62-65` — 동일
- `bench/iterative_scan_bench.py:19,87-91` — K=10, relaxed/strict
- `bench/head2head_fair.py:4,13` — 기본 work_mem=4MB 맥락
- `bench/REPORT_3way.md:67,79,146-163` — "iterative 붕괴" 헤드라인 + 메커니즘
- `bench/REPORT_scale.md:20,117-120` — iterative 두 경로, 저선택도 tail 폭발

## 6. 출처

- [pgvector/pgvector#678](https://github.com/pgvector/pgvector/issues/678) — 댓글 44개 전체
- 핵심 댓글: 2381458610, 2381838651, 2392576311, 2444063432, 2372612113, 2380877893, 2379953370, 2389665802, 2406345995, 2436628408
- [zilliztech/VectorDBBench](https://github.com/zilliztech/VectorDBBench)
- [qdrant/ann-filtering-benchmark-datasets](https://github.com/qdrant/ann-filtering-benchmark-datasets)
