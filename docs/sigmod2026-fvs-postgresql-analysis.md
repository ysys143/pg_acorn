# SIGMOD 2026 FVS in PostgreSQL — 논문 분석 및 pg_acorn 설계 지침

**논문**: "An In-Depth Study of Filter-Agnostic Vector Search on a PostgreSQL Database System: [Experiments & Analysis]"  
**출처**: arXiv:2603.23710 · Proc. ACM Manag. Data 4(3), Article 134 (SIGMOD 2026)  
**저자**: Duo Lu (Brown), Helena Caminal, Yannis Papakonstantinou, Yannis Chronis, Vaibhav Jain, Fatma Özcan (Google), Manos Chatzakis (Université Paris Cité)  
**작성 목적**: pg_acorn 구현 결정에 직접 영향을 미치는 발견을 정리하고 설계 지침을 도출한다.

---

## 1. 논문의 핵심 주장

기존 FVS(Filtered Vector Search) 연구는 대부분 **standalone library**(HNSWLib 등)에서 벤치마크를 수행하며, PostgreSQL 같은 실제 DBMS의 시스템 오버헤드를 무시한다. 이 논문은 production-grade PG 호환 시스템(AlloyDB 기반)에서 filter-agnostic 알고리즘들을 실측하여, library 결과가 DBMS 결과와 **질적으로 다르다**는 것을 최초로 체계적으로 증명한다.

**Figure 1 요약** (OpenAI-1M, 100 쿼리 평균 레이턴시):

```
Selectivity     1%      10%     100%
Library gap:    2x      1.5x    1x     (ACORN vs Sweeping)
PGVector gap:   역전    1.2x    1x
System cost:    최대 10x 높음
```

- PGVector 전체 레이턴시는 HNSWLib 대비 **최대 10x 높다**.
- 알고리즘 간 성능 교차점(crossover)이 library ↔ DBMS에서 **완전히 달라진다**.
- "distance computation 최소화"는 PG에서 end-to-end 성능의 신뢰할 수 없는 proxy다.

---

## 2. 왜 달라지는가 — 구조적 차이 (Table 1)

| 아키텍처 측면 | Library | PostgreSQL |
|-------------|---------|------------|
| **인덱스 저장** | raw pointer, 연속 메모리 | 8KB pages, 6-byte TID per neighbor |
| **노드 접근** | CPU 메모리 역참조 1회 | buffer pool lookup + page lock + 데이터 복사 |
| **필터 평가** | 단일 ID로 벡터/속성 동시 접근 | indextid → heap page → filter column (이중 조회) |
| **병렬성** | OpenMP, SIMD 자유 | 커넥션 당 1 쿼리, executor 제약 |

핵심: PG에서 **필터를 평가하려면 반드시 heap page를 추가로 읽어야 한다.** indextid(인덱스 내 포인터)가 heaptid(heap row 포인터)와 분리되어 있기 때문이다. ACORN의 2-hop 탐색은 이 비용을 기하급수적으로 증폭시킨다.

---

## 3. 2-hop 비용의 기하급수적 증가

ACORN/NaviX의 2-hop expansion이 PG에서 최악의 경우 유발하는 페이지 접근:

```
1 (현재 노드 page)
+ M (1-hop 이웃 M개의 page — indextid + heaptid 수집)
+ M × M (2-hop 이웃 M² 개의 page — heaptid 수집)
= 1 + M + M²
```

M=32 기준: **1 + 32 + 1024 = 1057 page accesses per traversal step**.  
각 page access는 buffer pool lookup + shared lock acquisition + data read + lock release를 포함한다.

이것이 library에서는 보이지 않는다. Library에서 2-hop expansion은 포인터 역참조 M²회다.

---

## 4. 논문이 제시한 두 가지 핵심 최적화

### 4.1 Translation Map (TM) — 필수

**정의**: 인덱스 빌드 시점에 `indextid → heaptid` 인메모리 해시맵을 생성한다. 쿼리 시 2-hop 이웃의 heaptid를 page access 없이 메모리에서 바로 조회.

**효과**: TM 없이는 heaptid 조회(page access)가 **총 CPU 사이클의 60-75%**를 차지한다 (Figure 13). TM 활성화 시 이 비용이 메모리 조회로 대체되어 다른 컴포넌트들이 병목이 된다.

```
TM 없음: [heaptid fetch: 70%] [filter: 15%] [distcomp: 10%] [기타: 5%]
TM 있음: [TM lookup: 8-17%] [filter: 20%] [distcomp: 35%] [metadata: 25%]
```

Translation Map 없이 ACORN을 PG에 구현하면 이론적 이점이 거의 상쇄된다.

### 4.2 Adaptive 2-hop Skip ("Hardening ACORN")

**정의**: 1-hop 이웃이 필터를 통과하면, 그 브랜치의 2-hop expansion을 건너뜀.

**근거**: 1-hop이 이미 유효한 후보라면 2-hop을 탐색해 연결성을 보완할 필요가 줄어든다.

**효과**: 선택도 10% 이상 구간에서 page access를 대폭 감소. NaviX의 복잡한 휴리스틱과 달리 단순한 조건문으로 구현 가능.

```c
/* acorn_scan.c 적용 위치 */
for each 1-hop neighbor v:
    if eval_predicate(v):
        add_to_W(v)
        /* 이미 유효 — 2-hop skip */
    else:
        /* 연결성 유지를 위해 2-hop 탐색 */
        for each 2-hop neighbor u of v:
            heaptid = translation_map_lookup(u.indextid)
            if eval_predicate_by_heaptid(heaptid):
                add_to_W(u)
            else:
                add_to_C(u)  /* 그래프 연결성 유지 */
```

---

## 5. ACORN-gamma의 pgvector 페이지 제약

pgvector의 neighbor tuple이 단일 8KB 페이지에 들어가야 한다는 제약이 있다:

```
(L_max + 2) × M × sizeof(ItemPointerData) ≤ 8192
```

`sizeof(ItemPointerData) = 6` bytes, 상수 2는 base layer의 2M connections 규격.

| M | L_max | 비고 |
|---|-------|------|
| 16 | 63 | 표준 pgvector 기본값 |
| 32 | 31 | 논문 실험 설정 |
| 40 | 25 | 논문 예시 |
| 64 | 15 | 실용 한계 |
| 80 (gamma=2) | **12** | L_max 절반으로 붕괴 |
| 128 (gamma=4) | **6** | 사실상 flat graph |

**pg_acorn Tier 2 함의**: `acorn_gamma=2`로 M을 두 배 늘리면 그래프 계층 깊이가 절반이 된다. `acorn_build.c`에서 이 trade-off를 계산하고 사용자에게 경고해야 한다. ACORN-gamma의 recall 이득(gamma=2가 1%보다 높음)이 계층 붕괴 손실보다 크다는 것은 논문의 실험 범위 밖이므로 pg_acorn 자체 벤치마크로 검증 필요.

---

## 6. 알고리즘별 성능 특성 (PG 환경)

### 6.1 선택도별 순위

| 선택도 | 최고 | 차선 | 하위 |
|--------|------|------|------|
| 1-5% | NaviX-Directed | ACORN | Sweeping, Iterative Scan |
| 5-30% | NaviX-Directed | ScaNN, ACORN | Sweeping > Iterative Scan |
| 50%+ | ScaNN | Sweeping ~ Iterative Scan | NaviX > ACORN |

**NaviX vs ACORN**: NaviX-Directed가 ACORN보다 1.2-1.7x 빠르다. 차이는 2-hop 탐색 순서다. NaviX-Directed는 1-hop 이웃을 거리 기준으로 정렬한 뒤 가장 가까운 이웃의 2-hop부터 탐색한다. ACORN은 임의 순서로 탐색한다. Directed 휴리스틱이 "더 유망한 방향"으로 먼저 이동하여 불필요한 탐색을 줄인다.

### 6.2 상관관계별 특성

| 상관관계 | 승자 | 특이점 |
|---------|------|--------|
| High Positive | NaviX-Directed | Sweeping/Iterative Scan도 10% 이상에서 경쟁력 |
| Low Positive | NaviX-Directed | ScaNN이 저선택도에서 강하나 고선택도 약화 |
| Negative | **ScaNN 압도적** | NaviX -53%, ACORN -44%, Iterative Scan -89% 하락 |

음의 상관관계(쿼리 근방 벡터들이 필터를 통과 못함): 그래프 방법은 쿼리 근방을 탐색하며 자원을 낭비한다. ScaNN의 클러스터 기반 구조는 그래프 근접성에 의존하지 않아 이 패널티를 피한다.

### 6.3 k 변화에 대한 강건성 (k=5 → k=100)

| 방법 | hop 증가율 |
|------|----------|
| NaviX | +86-106% |
| ACORN | 유사 |
| Sweeping | **+536-599%** |
| ScaNN | +220% |

대용량 k 쿼리(k=100)에서 NaviX/ACORN이 Sweeping, ScaNN 대비 훨씬 강건하다.

---

## 7. 시스템 오버헤드의 근원 — 병렬성과 무관

단일 스레드에서도 시스템 오버헤드가 압도적이다 (Table 7, OpenAI-5M, 10% selectivity, 95% recall@10):

| 방법 | System OH% (1T) | System OH% (16T) | 증가분 |
|------|----------------|-----------------|--------|
| NaviX | 55.9% | 73.5% | +48% |
| Sweeping | 81.8% | 91.0% | +68% |
| ScaNN | 84.4% | 86.9% | +59% |

단일 스레드 기준에서도 과반 이상이 시스템 오버헤드다. 이는 concurrency contention 문제가 아니라 **row-based page architecture의 구조적 비용**이다.

### ScaNN이 더 많은 필터 체크와 페이지 접근에도 불구하고 CPU 사이클이 적은 이유

1. **순차 접근 패턴**: 클러스터 leaf 내 벡터를 연속 페이지로 순차 스캔. hardware prefetcher 활용, cache line fill 효율적.
2. **SIMD 친화적**: 배치 bitmap probing + 거리 계산이 uniform access pattern으로 SIMD 벡터화 가능.
3. **낮은 "system tax"**: HNSW의 Translation Map 조회(8-17M cycles), neighbor metadata retrieval(15-25M cycles) 같은 고정 비용이 없다.
4. **캐시 locality**: 조밀한 클러스터 내 벡터들이 cache에 함께 존재. 그래프 방법의 임의 hop은 캐시를 오염시킨다.

그래프 방법이 distance computation을 적게 해도, **한 번의 page access 비용이 수십 번의 distance computation 비용과 맞먹는** PG 환경에서는 이 이점이 상쇄될 수 있다.

---

## 8. pg_acorn 구현에 대한 직접적 설계 지침

### 8.1 Translation Map — Step 3 구현 시 first-class 컴포넌트

`acorn_am.c` `ambuild` 콜백에서 인덱스 빌드 완료 후 TM을 생성한다.

```c
/* 빌드 시: acorn_build.c */
typedef struct AcornTranslationMap {
    HTAB *indextid_to_heaptid;  /* BlockNumber+OffsetNumber → HeapTupleData.t_self */
    MemoryContext mcxt;
} AcornTranslationMap;

/* TM 생성: 모든 인덱스 페이지를 순회하며 (indextid → heaptid) 엔트리 적재 */
AcornTranslationMap *acorn_build_translation_map(Relation index);

/* 스캔 시: acorn_scan.c */
ItemPointerData heaptid = acorn_tm_lookup(scan->tm, indextid);
bool passes = eval_predicate_heaptid(heaptid, scan->qual);
```

TM의 메모리 footprint: N개 노드, 엔트리당 약 18바이트(6 indextid + 8 heaptid + 4 overhead). 35M 벡터 기준 약 600MB. 벤치마크 시 GCP VM(128GB RAM)에서는 문제 없으나, 소규모 인스턴스에서는 pg_acorn.translation_map_enabled GUC로 on/off 제어 고려.

### 8.2 Adaptive 2-hop Skip — acorn_scan.c inner loop

```c
/* 현재 acorn_scan_execute() 내 이웃 평가 루프에 추가 */
foreach(lc, neighbors_1hop) {
    HnswElement elem = lfirst(lc);
    bool passes = acorn_eval_predicate(elem, predicate);

    if (passes) {
        pairingheap_add(W, &elem->ph_node);
        /* Hardening ACORN: 이 브랜치는 2-hop 탐색 불필요 */
        continue;
    }

    /* 연결성 유지: 2-hop 탐색 */
    pairingheap_add(C, &elem->ph_node);  /* traversal 유지 */
    if (acorn_should_expand_2hop(elem, selectivity_hint)) {
        acorn_expand_2hop(scan, elem, C, W, predicate);
    }
}
```

`acorn_should_expand_2hop()`은 현재 W 채움률로 heuristic 선택도를 추정해 결정한다.

### 8.3 ACORN-gamma 빌드 시 L_max 경고

```c
/* acorn_build.c */
int m_eff = opts->m * opts->acorn_gamma;
int l_max = (int)(8192.0 / (m_eff * sizeof(ItemPointerData)) - 2);

if (l_max < 10) {
    ereport(WARNING,
        errmsg("acorn_gamma=%d with m=%d produces L_max=%d (< 10). "
               "Graph navigability may degrade significantly.",
               opts->acorn_gamma, opts->m, l_max));
}
```

### 8.4 Benchmark — page I/O 측정 필수

논문이 증명했듯이 distance computation 수만으로는 PG 내 ACORN 이점을 측정할 수 없다.

```python
# bench/harness/cohere_bench.py 에 추가
def measure_page_io(conn, query_sql):
    conn.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query_sql)
    plan = conn.fetchone()[0][0]
    return {
        "shared_hit": plan["Planning"]["Shared Hit Blocks"] + plan["Execution"]["Shared Hit Blocks"],
        "shared_read": plan["Planning"]["Shared Read Blocks"] + plan["Execution"]["Shared Read Blocks"],
    }
```

벤치마크 리포트에 `QPS`, `recall@10`, `p99 latency`와 함께 **`pages_hit_per_query`** 컬럼 필수.

---

## 9. NaviX-Directed 휴리스틱 — 중장기 개선 방향

논문에서 NaviX가 ACORN보다 1.2-1.7x 빠른 핵심 이유는 **Directed** 휴리스틱이다.

**ACORN (Blind)**: 1-hop 이웃들의 2-hop 이웃을 임의 순서로 탐색.

**NaviX-Directed**: 1-hop 이웃들을 먼저 거리 기준으로 정렬. 가장 가까운 1-hop의 2-hop부터 탐색. 더 유망한 방향으로 탐색이 집중되어 불필요한 page access 감소.

pg_acorn에서 구현 시 `acorn_scan.c`의 2-hop expansion 루프 내 1-hop 이웃 정렬 1줄 추가로 구현 가능. Step 3 이후 독립 실험으로 진행 권장.

---

## 10. 장기 방향 — 컬럼나 엔진

논문의 결론이 명시적으로 제안하는 방향:

> "A promising approach is the use of columnar engines to host both the index as well as the columns that participate in filtered vector search. Columnar engines have shown their ability to drastically reduce the page access overhead and other system overheads while maintaining transactional consistency."

row-based PG의 double-lookup(indextid → heap page → filter column)을 근본적으로 해결하려면, 필터 컬럼을 인덱스 페이지와 동일한 컬럼나 청크에 함께 저장해야 한다.

pg_acorn의 현재 접근(CustomScan Tier 1)은 heap access를 minimization하는 방향에서 이 개념과 정렬한다. 더 나아가 `acorn_hnsw` 인덱스 내에 filter column 사본을 저장하는 "included columns" 방식(PG 11+의 `INCLUDE` 절 활용)을 검토할 수 있다.

---

## 11. pg_acorn의 차별화 포인트

이 논문은 AlloyDB(PG 호환 상용 시스템)에서의 실측이며 **소스 미공개**다.

pg_acorn은:
- **오픈소스**로 Translation Map + Adaptive 2-hop을 PostgreSQL community에 제공
- **Tier 1 hook**: zero-migration, 기존 hnsw 인덱스 재생성 없이 TM + adaptive 2-hop 즉시 활성화
- **Tier 2 acorn_hnsw**: ACORN-gamma를 native AM으로, TM을 인덱스 메타데이터로 영구 저장
- **Step 5 upstream PR**: 이 논문을 인용하며 "TM 없이는 PG에서 ACORN의 이론적 이점이 상쇄된다"를 정량 증명하는 자료로 활용

---

## 참고

- 논문 원문: https://arxiv.org/abs/2603.23710
- NaviX 논문 (Sweeping + adaptive): [44] (논문 내 참조)
- pgvector 0.8.0 Iterative Scan: [7]
- ScaNN: https://github.com/google-research/google-research/tree/master/scann
- RF-ANNS 서베이 (분류 체계): arXiv:2505.06501
- pg_acorn 아키텍처: `docs/architecture.md`
- 산업 지형: `docs/filterable-hnsw-landscape.md`
