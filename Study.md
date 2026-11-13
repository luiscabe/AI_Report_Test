# 과제보고서 — 학습·추론·경량화 과정 (MAC 런타임 CPU 최적화 AI 모델)

- 본 문서는 **학습 → 검증 → 탑재 추론 → 경량화** 전 과정의 현재 완료 상태 스냅샷이다 — 외부 문서 참조 없이 단독으로 읽을 수 있도록 자기완결적으로 작성했다
- 기준 산출물: `AI/evidence/train_model_final_20261113/` (2주기 재학습, 2026-11-13)
- 코드: `AI/MacCpuOpt_Model/`(학습 파이프라인)·`AI/MacCpuOpt_Inference/`(보드 추론 엔진)
- 이전 1주기 모델(2026-11-12, 학습 3계열)은 `AI/evidence/train_model_final_20261112/`에 기록 보관 — 본 문서는 **2주기 모델 기준**

---

## 0. 전체 파이프라인 한 장 그림

```
[보드] 11계열 트래픽 캠페인 수집(2026-11-12~13)
   │  visits_*.blocks.jsonl (PMU 9그룹 + SoC TAD/DSS 스냅샷)
   ▼
[랩 가공] pipeline/ extract·label·soc_merge
   │  사용량 라벨 + SE → 관측 8종 + 설정 인코딩 6×2
   ▼
[학습] select/train_final.py (8단계)
   │  ①감사 ②LOFO ③ŝ표 ④ablation 7종 ⑤선형/트리 대조
   │  ⑥단조성 ⑦최종 모델(6계열 전체) ⑧export
   ▼
[export] final_model.npz + model_spec.json + model_weights.h(C 헤더)
   │        + shat_table.csv + golden_vectors.csv(패리티 30벡터)
   ▼
[보드] mac_cpu_opt_inference(C 엔진) — model.c 가중치 교체 지점
   │  1초 폴링: /dev/shm pmu_snapshot 판독 → 전처리(공통 계약)
   │  → MLP 순전파 → 후보별 Δ̂ → argmin → inference_result 기록
   ▼
[적용] SHADOW(기록) / ACTIVE(레지스터 적용, 직렬화)
```

---

## 1. 학습 과정

### 1.1 학습 데이터 — 11계열 수집, 축 짝 분할

- **수집**: 트래픽 캠페인 엔진으로 11개 부하 계열을 전 격자(ORG + 10설정 + PF_DIS) 원칙으로 전수 수집 완료(2026-11-12~13). 1계열 = 부하 조성 1종(예: LTE 6cc+NR 9cc) × 전 설정 방문 × 재생.
- **축 짝(axis-pair) 분할** [결정-사용자, 2026-11-13]: 변동 축(① 조성 ② 방향 ③ RAT/부하)마다 "학습 형제 → 평가 홀드아웃" 짝을 구성.

| 구분 | 계열 | 방문 수 | 비고 |
|---|---|---|---|
| **학습 6계열** | lte6nr9·lte8nr8·lte9nr6 | 121+121+121 | 혼합 중부하군(상호 r≈0.9+) |
| | ulonly | 64 | 방향축(UL 특화) |
| | lteonly9(DCD OFF 정정 파생본)·nronly12 | 11+11 | 단일 RAT |
| **평가 풀 5계열** | lte9nr7·lte5nr11·dlonly·lteonly12·nronly9 | — | 개발 중 열람 금지(오프라인 평가 전용) |

- 총 학습 데이터: **449 방문(블록), SoC coverage 449/449, SUM 사용률 범위 25.4~63.99%** (`dataset_audit.json`). washout canary는 11계열 중 nronly9만 노이즈(초저부하 전이 잡음 의심) — 학습 6계열 내 플래그는 0~2개/계열로 클린.
- σ_a(재생잡음) = **0.041pp**(재생 재실행 간 사용률 편차 실측) — 라벨 SE 하한으로 전파.

### 1.2 라벨·피처 가공 — pipeline/

- **라벨**: y = ln(사용량(cand) / 사용량(cur)) — log 비는 부하·코어 수 규모가 지워지고 가법성(a→b + b→c = a→c)이 성립. 같은 재생 인덱스(round_no)끼리 짝지어 계열 내 노이즈 상쇄.
- **분모**: 레코드 슬롯 기준(records) — window 분모 대비 ablation에서 +1.7~+2.6pp 열화로 records 확증(window 분모는 빈 슬롯까지 분모에 포함돼 설정별 점유 밀도 차이가 사용률에 반영되기 때문).
- **관측 8종**(확정 집합 — PMU 비율 피처는 계열 단위 교차검증에서 의사결정 기여가 입증되지 않아 미채택):
  1. SUM 사용률(%)
  2. LTE 코어 수, 3. NR 코어 수 (DCD ON 코어 그룹 배정)
  4. SoC rd/wr, 5. REQ_OCC/s, 6. DSS_RD/REQ, 7. HIT/ALLOC, 8. IFB/REQ — TAD/DSS 스냅샷에서 v2 blocks `state.soc` 재계산
- **설정 인코딩 6×2**(cur·cand 각): 그룹별 [PF_MODE lo/4, hi/4, PF_DIS] — **pf_dis=1이면 (lo,hi)=[4,4] 강제 전인코딩**("가장 소극적 [3,3] 너머의 끄기" 가상 좌표, 정규화 /4·정의역 [0,1]). 변환은 원천 기록과 무관하게 pf_dis 플래그 기반 소비자 강제 적용.

### 1.3 시료 조립 — 전쌍(all-pairs)

- 학습기 입력 x = **[관측 8 | 현재 설정 6 | 후보 설정 6] = 20차원**, 출력 = Δ̂(SUM) (log 비).
- 시료 = (cur, cand) 전쌍(ORG 기준 라벨 차로 정확 유도). 학습 6계열 전체 **47,812시료**.
- 시료 가중 w = 1/SE²(SE는 방문 SE에 σ_a 하한 전파) — **무가중 채택**: 축 균형 손실 가중(축별 기여 1/3 균등화)은 ablation에서 무가중 대비 LOFO 계열 비가중 평균 2.20 → 1.39pp로 전 계열 열화해 **기각**(2026-11-13 결정, ablation `axis_weight_on`).

### 1.4 학습기 — 소형 MLP + 가중 Huber

- 구조: **MLP 20-16-1 (ReLU 은닉 1층 16유닛 → 선형 출력)**, numpy 단일 의존 구현(`select/mlp.py`).
- 손실: **시료 가중 Huber(δ=0.02, log 단위)** + L2(1e-4) — 이상치에 강건(MSE 대조 ablation에서 Huber 우위 확증, §1.6).
- 최적화: 전배치 Adam(lr 0.01, 800 epoch).
- **seed(0,1,2) 앙상블**(예측 평균) — 학습 노이즈 억제.
- 전처리: 표준화(μ, σ) + **OOD 클램프**(입력을 학습 범위 [lo, hi]로 클리핑 — ReLU MLP의 범위 밖 외삽 폭주 방지) — 무클램프 대비 LOFO regret 2.10→1.00pp·MAE 7.64→4.34pp 개선(1주기 실측).

### 1.5 검증 — train_final.py 8단계 파이프라인

`python3 select/train_final.py` 1회 실행으로 전 단계가 산출물로 기록된다:

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1/8 | 데이터 감사(방문 수·사용률 범위·SoC coverage·canary 목록) | `dataset_audit.json` |
| 2/8 | **LOFO**(계열 단위 leave-one-fold-out — 조성 외삽 검증) | `lofo_metrics.json` |
| 3/8 | ŝ 표(LOFO 잔차 q90, (cur,cand) 쌍별) | `shat_table.csv` |
| 4/8 | ablation 7종(설계 선택별 강건성) | `ablations.json` |
| 5/8 | 선형(Ridge 교차항)·GBDT(HistGB)·sklearn MLP 대조 + 비열등 CI + 모델 크기 | `comparisons.json` |
| 6/8 | 단조성 검사(라벨 격자 vs 모델 예측 격자) | `monotonicity.json` |
| 7/8 | 최종 모델(6계열 전체 학습) + 자체 평가·격자 재현 | `self_eval.json` |
| 8/8 | export(npz + spec + C 헤더 + golden 벡터) | §3.2 |

- **LOFO**: 6계열 각각을 hold-out 하고 나머지 5계열로 학습 → hold-out에서 평가. regret = 실제 Δ(선택) − 실제 Δ(오라클 최적) [%p]. 학습에 없는 조성·축에서의 외삽 성능을 재는 검증 게이트.
- **ablation 7종**: 설계 선택(SUM 피처·canary·전쌍·축 가중·은닉 크기·손실 함수)을 하나씩 제거/변경해 LOFO를 재실행 — base 스펙의 성능이 특정 선택에 우연히 기대는지(강건성) 확인.

### 1.6 최종 성능 (2주기, 2026-11-13)

**LOFO pooled regret 0.458pp (중앙 0.005pp)** — fold별(`lofo_metrics.json`):

| fold | regret 평균(pp) | MAE(pp) | 실측 최적 | 무조치 regret(pp) |
|---|---|---|---|---|
| lte6nr9 | 0.034 | 2.51 | XL07_N07 | 3.43 |
| lte8nr8 | 0.011 | 3.38 | XL08_N08 | 4.77 |
| lte9nr6 | 0.987 | 1.30 | XL07_N07 | 4.97 |
| ulonly | 0.009 | 1.58 | XL07_N07 | 5.14 |
| lteonly9 | 2.539 | 12.37 | XL03_N01 | 2.54 |
| nronly12 | 4.755 | 3.43 | XL03_N10 | 5.70 |
| **pooled** | **0.458** | **4.09** | — | — |

- 혼합군·방향축 fold는 regret ≈ 0.01~1.0pp. **단일 RAT fold(lteonly9·nronly12)가 약점** — 3분포군(혼합/방향/단일 RAT) 구조상 단일 RAT 분포는 혼합군 관측으로 외삽하기 어려움(2026-11-13 사전 분석). 평가 풀 5계열에서 재판정 예정.
- **최종 모델 자체**(6계열 전체 학습, in-sample — `self_eval.json`): 잔차 MAE 0.32pp·q90 0.65pp·bias −0.07pp, 격자 재현 Spearman 0.736(nronly12)~0.995, argmin 적중 2/6계열(미적중분도 인접 후보 오차 수준).

**ablation** (`ablations.json`, LOFO regret pp):

| 변형 | 무엇을 바꾸나 | 결과 | 판정 |
|---|---|---|---|
| base | — | 0.458 | 기준 |
| sum_excluded | SUM 사용률 피처 제외 | 0.818 | 열화 → SUM 포함 유지 |
| canary_excluded | washout 플래그 방문 제외 | 0.459 | 미미 → 포함 유지(플래그 기록 정책) |
| train_org_pairs | (ORG→후보) 쌍만 학습 | 2.186 | 열화 → 전쌍 유지 |
| axis_weight_on | 축 균형 손실 가중 on | 1.525 | 열화 → 무가중 확정 |
| hidden_32 / hidden_64 | 은닉 32 / 64 | 0.434 / 0.466 | 근소차 → 최소 모델 16 유지 |
| loss_mse | Huber → MSE | 1.354 | 열화 → Huber 유지 |

**모델류 대조** (`comparisons.json`):

| 학습기 | LOFO regret(pp) | 모델 크기 | 판정 |
|---|---|---|---|
| **MLP 20-16-1 (3seed)** | **0.458** | 8,472 B | 채택 |
| Ridge+교차항(68차원) | 2.795 | 552 B | 비열등 기각(Δ95% CI [2.19, 2.49] > 0.1pp) → MLP 유지 |
| HistGB(GBDT 300) | 0.143 | **929,278 B** | 성능 우위나 **256KB 예산 초과** + C 엔진 구현 부담 → 기각(기록) |
| sklearn MLP(독립 구현) | 3.778 | — | 독립 구현 대조 기록(무가중 학습 차이) |

- 단조성: 라벨 격자 위반율 12.0% → 모델 예측 격자 7.0%(라벨보다 평활 — 교차 설정 구조상 비단조는 정상).
- ŝ 표: 쌍 14,520행, shat 평균 3.5pp·q90 4.3pp(`shat_table.csv`) — 런타임 불확실도 판정 기준.

---

## 2. 추론 과정

### 2.1 런타임 회선 구성

```
┌─────────────────────── SUMB 프로세스 ────────────────────────┐
│ [각 isolated 코어 1~11의 TFW 스레드]                          │
│   MacCpuOpt_PmuPostProcess() (PAL post-process)              │
│     ① SystemRegConfig_RwPoll() — 대기 중 튜닝 트리거 소비      │
│     ② 1초 경과 판정 ③ PMU 판독(EL0 직접, syscall 없음)        │
│     ④ 자기 코어 전용 슬롯에 기록(seq 배리어)                  │
│ [SystemProcedureThread — LNX 타이머 1000ms]                   │
│   MacCpuOpt_IpcAgentFunction()                                │
│     ⑤ 전 코어 슬롯 판독 + 32bit wrap 보정 델타                │
│     ⑥ pmu_snapshot 원자 갱신(write tmp + rename, seq++)       │
│     ⑦ inference_result 판독·검증(magic/version/seq/lease)     │
│     ⑧ ACTIVE: SystemRegConfig_PrefetchRequest()로 트리거 arm │
└──────────────────────────────────────────────────────────────┘
     │⑥ 파일(SUMB→엔진)                ▲⑦ 파일(엔진→SUMB)
     ▼                                  │
┌──── mac_cpu_opt_inference (외부 C 프로세스, 보드 네이티브) ────┐
│  1초 폴링으로 pmu_snapshot 판독(seq 변화 감지)                │
│  → 전처리(공통 계약) → 모델 추론 → inference_result 원자 기록  │
└──────────────────────────────────────────────────────────────┘
```

- 통신은 전부 **tmpfs(`/dev/shm/mac_cpu_opt/`) 파일 IPC** — 소켓·이벤트 채널 없음, 파일 갱신 자체가 트리거. IPC 파일 포맷은 `Butterfly/Control/NR/MacCpuOptIpc.h`를 SUMB·엔진 양쪽이 직접 include해 **단일 원천**(어긋남 구조적으로 불가).
- **느슨한 결합**: 엔진이 죽어도 SUMB 측에 영향 없음 — lease(기본 5초) 만료 후 기본 설정 복귀만 수행.
- 수집→전달→적용 회선은 **보드 폐루프 검증 완료**(2026-11-03, 기지국 보드 실측).

### 2.2 1초 주기 추론 흐름 (엔진)

1. `pmu_snapshot` 폴링 — `seq`가 변하면 새 데이터로 판단.
2. **전처리(공통 전처리 계약)**: `preprocess.c` — 학습 파이프라인이 같은 C 구현을 같이 빌드해 사용하므로, 같은 원시 카운터가 학습과 런타임 양쪽에서 항상 같은 피처를 낸다(학습-서빙 일관성).
3. **모델 추론**: 입력 20 = [관측 8 | 현재 설정 6 | 후보 설정 6] → OOD 클램프(학습 범위 [lo, hi]) → 표준화 → MLP 순전파(ReLU) → **Δ̂(SUM) 출력**(log 비). Δ̂(cur→cur)=0 후처리. 런타임에서 클램프가 발생하면 OOD 플래그 권장.
4. **의사결정은 모델 외부 로직**: 후보 설정 열거 → 후보별 Δ̂ 예측 → argmin(최소 예상 사용량) → 선택. 불확실도 판정은 ŝ 표(쌍별 q90) 참조.
5. `inference_result` 원자 기록(write tmp + rename).

### 2.3 SHADOW / ACTIVE 적용

- **SHADOW 모드**(기본): 결과를 로그로만 출력 — 실제 레지스터 미접촉.
- **ACTIVE 모드**: 결과의 코어별 setting을 `SystemRegConfig_PrefetchRequest(cpu, 0, setting)`으로 트리거 슬롯에 arm → 각 isolated 코어의 post-process 루프(`SystemRegConfig_RwPoll()`)에서 소비, 해당 코어가 `/dev/sysreg` ioctl으로 자기 prefetcher 레지스터에 자가 적용.
- **적용 직렬화**(2026-11-03 보드 시험 반영): 레지스터 쓰기는 그 코어의 task isolation을 깨고 복구에 약 2초가 걸리므로 (a) 직전 setting과 같으면 재기입 스킵(steady state에서 레지스터 쓰기 0), (b) 최대 1코어씩·최소 2틱(2초) 간격 라운드로빈(전 코어 전파 최대 22초).

### 2.4 현재 상태와 교체 지점

- 엔진(`AI/MacCpuOpt_Inference/`, C)의 `model.c`는 현재 회선 연결 확인용 **고정 가중치 MLP stub**. 최종 모델 탑재는 `MacCpuOptModelInfer()`의 가중치를 `model_weights.h`(§3.2)로 교체하는 단일 지점에서 이뤄진다 — 인터페이스(관측 + cur/cand 설정 → Δ̂, 후보 열거·argmin·적용은 엔진 측 로직)는 이미 최종 설계 형태.
- 탑재 시퀀스: 오프라인 평가(평가 풀 5계열) 통과 후 SHADOW 탑재(엔진 r7 — TC_TABLE PF_DIS levels [4,4] 원천 기록 반영 번들).

---

## 3. 경량화 과정

추론은 비-MAC 코어에서 실행돼 계산 여유가 크고, 경량화는 병목이 아니다. 아래는 **경량화가 의사결정(스펙 선택)에 작동한 내역 + 탑재 구현상의 경량화 + 예약 항목**이다.

### 3.1 구조 경량화 — 최소 모델 원칙

- **은닉 1층·16유닛**(설계 제약: 은닉 2층 이하·유닛 64 이하). 파라미터 약 5.4k(20×16+16+16+1 = 353/seed).
- **모델 크기 예산 256KB** 대비:
  - 채택 MLP 20-16-1: **8,472 B**(seed 3개 앙상블 전부 포함, `comparisons.json` size_bytes) / seed 0 단일 헤더는 그 1/3 수준
  - float32 환산 약 22KB, int8 환산 약 5.4KB
  - **GBDT(HistGB 300)는 regret 0.143pp로 성능 최상이나 929,278 B — 예산 3.6배 초과** + C 엔진 구현 부담으로 기각(§1.6 대조표). 성능 우위를 확인하고도 크기 예산으로 기각한 것이 경량화 기준이 실제로 작동한 사례.
- **용량 ablation**: hidden 32(0.434pp)·64(0.466pp) vs 16(0.458pp) — 유의미한 개선 없음 → **최소 모델 원칙으로 16 유지**.

### 3.2 구현 경량화 — 의존성 없는 탑재

- **numpy 단일 의존 학습 파이프라인**(`select/` — PyTorch 등 프레임워크 없음): 랩·보드 포팅 용이 + C 구현과의 학습-서빙 일관성 유지.
- **C 헤더 export**(`model_weights.h`, auto-generated): 표준화 상수 + 가중치를 `static const double` 정적 배열(g_mu/g_sd/g_lo/g_hi/g_W1[16][20]/g_b1/g_W2/g_b2)로 직접 기록 — **런타임 의존성 0, 로더·파일 I/O 불필요**. 컴파일 타임 상수화로 탑재.
  - 헤더는 seed 0 단일 모델(탑재 시 앙상블 평균 또는 단일 시드 선택은 소비자 결정 사항 — 코드 주석 명시).
  - OOD 클램프 상한/하한(g_lo, g_hi)까지 동봉해 전처리 계약이 헤더 하나로 완결.
- **공통 전처리 계약**: `preprocess.c`(C)를 학습 파이프라인과 런타임 엔진이 **같이 빌드해 공유** — 피처 계산 단일 원천, 학습-서빙 스큐 구조적 차단.
- **IPC 포맷 단일 원천**: `MacCpuOptIpc.h` 직접 include(§2.1).

### 3.3 패리티 검증

- **golden_vectors.csv**: 대표 30벡터(입력 × 예측) — C 구현이 학습 numpy 구현과 같은 출력을 내는지 패리티 검증용. 1주기 산출에서 가중치 재현 오차 <1e-10 검증 완료, 2주기 산출에도 동반.
- 모델·스케일러는 `final_model.npz`(전체 seed 앙상블 + 스케일러)와 `model_spec.json`(입력 레이아웃·전처리·출력 규약 기계판독)으로 이중 보존.

### 3.4 추론 비용

- 도메인 하나 결정(피처 + 추론 + OOD 판정·기록): 약 115k cycle ≈ **58 µs**(클럭 2.0 GHz 가정) — 1초 결정 주기 대비 여유 충분.

### 3.5 잔존 경량화 — 2단계 예약

- **정수 양자화(int8)** — 현재 float64 헤더 → int8 시 약 5.4KB. 필요 시 착수.
- **증류**(대형 모델 → 소형 MLP).
- **후보 생성 전략** — 후보 64개 초과 시 전수 열거 비용 대응.
- 다변량 거리 기반 OOD 판정 고도화 — 현행은 입력 클램프 단순 정책.

---

## 4. 산출물·재현

### 4.1 2주기 학습 산출물 (`AI/evidence/train_model_final_20261113/`, 2026-11-13)

| 파일 | 내용 |
|---|---|
| `final_model.npz` | 최종 모델(스케일러 + seed 0,1,2 앙상블 가중치) |
| `model_spec.json` | 기계판독 스펙(입력 20 레이아웃·전처리·출력 규약·학습 계열·성능 요약 포함) |
| `model_weights.h` | C 헤더(정적 배열 — 엔진 탑재용, seed 0) |
| `golden_vectors.csv` | C 구현 패리티 검증용 30벡터 |
| `shat_table.csv` | (cur,cand) 쌍별 ŝ(14,520행) — 불확실도 기준 |
| `lofo_metrics.json` / `ablations.json` / `comparisons.json` / `monotonicity.json` | LOFO·ablation·모델류 대조·단조성 지표 |
| `self_eval.json` / `dataset_audit.json` | 자체 평가·데이터 감사 |
| `run.log` | 실행 기록(재현 로그) |

### 4.2 재현

```bash
# 전 파이프라인(학습 6계열 로드 → 8단계 → export) — Python3 + numpy 만 필요
cd SUMB/AI/MacCpuOpt_Model && python3 select/train_final.py

# 추론 엔진 크로스 빌드(aarch64, 보드 반입용)
cd SUMB/AI/MacCpuOpt_Inference && python3 build.py --target=cross --docker-image=default
```

- 학습 입력 blocks.jsonl은 `AI/evidence/` 동결 경로에서 직접 소비(이동·복사 금지). 평가 풀 5계열은 개발 중 열람 금지.

### 4.3 다음 단계

1. **오프라인 평가**: 평가 풀 5계열(lte9nr7·lte5nr11·dlonly·lteonly12·nronly9) 대상 2주기 모델 regret 측정 — nronly9(노이즈 계열) 제외 민감도 병기, 정적 기준선(TC_07/TC_08) 대비.
2. **SHADOW 탑재**: model_weights.h 교체 + TC_TABLE PF_DIS [4,4] 원천 기록 반영(엔진 r7) → ACTIVE 실운용 전환.
3. 필드 데이터에서 SoC 카운터 5종 재판정.




