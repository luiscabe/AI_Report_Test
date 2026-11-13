# MAC_RUNTIME_CPU_OPT — 바이너리 레벨 블록 구조

> 작성일 2026-11-13. 대상: 기지국 MAC 전용(isolated) 코어(1~11)의 CPU prefetcher/L2 설정을
> 런타임에 자동 선택하는 시스템(MAC_RUNTIME_CPU_OPT). 스냅샷 v3 기준 요약.

## 1. 블록 다이어그램 (컴파일 단위)

```
[빌드 타임 — 정적 구조]                        [런타임 — 동적 구조]

┌──────────────────┐   MacCpuOptIpc.h      ┌──────────────────────┐
│  sumb1_guc.ocn    │◄─────────────────────►│ mac_cpu_opt_inference │
│  (SUMB 본체)      │  SystemRegConfig.h    │ (외부 추론 엔진, C)    │
└───────┬───────────┘  (공유 헤더 = 포맷·인코딩 단일 원천)           │
        │              preprocess.c (학습-서빙 공통 전처리)          │
        │                                                ┌──────────┴──────────┐
        │                                                ▼                     ▼
        │                          [추론 모드: 1초 폴링 폐루프]   [수집 모드: --collect 캠페인]
        │                                                │                     │
        │         pmu_snapshot (W→R)                      │                     │ xlog 파이프
        ├──────────────────────────────────────────────►(파일)                 │ (popen, "sumb; prefetch_write N")
        │         inference_result (R←W)                 │                     │ + tad_dss_snapshot.sh
        ◄───────────────────────────────────────────────(파일)                 ▼
        │                                                                       [SoC CSR 직접 판독]
        │  /dev/sysreg ioctl (prefetch 레지스터 쓰기)                                 │
        ▼                                                                       │
┌──────────────────┐                                                            │
│  sysreg_kmod.ko   │  ← 카드 리부팅 시 재탑재 필요(Env/)                          │
│  (커널 모듈)      │                                                            │
└──────────────────┘                                                            ▼
                                                                   ┌──────────────────────┐
[보드 밖 — 호스트 PC]                                               │ visits_*.jsonl (보드) │
┌─────────────────────────────────────────┐                        └──────────┬───────────┘
│ Python 학습 파이프라인                     │◄──── 파일 다운로드(회수 최우선) ────────┘
│ scripts_v2 (오케스트레이션 스펙/변환)      │
│ MacCpuOpt_Model (label/train/select)     │── 가중치 export ──► model.c 교체 후 재빌드
└─────────────────────────────────────────┘
```

## 2. 블록별 정의

| 블록 | 산출물 | 비고 |
|---|---|---|
| **SUMB 본체** | `sumb1_guc.ocn(.gz)` — 프리셋 `gnbdu-guc-ltenr-cpu1` 크로스 빌드 | 수집 에이전트(MacCpuOptAgent)·트리거 소비·레지스터 적용·aitune CLI 내장 |
| **추론 엔진** | `mac_cpu_opt_inference` — 독립 CMake 빌드(같은 컨테이너·툴체인) | 이중 모드: 추론 폴링 / `--collect` 캠페인 오케스트레이터 |
| **커널 모듈** | `sysreg_kmod.ko` + `sysreg_cli` (Env/ 배포 자산) | 재부팅 후 수동 재탑재 필요 — 모듈 없으면 prefetch_write 전량 실패 |
| **학습 파이프라인** | Python (스크립트, 보드 외 호스트 실행) | 바이너리 아님 — 빌드 산출물이 아니라 가중치·데이터 처리 계열 |

## 3. 정적 구조 (컴파일 타임 결합)

| 결합 | 내용 | 효과 |
|---|---|---|
| `MacCpuOptIpc.h` | SUMB ↔ inference 양쪽이 **같은 헤더 include** | IPC 파일 포맷(96B/104B)이 어긋날 수 없음. `_Static_assert`가 양측 빌드에서 크기·오프셋 강제. 포맷 변경 시 헤더만 수정 + 양쪽 재빌드 |
| `SystemRegConfig.h` | inference가 직접 include | isolated 코어 범위·prefetch setting 인코딩(0~10) 단일 원천 |
| `preprocess.c` | inference 빌드 + 학습 파이프라인이 동일 C 소스 공유 | 학습-서빙 전처리 skew 원천 제거 (golden_test로 비트 일치 검증) |
| sysreg_kmod.ko | 완전 독립 빌드 | SUMB 빌드와 무관. 인터페이스는 `/dev/sysreg` ioctl ABI만 |

→ 4개 바이너리 중 **링크 타임 의존은 전혀 없고**, 결합은 헤더 3개(포맷·인코딩·전처리)가 전부. 나머지는 전부 런타임 계약.

## 4. 동적 구조 (런타임 인터페이스)

| 경로 | 방향 | 매체 | 성격 |
|---|---|---|---|
| pmu_snapshot | SUMB **W** → inference **R** | tmpfs 파일, write+rename 원자 교체, seq 폴링 | **느슨한 결합** — inference가 죽어도 SUMB 무영향 |
| inference_result | inference **W** → SUMB **R** | tmpfs 파일, lease(5s) 신선도 판정 | 모델 사망 시 lease 만료 → 기본값 복귀 (안전망) |
| prefetch 레지스터 쓰기 | SUMB → sysreg_kmod | `/dev/sysreg` ioctl | **유일한 ioctl 경로**. 전 코어 동시 쓰기는 kernel hang → 직렬화(1코어/2초) |
| 설정 기입(수집 모드) | inference → SUMB | xlog stdin 파이프 (`popen`, `sumb; prefetch_write N` CLI) | 캠페인 모드 전용 — 제어 채널이 파일이 아닌 **콘솔 CLI** |
| SoC 카운터(TAD/DSS) | inference → 하드웨어 | `tad_dss_snapshot.sh` 실행 (CSR 직접 판독) | SUMB 경유 없음 — 수집 경로 일원화 |
| 학습 데이터 | 보드 → 호스트 | visits JSONL 파일 다운로드 | 오프라인. 회수 최우선(재부팅 시 소실) |

## 5. SUMB ↔ 추론 엔진 전달 메커니즘 (상세)

두 프로세스 사이에 소켓, 파이프, 시그널 같은 별도 통신 채널은 없다. 유일한 매체는 tmpfs 파일 2개이고, **파일 갱신 자체가 트리거**다.

### 5.1 SUMB → 추론 엔진 (pmu_snapshot, 96B)

```
SUMB SystemProcedureThread (LNX 타이머, 1초)          추론 엔진
────────────────────────────────────                  ──────────
① 임시 파일에 write
② rename(tmp, pmu_snapshot)  ── 원자 교체 ──►          ③ 1초 폴링으로 파일 열어 seq 확인
                                                       ④ seq 변화 시 → 새 데이터로 추론 수행
```

- **원자성**: `write()+rename()` 조합. rename이 같은 파일시스템 내 경로 교체를 원자적으로 수행하므로 추론 엔진이 쓰다 만 "반쪽 파일"을 읽는 일이 구조적으로 불가능 — 잠금·핸드셰이크 프로토콜 불필요.
- **변화 감지 = 트리거**: 헤더의 `seq`가 갱신마다 +1. 추론 엔진은 매초 파일을 읽어 seq 변화를 보고 변했을 때만 추론. "새 데이터 있음" 신호를 별도로 보내지 않고 **seq 변화가 곧 신호**.

### 5.2 추론 엔진 → SUMB (inference_result, 104B)

```
추론 엔진                                               SUMB SystemProcedureThread (다음 1초 틱)
──────────                                              ─────────────────────────────────
① 임시 파일에 결과 write (setting[12], timestamp_ns,    ③ 매 틱 파일 판독
   lease_ms)                                            ④ 검증: magic/version/seq
② rename() 원자 기록 ─────────────────────────────►     ⑤ 신선도 판정: now − timestamp_ns ≤ lease_ms
                                                        ⑥ ACTIVE면 setting을 트리거 슬롯에 arm
```

- **역방향도 동일 원리**: SUMB가 결과 파일을 1초마다 폴링해 판독. 추론 엔진이 완료를 능동적으로 알리지 않고 SUMB가 다음 틱에 스스로 발견.
- **lease = 사망 감지 신호**: `timestamp_ns`가 lease_ms(기본 5초)보다 오래됐으면 결과 폐기, ACTIVE 모드면 부팅 기본값(ON + PF_MODE 3 = setting 4) 복귀 요청. 추론 엔진이 죽어도 별도 에러 통지 없이 lease 만료만으로 장애 감지.
- **데이터는 파일, 실행은 별도 경로**: 결과 파일로 전달되는 것은 setting 값뿐. 실제 레지스터 적용은 SUMB 내부 2단계 경로(LNX 타이머 arm → 각 isolated 코어가 `RwPoll()`에서 소비 → `/dev/sysreg` ioctl)로 수행 — 파일이 제어 신호를 직접 실행하지 않음.

### 5.3 설계 의도

| 특성 | 결과 |
|---|---|
| 느슨한 결합 | 추론 엔진이 죽어도 SUMB에 무영향 — 수집 계속. 반대로 SUMB 재기동 시 엔진은 파일 없음/seq 정지만 보고 조용히 대기 |
| 생존자 문제 해소 | 어느 쪽이 먼저 떠도 동작 — 파일 없으면 스킵, 생기면 소비 |
| 예외: 캠페인 수집 모드 | 역방향 **제어 채널** 별도 — 추론 엔진이 xlog 콘솔 stdin 파이프로 `sumb; prefetch_write N` CLI를 보내 SUMB에 설정 기입. 일반 운용 폐루프에서는 미사용(랩 전용) |

## 6. 요약

정적으로는 공유 헤더 3개로만 묶인 독립 빌드 4종(SUMB / inference / kmod / Python 파이프라인),
동적으로는 tmpfs 파일 2개(제어 데이터) + ioctl 1개(실행) + 콘솔 CLI 1개(캠페인 제어)로 통신하는 구조.
