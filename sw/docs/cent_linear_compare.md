# Linear 층 비교: CENT 변형(AiM 시뮬레이터) 대 보드 애플리케이션

> Llama-3.2-1B 한 층의 linear 연산 7개(q/k/v/o/gate/up/down_proj)를 두 가지 방법으로
> 돌리고, 사이클을 연산별로 비교한다.
>
> - **보드**: `sw/app/generate.py` 가 보드에서 실제로 돌린 PIM 프로그램의 RUN_CYC
> - **CENT 변형**: CENT 의 trace 생성기에 우리 GEMV 방식을 넣어 만든 trace 를, 보드
>   타이밍을 넣은 AiM 시뮬레이터로 돌린 사이클
>
> 비교는 2026-09-29 에 했다. 비교 스크립트와 데이터는
> [`docs/log/cent_linear_2026-09-29/`](../../docs/log/cent_linear_2026-09-29/) 에 있다.
> CENT 원래 방식(WR_BIAS 사용)과의 비교는 이 문서에서 다루지 않는다.

---

## 0. 결론

- 7개 연산 모두 **트레이스 패턴이 보드와 같았다.** 같은 것은 네 가지다.
  - 명령 순서
  - MAC 이 row 를 여는 순서
  - 각 벡터 싣기가 싣는 입력 조각
  - 반복 단위와 그 반복 간격

  다른 것은 주소의 시작 번호뿐이다(부록 A).
- 사이클 차이는 연산별로 −0.19% ~ +0.29% 다. 층 합계로는 latch 1개일 때 **+0.23%**,
  2개일 때 **+0.13%** 로, 보드가 조금 느리다.
- 트레이스 패턴이 같으므로, 남은 차이는 시뮬레이터 타이밍 모델의 오차다. 보드가
  실제로 쓴 row 번호 그대로 시뮬레이터에 돌려도, 사이클이 CENT 변형과 한 자리까지
  같다. 따라서 CENT 변형으로 우리 시스템의 linear 부분을 예측할 수 있다.

| 뱅크당 latch | 보드 RUN_CYC (한 층) | 시뮬레이터 | 차이 |
|---|---|---|---|
| 1개 | 4,652,603 | 4,642,155 | +0.23% |
| 2개 | 3,383,403 | 3,378,907 | +0.13% |

---

## 1. 비교한 두 쪽

| | 보드 | CENT 변형 |
|---|---|---|
| 무엇을 돌렸나 | `sw/app/generate.py` 가 Llama-3.2-1B-Instruct 로 글을 생성하면서 낸 PIM 프로그램 | CENT trace 생성기(`ref/CENT/cent_simulation/function_sim.py`)가 만든 linear GEMV trace |
| GEMV 순서 | 우리 런타임(`sw/runtime`) | `--GEMV reuse-bank` (우리 순서를 CENT 에 넣은 것) |
| 실행 | ch2 v2.0 보드 | AiM 시뮬레이터(`ref/aim_simulator`, ramulator2 기반) |
| 사이클 | 실행(launch)마다 CFR RUN_CYC | `memory_system_cycles` |

### "CENT 변형" 이 무엇인가

**latch(누산기 칸)**: 뱅크마다 있는 MAC 결과 저장 칸이다. 출력 하나를 계산하는 동안
한 칸을 차지하고, RD_MAC 으로 꺼내야 비워진다.

입력 벡터가 Global Buffer(한 번에 1,024개)보다 길면 여러 조각으로 나눠 실어야 한다.
이때 CENT 원래 방식과 우리 방식이 갈린다.

- **CENT 원래 방식**: 입력 조각을 한 번만 싣고 모든 출력을 계산한다. 각 출력의
  부분합은 조각마다 꺼내 두었다가, 다음 조각 때 WR_BIAS 로 누산기에 다시 넣는다.
- **우리 방식**: 출력 묶음마다 모든 조각을 누산기 안에서 끝까지 더하고, 마지막에 한
  번만 읽는다. 대신 입력 조각을 출력 묶음마다 다시 싣는다. RD_MAC 이 읽으면서
  누산기를 비우므로 WR_BIAS 는 쓰지 않는다. 부분합이 누산기 밖으로 나가지 않아서
  반올림도 한 번뿐이다.

CENT 생성기에는 `--GEMV` 선택지로 `reuse-bank` 라는 이름만 있고 구현이 없었다.
2026-09-29 에 그 자리에 우리 방식을 넣었다.

- 일반 층: [TransformerBlock.py:736](../../../ref/CENT/cent_simulation/TransformerBlock.py#L736)
- 활성화 함수가 붙는 층: [:808](../../../ref/CENT/cent_simulation/TransformerBlock.py#L808)

이 문서에서 "CENT 변형" 은 이것을 말한다. WR_BIAS 는 내지 않는다.

뱅크당 latch 수는 `--reuse-size` 로 정하고, 1 또는 2 만 받는다
([utils.py:22](../../../ref/CENT/cent_simulation/utils.py#L22)). 우리 하드웨어는 뱅크당
2칸이고, ISR 의 T 비트로 둘 중 하나를 고른다. 보드의 기본 실행은 1칸을,
`--dual-latch` 실행은 2칸을 쓴다.

---

## 2. 모델과 하드웨어

**모델**: `meta-llama/Llama-3.2-1B-Instruct`. 16층, hidden 2048, query head 32, KV head 8,
head 크기 64, FFN 8192.

| 연산 | 출력 × 입력 | 출력 묶음 × 입력 조각 = MAC 수 |
|---|---|---|
| q_proj | 2048 × 2048 | 64 × 2 = 128 |
| k_proj | 512 × 2048 | 16 × 2 = 32 |
| v_proj | 512 × 2048 | 16 × 2 = 32 |
| o_proj | 2048 × 2048 | 64 × 2 = 128 |
| gate_proj | 8192 × 2048 | 256 × 2 = 512 |
| up_proj | 8192 × 2048 | 256 × 2 = 512 |
| down_proj | 2048 × 8192 | 64 × 8 = 512 |

- **출력 묶음**: MAC 한 번에 뱅크 32개(2채널 × 16뱅크)가 하나씩 내는 출력 32개다.
- **입력 조각**: 입력 1,024개다.

**보드**: ch2 v2.0 이미지(09-29 PDI), 2채널 × 16뱅크, 주소 배치 RoChBaCo.

**CENT 생성기 인자**:

```
--Llama-GQA --n_heads 16 --n_kv_heads 4 --ffn_dim 8192
--only-FC --only-trace --num-channels 2 --FC-devices 1 --model-parallel
--GEMV reuse-bank --reuse-size {1|2}
```

CENT 생성기는 head 크기를 128 로 고정한다. 그래서 head 16개, KV head 4개로 주면
linear 모양이 Llama-3.2-1B 와 같아진다. 어텐션 모양은 달라지지만 이번 비교에는 쓰지
않는다. `--only-FC` 는 linear GEMV 만 만든다.

---

## 3. 타이밍을 어떻게 맞췄나

양쪽에 **같은 숫자**를 넣었다. 시뮬레이터의 타이밍 값이 보드 레지스터 값 그대로이므로,
시뮬레이터 사이클과 보드 RUN_CYC 를 같은 단위(보드 PL 클록 사이클)로 바로 비교한다.

### 3.1 보드

2026-09-29 오전, 09-29 PDI 를 올린 뒤 `hwdef/test/emu_timing --scale 7` 로 타이밍
레지스터 11개를 설정했다. 값은 기본 세트에 7을 곱한 것이다. 

| 레지스터 | 기본 | ×7 (보드 값) |
|---|---|---|
| T_FAW | 16 | 112 |
| T_RRD | 4 | 28 |
| T_RCD | 15 | 105 |
| T_CCD | 2 | 14 |
| T_RTP | 4 | 28 |
| T_RP | 15 | 105 |
| T_WR | 28 | 196 |
| T_RAS | 34 | 238 |
| T_MOD | 30 | 210 |
| T_RP_AB | 17 | 119 |
| T_GB | 2 | 14 |

- **애플리케이션은 타이밍을 설정하지 않는다.** 런타임이 엔진을 열 때 10개를 읽어
  기록하는데, 보드 실행 때 읽힌 값이 위 표와 같았다.
- **T_GB 는 런타임이 아직 모르는 레지스터라 기록이 없다.** 14 로 본 근거는 두 가지다.
  - 오전에 설정한 뒤 재프로그램한 기록이 없다.
  - 이번 결과가 0.2% 로 맞는다. T_GB 가 0 이었다면 벡터 싣기가 7배 빨랐을 것이고,
    벡터 싣기가 많은 latch 1개 쪽이 시뮬레이터보다 크게 빨랐을 것이다.

  직접 읽어 확인하는 일은 남아 있다.
- **RUN_CYC** 는 CFR 0x02C/0x030 ([emu_regs.h:144-145](../../hwdef/emu_regs.h#L144-L145))
  이다.
  - doorbell 부터 done 까지를 PL 클록으로 센다.
  - 런타임이 done 직후 읽어 덤프의 launch 헤더에 적는다.
  - done 뒤에 이어지는 EOS 되쓰기는 들어가지 않는다.

### 3.2 시뮬레이터

- **버전**: `ref/aim_simulator` 커밋 0f28a07. 로컬 변경 두 개는 환경 변수로 켜는
  진단 출력(`AIM_TIMING_DUMP`, `AIM_CMD_TRACE`)뿐이라 타이밍에 영향이 없다.
- **설정**: `test/example.yaml`(GDDR6_AiM_org, GDDR6_AiM_timing preset) 위에 보드
  레지스터 값을 덮어쓴 `aim.yaml` 이다.
  - 덮어쓰기는 [scripts/aim_compare.py](../../scripts/aim_compare.py#L33-L59) 의
    `sim_params` 로 했다.
  - ISR 단위 비교(`scripts/logs/aimisr-1hot`, ISR 마다 ±9 사이클 안)에 쓴 것과 같은
    규칙이다.

| 보드 레지스터 | 시뮬레이터 파라미터 | 값 |
|---|---|---|
| T_RCD | nRCDRD, nRCDRDMAC, nRCDEWMUL, nRCDRDAF, nRCDRDCP, nRCDWR, nRCDWRCP | 105 |
| T_CCD | nCCDL | 14 |
| T_GB | nCCDS | 14 |
| T_RTP | nRTP | 28 |
| T_RAS | nRAS | 238 |
| T_RP_AB | nRP | 119 |
| T_MOD | nMODCH | 210 |
| T_RRD | nRRDS, nRRDL | 28 |
| T_FAW | nFAW | 112 |
| T_WR | nWR = T_WR − nCWL(6) − nBL(2) | 188 |
| T_RAS + T_RP_AB | nRC | 357 |

- **T_GB 와 T_CCD 의 대응**: 시뮬레이터는 벡터 싣기(WR_GB)의 beat 간격을
  max(nBL, nCCDS) 로, MAC 의 beat 간격을 max(nCCDS, nCCDL) 로 둔다. 그래서 T_GB 를
  nCCDS 에, T_CCD 를 nCCDL 에 넣으면, T_GB ≤ T_CCD 인 동안 보드와 정확히 같아진다.
  여기서는 둘 다 14 다.
- **precharge 시간**: 시뮬레이터에는 하나뿐이라 T_RP_AB(119)를 넣었다. 뱅크 하나짜리
  T_RP(105)는 쓰지 않는다.
- **나머지 값**(nBL 2, nCWL 6 등)은 preset 그대로다.

---

## 4. 언제, 무엇을 비교했나

| 항목 | 시점 | 상태 |
|---|---|---|
| 보드 실행 (ISA 덤프) | 2026-09-29 15:04~15:05 | emulator_top 작업 트리(커밋 전). `--isa-trace` 와 `--dual-latch` 가 들어간 직후 |
| CENT trace 생성 | 2026-09-29 16:14 | ref/CENT 커밋 4c2aac6 + 로컬 변경(`reuse-bank` 추가, `--reuse-size` 는 1·2 만 허용) |
| 비교 실행 | 2026-09-29 16:14 | [compare_linear.py](../../docs/log/cent_linear_2026-09-29/compare_linear.py) |

보드 실행 뒤 q·K 스케줄이 바뀌었지만(15:28 완료), linear 연산과는 관계없다.

**보드 실행 조건** (pim-13 세션이 실행):
- 프롬프트 "The capital of France is"(6토큰), `--max-new-tokens 4`. 프롬프트 단계
  (step 0)와 디코드 3단계(step 1~3)다.
- latch 1개(기본)로 한 번, `--dual-latch` 로 한 번 돌렸다. 두 실행의 logits 는 모든
  단계에서 비트까지 같았다.
- `--isa-trace` 덤프는 단계마다 파일 하나다. 실행(launch)마다 헤더 한 줄(단계, 위치,
  층, 연산 이름, ISR 수, RUN_CYC)과 ISR 한 줄씩이 들어간다. 형식은
  [isa_trace.py](../app/pimllm/isa_trace.py) 맨 위에 있다.

**비교에 쓴 부분**:
- **step 1**(첫 디코드 단계, 위치 6)의 **0번 층** linear 연산 7개다. 트레이스 패턴 대조와
  사이클 비교 모두 이 층으로 했다.
- 16개 층 전체에서 연산별 보드 사이클이 똑같은 것도 확인했다. `results.csv` 에서
  board_min 과 board_max 가 같다.
- 이번 실행에서 linear 연산은 모두 연산당 launch 1번이었다.

---

## 5. 비교 기준

1. **latch 수를 같게 한다.**

   | 보드 | CENT |
   |---|---|
   | 기본 실행 | `--reuse-size 1` |
   | `--dual-latch` | `--reuse-size 2` |

2. **CENT 출력에 보드 규칙을 적용한다.**
   - 활성화 함수 명령(AF, RD_AF)을 뺀다. 보드에는 활성화 함수 장치가 없고, 이번에는
     GEMV 만 비교한다.
   - RD_MAC 을 채널별 두 줄로 나눈다. 보드의 RD_MAC 은 결과를 GPR 워드 하나에만
     놓으므로 채널마다 따로 낸다.
   - WR_BIAS 는 CENT 변형이 내지 않으므로 따로 할 일이 없다.
3. **트레이스 패턴이 같아야 한다.** CENT trace 는 연산별 MAC 수로 잘라 연산마다
   나눈 뒤, 연산마다 네 가지를 비교한다. 정의와 스니펫은 부록 A 에 있다.
   - 명령 열: (명령, OPSIZE, 채널 마스크)를 처음부터 끝까지 비교한다. 보드의 EOS 는
     뺀다.
   - row 패턴: 각 MAC 의 row 를 처음 나온 순서로 번호를 다시 매겨 비교한다.
   - 입력 조각 순서: 각 벡터 싣기가 몇 번째 입력 조각을 싣는지 비교한다.
   - 반복 단위: 단위를 이루는 명령, 반복 횟수, row 간격을 비교한다.
   - 보조 확인으로, 보드 row 번호 그대로 만든 trace 의 시뮬레이터 사이클이 같은지 본다.
4. **사이클을 비교한다.**
   - 보드 쪽은 그 연산의 RUN_CYC 다. 연산이 여러 launch 로 나뉘면 합한다.
   - 시뮬레이터 쪽은 그 연산의 trace 만 따로 돌린 사이클이다.
   - 차이 = (보드 − 시뮬레이터) ÷ 시뮬레이터.
5. **예외: latch 2개의 gate_proj.**
   - CENT 는 활성화 함수가 붙는 층에서 latch 절반을 그 결과용으로 남긴다. 그래서 CENT
     출력의 gate_proj 는 latch 1개로 돈다. 보드는 2개를 다 쓴다.
   - 이 행만, 같은 순서를 latch 2개로 만든 자체 trace(`compare_linear.py` 의
     `dual_both_latches`)로 돌렸다. 보드와 순서가 같은 것은 확인했다.
   - CENT 에 latch 를 전부 쓰는 설정을 넣으면 없어질 예외다.

---

## 6. 결과

step 1, 0번 층이다. 연산별 보드 사이클은 16개 층 모두 같다.

### latch 1개

| 연산 | 보드 | 시뮬레이터 | 차이 | 트레이스 패턴 (부록 A) |
|---|---|---|---|---|
| q_proj | 323,789 | 323,229 | +0.17% | 같음 |
| k_proj | 80,861 | 80,877 | −0.02% | 같음 |
| v_proj | 80,861 | 80,877 | −0.02% | 같음 |
| o_proj | 323,789 | 323,229 | +0.17% | 같음 |
| gate_proj | 1,295,501 | 1,292,637 | +0.22% | 같음 |
| up_proj | 1,295,501 | 1,292,637 | +0.22% | 같음 |
| down_proj | 1,252,301 | 1,248,669 | +0.29% | 같음 |
| **한 층 합계** | **4,652,603** | **4,642,155** | **+0.23%** | |

### latch 2개

| 연산 | 보드 | 시뮬레이터 | 차이 | 트레이스 패턴 (부록 A) |
|---|---|---|---|---|
| q_proj | 234,861 | 234,685 | +0.07% | 같음 |
| k_proj | 58,629 | 58,741 | −0.19% | 같음 |
| v_proj | 58,629 | 58,741 | −0.19% | 같음 |
| o_proj | 234,861 | 234,685 | +0.07% | 같음 |
| gate_proj | 939,789 | 938,461 | +0.14% | 같음 (자체 trace, 5의 예외) |
| up_proj | 939,789 | 938,461 | +0.14% | 같음 |
| down_proj | 916,845 | 915,133 | +0.19% | 같음 |
| **한 층 합계** | **3,383,403** | **3,378,907** | **+0.13%** | |

### 디코드 한 단계 안에서 linear 의 비중

보드 기준, step 1 의 16개 층 합계와 lm_head 를 포함한 값이다.

| | latch 1개 | latch 2개 |
|---|---|---|
| linear 7개 × 16층 | 74,441,648 (77.3%) | 54,134,448 (77.4%) |
| lm_head | 20,284,258 (21.1%) | 14,715,142 (21.0%) |
| 어텐션 (q·K + s·V) | 1,634,016 (1.7%) | 1,056,994 (1.5%) |
| 합계 | 96,359,922 | 69,906,584 |

위치 6, 즉 문맥이 짧을 때의 비중이다. 문맥이 길어질수록 어텐션 비중이 커진다.

---

## 7. 해석

- **보드가 모든 연산에서 0.1~0.3% 느리다.** ISR 단위 비교(`scripts/logs/aimisr-1hot`)
  에서, 모드 전환 직후의 MAC 은 보드가 8~9 사이클 더 걸렸다. 그 차이가 연산 크기에
  비례해 쌓인 것으로 보인다. 모드 전환이 적은 latch 2개 쪽의 오차가 더 작은 것도 같은
  방향이다.
- **이 차이는 시뮬레이터 타이밍 모델의 오차다.** 트레이스 패턴이 같고, 보드 row
  번호로 돌린 시뮬레이션도 CENT 변형과 같다. 그래서 GEMV 순서나 주소 배치의 차이로
  볼 여지가 없다.

---

## 8. 이번 비교에 없는 것

- **어텐션(q·K, s·V)**: 보류했다. CENT 생성기가 Llama-3.2-1B 의 어텐션 모양을 만들지
  못한다. head 크기가 128 로 고정돼 있고, K 저장 코드가 H_kv·D ≥ 1,024 를 가정한다.
  K·V 배치와 순서도 우리와 다르다.
- **lm_head**: 디코드 한 단계의 21% 다. CENT 에서는 별도 경로(`--embedding`)이고,
  보드에서는 launch 2번이다. 다음 비교 대상이다.
- **프롬프트 단계(step 0)**: 비교하지 않았다.
- **T_GB 직접 확인**: 3.1 참고.
- **CENT gate 설정**: 5의 예외 참고.

---

## 9. 다시 돌리려면

**1. 보드에서 덤프를 만든다.** 보드가 필요하다. 다른 세션이 보드를 쓰고 있지 않은지
먼저 확인한다.

```
cd sw/app
./generate.py --model llama-3.2-1b-instruct --prompt "The capital of France is" \
    --max-new-tokens 4 --isa-trace /tmp/isa_single
./generate.py --model llama-3.2-1b-instruct --prompt "The capital of France is" \
    --max-new-tokens 4 --isa-trace /tmp/isa_dual --dual-latch
```

**2. 비교한다.** 보드는 필요 없다.

```
python3 docs/log/cent_linear_2026-09-29/compare_linear.py \
    --single /tmp/isa_single --dual /tmp/isa_dual --out <결과 폴더>
```

- CENT 생성기는 torch 가 있는 conda 환경 `cent` 로 돈다. 스크립트 안의 `CENT_PYTHON`
  이 그 경로다.
- `ref/aim_simulator/build/ramulator2` 가 빌드돼 있어야 한다.
- `--timing` 에는 보드 실행 때의 레지스터 값이 든 `timing.txt` 가 있는 폴더를 준다.
  기본값 `scripts/logs/aimcmp-1hot` 은 이번 ×7 세트다. 타이밍이 다르면 바꿔 줘야 한다.

**3. 이번 결과물** (`docs/log/cent_linear_2026-09-29/run/`)

| 파일 | 내용 |
|---|---|
| `results.csv` | 연산별 보드·시뮬레이터 사이클과 차이, 트레이스 출처, 16층의 최소·최대. 패턴 판정 열(`cmds_same`, `rows_same`, `chunks_same`, `unit_same`, `regular`, `unit`, `repeats`, `unit_rows`, `row_step`)과 보드 row 로 돌린 사이클(`sim_board_rows`) |
| `launches_L1.tsv`, `launches_L2.tsv` | step 1 의 모든 launch 헤더(층, 연산, part, ISR 수, RUN_CYC) |
| `layer0_isrs_L1.tsv`, `_L2.tsv` | 0번 층 linear 연산의 보드 ISR(명령, OPSIZE, 채널, ROW, COL, T) |
| `sim_board_rows/*.trace` | 보드 ISR 을 보드 row 번호 그대로 옮긴 시뮬레이터 입력 |
| `steps_L1.tsv`, `steps_L2.tsv` | 보드 실행의 단계별 합계(pim-13 덤프에서 복사) |
| `cent_reuse_bank_L1.trace`, `_L2.trace` | CENT 생성기 출력 원본 |
| `sim/*.trace` | 연산별 시뮬레이터 입력(보드 규칙 적용 후) |
| `aim.yaml`, `timing.txt` | 시뮬레이터 설정과, 그 바탕이 된 보드 레지스터 값 |

원본 ISA 덤프는 pim-13 세션의 임시 폴더에 있어 보존되지 않는다. 비교에 필요한 부분은
위 파일들로 옮겨 두었다.

---

## 부록 A. 트레이스 패턴 비교: "같다" 의 뜻

본문에서 "트레이스 패턴이 같다" 고 한 것의 정확한 뜻과 근거를 적는다. 판정은 모두
`compare_linear.py` 로 다시 만들 수 있고, `run/results.csv` 의 해당 열에 남아 있다.

### A.1 기준

양쪽 trace 는 주소의 절대 번호부터 다르다. 보드는 우리 할당기가 준 row 를 쓰고,
CENT 는 CENT 의 메모리 배치가 준 row 를 쓴다. 그래서 줄 단위 문자열로 비교하지 않고,
아래 네 기준으로 비교했다.

| 기준 | 비교하는 것 | 방법 | results.csv 열 |
|---|---|---|---|
| 1. 명령 열 | (명령, OPSIZE, 채널 마스크)의 순서 | 처음부터 끝까지 한 줄씩 비교한다. 보드의 EOS 는 뺀다 | `cmds_same` |
| 2. row 패턴 | 각 MAC 이 여는 DRAM row | 처음 나온 순서대로 0, 1, 2, … 로 번호를 다시 매겨 비교한다. 같으면 새 row 를 여는 시점과 row 를 다시 여는 순서가 같다 | `rows_same` |
| 3. 입력 조각 순서 | 각 벡터 싣기(WRVEC)가 싣는 입력 조각 | 아래 설명 참고 | `chunks_same` |
| 4. 반복 단위 | 명령 열의 가장 짧은 반복 단위 | 단위의 명령, 반복 횟수, 단위 안 MAC 의 상대 row, 반복마다 row 가 넘어가는 간격이 같은지 본다. 모든 반복이 첫 단위를 그 간격만큼 옮긴 것인지도 본다 | `unit_same`, `regular` |

**기준 3 의 방법**
- 보드: WRVEC 의 GPR 워드 주소를 처음 나온 순서로 번호 매긴다. 조각마다 64워드
  간격이다.
- CENT: WR_GB 에는 주소가 없어서, 바로 다음 MAC 의 row 로 계산한다. reuse-bank
  순서에서 row = 시작 row + 묶음 × 조각 수 + 조각 이기 때문이다.

**보조 확인 (`sim_board_rows`)**: 보드 ISR 을 보드 row 번호 그대로 AiM trace 로 옮겨
시뮬레이터에 돌렸다. 이 사이클이 CENT trace 의 사이클과 같으면, row 번호의 절대값은
타이밍에 영향이 없고 패턴만 중요하다는 것이 직접 확인된다.

**비교하지 않은 것**
- GPR 주소(WRVEC 가 읽는 워드, RD_MAC 결과가 떨어지는 워드): CENT trace 는 늘 0 을
  적고, 시뮬레이터도 타이밍에 쓰지 않는다. WRVEC 쪽은 기준 3 에서 조각 번호로 바꿔
  비교했다.
- T 비트(latch 선택): CENT trace 와 시뮬레이터에 이 칸이 없다. 보드 값만 기록했다
  (A.3, A.4).
- COL: CENT trace 에 이 칸이 없다. 보드는 linear MAC 에서 늘 0 이었다. row 를
  처음부터 읽는다는 뜻이다.

### A.2 결과

14개 경우(latch 1·2 × 연산 7개) 모두 기준 1~4 를 만족했다. 보드 row 로 돌린
시뮬레이션도 모두 CENT 와 사이클이 같았다(A.6).

양쪽 값이 같아서 한 칸에 적는다. V 는 벡터 싣기(WRVEC, CENT 에서는 WR_GB), M 은 MAC,
R 은 RD_MAC 이다.

| latch | 연산 | 반복 단위 | 반복 횟수 | 단위 안 MAC 의 row (첫 MAC 기준) | 반복 간격 (row) |
|---|---|---|---|---|---|
| 1 | q_proj, o_proj | VMVMRR | 64 | +0, +1 | 2 |
| 1 | k_proj, v_proj | VMVMRR | 16 | +0, +1 | 2 |
| 1 | gate_proj, up_proj | VMVMRR | 256 | +0, +1 | 2 |
| 1 | down_proj | VMVMVMVMVMVMVMVMRR | 64 | +0, +1, …, +7 | 8 |
| 2 | q_proj, o_proj | VMMVMMRRRR | 32 | +0, +2, +1, +3 | 4 |
| 2 | k_proj, v_proj | VMMVMMRRRR | 8 | +0, +2, +1, +3 | 4 |
| 2 | gate_proj, up_proj | VMMVMMRRRR | 128 | +0, +2, +1, +3 | 4 |
| 2 | down_proj | (VMM)×8 RRRR | 32 | +0, +8, +1, +9, …, +7, +15 | 16 |

latch 2 의 gate_proj 는 본문 §5 의 예외대로 자체 trace 와 비교한 결과다.

**반복 단위가 뜻하는 것**
- **latch 1, 입력 조각 2개**: 출력 묶음 하나마다 다음을 한다.
  - [조각0 싣기, MAC, 조각1 싣기, MAC, 채널0 읽기, 채널1 읽기]
  - MAC 두 개는 그 묶음의 가중치 row 두 개(조각0용, 조각1용)를 연다.
  - 다음 묶음은 row 2개 뒤에서 시작한다. 반복 횟수는 출력 묶음 수다.
- **latch 1, down_proj (입력 조각 8개)**: 싣기와 MAC 을 8번 한 뒤 읽기를 2번 한다.
  다음 묶음은 row 8개 뒤다.
- **latch 2**: 출력 묶음 두 개(A, B)가 한 번 실은 조각을 같이 쓴다.
  - [조각0 싣기, MAC A, MAC B, 조각1 싣기, MAC A, MAC B, 읽기 4번]
  - A 의 row 는 +0(조각0), +1(조각1)이고 B 의 row 는 +2, +3 이다. 그래서 MAC 순서로
    보면 +0, +2, +1, +3 이다.
  - 다음 쌍은 row 4개 뒤다. 반복 횟수는 출력 묶음 쌍의 수다.
- **latch 2, down_proj**: 같은 구조에 조각이 8개다. row 는 A 가 +0~+7, B 가 +8~+15 이고,
  다음 쌍은 row 16개 뒤다.

### A.3 스니펫: q_proj, latch 1개

**보드 덤프** (step 1, 0번 층)
- 맨 끝의 256비트 원본 워드 칸은 "…" 로 줄였다.
- ROW 칸은 WRVEC·RD_MAC 에서는 GPR 워드 주소이고, MAC 에서는 DRAM row 다.

```
# launch 707  step 1  pos 6  layer 0  q_proj  part 1  isrs 385  run_cyc 323789  us 3240
  #     op      OPSIZE  ROW      COL  CH    PU     GBMC   T  word (bit 255 first)
  0     WRVEC   64      0        0    0x3   0      0      0  …
  1     MAC     64      0        0    0x3   0xffff 0xffff 0  …
  2     WRVEC   64      64       0    0x3   0      0      0  …
  3     MAC     64      1        0    0x3   0xffff 0xffff 0  …
  4     RD_MAC  0       32768    0    0x1   0      0      0  …
  5     RD_MAC  0       32769    0    0x2   0      0      0  …
  6     WRVEC   64      0        0    0x3   0      0      0  …
  7     MAC     64      2        0    0x3   0xffff 0xffff 0  …
  8     WRVEC   64      64       0    0x3   0      0      0  …
  9     MAC     64      3        0    0x3   0xffff 0xffff 0  …
  10    RD_MAC  0       32770    0    0x1   0      0      0  …
  11    RD_MAC  0       32771    0    0x2   0      0      0  …
  …
  378   WRVEC   64      0        0    0x3   0      0      0  …
  379   MAC     64      126      0    0x3   0xffff 0xffff 0  …
  380   WRVEC   64      64       0    0x3   0      0      0  …
  381   MAC     64      127      0    0x3   0xffff 0xffff 0  …
  382   RD_MAC  0       32894    0    0x1   0      0      0  …
  383   RD_MAC  0       32895    0    0x2   0      0      0  …
  384   EOS     0       0        0    0x3   0      0      0  …
```

**CENT 생성기 원본** (`run/cent_reuse_bank_L1.trace` 앞부분). 줄 형식은 다음과 같다.
- `AiM WR_GB <OPSIZE> <GPR> <채널>`
- `AiM MAC_ABK <OPSIZE> <채널> <row>`
- `AiM RD_MAC <GPR> <채널>`

```
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 3
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 4
AiM RD_MAC 0 0x3
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 5
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 6
AiM RD_MAC 0 0x3
```

**보드 규칙 적용 후** (`run/sim/q_proj_L1.trace` 앞부분). RD_MAC 한 줄이 채널별 두
줄이 됐다.

```
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 3
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 4
AiM RD_MAC 0 0x1
AiM RD_MAC 0 0x2
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 5
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 6
AiM RD_MAC 0 0x1
AiM RD_MAC 0 0x2
```

**나란히 놓으면** (첫 두 반복 단위, "보드 / CENT" 순서로 적음)

| # | 보드 | CENT (규칙 적용 후) | 기준 1 명령 | 기준 2 row (다시 매김) | 기준 3 입력 조각 |
|---|---|---|---|---|---|
| 0 | WRVEC 64, GPR 0 | WR_GB 64 | 같음 | | 0 / 0 |
| 1 | MAC 64, row 0 | MAC_ABK 64, row 3 | 같음 | 0 / 0 | |
| 2 | WRVEC 64, GPR 64 | WR_GB 64 | 같음 | | 1 / 1 |
| 3 | MAC 64, row 1 | MAC_ABK 64, row 4 | 같음 | 1 / 1 | |
| 4 | RD_MAC 채널0 | RD_MAC 0x1 | 같음 | | |
| 5 | RD_MAC 채널1 | RD_MAC 0x2 | 같음 | | |
| 6 | WRVEC 64, GPR 0 | WR_GB 64 | 같음 | | 0 / 0 |
| 7 | MAC 64, row 2 | MAC_ABK 64, row 5 | 같음 | 2 / 2 | |
| 8 | WRVEC 64, GPR 64 | WR_GB 64 | 같음 | | 1 / 1 |
| 9 | MAC 64, row 3 | MAC_ABK 64, row 6 | 같음 | 3 / 3 | |
| 10 | RD_MAC 채널0 | RD_MAC 0x1 | 같음 | | |
| 11 | RD_MAC 채널1 | RD_MAC 0x2 | 같음 | | |

- **row**: 보드는 0 부터, CENT 는 3 부터 시작한다. CENT 의 메모리 배치에서 q 가중치가
  row 3 부터 놓이기 때문이다. 다시 매기면 둘 다 0, 1, 2, 3 이다.
- **입력 조각**: 보드는 GPR 0 이 조각 0, GPR 64 가 조각 1 이다. CENT 는 다음 MAC 의
  row 에서 계산한다(row 3 → (3−3) mod 2 = 0, row 4 → 1). 둘 다 0, 1, 0, 1 이다.
- **반복**: 이 6줄 단위가 64번 반복되고, 반복마다 row 가 2씩 넘어간다. 마지막 단위의
  MAC row 는 보드가 126, 127, CENT 가 129, 130 이다.

### A.4 스니펫: q_proj, latch 2개

**보드 덤프**. T 칸이 latch 번호다.

```
# launch 707  step 1  pos 6  layer 0  q_proj  part 1  isrs 321  run_cyc 234861  us 2350
  #     op      OPSIZE  ROW      COL  CH    PU     GBMC   T  word (bit 255 first)
  0     WRVEC   64      0        0    0x3   0      0      0  …
  1     MAC     64      0        0    0x3   0xffff 0xffff 0  …
  2     MAC     64      2        0    0x3   0xffff 0xffff 1  …
  3     WRVEC   64      64       0    0x3   0      0      0  …
  4     MAC     64      1        0    0x3   0xffff 0xffff 0  …
  5     MAC     64      3        0    0x3   0xffff 0xffff 1  …
  6     RD_MAC  0       32768    0    0x1   0      0      0  …
  7     RD_MAC  0       32769    0    0x2   0      0      0  …
  8     RD_MAC  0       32770    0    0x1   0      0      1  …
  9     RD_MAC  0       32771    0    0x2   0      0      1  …
  10    WRVEC   64      0        0    0x3   0      0      0  …
  11    MAC     64      4        0    0x3   0xffff 0xffff 0  …
  12    MAC     64      6        0    0x3   0xffff 0xffff 1  …
  …
```

**CENT 생성기 원본** (`run/cent_reuse_bank_L2.trace` 앞부분)

```
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 3
AiM MAC_ABK 64 0x3 5
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 4
AiM MAC_ABK 64 0x3 6
AiM RD_MAC 0 0x3
AiM RD_MAC 0 0x3
```

**보드 규칙 적용 후** (`run/sim/q_proj_L2.trace` 앞부분). RD_MAC 두 줄이 네 줄이 됐다.

```
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 3
AiM MAC_ABK 64 0x3 5
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 4
AiM MAC_ABK 64 0x3 6
AiM RD_MAC 0 0x1
AiM RD_MAC 0 0x2
AiM RD_MAC 0 0x1
AiM RD_MAC 0 0x2
```

**나란히 놓으면** (첫 반복 단위)

| # | 보드 | CENT (규칙 적용 후) | 기준 1 명령 | 기준 2 row (다시 매김) | 기준 3 입력 조각 | 보드 T |
|---|---|---|---|---|---|---|
| 0 | WRVEC 64, GPR 0 | WR_GB 64 | 같음 | | 0 / 0 | |
| 1 | MAC 64, row 0 | MAC_ABK 64, row 3 | 같음 | 0 / 0 | | 0 |
| 2 | MAC 64, row 2 | MAC_ABK 64, row 5 | 같음 | 1 / 1 | | 1 |
| 3 | WRVEC 64, GPR 64 | WR_GB 64 | 같음 | | 1 / 1 | |
| 4 | MAC 64, row 1 | MAC_ABK 64, row 4 | 같음 | 2 / 2 | | 0 |
| 5 | MAC 64, row 3 | MAC_ABK 64, row 6 | 같음 | 3 / 3 | | 1 |
| 6 | RD_MAC 채널0 | RD_MAC 0x1 | 같음 | | | 0 |
| 7 | RD_MAC 채널1 | RD_MAC 0x2 | 같음 | | | 0 |
| 8 | RD_MAC 채널0 | RD_MAC 0x1 | 같음 | | | 1 |
| 9 | RD_MAC 채널1 | RD_MAC 0x2 | 같음 | | | 1 |

- **row**: 상대 위치 +0, +2, +1, +3 이 양쪽 같다. 보드는 0, 2, 1, 3 이고 CENT 는 3, 5,
  4, 6 이다.
- **보드 T**: MAC 은 latch 0 과 1 을 번갈아 쓴다. 읽기는 latch 0 의 두 채널, 그다음
  latch 1 의 두 채널 순서다. CENT trace 에는 latch 칸이 없지만, 같은 자리에 같은
  명령이 온다.
- **반복**: 이 10줄 단위가 32번 반복되고, 반복마다 row 가 4씩 넘어간다. 마지막 단위의
  MAC row 는 보드가 124, 126, 125, 127, CENT 가 127, 129, 128, 130 이다.

### A.5 스니펫: down_proj, latch 1개 (입력 조각 8개)

**보드 덤프**. 256비트 워드 칸과 그 앞의 PU·GBMC·T 칸은 줄였다.

```
# launch 715  step 1  pos 6  layer 0  down_proj  part 1  isrs 1153  run_cyc 1252301  us 12525
  #     op      OPSIZE  ROW      COL  CH
  0     WRVEC   64      0        0    0x3   …
  1     MAC     64      1344     0    0x3   …
  2     WRVEC   64      64       0    0x3   …
  3     MAC     64      1345     0    0x3   …
  4     WRVEC   64      128      0    0x3   …
  5     MAC     64      1346     0    0x3   …
  6     WRVEC   64      192      0    0x3   …
  7     MAC     64      1347     0    0x3   …
  8     WRVEC   64      256      0    0x3   …
  9     MAC     64      1348     0    0x3   …
  10    WRVEC   64      320      0    0x3   …
  11    MAC     64      1349     0    0x3   …
  12    WRVEC   64      384      0    0x3   …
  13    MAC     64      1350     0    0x3   …
  14    WRVEC   64      448      0    0x3   …
  15    MAC     64      1351     0    0x3   …
  16    RD_MAC  0       32768    0    0x1   …
  17    RD_MAC  0       32769    0    0x2   …
  …
  1149  MAC     64      1855     0    0x3   …
  1150  RD_MAC  0       32894    0    0x1   …
  1151  RD_MAC  0       32895    0    0x2   …
  1152  EOS     0       0        0    0x3   …
```

**CENT, 보드 규칙 적용 후** (`run/sim/down_proj_L1.trace` 첫 단위)

```
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1557
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1558
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1559
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1560
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1561
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1562
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1563
AiM WR_GB 64 0 0x3
AiM MAC_ABK 64 0x3 1564
AiM RD_MAC 0 0x1
AiM RD_MAC 0 0x2
```

- **입력 조각**: 보드는 GPR 0, 64, …, 448 로 조각 0~7 을 싣는다. CENT 는 row
  1557~1564 에서 (row − 1557) mod 8 로 계산해 역시 0~7 이다.
- **row**: 보드 1344~1351, CENT 1557~1564 이다. 다시 매기면 둘 다 0~7 이다.
- **반복**: 이 18줄 단위가 64번 반복되고, 반복마다 row 가 8씩 넘어간다. 마지막 MAC
  row 는 보드가 1855(= 1344 + 511), CENT 가 2068(= 1557 + 511) 이다.

### A.6 보드 row 로 돌린 시뮬레이션

보드 ISR 을 보드 row 번호 그대로 옮긴 trace(`run/sim_board_rows/`)와 CENT trace
(`run/sim/`)의 시뮬레이터 사이클이다.

| 연산 | latch 1: CENT trace | latch 1: 보드 row | latch 2: CENT trace | latch 2: 보드 row |
|---|---|---|---|---|
| q_proj, o_proj | 323,229 | 323,229 | 234,685 | 234,685 |
| k_proj, v_proj | 80,877 | 80,877 | 58,741 | 58,741 |
| gate_proj, up_proj | 1,292,637 | 1,292,637 | 938,461 | 938,461 |
| down_proj | 1,248,669 | 1,248,669 | 915,133 | 915,133 |

모든 경우 한 자리까지 같다. row 의 시작 번호가 달라도, 새 row 를 여는 패턴이 같으면
시뮬레이터 타이밍이 같다는 뜻이다. 그래서 본문 §6 의 차이(보드 대 시뮬레이터
0.1~0.3%)는 trace 의 차이가 아니라, 보드와 시뮬레이터의 타이밍 차이다.
