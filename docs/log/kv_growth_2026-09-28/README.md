# KV 캐시 grow 실측 (2026-09-28 ~ 09-29)

KV 캐시를 토큰이 들어오는 만큼 늘리도록 바꾼 뒤(growable `pim_tensor`), 모델마다
K와 V 페이지가 몇 번째 토큰에서 할당되는지, 카드 메모리가 어디서 바닥나는지를 보드에서 잰 기록이다.

보고서: https://claude.ai/artifact/82PXWsAjnNPvuQAYN4XkSk (원본은 이 폴더의 `kv_report.html`)

## 측정 조건

- 보드: ch2 이미지, 2채널 × 16뱅크, DRAM 8 GiB, 할당 단위 64 KiB, 주소 매핑 RoChBaCo
- DRAM 타이밍: 다른 작업이 걸어 둔 `emu_timing` ×7 설정 상태. 토큰당 시간은 이 설정 기준이다.
- 코드: growable KV 변경을 적용한 작업 트리(측정 시점에는 커밋 전)
- 모델: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Qwen3-0.6B

## 파일

| 경로 | 내용 |
|---|---|
| `data/<모델>.alloc.jsonl` | 할당 전용 실행의 grow 이벤트, 한 줄에 한 건 |
| `data/<모델>.alloc.summary.json` | 할당 실행 요약: 메모리 추정, 마지막 성공 토큰, 실패 토큰과 오류, DRAM 사용량 |
| `data/<모델>.gen.jsonl` | 실제 생성 기록, 한 줄에 토큰 하나 |
| `data/<모델>.gen.summary.json` | 생성 실행 요약: 프롬프트 길이, 생성 토큰 수, 시간, 총 ISA, 실행 횟수, DRAM 사용량 |
| `data/<모델>.gen.log` | 생성 실행의 표준 출력 |
| `data/overnight.log` | 생성 실행 세 개의 시작과 종료 시각, 종료 코드 |
| `scripts/kv_experiment.py` | 실험 드라이버. `--mode alloc` 또는 `--mode gen` |
| `scripts/run_overnight.sh` | 세 모델의 생성 실행을 순서대로 돌린 스크립트 |
| `scripts/report_gen.py` | `data/`로 `kv_report.html`을 만든다 |
| `scripts/kv_grow_board.py` | growable과 고정 크기 텐서의 q·K, s·V 결과가 비트 단위로 같은지 보는 보드 검증 |

## 기록 형식

`*.alloc.jsonl` — grow 한 건마다 한 줄. 한 토큰에서 모든 레이어가 grow하므로 같은 `tokens`가 레이어 수만큼 나온다.

| 필드 | 뜻 |
|---|---|
| `tokens` | 이 grow가 일어났을 때 캐시가 담아야 하는 토큰 수 |
| `layer`, `which` | 레이어 번호, `K` 또는 `V` |
| `units`, `bytes` | 이번에 할당한 64 KiB unit 수와 바이트 |
| `room` | 할당 후 그 텐서가 담을 수 있는 토큰 수 |
| `t`, `dt` | 시각(monotonic), 실행 시작 후 초 |

`*.gen.jsonl` — 첫 줄은 `kind: "prefill"`(프롬프트 토큰 수), 이후 `kind: "token"`이 생성 토큰마다 한 줄.

| 필드 | 뜻 |
|---|---|
| `i`, `seq` | 생성 순번, 시퀀스 위치(프롬프트 포함) |
| `t`, `dt` | 시작 후 초, 이 토큰에 걸린 초 |
| `isr` | 이 토큰에 실행된 ISA 수 |
| `grows` | 이 토큰을 계산하는 동안 일어난 grow (필드는 alloc.jsonl과 같음) |

캐시가 N토큰이 되는 grow는 N+1번째 출력 토큰을 계산할 때 일어나므로, `tokens: N`인 grow는 `seq: N+1` 줄에 기록된다.
프롬프트 처리 중의 grow는 첫 생성 토큰(`i: 1`) 줄에 들어 있다.

## 다시 돌리기

pim.ko와 QDMA 큐가 준비된 상태에서, 한 번에 하나씩(보드는 동시에 두 작업을 못 한다):

```sh
cd emulator_top/docs/log/kv_growth_2026-09-28
PY=/home/kjy/miniconda3/envs/pim/bin/python

# 할당 전용: 가중치를 올린 뒤 카드가 거부할 때까지 grow (모델당 1분 안팎)
$PY scripts/kv_experiment.py --model meta-llama/Llama-3.2-1B-Instruct --mode alloc --out data

# 실제 생성: 시간 제한까지 (세 모델 합쳐 약 11.5시간)
bash scripts/run_overnight.sh

# 보고서
python3 scripts/report_gen.py
```
