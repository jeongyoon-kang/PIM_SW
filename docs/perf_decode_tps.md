# Decode 속도 측정 — Llama-3.2 Instruct (ch2, DRAM 타이밍 ×3)

`sw/` 스택으로 Llama-3.2-1B-Instruct 와 3B-Instruct 를 보드에서 돌려, **prefill 을 뺀
decoding 단계에서 토큰 하나가 나오는 속도(tok/s)** 를 잰 기록이다. 코드는 이 저장소
`main` 그대로이고, 측정을 위해 바꾼 것은 보드의 DRAM 타이밍 레지스터 하나뿐이다.

## 1. 결과

| 모델 | 생성 토큰 | 첫 토큰 (prefill + KV 할당, 입력 토큰: 41, Weight load 제외) | **decode tok/s** | 토큰당 시간 (중앙값 / p95) | 토큰당 ISA (처음 → 끝) |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 512 | 10.00 s | **2.53** | 0.390 s / 0.44 s | 114,306 → 138,882 |
| Llama-3.2-3B-Instruct | 512 | 27.39 s | **0.97** | 1.020 s / 1.09 s | 266,175 → 298,431 |

decode tok/s 는 **decode 스텝 511개 ÷ 그 시간 합계** 다 (1B 202.22 s, 3B 525.20 s).
두 모델 모두 EOS 없이 512 토큰을 채웠다.

문맥이 길어질수록 조금씩 느려진다. q·K 와 s·V 가 지금까지 쌓인 토큰 수만큼 길어지기
때문이고, 토큰당 ISA 가 같은 방향으로 늘어난 것이 그 흔적이다.

| 구간 (출력 토큰 번호) | 1B tok/s | 3B tok/s |
|---|---|---|
| 2 – 64 | 2.64 | 0.98 |
| 65 – 256 | 2.55 | 0.98 |
| 257 – 512 | 2.49 | 0.97 |

실행 전체 (prefill 포함)의 `generate.py` 요약 줄:

| 모델 | 전체 | 도어벨 대기 | ops | launches | ISAs |
|---|---|---|---|---|---|
| 1B | 512 tokens in 212.32 s (2.41 tok/s) | 134.71 s (63 %) | 62,336 | 79,232 | 68,228,480 |
| 3B | 512 tokens in 552.52 s (0.93 tok/s) | 345.84 s (63 %) | 108,704 | 138,400 | 153,643,168 |

도어벨 대기는 카드가 프로그램을 실행하는 동안 호스트가 기다린 시간의 합이다. 나머지
약 37 % 는 호스트 쪽 (프로그램 생성, PCIe 전송, RMSNorm·RoPE·softmax 등 호스트에 남은
연산, Python) 이다.

## 2. 조건

| 항목 | 값 |
|---|---|
| 날짜 | 2026-09-29 |
| 코드 | `main` @ `cda1f04` |
| 비트스트림 | `hw/ch2/version1.0` (`sudo scripts/reprogram.sh --ch 2`) |
| 보드 | 2채널 × 16뱅크, DRAM 8 GiB, RoChBaCo |
| 드라이버 | `sudo make -C sw/drv load CH=2 MAP=1` |
| **DRAM 타이밍** | `emu_timing` base **×3**: `faw 48 · rrd 12 · rcd 45 · ccd 6 · rtp 12 · rp 51 · wr 84 · ras 102` |
| KV 캐시 | 고정 예약 `S_max = 8192` (`generate.py` 기본값). 1B 384 MiB, 3B 896 MiB |
| Python 환경 | [`sw/app/environment.yml`](../sw/app/environment.yml) (Python 3.11.15, torch 2.13.0+cpu, transformers 5.14.1) |
| 호스트 | AMD Ryzen 9 7900X (12코어 24스레드), RAM 61 GiB (`free` 기준) |
| 프롬프트 | `generate.py` 기본값 `"Walk me through SSD architecture."` + `--chat` (chat template 적용 후 41 토큰) |
| 디코딩 | greedy (`do_sample=False`), batch 1 |
| 반복 | 모델당 1회 |

### prefill 입력

사용자 문장은 `Walk me through SSD architecture.` 하나지만, `--chat` 이라 Llama-3.2
Instruct 의 chat template 이 씌워진 아래 텍스트 전체가 prefill 로 들어갔다. 1B 와 3B 는
tokenizer 와 template 이 같아 입력도 같다.

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 29 Sep 2026

<|eot_id|><|start_header_id|>user<|end_header_id|>

Walk me through SSD architecture.<|eot_id|><|start_header_id|>assistant<|end_header_id|>


```

system 부분은 지정하지 않았고 template 이 자동으로 넣었다. `Today Date` 는 template 이
**실행한 날짜**를 넣으므로, 다른 날 다시 재면 이 줄이 바뀐다 (토큰 수는 같다).

| 부분 | 토큰 수 | 내용 |
|---|---|---|
| 시작 + system 헤더 | 5 | `<\|begin_of_text\|>` `<\|start_header_id\|>` `system` `<\|end_header_id\|>` `\n\n` |
| system 본문 | 20 | `Cutting Knowledge Date: December 2023` / `Today Date: 29 Sep 2026` 과 줄바꿈 |
| system 끝 + user 헤더 | 5 | `<\|eot_id\|>` `<\|start_header_id\|>` `user` `<\|end_header_id\|>` `\n\n` |
| **사용자 문장** | **6** | `Walk` ` me` ` through` ` SSD` ` architecture` `.` |
| user 끝 + assistant 헤더 | 5 | `<\|eot_id\|>` `<\|start_header_id\|>` `assistant` `<\|end_header_id\|>` `\n\n` |
| **합계** | **41** | |

마지막 (512번째) 토큰을 만드는 스텝의 어텐션은 프롬프트 41 + 앞서 생성한 511 = 552
토큰 위에서 계산된다.

### DRAM 타이밍 ×3 에 대해

타이밍 레지스터 8개는 [`hwdef/test/emu_timing.c`](../hwdef/test/emu_timing.c) 의 base
(`16, 4, 15, 2, 4, 17, 28, 34` — HBM2 소자 값을 컨트롤러 사이클로 옮긴 것)에 3 을
곱한 값으로 두었다.

```sh
hwdef/test/emu_timing --scale 3 --no-run --keep    # 설정하고 남겨 둔다
hwdef/test/emu_timing --show                       # 보드에 걸린 값 확인 (쓰지 않음)
```

**평소 보드 값과 다르다.** `scripts/setup.sh` 와 `hwdef/test/emu_sanity` 는 타이밍을
`EMU_TIMING_SIM` (`30, 6, 4, 2, 3, 3, 4, 6`) 으로 쓴다. 그래서 이 측정을 다시 하려면
`setup.sh` 나 `emu_sanity` 를 **마지막으로** 돌린 뒤에 `emu_timing --scale 3 --keep` 을
걸어야 하고, 이 순서가 바뀌면 다른 타이밍에서 잰 것이 된다. 이번 측정은 두 모델을
돌린 뒤 `--show` 로 ×3 이 그대로인 것을 확인했다.

## 3. 측정 방법

### 실행

```sh
conda activate pim                  # sw/app/environment.yml 로 만든 환경
cd sw/app
./generate.py --model llama-3.2-1b-instruct --chat --max-new-tokens 512 > 1b.log
./generate.py --model llama-3.2-3b-instruct --chat --max-new-tokens 512 > 3b.log
```

### prefill 과 decode 를 가르는 법

`generate.py` 는 토큰이 나올 때마다 한 줄씩 찍는다
([`pimllm/model.py`](../sw/app/pimllm/model.py) 의 `TimedStreamer`).

```
                ''   [  1  10.00s  3723266 ISA]      ← prefill (첫 토큰이 여기서 나온다)
                ''   [  2   0.40s   114306 ISA]      ← decode 1 스텝
            'SSD '   [  3   0.37s   114306 ISA]
```

가운데 숫자는 **바로 앞 줄부터 이 줄까지 걸린 시간**, 오른쪽은 그 사이에 카드에 보낸
ISA 수다. 1번 줄만 기준점이 streamer 를 만든 시점이다. 그래서 **2번 줄부터가 decode 한
스텝씩** 이고, decode tok/s 는 그 줄 수를 시간 합으로 나눈 것이다.

1번 줄 (첫 토큰) 에 들어가는 것과 들어가지 않는 것:

- **들어가지 않는다 — weight 적재.** `PimModel` 을 만들 때 모든 weight 를 카드에 올리고
  (`113 linear layers, 2.30 GiB on the card`), 그다음 `generate()` 가 프롬프트를
  토큰화한 **뒤에** streamer 를 만든다 (`pimllm/model.py` 의 `generate`). weight 적재와
  토큰화는 시계가 돌기 전에 끝난다.
- **들어간다 — prefill forward 한 번.** transformers 의 `generate` 는 프롬프트 41 토큰을
  한 번의 forward 로 처리하고 그 출력으로 1번 토큰을 뽑는다. 따로 decode 스텝을 한 번 더
  돌지 않는다.
- **들어간다 — KV 캐시 할당.** K/V 는 첫 `update` 에서 레이어마다 `S_max` 크기로 한 번에
  잡히고, V 의 첫 1024 토큰 chunk 를 0 으로 채우는 쓰기도 이때 한다 (`pimllm/cache.py`).

그래서 첫 토큰 시간은 "prefill 계산" 만이 아니라 "prefill + KV 할당" 이다. 이 문서의
decode tok/s 는 2번 줄부터 계산하므로 어느 쪽도 들어가지 않는다.

```sh
grep -E 'ISA\]$' 1b.log \
  | sed -E 's/.*\[ *([0-9]+) +([0-9.]+)s +([0-9]+) ISA\]$/\1 \2 \3/' \
  | awk '$1>1 {n++; s+=$2} END {printf "%d decode steps, %.2f s, %.3f tok/s\n", n, s, n/s}'
# 511 decode steps, 202.22 s, 2.527 tok/s
```

`generate.py` 가 마지막에 찍는 `N tokens in T s (x tok/s)` 는 prefill 이 들어간 값이라
decode 속도가 아니다 (1B 2.41, 3B 0.93).

## 4. 이 숫자를 읽을 때

- **에뮬레이터의 숫자다.** FPGA 위 PIM 에뮬레이터를 ×3 타이밍으로 둔 결과이고, 실제
  HBM-PIM 소자의 속도를 뜻하지 않는다. 같은 코드도 타이밍 설정이 다르면 다른 값이
  나온다. 참고로 `EMU_TIMING_SIM` 에서 같은 1B base 모델로 1024 토큰을 뽑았을 때
  전체 3.90 tok/s 였지만, 프롬프트 · 토큰 수 · 모델 (base) 이 달라 이 표와 바로 비교할
  수 없다.
- **호스트가 약 37 % 다.** 도어벨 대기가 전체의 63 % 이므로, 호스트 코드나 머신이
  바뀌면 같은 타이밍에서도 값이 달라진다.
- **KV 캐시는 고정 크기다.** 이 스택은 시작할 때 `S_max` 만큼 K/V 를 한 번에 잡는다
  (`pimllm/cache.py`). 생성 중에 메모리를 새로 받지 않으므로 토큰당 시간에 할당 비용이
  섞이지 않는다. V 의 새 1024 토큰 chunk 에 처음 쓸 때 0 으로 채우는 쓰기가 들어가지만,
  512 토큰은 첫 chunk 안이라 이번 측정에는 나타나지 않는다.
- **모델당 한 번 돌렸다.** 토큰당 시간의 p95 가 중앙값의 1.13 배 이하라 흔들림은
  작지만, 반복 측정으로 얻은 분산은 아니다.
- **출력 텍스트는 판정하지 않았다.** 정확도는 `check_vs_torch.py` (torch 와 logit 비교)
  의 일이고, 이 문서는 속도만 다룬다.
