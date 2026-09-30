# PIM emulator stack — version1.0 (ch2)

FPGA 위의 PIM 에뮬레이터(HBM 2채널 × 16뱅크)에 비트스트림을 올리고, 그 위에서 도는
소프트웨어를 빌드·실행하는 저장소다. 이 브랜치(`main`)는 **version1.0 이미지를 ch2 로**
쓰는 상태를 담는다. 이미지는 [`hw/ch2/version1.0/`](hw/ch2/version1.0/) 에 있다.

소프트웨어는 두 갈래다.

| | `hwdef/` + `runtime/` | `sw/` |
|---|---|---|
| 하는 일 | 보드 검증 프로그램(`emu_*`)과 `libpim.so` | 커널 드라이버 `pim.ko` + `libpim`/`libpimrt` + Python 앱(Llama 실행) |
| 채널 수가 들어가는 곳 | 빌드 시 `-D` (`platform/ch2.conf` → `scripts/setup.sh`) | `insmod` 인자 (`make load CH=2`) |
| 자세한 문서 | [`hwdef/README.md`](hwdef/README.md), [`hwdef/test/README.md`](hwdef/test/README.md) | [`sw/README.md`](sw/README.md), [`sw/app/README.md`](sw/app/README.md) |

---

## 0. 의존성

| 항목 | 기본 경로 / 값 | 바꾸는 법 |
|---|---|---|
| QDMA 드라이버 빌드 트리 | `/home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel` (이 저장소 밖) | `QDMA_TREE=...` |
| 카드 PCIe 주소 | `0000:01:00.0` (`lspci -nn \| grep -i xilinx` 로 확인) | `BDF=...` |
| Python (앱만) | conda 환경 `pim` — [`sw/app/environment.yml`](sw/app/environment.yml) | `make PYTHON=...` |

기본값은 [`platform/common.conf`](platform/common.conf) 에 모여 있고, 표의 환경 변수로
한 번씩 덮을 수 있다.

QDMA 드라이버 초기에 별도 설치할 것.



## 1. 한 번만: 카드 접근 권한

일반 사용자가 카드의 BAR 와 QDMA 큐를 쓸 수 있도록 udev 규칙을 설치한다. 사용자는
`plugdev` 그룹에 들어 있어야 한다.

```sh
sudo scripts/setup_permissions.sh
scripts/setup_permissions.sh --check      # 확인만
```

## 2. HW: ch2 이미지 프로그래밍
Main PC Cold Boot한 상태라면 Vivado HW Manager 열고 PDI 적재 후, 재부팅할 것.

```sh
sudo scripts/reprogram.sh --ch 2
```

- `platform/ch2.conf` 의 `HW_DIR`(= `hw/ch2/version1.0`) 에 있는 `.pdi` 를 JTAG 로
  올리고, 같은 폴더의 `.ltx` 도 함께 넘긴다.
- PCIe 장치를 떼었다가 다시 붙이고, BAR2 가 잡혔는지 확인한 뒤 권한을 다시 건다.
- 마지막에 `platform/active` 를 `ch2.conf` 로 가리킨다. 이후 빌드는 이 링크를 읽는다.
- Vivado 로그는 `scripts/logs/reprogram.vivado.log` 에 남는다.

보드를 건드리지 않고 ch2 선택만 기록하려면 (빌드만 해 볼 때):

```sh
scripts/reprogram.sh --ch 2 --select-only
```

프로그래밍은 PCIe 재열거를 일으키므로 QDMA 큐가 사라진다. 다음 단계에서 다시 만든다.

## 3. QDMA 큐

> **먼저: Xilinx QDMA 드라이버를 받아 빌드하고, 그 경로로 스크립트를 고쳐야 한다.**
> `qdma_queues.sh` 는 드라이버를 이 저장소에 담지 않고, Xilinx 의
> [dma_ip_drivers](https://github.com/Xilinx/dma_ip_drivers) 빌드 트리에서
> `bin/qdma-pf.ko` 와 `bin/dma-ctl` 을 그대로 가져다 쓴다. 기본 경로
> (`/home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel`)는 원래 작업 머신의 것이므로,
> 다른 머신에서는
>
> ```sh
> git clone https://github.com/Xilinx/dma_ip_drivers.git
> make -C dma_ip_drivers/QDMA/linux-kernel      # bin/ 에 qdma-pf.ko, dma-ctl 이 생긴다
> ```
>
> 로 빌드한 뒤 [`scripts/qdma_queues.sh`](scripts/qdma_queues.sh) 의 `QDMA_TREE=` 기본값을
> 그 `QDMA/linux-kernel` 경로로 고친다.

```sh
sudo scripts/qdma_queues.sh setup     # 드라이버 insmod + MM 큐 쌍 생성
scripts/qdma_queues.sh status         # 확인
```

`/dev/qdma01000-MM-0`(H2C), `/dev/qdma01000-MM-1`(C2H) 이 생긴다. 보드를 다시
프로그래밍할 때마다 다시 실행한다.

## 4. SW 빌드 (`hwdef/` + `runtime/`)

```sh
scripts/setup.sh
scripts/setup.sh --status             # 무엇이 선택됐고 무엇으로 빌드됐는지
```

- `platform/active`(ch2) 를 읽어 `hwdef/gen_config.sh --defs` 로 `-D` 목록을 만들고,
  `hwdef`, `hwdef/test`, `runtime`, `runtime/test` 를 차례로 빌드한다.
- 상수가 바뀌었으면 먼저 `clean` 한다.
- 빌드가 끝나면 `hwdef/test/emu_sanity` 를 실행해 보드의 채널 주소 방식 레지스터를
  ch2.conf 의 `ADDR_MAP=2`(RoChBaCo, 32 KiB 마다 채널이 바뀜)로 맞춘다.
  

2~4단계를 한 번에: `sudo scripts/reprogram.sh --ch 2 && sudo scripts/setup.sh --all`
(`--all` 은 빌드에 더해 큐 생성과 권한 적용까지 한다).

## 5. 구동 확인 (`hwdef/test`, `runtime/test`)

각 프로그램의 첫 두 줄에 무엇으로 빌드됐는지가 찍힌다.

```
platform ch2 (compiled in)  —  2 ch x 16 bank, window 256 MiB, stride 1024 MiB
```

작은 것부터 순서대로:

```sh
cd hwdef/test
./emu_sanity                  # 보드가 응답하나 (CFR 쓰고 읽기)
./emu_gpr_loop                # GPR 4 MiB 를 QDMA 로 쓰고 되읽기
./emu_hbm_direct --ch 0,1     # 두 채널 16뱅크 창에 각각 닿나
./emu_gemv --chs 0,1          # 두 채널 GEMV 타일, 16 lane 검증
./emu_ewmul --ch 0            # EWMUL 
./emu_chain --test all        # 한 ISR 프로그램 안의 조합
cd ../..

make -C runtime/test run      # pim_test, load_test (libpim.so 수용 시험)
```

모두 종료 코드로 판정한다: 0 = pass, 1 = fail, 2 = 사용법 오류. 옵션은
[`hwdef/test/README.md`](hwdef/test/README.md) 에 있다.

## 6. LLM 스택 (`sw/`)

### 빌드

```sh
make -C sw                    # libpim + libpimrt (유저 공간)
make -C sw test               # 보드 없이 도는 단위 시험
make -C sw drv                # pim.ko
```

### 드라이버 로드 (ch2)

```sh
sudo make -C sw/drv load CH=2 MAP=1   # insmod + dmesg 라 root 가 필요하다
cat /proc/pim                 # DRAM·GPR 두 pool 상태
```

- `CH=2` 가 채널 수다. 이 스택은 채널 수를 빌드가 아니라 이 인자로 받는다.
- `MAP=1` 은 드라이버 쪽 표기로 RoChBaCo 다. 4단계에서 보드에 건 주소 방식과 같아야
  하고, 이 스택은 RoChBaCo 만 받는다. 다르면 `libpimrt` 가 열 때 거부하며 어느 쪽을
  맞추라고 알려 준다.

### Python 앱

version1.0 때 쓰던 conda 환경을 그대로 고정해 둔 파일로 만든다. Python 3.11.15,
torch 2.13.0+cpu, transformers 5.14.1, pybind11 3.1.0 을 포함해 모든 패키지의 버전이
당시와 같다 (linux-64).

```sh
conda env create -f sw/app/environment.yml     # 환경 이름: pim
conda activate pim
```



```sh
cd sw/app
make PYTHON=$(which python)                              # _pim.so 바인딩
make check                                               # import 되고 geometry 가 나오나
./test_ops.py                                            # torch 에서 바인딩 끝까지
./generate.py --dry-run --model llama-3.2-1b             # 배치 계산만 (보드·다운로드 없음)
./generate.py --model llama-3.2-1b --prompt "The capital of France is"
./check_vs_torch.py                                      # torch 와 logit 비교
```

모델은 Hugging Face 에서 받는다. Llama 3.2 는 접근 승인이 필요한 모델이라
`huggingface-cli login` 이 되어 있어야 한다.

### 성능 측정 (decode tok/s)

Instruct 모델 두 개로 512 토큰씩 생성해, prefill 을 뺀 decoding 단계의 속도를 쟀다.
보드 DRAM 타이밍을 `emu_timing` base ×3 으로 둔 에뮬레이터 수치다.

| 모델 | decode tok/s | 토큰당 시간 (중앙값) |
|---|---|---|
| Llama-3.2-1B-Instruct | 2.53 | 0.39 s |
| Llama-3.2-3B-Instruct | 0.97 | 1.02 s |

조건 (타이밍 값과 거는 순서, 프롬프트, 환경), prefill 과 decode 를 가르는 법, 다시 재는
명령, 그리고 이 숫자를 읽을 때 주의할 점은
[`docs/perf_decode_tps.md`](docs/perf_decode_tps.md) 에 있다.

---

## 순서 요약

```sh
# 한 번만
make -C /home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel
sudo scripts/setup_permissions.sh

# 보드를 올릴 때마다
sudo scripts/reprogram.sh --ch 2
sudo scripts/qdma_queues.sh setup
scripts/setup.sh
hwdef/test/emu_sanity && hwdef/test/emu_gemv --chs 0,1

# LLM
conda env create -f sw/app/environment.yml && conda activate pim     # 한 번만
make -C sw && make -C sw drv && sudo make -C sw/drv load CH=2 MAP=1
cd sw/app && make PYTHON=$(which python) && ./generate.py --model llama-3.2-1b
```

## 디렉토리

```
hw/ch2/version1.0/   ch2 비트스트림 (.pdi) 과 ILA probe 파일 (.ltx)
platform/            ch1/ch2/ch4.conf — 채널 수·주소·정책,  common.conf — 보드 값
scripts/             reprogram.sh, qdma_queues.sh, setup.sh, setup_permissions.sh
hwdef/               레지스터 정의·플랫폼 상수 (libhwdef.a) 와 보드 검증 프로그램 (test/)
runtime/             libpim.so — 할당·배치·스케줄·전송
sw/                  pim.ko + libpim + libpimrt + Python 앱
docs/                LLM 스택 제안서, decode 속도 측정 (perf_decode_tps.md)
```
