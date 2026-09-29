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

## 0. 준비물

| 항목 | 기본 경로 / 값 | 바꾸는 법 |
|---|---|---|
| Vivado 2025.2 | `/tools/Xilinx/2025.2/Vivado/settings64.sh` | `VIVADO_SETTINGS=...` |
| JTAG 프로그래밍 Tcl | `/home/kjy/pim/bank_controller/program.tcl` (이 저장소 밖) | `PROGRAM_TCL=...` 또는 `--tcl FILE` |
| QDMA 드라이버 빌드 트리 | `/home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel` (이 저장소 밖) | `QDMA_TREE=...` |
| 카드 PCIe 주소 | `0000:01:00.0` (`lspci -nn \| grep -i xilinx` 로 확인) | `BDF=...` |
| 카드 접근 그룹 | `plugdev` | `GROUP=...` |
| 커널 헤더 | `/lib/modules/$(uname -r)/build` | `sw/drv` 에서 `KDIR=...` |
| Python (앱만) | `torch`, `transformers`, `pybind11` | `make PYTHON=...` |

기본값은 [`platform/common.conf`](platform/common.conf) 에 모여 있고, 표의 환경 변수로
한 번씩 덮을 수 있다.

QDMA 드라이버는 설치하지 않고 빌드 트리에서 바로 쓴다. 처음 한 번 빌드해 둔다.

```sh
make -C /home/kjy/pim/dma_ip_drivers/QDMA/linux-kernel
```

## 1. 한 번만: 카드 접근 권한

일반 사용자가 카드의 BAR 와 QDMA 큐를 쓸 수 있도록 udev 규칙을 설치한다. 사용자는
`plugdev` 그룹에 들어 있어야 한다.

```sh
sudo scripts/setup_permissions.sh
scripts/setup_permissions.sh --check      # 확인만
```

## 2. HW: ch2 이미지 프로그래밍

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
  카드가 아직 없으면 빌드는 성공으로 두고 그 사실만 알려 준다.

`make` 를 각 디렉토리에서 직접 돌리지 않는다. 그렇게 만든 바이너리는
[`hwdef/pim_config.h`](hwdef/pim_config.h) 의 기본값으로 컴파일되고, 실행할 때
`pim_platform_check()` 가 거부한다.

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
./emu_ewmul --ch 0            # EWMUL 한 조
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
make -C sw/drv load CH=2 MAP=1
cat /proc/pim                 # DRAM·GPR 두 pool 상태
```

- `CH=2` 가 채널 수다. 이 스택은 채널 수를 빌드가 아니라 이 인자로 받는다.
- `MAP=1` 은 드라이버 쪽 표기로 RoChBaCo 다. 4단계에서 보드에 건 주소 방식과 같아야
  하고, 이 스택은 RoChBaCo 만 받는다. 다르면 `libpimrt` 가 열 때 거부하며 어느 쪽을
  맞추라고 알려 준다.
- 내릴 때: `make -C sw/drv unload`

### Python 앱

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
make -C sw && make -C sw drv && make -C sw/drv load CH=2 MAP=1
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
docs/                LLM 스택 제안서
```
