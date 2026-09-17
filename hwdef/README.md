# emulator_top — 주소 맵

ISR 인코딩의 정본은 [`../HW/r1p0/docs/pim_isr_opcode_사용가이드.md`](../HW/r1p0/docs/pim_isr_opcode_사용가이드.md)
이고, 이 파일은 호스트가 실제로 쓰는 주소를 한 곳에 모아둔 것이다.

각 항목의 출처를 표시했다 — `[문서]` ISR 가이드, `[실측]` 보드에서 확인,
`[유도]` 문서의 규칙에서 계산했으나 전체를 확인하지는 못함.

---

## 1. 호스트가 닿는 창

| AXI 주소 | 크기 | 무엇 | 접근 |
|---|---|---|---|
| `0x0040_0000_0000` | 32 GiB | HBM 직접 (경로 A) | QDMA 만 |
| `0x0204_0000_0000` | 4 GiB × 채널 | MC `s_axi` (경로 B) | QDMA 만 |
| `0x0202_0000_0000` | 8 MB | 제어 평면 (BAR2) | MMIO 또는 QDMA |

**주소 값의 정본은 [`../platform/*.conf`](../platform/) 이다.** 여기에는 공식만 싣는다 —
채널 수마다 다른 16/32/64줄짜리 표를 두 곳에 두면 반드시 갈린다.

### 1.1 제어 평면 — BAR2 안 (채널 수와 무관)

BAR → AXI 는 단순 베이스 덧셈이고 베이스가 8 MB 정렬이라 **하위 6자리가 그대로
BAR2 오프셋**이다.

```
AXI = 0x0202_0000_0000 + (BAR2 오프셋)
```

| BAR2 오프셋 | AXI | 크기 | 무엇 | 블록 | 전송 |
|---|---|---|---|---|---|
| `+0x00_0000` | `0x0202_0000_0000` | 4 MiB | GPR | `gpr_wrap_0/s_axi` | QDMA |
| `+0x40_0000` | `0x0202_0040_0000` | 4 KB | CFR | `dispatcher_top_0/s_lite` | MMIO |
| `+0x40_1000 + ch·0x1000` | `0x0202_0040_1000`.. | 4 KB | 위반 CSR, **채널당 하나** | `viol_csr_chN/s_axi` | MMIO |
| `+0x60_0000` | `0x0202_0060_0000` | 512 KB | IMEM (쓰기만) | `dispatcher_top_0/s_axi` | QDMA |

**GPR·CFR·IMEM 은 채널 수가 늘어도 하나다.** dispatcher 도 하나이므로 **명령 스트림과
도어벨도 하나**다 — 채널 선택은 별도 doorbell 이 아니라 ISR 안의 `CH_MASK` 로 한다.
채널 수에 따라 늘어나는 것은 위반 CSR 뿐이다.

`[실측 2026-08-10]` 네 창 전부 보드에서 답했다 — AXI-Lite 둘은 `emu_sanity` 의
write/readback, GPR 은 `emu_gpr_loop`, IMEM 은 GEMV 실행이 증거다. 사이
구간(CH=1 기준 `+0x40_2000`..`0x5F_FFFF`, `+0x68_0000`..`0x7F_FFFF`)은 미디코드로
`0xffffffff` 로 읽힌다 — `emu_sanity` 의 all-ones 검사가 여기에 걸린다.

```
GPR   0x0202_0000_0000 + (word << 5)     word 17 b, 131072 word
IMEM  0x0202_0060_0000 + (word << 5)     word 14 b,  16384 word
한 word = 256 b = 32 B
```

---

## 2. DRAM — 두 경로, 같은 메모리

| | 주소 | 창 | 간격 | 거치는 것 |
|---|---|---|---|---|
| **경로 A** 직접 | `HBM_BASE + ch·CH_SPAN + bank·BANK_STRIDE` | **256 MiB** | `BANK_STRIDE` | NoC → HBM |
| **경로 B** MC | `0x0204_0000_0000 + ch·4 GiB + A` (`A` = 채널 오프셋) | — | — | **MC 경유** (세그먼트 분할 + 타이밍 모델). 컨트롤러가 `A` 를 `ROW\|BA\|CO` 로 쪼갠다 |

`offset < 256 MiB` 범위에서 두 주소가 같은 바이트를 가리킨다.

**창과 간격은 다른 수다.** 경로 A 는 bank 마다 **256 MiB 만 디코드**하는데 bank 사이
간격은 그보다 넓다 — 그 사이는 **아무 데도 디코드되지 않는 gap** 이다. 경로 B 는
빽빽하게 붙어 있어 창 = 간격이다.

```
경로 A : [bank 256M][── gap ──][bank 256M][── gap ──] ...
경로 B : [bank 256M][bank 256M][bank 256M][bank 256M] ...
```

이 둘을 상수 하나로 뭉뚱그린 것이 예전 `EMU_HBM_BANK_SPAN = 1 GiB` 였고, 그래서
`emu_hbm_direct` 가 256 MiB 창에 1 GiB 를 찔렀고 `pim_dma_check()` 가 네 배 큰 전송을
통과시켰다. 지금은 `EMU_BANK_WINDOW` 와 `EMU_BANK_STRIDE` 로 나뉘어 있다.

### 2.1 이미지별 파라미터

| | `NCH` | `CH_SPAN` | `BANK_STRIDE` | `BANK_WINDOW` |
|---|---|---|---|---|
| CH=1 | 1 | 16 GiB | 1 GiB | 256 MiB |
| CH=2 | 2 | 16 GiB | 1 GiB | 256 MiB |
| **CH=4** | 4 | **8 GiB** | **512 MiB** | 256 MiB |

**CH=4 에서 간격이 절반이 된다.** 64개 bank 가 같은 32 GiB 안에 들어가기 때문이다.
창은 안 변한다. 이것이 창과 간격을 절대 같다고 가정하면 안 되는 이유다.

### 2.2 bank 안의 오프셋

```
offset = row × 2048 + col × 32          한 row = 2048 B = 64 beat
```

`256 MiB = 2^28` 은 **bank 실제 용량**이고, MC 의 packed 주소 맵이 쓰는 `BANK_OFF_W`
이고, ISR 의 `ROW` 필드(17 b × 2048 B = 2^28)가 닿는 범위다 — **셋이 정확히 일치한다.**

### 2.3 런타임의 주소 모델은 RoBaCo 다

경로 B 는 채널 내부에서 **RoBaCo** 를 디코드한다 — 런타임과 같은 모델이다.

```
 31                    15 14      11 10       5 4      0
| ROW (17b)              | BA (4b)  | CO (6b)  | byte   |
```

bank 가 최상위가 아니라 **중간 필드**이므로 bank 별 베이스 주소라는 것이 없다.
`pim_mc_bank()` 도 `pim_mc_addr(ch, bank, off)` 도 없고 `pim_mc_ch_base(ch) + A` 뿐이다
— bank 를 손에 들고 있는 호출자는 직접 창의 좌표를 들고 있는 것이고, 그 사고방식이
이쪽으로 넘어오면 안 된다.

그 결과 한 bank 안의 연속 구간이 이 창에서는 연속이 아니다(32768 B 마다 2048 B).
반대로 **연속 32 KiB 전송 하나가 16개 bank 의 page 를 전부 덮는다** — all-bank MAC 이
읽는 모양 그대로다:

```
robaco(row, bank, col) = row·32768 + bank·2048 + col·32
RoBaCo row = 32 KiB = 16 bank × 1 page = all-bank MAC 한 발의 피연산자 전체
column(전 bank) = 512 B = 할당 단위
```

RoBaCo 의 근거와 그것이 배치에 갖는 의미는 `../runtime/pim_layout.h` 헤더에 있다.

---

## 3. 슬레이브 제약 — 호스트 코드가 지켜야 한다

IMEM 과 GPR 두 AXI4-Full 슬레이브:

- **INCR 전용.** `AWBURST`/`ARBURST` 를 받기는 하지만 해석하지 않는다. WRAP 을 보내면
  INCR 로 취급되어 다른 곳에 쓴다.
- ~~**`WSTRB` 포트가 없다.**~~ **`[실측 2026-08-19]` 지금은 있다.** GPR / 직접 어퍼처 /
  MC 세 창 모두에서 1 B 쓰기가 정확히 1 B 만 바꾼다 — DMA 도 MMIO 도 그렇다. 32 B 워드
  경계를 걸치는 쓰기도 흘러넘치지 않는다. 이전 이미지의 관측(`emu_regs.h` 참고)은
  더 이상 이 보드에 해당하지 않는다.
- ~~**32 B 정렬.**~~ 필요 없다. `pim_dma_check()` 도 더 이상 정렬을 보지 않는다 —
  버스가 제대로 수행하는 것을 거부하면 호출자가 이 함수를 우회하게 될 뿐이다.
  런타임 자신은 여전히 32 B 배수로만 움직이지만(ISR 워드·GPR 워드·DRAM beat 이 전부
  32 B), 그건 런타임의 성질이지 가드가 강제할 일이 아니다.

가드가 계속 보는 것은 **소프트웨어가 복구할 수 없는 것들**이다 — 아무 데도 디코드되지
않는 주소, bank 창 위의 미디코드 gap(여기에 DMA 를 쏘면 H2C 엔진이 latch 된다), 창을
넘어가는 전송, 그리고 AXI-Lite 블록(GPR 이 끝나는 한 beat 뒤가 CFR 이고 그 오프셋 0 이
도어벨이다).

`[문서 §1.2]`

CFR 은 `addr[7:0]` 만 디코드한다 — 4 KB 안에서 **256 B 마다 반복**된다. 하위 바이트가
`0x00` 인 오프셋은 어디든 `CTRL` 이고 그 비트 0 이 도어벨이다.

---

## 4. 현재 파일

| 파일 | 층 | 무엇 |
|---|---|---|
| `emu_regs.h` | 0 | **이미지마다 같은 것만** — BAR2 제어 평면, 256 b geometry, ISR 인코딩 |
| `pim_platform.{h,c}` | 0 | **이미지마다 다른 것 전부** — 채널 수, bank stride/window, MC base, 정책. 전부 **컴파일 상수**다 (아래 참조). `runtime/libpim.so` 도 이 파일을 링크한다 |
| `pim_config.h` | 0 | 그 상수들의 **정적 기본값(ch4)**. `#ifndef` 가드가 걸려 있어 `-D` 로 덮인다 |
| `gen_config.sh` | 0 | 활성 `.conf` → `--defs` 로 `-D` 목록 출력. conf 키↔매크로 매핑이 사는 유일한 곳 |
| `../platform/select.sh` | 0 | 어느 `.conf` 가 활성인지 결정. `reprogram.sh` 와 `setup_permissions.sh` 가 source 한다 (`setup.sh` 는 conf 를 직접 읽는다) |
| `../platform/common.conf` | 0 | 보드 값 (BDF, 드라이버, 큐, 권한) |
| `../platform/ch{1,2,4}.conf` | 0 | **채널 수·주소·정책.** 채널 의존성이 사는 유일한 곳 |
| `setup.sh` | 0 | `--platform chN` 으로 활성 `.conf` 선택 **+ `-D` 를 만들어 4번 make**. 상수가 바뀌면 `clean` 을 선행한다 (+ 큐/권한) |
| `reprogram.sh` | 0 | PDI 프로그래밍 + PCIe 재열거 + sanity. `HW_DIR` 은 활성 `.conf` 가 준다 |
| `qdma_queues.sh` | 0 | MM 큐 생성/삭제/상태 |
| `setup_permissions.sh` | 0 | udev 권한 (한 번만) |
| `emu_sanity.c` | 1 | CFR write/readback + 위반 CSR 읽기 |
| `emu_gpr_loop.c` | 2 | GPR 4 MiB QDMA 루프백 |
| `emu_hbm_direct.c` | 2 | HBM 주소 범위 QDMA 도달 확인 |
| `emu_mc.c` | 2 | MC normal path — **양방향**. 직접 창을 기준면으로 쓴다 |
| `emu_gemv.c` | **3** | **GEMV 타일 하나, 16 lane 검사.** GB 채우는 두 경로 |
| `emu_ewmul.c` | **3** | **EWMUL 한 조.** 결과가 제3 뱅크 DRAM 에 착지 |
| `emu_chain.c` | **3** | **한 프로그램 안에서 무엇이 합성되는가** — 누산 연쇄 / GB rewind / K 분할 / GEMV 전체 |
| `pim_layout.{h,c}` | 2 | **실제 행렬을 bank 별로 쪼개는 계층.** RoBaCo 배치 + 채널·bank 당 한 번의 연속 전송. 파일을 열지 않는다 (전송 함수를 받는다) |
| `../runtime/pim.h` 외 | 2~4 | `libpim.so` — 할당·배치·스케줄·전송. 이 디렉터리에만 의존한다 |
| `../runtime/test/pim_test.c` | 4 | `pim.h` 만으로 도는 수용 시험 |
| `../runtime/test/load_test.c` | **3** | **실제 행렬을 올리고 그걸로 GEMV.** 되읽기(배치) + GEMV(ISA가 보는 위치) 두 검사 |

```sh
../../scripts/setup.sh --platform ch2      # 선택 + 전체 재빌드.  이것만 쓴다
```

**채널 수·stride·window 는 컴파일 상수다.** `setup.sh` 가 활성 conf 를
`gen_config.sh --defs` 로 `-D` 목록으로 만들어 네 번의 make 에 넘긴다. `pim_config.h`
에 ch4 기본값이 `#ifndef` 로 들어 있어 맨손 `make` 도 컴파일은 되지만, 그런 빌드는
`PIM_CONFIG_FROM_CONF` 가 0 이라 **모든 도구가 실행을 거부한다** — 아무도 고르지 않은
채널 수로 도는 것이 가장 나쁜 실패이기 때문이다.

`-D` 는 make 가 못 본다. 그래서 `setup.sh` 는 상수가 바뀌었으면(또는 이 스크립트를
거치지 않고 빌드된 흔적이 있으면) **`clean` 을 먼저** 하고, 성공한 뒤에만
`platform/.built` 에 무엇으로 빌드했는지 남긴다. `--status` 가 그걸 읽어 활성 conf 와
대조한다 — `cat pim_config.h` 는 이제 항상 ch4 라고 답하므로 그 자리를 대신할 것이
필요하다.

도구마다 첫 두 줄에 무엇으로 빌드됐는지 찍는다:

```
platform ch2 (compiled in)  —  2 ch x 16 bank, window 256 MiB, stride 1024 MiB
         upload=direct schedule=group ntail=pad T_latch=off
```

| 옵션 | 어느 도구 |
|---|---|
| `--ch LIST` | `emu_hbm_direct` (기본 전 채널) |
| `--ch N` | `emu_mc`, `emu_gemv`, `emu_ewmul`, `emu_chain` (한 번에 한 채널) |
| `--chs LIST` | `emu_gemv` — **채널을 묶어 한 프로그램으로.** multicast `CH_MASK` |
| 없음 | `emu_sanity` (제어 평면은 채널 불변, 위반 CSR 만 `NCH` 개 순회), `emu_gpr_loop` (GPR 은 채널 공용) |

`--platform NAME` 은 **없다.** 여러 도구의 `--help` 에 아직 남아 있지만 getopt 테이블에
없는 잔재다 — 런타임 conf 파싱이 폐지될 때 도움말이 따라오지 않았다. 플랫폼은
`setup.sh --platform` 으로 고르고 다시 빌드하는 것이지 실행 시 고르는 것이 아니다.

---

## 5. 도구별 — 어느 주소를 건드리는가

**규칙: 도구를 추가하면 이 절에 접근 주소 표를 반드시 같이 넣는다.** 무엇을
증명한다는 산문보다, 어디를 읽고 쓰는지가 검토 가능한 사실이다.

### 5.1 `emu_sanity` — 1층, AXI-Lite 두 슬레이브

DMA 를 쓰지 않는다. BAR2 mmap 만. 쓰기 16회, 읽기 25회.

| 대상 | BAR2 오프셋 | AXI | 동작 |
|---|---|---|---|
| CFR `T_FAW`..`T_RAS` | `+0x400008`..`+0x400024` | `0x0202_0040_0008`.. | 쓰기 → 되읽기 (1차: `0x11`..`0x88`) |
| 〃 | 〃 | 〃 | 쓰기 → 되읽기 (2차: `30 6 4 2 3 3 4 6`) |
| CFR `STATUS` | `+0x400004` | `0x0202_0040_0004` | 읽기 |
| CFR `PROG_LEN` | `+0x400028` | `0x0202_0040_0028` | 읽기 |
| 위반 `ANY` | `+0x401404` | `0x0202_0040_1404` | 읽기 |
| 위반 `CTRL` | `+0x401400` | `0x0202_0040_1400` | 읽기 |
| 위반 bank0 `STICKY/CNT_A/MAX_A` | `+0x401000/04/0C` | 〃 | 읽기 |
| 위반 bank15 `STICKY` | `+0x4013C0` | 〃 | 읽기 |
| 위반 bank0 미매핑 | `+0x401014` | 〃 | 읽기 (0 이어야 함) |

`+0x400000`(CTRL)에는 쓰지 않는다 — 비트 0 이 도어벨이다. CFR 이 `addr[7:0]` 만
디코드하므로 하위 바이트가 `0x00` 인 오프셋은 어디든 CTRL 이다.

1차가 검사다. 8개 값이 전부 달라야 두 레지스터가 같은 주소로 디코드되는 것을
잡는다. 2차의 `30 6 4 2 3 3 4 6` 은 `6`·`4`·`3` 이 각각 두 번씩이라 그것만으로는
못 잡는다.

### 5.2 `emu_gpr_loop` — 2층, GPR

| 단계 | 경로 | 주소 | 크기 |
|---|---|---|---|
| 1 | QDMA H2C | `0x0202_0000_0000` | 4 MiB |
| 2 | QDMA C2H | `0x0202_0000_0000` | 4 MiB |

GPR 은 256 b 슬레이브라 DMA 전용이다(§5.7).

패턴: 32 B 워드마다 자기 word index. 불일치 시 "word 2851 이 word 2850 의 데이터를
갖고 있다(off by −1)" 로 나온다.

### 5.3 `emu_hbm_direct` — 2층, HBM 도달 범위

`NCH×16` 개 베이스 × 3 오프셋. **전부 쓴 다음에 전부 읽는다** — 베이스마다 쓰고 바로
읽으면 16개가 같은 메모리를 가리켜도 통과한다.

| 오프셋 | 왜 |
|---|---|
| `+0` | ISR `ROW` 가 세는 시작점 |
| `+128 MiB` | 가운데. 양 끝만 보면 가장자리만 디코드하는 창에서도 통과한다 |
| `+256 MiB − blk` | **디코드되는 마지막 바이트.** 창 = bank 실제 용량 = ISR `ROW` 도달 범위 |

**gap 은 전송하지 않는다.** bank 창 위쪽은 아무 데도 디코드되지 않고, 거기로 DMA 를
보내면 EIO 와 함께 **H2C 엔진이 latch** 된다 `[관측]`. 대신 `pim_dma_check()` 가 그
주소를 **거부하는지**를 소프트웨어로 확인한다 (phase 0) — 모든 도구가 실제로 기대는
것이 그 가드이므로, 시험할 가치가 있는 것도 그것이다.

예전 버전은 `+1 GiB − blk` 를 찔러서 16 뱅크 전부 통과시켰다. 창이 256 MiB 라면
통과했으면 안 된다. **그 전송을 다시 돌리지 않고** 거부 시험으로 대체한 이유가 위와
같다.

기본 블록 64 KiB.

### 5.4 `emu_mc` — 2층, MC normal path (양방향)

**MC 가 ROW/BA/CO 를 호스트와 같게 디코드하는가**, 그리고 **얼마나 빠른가**. 이 둘이
목적이다. 직접 어퍼처는 기준면 — MC 를 거치지 않고 bank 에 닿으므로, 검증 대상을
믿지 않고도 알려진 바이트를 알려진 자리에 놓을 수 있다.

| 방향 | 0 poison | 1 전송 | 2 되읽기 |
|---|---|---|---|
| `--read` | 직접 | 직접 | **MC** ← 측정 |
| `--write` | 직접 | **MC** ← 측정 | 직접 |

**범위는 bank 의 것이 아니라 채널의 것이다.** MC 전송이 주소로 가리키는 게 그거다.
예전에는 `--bank N` 으로 bank 를 고르고 크기가 bank 당이었는데, 그건 MC 가 bank 마다
256 MiB 슬롯을 따로 두던 시절 얘기다. 지금은 "한 bank 만 덮는 전송" 이라는 게
표현되지 않으므로 그 옵션은 없다.

| 옵션 | 무엇 |
|---|---|
| `--ch N` | 채널 |
| `--at OFF` | 채널 RoBaCo 공간의 시작 오프셋. **MC 주소가 곧 이 값** |
| `--mib N` / `--kib N` / `--bytes N` | 크기. **1 바이트부터 아무 크기나** — row 도 beat 도 8 B 도 배수일 필요 없다 |
| `--ref` | (read) 직접 왕복을 먼저 해서 **기준면 자체를 먼저 증명** |
| `--block N` | MC 전송 크기(MiB, 기본 4). 호스트 버퍼만 제한한다 |
| `--one` | 블록 없이 전 범위를 **한 번의 전송**으로 |
| `--beats N` | 끝에서 N beat 을 32 B 씩 하나하나. **이웃 beat 오염(spill) 검사 포함** |
| `--settle-ms N` | 전송과 되읽기 사이 대기 |

주소:

```
DIRECT = pim_bank(ch, b) + off     bank 마다 창이 따로.  b 와 off 를 계산해야 함
MC     = mc_base(ch) + A           A 가 채널 오프셋 그대로.  계산 없음
```

**전송 모양이 두 면에서 다르다.** 페이로드 하나를 놓을 때:

| | DIRECT | MC |
|---|---|---|
| 전송 횟수 | **bank 당 1회** (범위가 닿는 bank 만) | **1회** (또는 블록) |
| bank 경계에서 | 창을 바꿔야 함 | 아무 일도 안 일어남 |
| 호스트가 아는 것 | bank 수, stride, RoBaCo 공식 | 시작 주소와 길이 |

범위가 row 배수가 아니어도 된다. **한 bank 의 몫은 어떤 범위에서도 그 bank 안에서
한 덩어리**다 — 부분 page 는 범위의 양 끝에만 생기고, 앞쪽 부분은 page 끝까지 닿아
다음 row 가 bank 공간에서 바로 이어받기 때문이다. 그래서 direct 도 bank 당 1회다.
시작 오프셋과 길이만 bank 마다 다르다.

```
at=0x7e0  len=64    -> bank0:(bkoff 2016, 32B)  bank1:(bkoff 0, 32B)
at=0xffe0 len=96    -> bank15:(4064, 32B)       bank0:(4096, 64B)
```

두 가지가 **양방향 모두에** 적용된다:

- **poison 을 먼저 깐다.** 안 그러면 이전 실행이 남긴 값이 마침 기대값과 같을 때
  아무것도 안 한 전송이 통과한다.
- **전부 전송한 다음 전부 읽는다.** 조각마다 `poison→전송→읽기` 를 붙여 돌면
  **순열을 못 잡는다** — 엉뚱한 자리에 쓴 전송을 그 엉뚱한 자리에서 읽는다.

패턴은 **채널 오프셋을 키로 하는 바이트 스트림**이다. 오프셋 `A` 의 바이트는 그 8 B
그룹 워드의 `A%8` 번째 바이트라, 어떤 범위든 잘 정의된 조각이 된다. 불일치 시
`FIRST DIVERGENCE` 가 기대/실제 바이트와, 8 B 그룹이 범위 안에 온전히 들어 있으면
**그 데이터가 원래 어느 오프셋 것인지** 를 역산해 bank·row·col 로 찍는다 — bank 가
틀렸는지 row 가 틀렸는지가 갈린다.

`[실측 2026-08-19]` ch1, 16 MiB: **MC 읽기 623 MB/s, 쓰기 882 MB/s**. 같은 범위를
직접 어퍼처로는 2976 / 3754 MB/s. 블록 크기를 1/4/16 MiB 로 바꿔도 MC 값은 변하지
않는다 — 병목이 syscall 이 아니라 MC 자신이라는 뜻이다.

### 5.5 `emu_gemv` — 3층, ISR 실행

| 단계 | 경로 | 주소 | 크기 |
|---|---|---|---|
| 1 타이밍 | MMIO | `0x0202_0040_0008`..`0024` | 8×4 B |
| 2 행렬 | **QDMA** | `0x0040_0000_0000 + b×1GiB + row×2048` | 16 × L×32 B |
| 3 벡터 (`--fill wrvec`) | **QDMA** | `0x0202_0000_0000 + (vec_word<<5)` GPR | L×32 B |
| 3 벡터 (`--fill copy`) | **QDMA** 직접 | `0x0040_0000_0000 + vec_row×2048` bank 0 | L×32 B |
| 4 프로그램 | **QDMA** | `0x0202_0060_0000` (word 0~3) | 4×32 B |
| 5 poison | **QDMA** | `0x0202_0000_0000 + (dst_word<<5)` | 32 B |
| 6 PROG_LEN | MMIO | `0x0202_0040_0028` ← 4 | 4 B |
| 7 도어벨 | MMIO | `0x0202_0040_0000` ← 1 | 4 B |
| 8 폴링 | MMIO | STATUS (AXI-Lite, 공짜) | 4 B |
| 9 결과 | **QDMA** | `0x0202_0000_0000 + (dst_word<<5)` | 32 B |
| 10 위반 | MMIO | `0x0202_0040_1404` + bank 블록 | |

**벡터는 언제나 Global Buffer 를 거쳐** PU 에 닿고, `--fill` 이 GB 를 채우는 길을
고른다.

**`--fill both` 은 프로그램 하나가 아니라 둘이다.** `[WRVEC | MAC | RD_MAC… | EOS]`
와 `[COPY | MAC | RD_MAC… | EOS]` 를 **도어벨 두 번**으로 따로 쏴서 lane 단위로
대조한다. GEMV 에 GB fill 은 하나면 되고, 이건 그 하나를 채우는 두 가지 방법이다 —
COPY 가 WRVEC 뒤에 추가로 들어가는 것이 아니다. word 0 이후가 동일하므로 lane 이
갈리면 원인이 fill 경로밖에 없고, 그래서 이 도구가 `ISR_COPY` read-half 를 검증한
수단이 되었다 (`../hw/ch1/r1p0/docs/report.md` §1.1).

**멀티채널에서 두 경로의 비용이 다르다** `[실측 2026-08-23, ch2]`:

| | 벡터가 있는 곳 | 채널 N 개일 때 호스트가 하는 일 |
|---|---|---|
| `WRVEC` | GPR | GPR 에 **한 번** 쓰고 `CH_MASK` 가 fan-out |
| `COPY` | 각 채널의 bank DRAM | **채널마다 한 번씩** — 총 N 번 |

`COPY` 도 `CH_MASK` 로 multicast 된다 — `--chs 0,1` 에서 `CH_MASK=0x3`, **32/32 lane
전부 WRVEC 과 일치**. 다만 **multicast COPY 는 각 채널이 자기 bank 를 읽는다.** ISR 에
ROW 필드가 하나뿐이라 채널들이 같은 자리에서 가져오고, 그래서 공유 벡터는 실행 전에
채널마다 **복제**돼 있어야 한다. 즉 공유 operand 에는 `WRVEC` 이 순수하게 일이 적다.

`COPY` 가 값을 하는 경우는 **벡터가 이미 DRAM 에 있을 때**뿐이고, 이 데이터패스에서
그건 `EWMUL` 결과다 — `RD_MAC` 은 GPR 에 쓰기 때문이다.

| `--fill` | word 0 | 벡터가 놓이는 곳 |
|---|---|---|
| `wrvec` | `ISR_WRVEC` `ROW`=GPR word | GPR |
| `copy` | `ISR_COPY` `ROW`=DRAM row, `route[0]=16`(GB) | bank 0 의 DRAM |
| `both` (기본) | 둘 다 돌리고 **lane 단위로 대조** | |

`both` 가 검사다 — MAC 이후가 완전히 같으므로 lane 이 갈리면 그건 fill 경로
차이뿐이다. GB 채운 개수와 MAC 이 꺼내는 개수는 같아야 한다 (`fill.OPSIZE ==
MAC.OPSIZE`).

폴링은 두 단계다. **STATUS 는 MMIO 로** 돈다 — AXI-Lite 라 한 번에 32 비트
읽기 하나고 커널 로그도 안 남는다. **결과 워드는 done 이 선 뒤에 C2H 로 한 번**
읽는다 — 드라이버가 전송마다 `printk` 하므로 폴링마다 DMA 를 걸면 로그가
넘친다. done 은 held level 이라 이전 실행 값이 남아 있을 수 있어 판정하지
않는다. 판정하는 것은 poison 이다.

### 5.6 `emu_ewmul` — 3층, EWMUL 한 조

MAC 과 달리 **누산하지 않고 결과를 제3 뱅크의 DRAM 에 쓴다.** RD_MAC 도 GPR 도
쓰지 않으므로 검사는 **직접 창 되읽기**다. 프로그램은 `EWMUL` + `EOS` 두 워드.

조는 route 가 정한다 — `feed s: route[s]=i`, `exec i: route[i]=d`, `dst d`.
`pu_mask = 1<<i` (한 번에 한 조), `gb_mc_mask = 0` (peer 소싱).

| 단계 | 경로 | 주소 | 크기 |
|---|---|---|---|
| 0 V | **QDMA** 직접 | `0x0040_..` + feed×1GiB + row×2048 | L×32 B |
| 0 W | **QDMA** 직접 | 〃 exec | L×32 B |
| 0 poison | **QDMA** 직접 | 〃 dst | L×32 B |
| 1 프로그램 | **QDMA** | `0x0202_0060_0000` (word 0~1) | 2×32 B |
| 2 PROG_LEN·도어벨 | MMIO | `0x0202_0040_0028` / `0x0202_0040_0000` | 4 B |
| 3 결과 | **QDMA** 직접 | 〃 dst | L×32 B |

세 뱅크가 **같은 `(ROW, COL)`** 을 쓴다 — ISR 에 `ROW` 필드가 하나뿐이라 구조적으로
강제된다. `--sweep` 은 (feed, exec, dst) 를 16 뱅크에 돌려가며 16개 조를 검사한다.

피연산자: `V[k]=2.0`, `W[k]=k+1` → 곱이 `2(k+1)` 로 전부 작은 정수라 BF16 에서
정확하다. 불일치는 오차 논쟁이 아니라 결함이다.

### 5.6b `load_test` — 실제 행렬을 올리고 그걸로 GEMV

**여기 있지 않다.** `../runtime/test/load_test.c` 다. 이 표의 다른 도구들과 달리
`libpim.so` 를 링크하지만, 쓰는 것은 `pim_matrix_*`(`runtime/pim_layout.c`) 와
`pim_platform_*` 뿐이고 BAR2 는 직접 mmap, ISR 은 `emu_isr_build` 로 손수 조립한다 —
그래서 주소를 건드리는 도구 목록에 남겨 둔다.

다른 probe 들은 전부 **자기가 지어낸 피연산자**를 쓴다 — bank 당 값 하나, 답이 작은
정수가 되도록 고른 것. 데이터패스는 증명하지만 **실제 텐서를 제자리에 넣는 것**은
아무것도 증명하지 않는다. 이 도구가 그 부분이고, 검증 축이 둘이다:

- `pim_matrix_verify` 의 **산술 없는 바이트 되읽기** — 바이트가 도착했나
- 단일 도어벨 **ISR GEMV** — ISA 가 찾는 자리에 도착했나

주소: `pim_bank(ch, b)` (bank 당 1회 전송), IMEM, GPR, CFR. 실행은
`../runtime/test/` 에서 `make run` 또는 `./load_test`.

### 5.7 `emu_chain` — 3층, 한 프로그램 안에서 무엇이 합성되는가

`emu_gemv` 는 MAC 을 한 발만 쏜다. 프로그램이 4 워드보다 길어지는 순간 걸리는 규칙
넷은 거기서 관측되지 않는다. 넷을 따로 물어서, 실패가 자기 원인을 말하게 한다.

```
./emu_chain                                    # 합성 규칙 넷 + 산술 둘
./emu_chain --test chain                       # 누산기가 ISR 을 건너 사나
./emu_chain --test rewind                      # 한 WRVEC 이 MAC 여럿을 먹이나
./emu_chain --test kchunk                      # K 를 쪼갤 수 있나
./emu_chain --test gemv  --l 64 --g 1024       # GEMV 전체가 한 doorbell 에
./emu_chain --test gemv  --g 128 --kchunks 4 --order chunk
./emu_chain --test acc                         # 누산기 폭 — BF16 보다 넓은가
./emu_chain --test order --l 64 --g 256 --dist cancel       # 누산 순서
./emu_chain --test split --l 64 --g 64 --kchunks 4          # 두 배치의 속도·정확도
```

| 옵션 | |
|---|---|
| `--l N` | MAC 당 beat, 1~64. `K = 16·N` |
| `--g N` | 출력 그룹 수. `N_out = 16·G` |
| `--kchunks C` | 출력당 K 분할. `K = C·16·L` |
| `--order group\|chunk` | 어느 루프가 바깥인가 (§아래) |
| `--dist ...` | `order` 프로브의 피연산자 — 아래 |

**`--test acc` — 누산기 폭.** 진행 중인 합의 half-ulp 아래인 항(`2^-9`)만 더한다.
BF16 누산기라면 전부 버려져 정확히 `1.0` 이 나오고, 더 넓으면 `1 + (K-1)/512` 가
나온다. K = 16/64/256/1024 네 지점과, `MAC` 을 여러 발 이어서 ISR 을 건널 때도 묻는다.

**`--test order` — 누산 순서.** 무작위 데이터로는 **판정이 안 된다**: 출력이 BF16
8 비트뿐이라 fp32 순서 차이가 묻히고, 현실 분포에서 모델끼리 갈리는 lane 이 4096 중
1 개다. 그래서 `--dist` 로 피연산자를 고른다.

| `--dist` | 무엇 |
|---|---|
| `normal` (기본) | `w~N(0,0.02)`, `x~N(0,1)` — 트랜스포머가 실제로 담는 값 |
| `uniform` / `wide` | 참고용. 판별력이 거의 없다 |
| `eps-only` | **대조군.** `ε` lane 만 남기고 나머지를 0 으로. 모든 모델이 같은 값을 예측하므로, 디바이스가 그 값을 안 내면 `ε` 가 flush 되는 것이고 아래 둘은 무의미해진다. **먼저 돌린다** |
| `cancel` | `ε` 를 lane 0,1 에 — adjacent tree `(2i,2i+1)` 를 겨냥 |
| `cancel-half` | `ε` 를 lane 0,8 에 — halves tree `(i,i+n/2)` 를 겨냥 |

`cancel` 계열은 무작위가 아니라 **지어낸 벡터**다. tree 가 먼저 짝짓는 lane 에 `ε`,
나머지에 `±A` 를 두어 다른 짝이 정확히 상쇄되게 하면, 후보 모델들이 1 ulp 차이가
아니라 **아예 다른 답**을 낸다. BF16 출력으로 순서를 물을 수 있는 유일한 방법이다.

`[실측 2026-08-10]` 대조군이 통과했고, `cancel` 과 `cancel-half` 가 둘 다 `0` 을 내서
**adder tree 배선 두 가지가 모두 배제되고 순차 fp32 가 전부 일치**했다. golden 모델은
따라서 lane 순서 그대로의 fp32 누산이다.

**`--test split` — 두 배치의 대가.** 같은 무작위 데이터를 group-outer 와 chunk-outer
로 각각 돌려 golden 과 대조한다. group-outer 는 **bit-exact** 이고, chunk-outer 는
출력의 9~27 % 가 **1 ulp** 어긋난다 (C = 2~8 어디서도 그 이상은 없다). 속도는
chunk-outer 가 1.4~1.6 배다.

**`--order` 가 이 도구의 요점이다.** 누산 latch 가 뱅크당 하나라 두 배치가 가능하다:

| | WRVEC 횟수 | RD_MAC 횟수 | 반올림 |
|---|---|---|---|
| `group` | `G·C` | `G` | 출력당 **한 번** (fp32 로 이어 누산) |
| `chunk` | **`C`** | `G·C` | 출력당 **`C` 번** (부분합을 호스트가 더함) |

64-beat WRVEC 이 약 1.37 µs 로 MAC 보다 비싸므로 `chunk` 가 **1.63 배** 빠르다
(`N=2048, K=4096`: 1141 → 698 µs). 대신 반올림이 늘어난다. 정확도 대가는 아직
측정되지 않았다 — 지금 시험 피연산자는 BF16 에서 정확해서 두 배치가 같은 답을 낸다.

결과 GPR 워드는 항상 **연속**이고 한 번의 `pread` 로 걷힌다. 워드마다 따로 읽으면
연산보다 전송이 비싸진다.

### 5.8 전송 정책 — AXI-Lite 만 MMIO, 나머지는 DMA

**창 하나에 마스터 하나.**

| 대상 | 전송 | 이유 |
|---|---|---|
| CFR, 위반 CSR (AXI-Lite) | **MMIO 만** | 32 비트 단일 TLP, 쪼개질 것이 없다. DMA 큐 없이도 판정할 수 있어야 한다 (`emu_sanity`) |
| IMEM, GPR, HBM, MC (256 b) | **QDMA 만**, 32 B 정렬 | 아래 |

DMA 에 정렬 제약은 없다 `[실측 2026-08-19]` — narrow transfer 가 동작해서 1 B 접근도
정확히 1 B 만 건드린다. `pim_dma_check()` 가 창 검사보다
먼저 이것을 거른다 — 주소 하위 5 비트는 디코드조차 되지 않아서 어긋난 주소는
거절되는 게 아니라 **조용히 내림**되고, 부분 beat 은 `WSTRB` 가 없어 이웃
바이트를 덮는다.

**쓰기가 MMIO 로 안 되는 이유 — 실측**

이 호스트는 **32 B MMIO 스토어를 16 B 이하 TLP 로 쪼갠다.** `memcpy` 든 단일 AVX
32 B 스토어든 같다. 그리고 GPR / IMEM 슬레이브에는 `WSTRB` 포트가 없어서 **조각
하나하나가 32 B 워드 전체 쓰기로 확장**된다. 두 번째 조각이 이기고, 그것이 싣지
않은 바이트는 0 이 되어 돌아온다.

```
GPR word 2000 에 32 B 쓰기
  A: DMA write           ok
  B: memcpy write        want b0..bf c0..cf   got 0000..0000 c0..cf
  C: one 32 B AVX store  want c0..cf d0..df   got 0000..0000 d0..df
```

**따라서 GPR·IMEM 쓰기는 반드시 DMA 로 한다.** HANDOFF §1.0a 가 "MMIO 또는 DMA"
라고 허용하지만, 이 호스트에서 MMIO 쓰기는 선택지가 아니다.

- **CFR 은 무사하다** — 32 비트 레지스터라 4 바이트 단일 TLP 다.
- **읽기도 MMIO 로 하지 않는다.** 쪼개진 읽기 자체는 각 절반이 맞지만, 창마다
  마스터가 하나여야 두 주소 뷰가 일치하는지 따질 일이 없다. 찢긴 읽기(32 B 가
  16 B 두 조각으로 와서 RD_MAC 을 중간에 잡는 것)도 그래서 없다 — C2H 는 beat 을
  통째로 준다.

이것이 첫 GEMV 실행을 통째로 무효화했다. 벡터·프로그램·poison 이 전부 같은
방식으로 깨졌고, IMEM 워드의 `[63:0]`(opcode·OPSIZE·ROW) 이 하위 16 B 에 있어서
**opcode 가 0 인 쓰레기가 실행됐다.**

### 5.9 측정된 대역폭

`[실측 2026-08-10]`

| 경로 | 전송량 | 쓰기 | 읽기 |
|---|---|---|---|
| GPR | 1 MiB | 1949 MB/s | 886 MB/s |
| GPR | 4 MiB | 1978 MB/s | 909 MB/s |
| HBM 직접 | 3 MiB (48 지점) | 3256 MB/s | 1970 MB/s |
| HBM 직접 | 16 MiB 연속 | 7145 MB/s | — |
| MC `s_axi` | 16 MiB | — | **651 MB/s** |

MC 경로가 여전히 직접 경로보다 느리다 — 세그먼트 분할과 타이밍 모델을 지나기
때문이다. 다만 **비율이 1/11 수준**이고, 작은 전송에서는 syscall 고정비가 섞여
있으므로 큰 전송으로 다시 재야 정확하다.

---
