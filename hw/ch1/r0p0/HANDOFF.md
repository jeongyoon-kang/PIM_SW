# emulator_top FPGA 인계 — 단일 채널 플랫폼

호스트 코드를 쓰는 데 필요한 것만 담았다. 이 문서와 아래 RTL 헤더가 정본이고,
문서와 헤더가 어긋나면 **헤더가 이긴다**.

| 정본 | 무엇 |
|---|---|
`bd/emu_top_fpga_bd.tcl` | 주소 맵 (`assign_bd_address`) |
`.../dispatcher_top/rtl/dispatcher_top.v` 헤더 | CFR(도어벨/상태/타이밍) 맵, IMEM 창 |
`.../emu_gpr_wrap/rtl/emu_gpr_wrap.v` 헤더 | GPR 호스트 슬레이브, 2-master 중재 |
`rtl/emu_viol_csr.v` 헤더 | 위반 CSR 레지스터 맵 |
`.../emulator_controller/rtl/pim_isr_defs.vh` | **ISR 필드 위치 + opcode** (RTL·TB 가 전부 여기서 읽는다) |

플랫폼 사실 두 가지: **설계 클럭 150 MHz**, **ILA 없음**(DEBUG=0 빌드라
Hardware Manager 에서 내부 신호를 볼 수 없다. 아래 모든 관측은 BAR2 읽기로 한다).

---

## 1. 주소 맵

### 1.0 BAR → AXI 변환

호스트가 보는 주소와 이 문서가 적는 주소는 **다른 공간**이다. 둘을 잇는 것은
CPM QDMA 의 BAR-to-AXI 변환 레지스터 하나뿐이고, **단순 베이스 덧셈**이다.

```
AXI 주소 = 0x0202_0000_0000 + (BAR2 안의 오프셋)
```

BD 설정 (정본):

| 설정 | 값 | 뜻 |
|---|---|---|
`CPM_PCIE0_MODES` | `DMA` | CPM0 를 QDMA 로 쓴다 |
`CPM_PCIE0_PF0_BAR2_QDMA_ENABLED` | `1` | BAR2 사용 |
`CPM_PCIE0_PF0_BAR2_QDMA_SIZE` / `SCALE` | `8` / `Megabytes` | **BAR2 = 8 MB** |
`CPM_PCIE0_PF0_BAR2_QDMA_64BIT` | `1` | 64-bit BAR (BAR2+BAR3 한 쌍으로 잡힌다) |
`CPM_PCIE0_PF0_PCIEBAR2AXIBAR_QDMA_2` | `0x0202_0000_0000` | **BAR2 오프셋 0 이 닿는 AXI 주소** |
`CPM_PCIE0_PF0_BAR4_QDMA_ENABLED` | `0` | BAR4 미사용 |

**변환표 — 호스트가 BAR2 의 이 오프셋을 건드리면 이 AXI 주소가 된다:**

| BAR2 오프셋 | → AXI 주소 | 크기 | 무엇 |
|---|---|---|---|
`+0x00_0000` | `0x0202_0000_0000` | 512 KB | **IMEM** |
`+0x18_0000` | `0x0202_0018_0000` | 4 KB | **CFR** |
`+0x18_1000` | `0x0202_0018_1000` | 4 KB | **위반 CSR** |
`+0x40_0000` | `0x0202_0040_0000` | 4 MiB | **GPR** |
`+0x7F_FFFF` | `0x0202_007F_FFFF` | — | BAR2 끝 (GPR 끝과 일치) |

**BAR2 안에서는 오프셋과 AXI 주소의 하위 자릿수가 같다.** 베이스가
`0x0202_0000_0000` 으로 8 MB 정렬이라 캐리가 생기지 않기 때문이다 — 그래서 이 문서가
AXI 주소로 적어둔 값의 **하위 6 자리를 그대로 BAR2 오프셋으로 읽으면 된다.**

```c
/* BAR2 를 mmap 한 뒤 */
volatile uint8_t *bar2 = mmap(... resource2 ...);   /* = AXI 0x0202_0000_0000 */

#define IMEM   (bar2 + 0x000000)     /* AXI 0x0202_0000_0000 */
#define CFR    (bar2 + 0x180000)     /* AXI 0x0202_0018_0000 */
#define VIOL   (bar2 + 0x181000)     /* AXI 0x0202_0018_1000 */
#define GPR    (bar2 + 0x400000)     /* AXI 0x0202_0040_0000 */

*(volatile uint32_t *)(CFR + 0x10) = 4;          /* T_RCD = 4 */
memcpy((void *)(IMEM + (word << 5)), isr, 32);   /* ISR 한 칸 */
```

### 1.0a BAR2 에 없는 것 — DMA 로만 닿는다

**HBM(오퍼랜드)과 MC `s_axi` 는 BAR2 밖이다.** MMIO 로는 못 닿고, QDMA 의 DMA
엔진에 **AXI 주소를 직접 실어서** 접근해야 한다.

| 대상 | AXI 주소 | BAR2 | 접근 수단 |
|---|---|---|---|
IMEM / CFR / 위반 CSR / GPR | `0x0202_0000_0000` + | **안에 있음** | MMIO (load/store) **또는** DMA |
HBM (bank 데이터) | `0x40_0000_0000` + `b`×1 GiB | 밖 | **DMA 만** |
MC `s_axi` (normal path) | `0x0201_0000_0000`, 4 GB | 밖 | DMA 만 — §1.4, 쓰지 마라 |

오퍼랜드는 크기가 커서 어차피 DMA 가 맞다. 도어벨·CFR·결과 읽기처럼 작고 잦은
접근이 BAR2 쪽이다.

### 1.1 제어 평면 — 슬레이브별 상세

| AXI 주소 | 크기 | 무엇 | 접근 |
|---|---|---|---|
`0x0202_0000_0000` | 512 KB | **IMEM** (AXI4-Full, 256b) | 쓰기만 |
`0x0202_0018_0000` | 4 KB | **CFR** (AXI4-Lite, 32b) | 읽기 / 쓰기 |
`0x0202_0018_1000` | 4 KB | **위반 CSR** (AXI4-Lite, 32b) | 읽기 (+ CTRL 쓰기) |
`0x0202_0040_0000` | **4 MiB** | **GPR** (AXI4-Full, 256b) | 읽기 / 쓰기 |

```
IMEM  0x0202_0000_0000 + (word << 5)      word 14 b, 16384 word   = 512 KB
GPR   0x0202_0040_0000 + (word << 5)      word 17 b, 131072 word  = 4 MiB

한 word = 256 b = 32 B.  INCR 버스트는 beat 마다 word 를 1 증가시킨다.
```

**GPR 이 512 KB 에서 4 MiB 로 커졌고 베이스도 옮겨갔다.** 4 MiB 창은 4 MiB 배수
주소에서만 시작할 수 있어서(AXI 세그먼트는 자기 크기로 정렬돼야 한다) `+0x40_0000`
에 놓았다. CFR / 위반 CSR 주소는 그대로다. BAR2 도 4 MB → 8 MB 로 키웠다 — 네 창의
합이 4 MB 를 넘었다.

창 크기는 디코드 폭과 정확히 같다 — IMEM 은 `addr[18:5]`, GPR 은 `addr[21:5]`.
슬레이브는 그 위 비트를 아예 안 본다(포트에 없다). **창을 더 크게 잡으면 같은
메모리가 두 번 보인다** — 예를 들어 GPR 을 8 MiB 로 잡으면 `+0x40_0000` 과
`+0x80_0000` 이 똑같이 word 0 을 가리키고, 두 번째로 쓴 값이 첫 번째를 덮는다.
경고도 에러도 없다.

### 1.2 IMEM / GPR 슬레이브 제약 — 호스트 코드가 지켜야 한다

두 슬레이브 모두 **INCR 전용, 32 B 풀폭 beat 전용**이다.

- **WRAP 버스트를 보내지 마라.** `ARBURST`/`AWBURST` 를 받기는 하지만 해석하지
  않는다. INCR 로 취급되어 조용히 다른 곳에 쓴다.
- **부분 바이트 쓰기가 불가능하다.** GPR 슬레이브에는 **`WSTRB` 포트가 아예 없다**
  (`emu_gpr_wrap.v` 포트 목록). 32 B 워드 하나가 통째로 쓰인다. 4 바이트만 고치려
  해도 나머지 28 바이트가 마스터가 실어보낸 값으로 덮인다. **read-modify-write 를
  호스트가 직접 해야 한다.**
- **32 B 정렬을 지켜라.** 주소 하위 5 비트는 디코드되지 않는다.

### 1.3 HBM (bank 데이터)

bank `b` 의 베이스 = `0x40_0000_0000 + b × 1 GiB`, 창 **1 GiB**.

| bank | 베이스 |
|---|---|
0 | `0x40_0000_0000` |
1 | `0x40_4000_0000` |
… | `+1 GiB` 씩 |
15 | `0x43_C000_0000` |

호스트도 **같은 주소**로 HBM 에 직접 닿는다. 즉 오퍼랜드를 호스트가 심고, 같은
주소를 bank 가 읽는다. bank 안 오프셋 = `row × 2048 + col × 32`
(한 row = 2048 B = 64 beat).

**ISR 로 닿을 수 있는 범위는 bank 당 256 MiB 다.** `ROW` 필드 17 비트 ×
row 2048 B = 2^28. 이 값은 MC 의 packed 주소 맵이 쓰는 `BANK_OFF_W`(28) 와 같아서
**ISR 경로와 normal 경로가 bank 크기를 같게 본다.**  (`ROW` 가 14 비트이던 시절엔
ISR 만 32 MiB 로 좁아 8 분의 1 만 닿았다.)  1 GiB 창의 나머지 768 MiB 는 호스트가
위 주소로 직접 쓸 때만 쓸 수 있고, ISR 로는 지목할 수 없다.

### 1.4 BAR2 밖에 있는 창 — MC `s_axi`

`emulator_controller_0/s_axi` 가 `0x0201_0000_0000` 에 4 GB 로 **AXI 주소 맵에는
올라가 있다.** 호스트가 DRAM 을 MC 경유로(= 세그먼트 분할·타이밍 모델을 거쳐)
읽고 쓰는 normal path 다.

**BAR2 는 `0x0202_0000_0000` 부터 8 MB 라 이 창을 덮지 않는다.** 따라서

- **MMIO(BAR 매핑 로드/스토어)로는 닿지 않는다.**
- QDMA 의 DMA 엔진은 AXI 주소를 직접 실으므로 닿을 수는 있다.

PIM 브링업에서는 쓰지 마라. 오퍼랜드는 §1.3 의 HBM 직접 주소로 심는 것이 정본
경로이고, 그쪽이 검증된 경로다.

### 1.5 인접 bank 가 같은 HBM PC 를 공유한다

`HBM00_AXI`/`HBM01_AXI` 가 각각 `HBM0_PC0`/`HBM0_PC1` 로 간다. 16 개 연속 포트를
쓰므로 두 bank 가 한 PC 의 대역을 나눠 쓸 수 있다. 물리 타이밍 에뮬레이터에서는
이것이 모델이 계산하지 않는 지연으로 나타난다. **단, 위반이 쌓인다고 여기부터
의심하지 마라 — §2 의 tRCD 항을 먼저 읽어라.**

---

## 2. CFR — 도어벨 / 상태 / 타이밍

`0x0202_0018_0000` 기준 오프셋. 32 b 레지스터.

| 오프셋 | R/W | 무엇 |
|---|---|---|
`0x00` | W | `CTRL` — `[0]` 도어벨(자동 클리어) `[1]` mode (0 NORMAL / 1 ALL_BANK) |
`0x04` | R | `STATUS` — `[3:0]` fetch FSM 상태 `[30:4]` 예약(0 을 읽음) `[31]` done |
`0x08` | W | `T_FAW` |
`0x0C` | W | `T_RRD` |
`0x10` | W | `T_RCD` |
`0x14` | W | `T_CCD` |
`0x18` | W | `T_RTP` |
`0x1C` | W | `T_RP` |
`0x20` | W | `T_WR` |
`0x24` | W | `T_RAS` |
`0x28` | W | `PROG_LEN` — 실행할 ISR 개수 |

타이밍 8 개는 전부 **읽기도 된다** — 브링업 2 단계에서 되읽어 BAR 매핑을 확인하는
데 쓴다. `T_FAW`/`T_RRD` 는 MC 의 rank gate 로, 나머지 6 개는 16 bank 전체에 공통.

**시작값** (시뮬레이션이 쓰는 값):

```
T_FAW=30  T_RRD=6  T_RCD=4  T_CCD=2  T_RTP=3  T_RP=3  T_WR=4  T_RAS=6
```

> **`T_CCD` 를 1 로 낮추지 마라.** `mac_top.sv:26-33` — 같은 latch 로 두 beat 가
> 연속으로 들어가면 **격 beat 만 누산되고 진단이 없다.** 2 가 최소값이다.

> **`T_RCD=4` 는 위반을 낸다. 정상이다.**
> `tb_emu_ctrl_normal` 로 측정한 결과, 지연이 0 인 이상적인 메모리 모델 상대로도
> `T_RCD=4` 는 모자라서 `RCD_RD` 위반이 overrun 3 으로 뜬다. `T_RCD=8` 이면 안 뜬다.
> 실제 HBM 은 더 느리므로 **기본값으로 돌리면 위반 카운터가 0 이 아니다.**
> 에뮬레이터가 "메모리가 모델보다 느렸다" 고 정직하게 보고하는 것이지 고장이 아니다.
> 결과 데이터는 영향받지 않는다. 위반을 없애고 싶으면 `T_RCD` 를 올려라.

---

## 3. ISR — 256 b 명령어

한 워드 = 256 b = IMEM 한 칸. 필드 위치의 정본은 `pim_isr_defs.vh` 이고,
아래는 §3.5 의 실제 통과 워드를 디코드해 검산한 것이다.

> **2026-07-31 자로 배치가 바뀌었다. 예전 인코딩으로 만든 프로그램은 다시 만들어야 한다.**
>
> `ROW` 가 14 → 17 비트가 되면서 그 위의 `BK` / `CH_MASK` / `T` / `OPSIZE` 가 전부
> 3 비트씩 올라갔다. `opcode` 와 `COL`, 그리고 bit 64 이상
> (`pu_mask` / `route` / `gb_mc_mask`) 은 자리가 그대로다.
>
> **그 3 비트는 `C[58]` / `IO[57:56]` 을 없애서 만들었다.** AiM slide-11 하위 64b 에
> 있던 필드인데, 의미가 확정된 적이 없고 RTL 이 읽은 적도 없다(decode 에서 latch 만
> 했다). `G[45]` 도 같은 이유로 `RESERVED` 에 흡수됐다. 그래서 **이 플랫폼의 하위 64b
> 는 더 이상 AiM slide-11 포맷과 같지 않다** — AiM 쪽 호스트 코드를 가져다 쓸 수 없다.
>
> `C`/`IO` 를 같이 밀어올리지 않은 이유는 그러면 `opcode` 가 `[66:62]` 로 가면서
> 64 비트 경계를 넘고, 스칼라 필드를 `uint64_t` 하나로 조립하는 §3.4 의 방식이
> 깨지기 때문이다.

### 3.1 필드 일람

명시하지 않은 비트는 전부 0 이다.

| 비트 | 필드 | 폭 | 옛 위치 | 쓰는 ISR |
|---|---|---|---|---|
`[63:59]` | `opcode` | 5 | 그대로 | 전부 |
`[58:49]` | `OPSIZE` | 10 | `[55:46]` | MAC, WRVEC, COPY, EWMUL |
`[48:36]` | — | 13 | `[45:33]` | 빈 자리 |
`[35]` | `T` | 1 | `[32]` | MAC, RD_MAC |
`[34:27]` | `CH_MASK` | 8 | `[31:24]` | 전부 |
`[26:23]` | `BK` | 4 | `[23:20]` | 단일 bank 를 지목하는 ISR |
`[22:6]` | `ROW` / `GPR_ADDR` | **17** | `[19:6]` (14) | 전부 (의미가 갈린다 — 아래) |
`[5:0]` | `COL` | 6 | 그대로 | MAC, COPY, EWMUL |
`[93:78]` | `pu_mask` | 16 | 그대로 | MAC, EWMUL |
`[175:96]` | `route[16]` | 5×16 | 그대로 | 크로스바를 쓰는 ISR |
`[191:176]` | `gb_mc_mask` | 16 | 그대로 | MAC, EWMUL |

### 3.2 필드 설명

**`OPSIZE` `[58:49]` — beat 수**
"몇 개" 가 아니라 **몇 beat** 다. 1 beat = 256 b = **BF16 16 개**.
길이 `K` 인 내적은 `OPSIZE = K/16` (`K` 는 16 의 배수여야 한다).

`OPSIZE ≤ 64` 여야 한다. 셋이 동시에 걸리는 한계다 — GB 버퍼 64 entry,
DRAM 한 row 64 beat, 그리고 마디를 쪼개지 않으려면 `COL + OPSIZE ≤ 64`.
**65 이상을 주면 `WRVEC` 이 영구 정지하고 아무 메시지도 나오지 않는다.**

**`T` `[35]` — accumulator latch 선택**
PU 안에 latch 가 2 개 있고 이 비트가 고른다는 것이 설계인데, **지금 MC 가
`latch_sel` 을 0 으로 하드타이해 놓았다** (`emulator_controller.v:1166`).
이 필드는 현재 **아무것도 고르지 않는다.** 항상 0 으로 두라 — 1 을 넣어도 latch 0 이
쓰이므로 두 thread 를 번갈아 쓰는 프로그램은 조용히 틀린다.

**`CH_MASK` `[34:27]` — 목적지 채널 비트마스크**
설계 의도는 비트 `c` 가 서면 채널 `c` 로 가는 multicast 인데, **지금 하드웨어는
단일 채널 고정이고 이 필드를 읽는 로직이 없다.** `0x01` 로 두라. 0 을 넣어도
막지 않지만 다중 채널이 붙는 순간 "목적지 없음" 이 된다.

**`BK` `[26:23]` — bank index**
채널 안에서 0..15. all-bank MAC 처럼 `pu_mask` 로 대상을 정하는 ISR 은 안 쓴다.

**`ROW` / `GPR_ADDR` `[22:6]` (17 b) — ISR 마다 의미가 다르다**
같은 비트 자리를 두 용도로 쓴다. **여기를 헷갈리면 조용히 엉뚱한 데이터를 읽는다.**

| ISR | 의미 |
|---|---|
`MAC`, `COPY`, `EWMUL` | **DRAM row** — bank 안 오프셋 = `ROW × 2048 + COL × 32` |
`WRVEC` | **GPR word index** — 벡터를 읽어올 시작 word. `OPSIZE` 개 연속으로 읽는다 |
`RD_MAC` | **GPR word index** — 16 lane 결과가 착지할 word |

17 비트라 **DRAM 은 bank 당 256 MiB, GPR 은 131072 word (4 MiB)** 까지 지목한다.
둘 다 하드웨어의 실제 크기와 정확히 맞는다.

**`COL` `[5:0]`** — row 안의 beat 오프셋 0..63. `COL + OPSIZE ≤ 64`.

**`pu_mask` `[93:78]`** — 연산에 참여하는 bank. all-bank MAC 은 `0xFFFF`.

**`gb_mc_mask` `[191:176]`** — GB 가 벡터를 뿌릴 bank 들.
**`pu_mask` 와 반드시 같아야 한다.** `pim_decode.v:217` 이
`wr_mask = (wr_dsts | gb_mc) & ~pu_norm` 으로 파생 마스크를 만들기 때문에, 둘이
어긋나면 GB 를 받는데 연산은 안 하는 bank(또는 그 반대)가 생겨 결과가 조용히 틀린다.

**`route[16]` `[175:96]`** — bank `i` 의 필드는 `[96 + 5i +: 5]`. 값은 포트 번호,
**`31` 이 "안 씀"(`GB_NULL_PORT`)**. GB 소싱 MAC 은 목적지를 `gb_mc_mask` 로 정하므로
route 를 쓰지 않는다 — **16 개 전부 31 로 채워라.** 0 은 "bank 0 으로 보내라" 다.

### 3.3 opcode

| 값 | 이름 | 상태 |
|---|---|---|
`0x0C` | `MAC` | **검증됨** — all-bank, GB 소싱 |
`0x0F` | `WRVEC` | **검증됨** — GPR → GB 벡터 적재 |
`0x10` | `RD_MAC` | **검증됨** — 16 lane BF16 결과를 GPR 에 착지 |
`0x11` | `EOS` | **검증됨** — 프로그램 종료 |
`0x0E` | `COPY` | 읽기 절반만 검증(bank → GB). 쓰기 절반 미검증 |
`0x0D` | `EWMUL` | FPGA 미검증 |

### 3.4 바이트 순서

한 ISR = 32 B, IMEM 한 칸. 주소는 `0x0202_0000_0000 + (word << 5)`.

**리틀 엔디언이다.** ISR 의 `bit[0]` 이 32 B 배열 `byte[0]` 의 최하위 비트다.
아래 §3.5 의 hex 문자열은 관례대로 **왼쪽이 MSB(bit 255)** 라 뒤집어야 한다.

```c
void isr_to_bytes(const uint8_t hex_msb_first[64], uint8_t out[32]) {
    for (int i = 0; i < 32; i++)
        out[i] = hexbyte(hex_msb_first, 31 - i);   /* 바이트 순서를 뒤집는다 */
}
```

C 구조체라면 `uint64_t w[4]`, `w[0]` 이 `bit[63:0]`, `w[3]` 이 `bit[255:192]`.
`opcode`/`OPSIZE`/`ROW`/`COL`/`CH_MASK`/`T` 는 전부 `w[0]`, `pu_mask` 는 `w[1]`,
`route[16]` 는 `w[1]`~`w[2]`, `gb_mc_mask` 는 `w[2]` 상위에 걸친다.

### 3.5 검증된 프로그램 — 실제 워드

시뮬레이션에서 **실제로 IMEM 을 통과해 280/0 을 낸** 4 워드를 `TRACE=isr` 로 뽑았다
(왼쪽이 MSB). qwen3_0p6b layer0 `k_proj` GEMV 타일 하나 — `L=64` / DRAM row 100 /
벡터는 GPR word 0.. / 결과는 GPR word 1041.

**새 필드 배치의 값이다.** MAC / RD_MAC 워드는 시뮬레이션이 IMEM 에서 읽어낸 hex 를
그대로 옮긴 것이고, 나머지 둘은 같은 빌더로 만든 값이다.

```
word 0  WRVEC
  00000000000000000000ffffffffffffffffffff000000007880000008000000
word 1  MAC
  0000000000000000ffffffffffffffffffffffff3fffc0006080000008001900
word 2  RD_MAC
  00000000000000000000ffffffffffffffffffff000000008000000008010440
word 3  EOS
  00000000000000000000ffffffffffffffffffff000000008800000008000000

CFR PROG_LEN (0x28) = 4
```

디코드하면:

| | opcode | OPSIZE | ROW | COL | T | CH_MASK | pu_mask | gb_mc_mask | route[16] |
|---|---|---|---|---|---|---|---|---|---|
WRVEC | `0x0F` | 64 | 0 | 0 | 0 | `0x01` | — | — | 전부 31 |
MAC | `0x0C` | 64 | 100 | 0 | 0 | `0x01` | `0xFFFF` | `0xFFFF` | 전부 31 |
RD_MAC | `0x10` | 0 | 1041 | 0 | 0 | `0x01` | — | — | 전부 31 |
EOS | `0x11` | 0 | 0 | 0 | 0 | `0x01` | — | — | 전부 31 |

세 워드에서 `ROW` 의 의미가 서로 다른 것에 주의하라 — **MAC 은 DRAM row,
WRVEC/RD_MAC 은 GPR word index** 다.

### 3.6 프로그램 모양

한 프로그램 = 도어벨 1 회 = ISR 4 개.

```
word 0   WRVEC   OPSIZE=L  ROW=<GPR word 시작>
word 1   MAC     OPSIZE=L  ROW=<DRAM row>  COL=0  pu_mask=0xFFFF  gb_mc_mask=0xFFFF
word 2   RD_MAC  OPSIZE=0  ROW=<결과가 착지할 GPR word>
word 3   EOS
PROG_LEN = 4
```

이것이 GEMV 타일 하나다:

```
GB 벡터      = 호스트가 GPR 에 올린 L beat, 16 bank 에 브로드캐스트
bank b 의 A  = 자기 DRAM row 의 L beat
GPR lane b   = bank b 의 누산 결과 (BF16, [b*16 +: 16])
```

`L = K/16`, **`L ≤ 64`**.

### 3.7 소프트웨어가 반드시 지켜야 하는 것

validity gate 가 타이밍 때문에 fetch 경로에서 제외됐다(2026-07-29). **하드웨어는
ISR 을 검사하지 않고, 어긴 것에 대한 진단도 없다.** 권고가 아니라 전제다.

1. **`WRVEC.OPSIZE == MAC.OPSIZE`** — 다르면 bank peri-in skid 에 벡터가 남고
   (그 skid 에는 flush 포트가 없다) 다음 MAC 의 벡터가 `vector[OPSIZE]` 부터
   시작한다. 조용한 오답이다.
2. **`gb_mc_mask == pu_mask`**
3. **`COL + OPSIZE ≤ 64`**
4. **`OPSIZE ≤ 64`** — `WRVEC` 이 65 이상이면 **영구 정지**하고 메시지가 없다
5. **`T = 0`**
6. **`CH_MASK ≠ 0`** (`0x01` 로 두라)

---

## 4. 위반 CSR

`0x0202_0018_1000` 기준 오프셋. 정본은 `emu_viol_csr.v` 헤더.

```
bank b, base = b*0x40 :
  +0x00  STICKY   [0] RCD_RD [1] CCD_RD [2] RCD_WR [3] CCD_WR
                  [4] RECOVERY_WR   [8] ewmul_drop   [31] 위 중 하나라도
  +0x04  CNT_A    [7:0] RCD_RD [15:8] CCD_RD [23:16] RCD_WR [31:24] CCD_WR
  +0x08  CNT_B    [7:0] RECOVERY_WR [15:8] ewmul_drop
  +0x0C  MAX_A    [7:0] RCD_RD [15:8] CCD_RD [23:16] RCD_WR [31:24] CCD_WR
  +0x10  MAX_B    [7:0] RECOVERY_WR
전역 :
  0x400  CTRL     [0] clrstats — 1 을 쓰면 전 bank 초기화 (자동 클리어)
  0x404  ANY      [15:0] bank 당 1 비트, 그 bank 의 STICKY[31] 이 서 있으면 1
```

**`0x404` 를 먼저 읽어라.** 한 번의 읽기로 범인 bank 를 지목한다. 카운트는
0xFF 에서 포화하고 래핑하지 않는다. 통계는 읽기 전용이라 호스트가 깨끗한 실행을
위조할 수 없다.

---

## 5. 브링업 순서

```
1. CFR 타이밍 8 개 기록 + 되읽기 검증        0x0202_0018_0000 + 0x08..0x24
2. HBM 에 오퍼랜드 기록                      0x40_0000_0000 + b*1GiB + row*2048 + col*32
3. HBM 되읽기로 주소 뷰 일치 확인             (같은 주소)
4. GPR 에 벡터 기록                          0x0202_0040_0000 + (word << 5)
5. IMEM 에 프로그램 기록                     0x0202_0000_0000 + (word << 5)
6. PROG_LEN 기록                            CFR 0x28
7. 도어벨                                   CFR 0x00 <- 1
8. STATUS[31] (done) 폴링                   CFR 0x04
9. 결과 읽기                                GPR 의 RD_MAC 이 지정한 word
10. 위반 CSR 0x404 (ANY) 확인               0x0202_0018_1000 + 0x404
```

**1 단계의 되읽기가 BAR 매핑 · NoC · AXI-Lite 를 한 번에 증명한다.** CFR 타이밍
레지스터는 전부 읽히므로, 쓴 값이 그대로 돌아오면 제어 평면이 살아 있는 것이다.

**3 단계가 가장 중요하다.** 호스트가 보는 HBM 주소와 bank 가 구동하는 주소는 BD 상
같게 만들어져 있지만, 어긋나면 **MAC 이 쓰레기를 누산하고 아무 신호도 안 뜬다.**

**9 단계는 GPR 슬레이브에서 그냥 읽으면 된다.** `0x0202_0040_0000 + (word << 5)`
에서 32 B 를 읽으면 16 lane BF16 이 들어 있다 (lane b = bank b, `[b*16 +: 16]`).
시뮬레이션이 이 경로를 계층 참조 값과 대조해 검증하므로, 두 값이 어긋나면 그것이
곧 슬레이브 버그다.

**10 단계에서 위반이 보이는 것은 정상이다** — §2 의 `T_RCD` 항을 읽어라.

---

## 6. 지금 안 되는 것 / 미검증

| | 상태 |
|---|---|
dispatcher AR/R | zero-return 스텁. **IMEM 되읽기는 안 된다** (GPR 은 자기 슬레이브로 읽는다) |
normal read/write 경로 (MC `s_axi`) | **BAR2 밖이다** — §1.4 참조. MMIO 로는 못 닿고, 시뮬레이션에서만 검증됐다 |
`emu_viol_csr` | **테스트벤치가 없다.** 레지스터 읽기를 아무도 돌려본 적 없다 |
`EWMUL` | FPGA 미검증 |
`COPY` 쓰기 절반 | 미검증 |
bank 당 256 MiB 초과 | ISR 의 `ROW` 로 지목 불가 — 1 GiB 창의 나머지 (§1.3) |
높은 `ROW` 값 | **`ROW` 17 b 확장은 디코더까지만 검증됐다.** `tb_pim_decode` 가 `0x1ABCD` 를 확인하지만, 지금 도는 어떤 TB 도 14 b 를 넘는 row 를 실제 bank 의 `m_axi_araddr` 까지 흘려보내지 않는다 (쓰는 row 가 40 / 42 / 100 뿐). 산술상 `2^28-1` 이 30 b 오프셋 필드에 들어가는 것은 확인했지만 **돌려본 적은 없다** |
다중 채널 | 하드웨어가 단일 채널 고정. `CH_MASK` 를 아무도 읽지 않는다 |
ILA | 없음 (DEBUG=0). 모든 관측은 BAR2 읽기로 |

---

## 7. 보낼 것

```
emu_top_fpga_wrapper.pdi     device image
HANDOFF.md                   이 문서
rtl/emu_viol_csr.v           위반 CSR 맵 (읽히려고 보낸다, 단독 컴파일 안 됨)
dispatcher_top.v             CFR / IMEM 창 맵 (같은 이유)
emu_gpr_wrap.v               GPR 호스트 슬레이브 + 2-master 중재 (같은 이유)
```
