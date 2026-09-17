# sw/ — PIM 소프트웨어 스택

설계 정본은 [`docs/pim_sw_design.md`](docs/pim_sw_design.md) 이다. 이 문서는 그 설계가
디렉토리로 어떻게 나뉘었는지, 지금 어디까지 되어 있는지만 적는다.

```
sw/
  include/    두 개의 계약.  이 두 파일이 개발을 나눌 수 있게 만드는 전부다
    uapi/pim_ioctl.h    lib <-> drv        (커널·유저 양쪽이 같이 컴파일)
    pim/pim.h           app <-> lib
    pim/pim_geometry.h  주소 산술 (순수 함수, 상태 없음)
    pim/pim_rt.h        app <-> runtime
  drv/        pim.ko — device memory 장부 2개 (dram, gpr).  설계 §6, §9.4
  lib/        libpim — 할당 · 주소 변환 · 데이터 이동.  설계 §7, §8, §9.4
  runtime/    libpimrt — 실행 엔진 + GEMV.  설계 §9
  docs/       설계 정본
```

의존 방향은 한 방향뿐이다.

```
app ──► runtime ──► lib ──► /dev/pim (drv)
                      └──► /dev/qdma01000-MM-{0,1}
```

`runtime` 은 `lib` 의 **공개 API 만** 부른다 (`libpim.so` 를 링크하지 오브젝트를
링크하지 않는다). `lib` 은 `drv` 와 헤더 하나만 공유한다. `drv` 는 위를 전혀 모른다.

---

## 1. 빌드

```sh
make            # lib + runtime  (유저 공간만)
make drv        # pim.ko         (커널 빌드 트리 필요)
make all
```

**driver 가 기본 목표에 없는 것이 의도한 것이다.** 유저 공간 절반은 빌드 시점에
driver 를 전혀 필요로 하지 않는다 — 공유하는 것이 헤더 하나뿐이므로. 한쪽만 따로
빌드되는 것이 그 분리가 진짜라는 검사다.

`libpim.a` 를 정적 링크하는 쪽은 `-lpthread` 가 필요하다.

## 2. 실행 준비

```sh
cd drv && make load CH=4        # insmod pim.ko pim_channels=4 ...
cat /proc/pim                   # 두 pool 의 장부 상태
../../scripts/qdma_queues.sh setup
```

**채널 수는 insmod 인자다** (설계 §6.4). 재빌드가 아니다. 이것이 hwdef 방식과의
가장 큰 차이고, 이 스택을 새로 만드는 이유의 절반이다.

| | hwdef/ + runtime/ (기존) | sw/ (이것) |
|---|---|---|
| 채널 수 출처 | `platform/*.conf` → `-D` → 컴파일 상수 | `insmod` 인자 → `GET_INFO` |
| 채널 바꾸기 | `setup.sh` 가 전체 재빌드 | `rmmod` / `insmod` |
| 불일치 사고 | 빌드와 보드가 갈릴 수 있음 | 진실이 한 곳 |

## 3. 두 개의 pool (설계 §9.4)

|  | DRAM | GPR |
|---|---|---|
| 크기 | `nch × 4 GiB` | 4 MiB |
| 담는 것 | weight, KV cache | vector, result 버퍼 |
| 물리 배치 | RoChBaCo interleave | 선형 |
| driver 단위 | 2 MiB chunk | 4 KiB page |
| libpim 단위 | 128 KiB broadcast unit | 4 KiB page (= driver 단위) |

```c
uint16_t *w = pim_alloc(c, n*k*2, PIM_MEM_DRAM);   /* weight  */
uint16_t *x = pim_alloc(c, k*2,   PIM_MEM_GPR);    /* vector  */
uint16_t *y = pim_alloc(c, n*2,   PIM_MEM_GPR);    /* results */

pim_memcpy(c, x, host_vec, k*2, PIM_TO_DEV, 0);    /* 별도 API 없음 */
pim_rt_gpr_words(c, x, k*2, &word, &nwords);       /* WRVEC 이 쓰는 좌표 */
```

**같은 코드가 둘 다 돌린다.** driver 의 `struct pim_ledger` 는 이제 인스턴스이고
(`pim_led[PIM_NREGION]`), `pim_blocks_alloc()` 은 자기가 받은 ledger 에서 상수를
읽는다. libpim 쪽도 pool 이 `pool[PIM_NMEM]` 하나짜리 배열이다. GPR 은
`grans_per_block == 1` 이라 안쪽 비트맵이 1비트인 **퇴화(degenerate) 사례일 뿐 특수
사례가 아니다.**

그 퇴화의 관측 가능한 결과 하나: **GPR 할당은 매번 ioctl 을 한 번 쓴다** (채울 부분
블록이 없으므로). 설계상 GPR 객체는 오래 사는 것들이라(벡터 하나 올려두고 WRVEC 이
수없이 재사용) 지금은 문제가 아니고, 프로파일에 churn 이 보이면 여기서 한 번에 여러
page 씩 받아오면 된다 — 다른 곳은 안 바뀐다.

**전송은 손댈 게 없었다.** GPR 은 같은 MM 큐에서 같은 AXI 주소 공간의 다른 구간이라
`pim_memcpy` 가 extent 를 걷는 루프를 그대로 돈다. DRAM 이 여러 run 을 찾는 자리에서
GPR 은 보통 하나를 찾을 뿐이다.

**주소 좌표는 두 개다.** `pim_resolve()` 가 `pim_loc { mem, axi, off }` 를 준다 —
`axi` 는 pread/pwrite 가 받는 것, `off` 는 region 안 offset 이고 GPR 에서는
`off / 32` 가 WRVEC/RD_MAC 이 이름 부르는 **word 번호**다. 서로 다른 pool 의 포인터를
잘못 쓰면 `pim_rt_unit()` 과 `pim_rt_gpr_words()` 가 각각 거부한다.

**GB 는 allocator 의 대상이 아니다.** 채널당 global buffer 는 host 직접 접근 경로가
없고 WRVEC/RD_MAC 으로만 읽고 쓰인다. "GB 에 올린 operand 를 누가 소비하고 언제 다음
WRVEC 이 덮어써도 되는가" 는 메모리 관리가 아니라 **스케줄러의 의존성 문제**라서 이
스택에는 들어오지 않는다.

## 4. 지금 어디까지 되어 있는가

| | 상태 |
|---|---|
| `drv/` 장부 2개, 2단 first-fit, per-fd 회수, `/proc/pim` | **구현됨.** 5.15.165 에서 warning 없이 빌드 |
| `lib/` PROT_NONE 예약, pool + 비트맵, fill-partial-first, trim hysteresis | **구현됨**, 두 pool 다 |
| `lib/` 주소 변환, extent 합치기, QDMA pread/pwrite | **구현됨**, 두 pool 다 |
| `include/pim_geometry.h` RoChBaCo · ChRoBaCo 디코드, GPR word 변환 | **구현됨 + 검증됨** (아래) |
| `runtime/` unit 단위 피연산자 해석, GPR word 해석 | **구현됨** |
| `runtime/` 엔진 (IMEM 적재 · PROG_LEN · 도어벨 · STATUS · 위반 CSR) | **구현됨** — 아래 §8 의 잠정 결정 위에 |
| `runtime/` 프로그램 빌더 + 검증기 4종 | **구현됨 + 검증됨** |
| `runtime/` GEMV — supergroup 여러 개, 두 스케줄 | **구현됨 + 보드 검증됨** (2026-08-21, ch2) |
| `runtime/` 누산기 scrub | **구현됨** — 프로세스를 건너 사는 latch 를 비운다 |
| `runtime/` RTL 정확 레퍼런스 (`pim_mac_exact.c`) | **이식됨** — 구버전 것 그대로, 보드와 일치했던 것 |
| 여러 supergroup (`n = 16×nch×ngroups`) | **구현됨.** `n=1024` 까지 확인 |
| `lib/test/` 오프라인 테스트 2개 | **구현됨.** 보드도 모듈도 필요 없음 |
| 보드 위 실행 | **됨.** 2026-08-21, ch2 / RoChBaCo / QDMA MM 큐 |

```sh
make test        # lib/test 둘 + runtime/test/gemv_test
```

**주소 산술은 이미 검증되어 있다.** `pim_geom_decode()` 를 hwdef 의
`pim_decode_as()` 와 대조했다 — 후자는 `emu_mc` 로 보드에서 8 GiB 까지 확인된
것이다.

```
round trip RoChBaCo ch4 : 0 of 549493 mismatched
round trip ChRoBaCo ch4 : 0 of 549493 mismatched
cross-check vs hwdef    : 524625 points, ch 0 bad, bank 0 bad, bank-offset 0 bad
```

§4.1 불변식 위반(unit 이 chunk 를 안 나눔, unit/chunk 가 64 초과, nch 가 2의 거듭제곱이
아님)은 세 경우 모두 `pim_geom_check()` 가 거부하는 것도 확인했다.

**할당기는 가짜 driver 로 검증했다.** `pool_test` 가 진짜 `pim_alloc.c`/`pim_geom.c`/
`pim_rt.c` 를 링크하고 ioctl 두 개만 시뮬레이션한다 — `pim_dev.c` 만 빠지는 것이
이 분할의 이유다. 찾는 것은
**aliasing** 이다 — 인덱스가 미묘하게 틀려도 나머지 성질은 다 통과하는데, 살아 있는 두
할당이 같은 카드 주소로 풀리는 것만은 통과하지 못한다. 그리고 그게 나중에 GEMV
결과에서 역추적하기 제일 어려운 실패다. pool 이 둘이 되면서 같은 걱정의 두 번째 판본
(DRAM 인덱스를 GPR granule 에 적용) 이 생겼으므로, alias 검사는 pool 별이 아니라
**두 pool 을 한꺼번에** 훑는다.

```
1. dram 1 B -> 131072 B, gpr 1 B -> 4096 B; pools reported correctly
2. a 4096 B vector is GPR word 0..127 (128 words of 32 B)
   a DRAM pointer is refused / a GPR pointer is refused
3. 80 mixed allocations, 0 aliases
   dram 32 MiB in 17 chunks (17 ioctls)   gpr 368 KiB in 92 pages (27 ioctls)
4. after freeing every other, an 8 MiB dram alloc still succeeds
5. 2 MiB dram -> one run of 2097152 B;  64 KiB gpr -> one run of 65536 B
6. after trim: dram 0 blocks out, gpr 0 blocks out      (누수 없음)
```

## 5. GEMV — 임의 크기

```c
pim_gemv_alloc(c, n, k, &w);              /* n, k 아무 값이나 → 타일링 결정 */
pim_gemv_upload(c, &w, W);                /* 호스트 순열 + 전송 한 번       */

size_t xb, yb;
pim_gemv_gpr_bytes(&w, mode, pim_exec_max_isrs(e), &xb, &yb);
void *xg = pim_alloc(c, xb, PIM_MEM_GPR);         /* 벡터 */
void *yg = pim_alloc(c, yb, PIM_MEM_GPR);         /* 결과 */

pim_gemv_ex(c, e, &w, x, xg, yg, y, mode, &st);   /* 프로그램 → 도어벨 → 회수 */
```

**`n` 도 `k` 도 아무 값이나 됩니다.** 둘 다 하드웨어가 주소를 매기는 단위로 0 패딩합니다 —
`n` 은 supergroup(`16 × nch` 출력), `k` 는 beat(16 BF16). 0 가중치는 어떤 순서로 더해도
정확히 `0.0f` 이라 실제 출력을 건드리지 않습니다. `k` 를 1024 까지 올리지는 **않습니다** —
마지막 청크가 짧은 `OPSIZE` 를 지고 갑니다.

**IMEM 을 넘으면 supergroup 단위로 launch 를 쪼갭니다.** supergroup 은 첫 MAC 부터
RD_MAC 까지 자기 누산기를 소유하므로, 다른 데서 자르면 진행 중인 합이 도어벨을 건너
latch 에 갇힙니다. `pim_exec_config.max_isrs` 로 한계를 낮춰 그 경로를 시험할 수 있습니다:

```
  n=1024 k=4096  32 grp x 4 chunk |  321 ISR  128 WRVEC   1 launch  292 us | 1024/1024 exact
                                  |  322 ISR  128 WRVEC   2 launch  292 us | 1024/1024 exact
                                  |  324 ISR  128 WRVEC   4 launch  298 us | 1024/1024 exact
                                  |  352 ISR  128 WRVEC  32 launch  327 us | 1024/1024 exact
```

분할 대가는 launch 당 EOS 하나와 도어벨 왕복 — 32 조각으로 잘라도 12% 입니다.

프로그램은 이렇게 나온다 (`nchunks` 개 청크, `nch` 개 채널):

```
WRVEC  OPSIZE=L0  ROW=xword+0     GPR -> GB, 전 채널
MAC    OPSIZE=L0  ROW=row0        pu_mask = gb_mc_mask = 0xFFFF
WRVEC  OPSIZE=L1  ROW=xword+64
MAC    OPSIZE=L1  ROW=row1        ← 누산기가 fp32 로 이어받음
...
RD_MAC OPSIZE=0   ROW=yword+ch    CH_MASK = 1<<ch   (1-hot, 채널당 하나)
EOS                               ← done 은 마지막 ISR 을 *수락*할 때 뜬다
```

**`ROW` 은 같은 비트에서 두 가지를 뜻한다** — WRVEC/RD_MAC 에서는 GPR word 번호,
MAC 에서는 DRAM row. 뒤바꿔도 하드웨어는 아무 말도 하지 않는다.

**가중치 업로드가 전송 한 번인 이유**: RoChBaCo 에서 선형 `unit_bytes` 한 덩이가 전
채널 × 전 bank 의 같은 row 페이지 = all-bank MAC 한 발의 피연산자 전체다. 그래서 배치가
**호스트 순열 + `pim_memcpy` 한 번**이지 (채널, bank) 마다 전송하는 scatter 가 아니다.
구버전이 채널당 16 전송을 하던 건 직접 어퍼처가 bank 마다 base 를 따로 줘서였다.

### 실측 (ch2, 2026-08-21)

| shape | | ISR | WRVEC | us | 정확도 |
|---|---|---|---|---|---|
| `[2048 × 1024]` q_proj | SINGLE | 194 | 1 | 86 | 2048/2048 |
| `[3072 × 1024]` gate/up | SINGLE | 290 | 1 | 128 | 3072/3072 |
| `[1024 × 3072]` down | SINGLE | 257 | 96 | 225 | 1024/1024 |
| | **DUAL** | 209 | **48** | **187** | 1024/1024 |
| `[4096 × 4096]` | SINGLE | 1281 | 512 | 1166 | 4096/4096 |
| | **DUAL** | 1025 | **256** | **999** | 4096/4096 |
| `[2048 × 8192]` | SINGLE | 1153 | 512 | 1166 | 2048/2048 |
| | **DUAL** | 897 | **256** | **998** | 2048/2048 |

`k=1024` 인 shape 은 청크가 하나라 WRVEC 이 이미 1 개 — DUAL 이 줄일 게 없습니다.

```sh
./gemv_test --n 4096 --k 4096 --dual
./gemv_test --n 1024 --k 4096 --max-isrs 50    # 분할 경로
```

**MAC 이 이어지는 이유**: 누산기가 BF16 보다 넓고 ISR 을 건너 산다. 그래서 청크가 몇
개든 **최종 반올림은 RD_MAC 에서 한 번**이다.

**레퍼런스는 fp32 루프가 아니다.** beat 하나가 **block floating point** 다 — 16 lane 을
그 beat 의 **최대 지수**로 전부 오른쪽 시프트한 다음 정렬된 정수로 정확히 더한다.
그래서 비트를 잃는 곳은 덧셈이 아니라 **정렬 시프트**이고, 24 자리 이상 밀리는 lane 은
통째로 죽는다. `[+1, -1, 2^-30]` 이 실리콘에서 정확히 0 이 나오는 이유고 (eps 가
상쇄쌍을 만나기 전에 정렬에서 소멸), fp32 모델은 **어떤 순서로 더해도** 재현 못 한다.

그래서 레퍼런스는 RTL 을 옮긴 `pim_mac_exact.c` (반올림 지점 11곳, 각각 RTL 줄 번호
인용) 이고, 구버전 runtime 이 쓰던 것과 같은 파일 — 보드와 일치했던 것 — 을 그대로
이식했다. 테스트가 `[+1, -1, 2^-30] -> 0000` 을 확인하므로, 누가 이걸 float 루프로
바꿔놓으면 즉시 걸린다.

### 보드 실측 (2026-08-21, ch2 · RoChBaCo · `pim.ko` abi 2)

```
2 ch x 16 bank, unit 64 KiB, n = 32 outputs per tile
accumulators scrubbed (2 channel(s) drained into GPR word 0)
k=16    chunks=1  last_beats=1    readback 0 bad  lanes 32/32 exact
k=1024  chunks=1  last_beats=64   readback 0 bad  lanes 32/32 exact
k=2048  chunks=2  last_beats=64   readback 0 bad  lanes 32/32 exact
k=1500  chunks=2  last_beats=30   readback 0 bad  lanes 32/32 exact
```

**32 lane 전부 bit-exact**, 청크 체이닝(`k=2048`)과 짧은 마지막 청크 + 패딩(`k=1500`)
포함. 랜덤 BF16 피연산자에 톨러런스 0.

**위반 CSR 은 단계별로 깨끗하게 갈렸다.** CSR 을 단계 사이마다 클리어하고 측정:

| 단계 | `RCD_RD` | `RECOVERY_WR` |
|---|---|---|
| scrub 만 | 0 | 0 |
| MC 업로드만 | 0 | **384** |
| MC 되읽기만 | **272** | 0 |
| GEMV 런치만 | **32** = 2 chunk × 16 bank | 0 |

즉 `RECOVERY_WR` 은 **MC 쓰기 경로에서만**, `RCD_RD` 은 **읽기/MAC 에서만** 난다.
런치의 32 는 (bank, chunk) 마다 row activate 하나씩 — 정확히 예상되는 개수다.

**`RCD_RD` 은 결함이 아니라 타이밍 모델의 리포트다.** `T_RCD` 을 쓸어보면:

| `T_RCD` | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| `RCD_RD` (채널당) | 32 | 32 | 32 | 4 | 4 |
| worst overrun (cy) | 81 | 70 | 61 | 37 | 15 |
| lanes exact | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 |

허용치를 키우면 overrun 이 줄어든다 — 에뮬레이터가 "메모리가 모델보다 느렸다" 고
말하는 것이고, **어느 설정에서도 답은 bit-exact** 다. 참고로 `hwdef/emu_regs.h` 의
`EMU_TIMING_QUIET`(`T_RCD=8`) 은 **이 워크로드를 조용하게 만들지 못한다** — 32 쯤
필요하다. 그 값은 단일 beat 프로브에서 측정된 것이라 여기엔 안 맞는다.

### 누산기 latch 두 개 (ISR[35], "T") — 2026-08-21 측정

**무엇에 쓰는 물건인가.** GB-sourced MAC 은 시작할 때 GB 읽기 포인터를 **rewind** 하므로
WRVEC 하나가 여러 MAC 을 먹일 수 있습니다 (`emu_chain --test rewind` 실측). K 가 한
페이지에 들어가면 이건 공짜입니다 — `WRVEC | MAC | RD_MAC | MAC | RD_MAC | ...`.

**K 가 1024 를 넘는 순간 공짜가 아니게 됩니다.** 각 출력 그룹이 자기 청크들을 누산기에
이어야 하는데 `RD_MAC` 은 **읽고 지우기**라, 두 번째 그룹을 사이에 끼우면 첫 그룹의
진행 중인 합이 날아갑니다. latch 하나로 가능한 유일한 정확 스케줄은 group-outer 이고,
**물리적으로 벡터 하나를 그룹 수만큼 다시 올립니다.**

latch 둘이면 WRVEC 한 번 올리고 **ROW 와 T 만 바꿔서** 그룹 둘을 처리합니다:

```
SINGLE                       DUAL
WRVEC(chunk 0)               WRVEC(chunk 0)
MAC(A, 0)                    MAC(A, 0, T=0)
WRVEC(chunk 1)               MAC(B, 0, T=1)     ← ROW·T 만 다름
MAC(A, 1)                    WRVEC(chunk 1)
RD_MAC(A)                    MAC(A, 1, T=0)
WRVEC(chunk 0)  ← 또         MAC(B, 1, T=1)
MAC(B, 0)                    RD_MAC(A), RD_MAC(B)
WRVEC(chunk 1)  ← 또
MAC(B, 1)
RD_MAC(B)
```

**둘 다 정확합니다.** 각 출력은 여전히 자기 latch 에서 전 청크를 누산하고 자기 RD_MAC
에서 한 번 반올림합니다 — 속도-정확도 교환이 아닙니다. 다른 건 벡터가 GB 로 몇 번
넘어가느냐 뿐입니다.

**실측** (ch2, `runtime/test/tlatch_test.c`, 20 reps):

| shape | WRVEC | us/launch | golden 대비 |
|---|---|---|---|
| `g4 k=1024` (1 청크) | 1 → 1 (**0%**) | 7.5 → 7.4 | 0/128, 0/128 |
| `g4 k=4096` | 16 → 8 (**50%**) | 37.6 → 31.9 (**15.3%**) | 0/128, 0/128 |
| `g16 k=4096` | 64 → 32 (**50%**) | 147.3 → 125.6 (**14.7%**) | 0/512, 0/512 |
| `g32 k=8192` | 256 → 128 (**50%**) | 584.0 → 499.9 (**14.4%**) | 0/1024, 0/1024 |

**청크가 1개면 이득이 0 인 게 맞습니다** — 그때는 single latch 도 WRVEC 하나로 전
그룹을 처리할 수 있고, `PIM_ACC_SINGLE` 도 그렇게 hoist 합니다. 비교를 공정하게
만들려고 넣은 것이지, DUAL 을 좋아 보이게 하려고 뺀 게 아닙니다.

**ISR[35] 이 실제로 디코드된다는 증거**는 이 결과 자체입니다: latch_sel 이 묶여 있으면
그룹 A 와 B 가 같은 latch 에 쌓여 두 답이 서로 오염됩니다. 128 개 출력이 전부
bit-exact 로 나올 수가 없습니다. **conf 의 `FEATURE_T_LATCH=0` 은 낡았습니다.**
(ch2 에서만 확인 — ch1/ch4 conf 에 그렇게 적어뒀습니다.)

**T 가 `T_CCD >= 2` 를 풀어주지는 않습니다.** 별도로 측정했는데, `T_CCD=1` 에서 SINGLE
과 DUAL 이 **똑같이** 깨집니다 (32/32 틀림). 그 제약은 *연속 MAC* 이 아니라 **한 MAC
안의 beat** 에 대한 것이고, MAC 의 `OPSIZE` beat 이 전부 그 ISR 의 T 가 가리키는 latch
로 가는데 T 는 ISR 단위라 beat 단위로 번갈 수가 없습니다.

## 6. 결정된 것 — 코드에 이미 박혀 있는 것들

* **PIM 은 한 프로세스가 점유한다** (2026-08-21). 실행 엔진 — dispatcher, IMEM,
  도어벨, **프로세스를 넘어 살아남는 누산기 latch** — 은 싱글턴이라 나누는 게 아니라
  중재해야 하고, 그건 나중이다. **메모리 쪽은 이 전제에 영향받지 않는다**: per-fd
  장부는 crash 회수 수단이지 멀티프로세스 기능이 아니라 그대로 있다.

* **RoChBaCo 만 지원한다.** `pim_open()` 이 ChRoBaCo 를 보면 거부한다. 디코드 자체는
  두 맵 다 있지만(세 줄 차이라 빼면 오히려 둘이 더 달라 보인다) 위 계층은 RoChBaCo
  로만 쓰여 있다.
* **BAR 를 유저 공간에 매핑하지 않는다.** 그래서 `drv` 에는 `.mmap` 이 없고 `lib` 에는
  레지스터 접근이 없다. 대가는 §6 참조.
* **`pim_alloc` 은 진짜 `void *` 를 준다** — `PROT_NONE` mmap 예약. 물리 메모리 0,
  page table entry 0, VMA 하나. 역참조하면 그 자리에서 SIGSEGV.
* **DRAM 요청은 broadcast unit 으로 올림된다** (4채널이면 128 KiB). all-bank 연산이
  unit 전체를 건드리므로 한 unit 을 두 buffer 가 나눠 쓰면 둘 다 피연산자가 될 수 없다.
  **GPR 은 4 KiB 로 올린다** — 이건 피연산자 모양 때문이 아니라, 크기 분포를 재보기
  전에 size-class 같은 가정을 굽지 않기 위해서다 (설계 §9.4).
* **GPR 내용은 persist 한다.** launch 경계 무효화 같은 조항은 없다 — 수명은 alloc/free
  가 지배한다. 그래서 vector 를 한 번 올리고 WRVEC 이 계속 재사용하는 게 성립한다.
* **`pim_memcpy` 는 동기다.** 반환 시점에 전부 착지해 있다 — 그래서 바로 뒤에 연산을
  걸어도 된다. `pim_sync()` 는 캐시 flush 가 아니라 fence 로 예약해 뒀고 지금은
  즉시 반환한다.

## 7. 기존 `runtime/` 과의 관계

`emulator_top/runtime/` (libpim.so 구버전) 은 **아직 지우지 않았다.** 남긴 이유는
지금 지우면 같이 손봐야 하는 것들이 있기 때문이다:

* `scripts/setup.sh` 가 `runtime` 과 `runtime/test` 를 빌드한다 (line 235-236)
* `runtime/test/` 두 개가 구버전 `libpim.so` 를 링크한다
* `hwdef/test/` 프로브들은 **영향 없다** — 원래 libpim 을 링크하지 않는다

**두 트리를 한 바이너리에 같이 링크하면 안 된다.** 양쪽 다 `pim_alloc` / `pim_free` 를
내보내는데 시그니처가 다르다 (새 쪽은 region 인자를 받는다). 헤더 가드도 다르고(`PIM_H` vs `PIM_PIM_H`) 서로를 막지
않으므로, 막는 것은 이 문장뿐이다.

폐기하려면 `setup.sh` 에서 두 줄을 빼고 `runtime/` 을 옮기면 된다 — 한 번에 하겠다고
하면 그렇게 한다. **git 저장소가 아니라서 지우면 복구가 안 되므로** 이번에는 건드리지
않았다.

`hwdef/` 는 그대로 둔다. 프로브(`emu_mc`, `emu_gemv` …)는 보드 검증 수단이고 이
스택과 무관하게 계속 쓸모가 있다. `runtime/` 이 `hwdef/emu_regs.h` 를 쓰게 될 것이다 —
ISR 인코딩은 보드의 사실이고 집이 하나여야 한다. 다만 `hwdef/pim_platform.h` 는
**쓰면 안 된다**: 채널 수를 컴파일 상수로 들고 있어서, 한 바이너리가 채널 수에 대해
두 개의 답을 갖게 된다.

## 8. 잠정 결정 하나 — 도어벨

**`sw/runtime/pim_exec.c` 가 CFR 페이지(32 KiB)를 user 공간에 mmap 한다.** 설계 §6.1 의
"BAR 를 user 에 매핑하지 않는다" 를 그만큼 되돌린 것이고, GEMV 를 끝까지 돌려보려면
도어벨·PROG_LEN·STATUS 에 닿아야 하는데 지금은 다른 길이 없어서다. 벌크는 전부 QDMA
그대로다 — IMEM 에 프로그램, GPR 에 벡터·결과.

| | |
|---|---|
| 지금 받아들일 수 있는 이유 | §6.1 의 근거 둘 중 **프로세스 간 중재**는 단일 프로세스 전제에서 무의미하고, **임의 프로세스의 레지스터 접근 차단**은 §11 의 격리 비고려 결정에 이미 포함된다 |
| 대가 | 다른 프로세스가 도어벨을 누를 수 있고 **아무것도 막지 않는다** |
| 되돌리는 법 | MMIO 는 전부 `cfr_rd`/`cfr_wr` 과 `pim_exec_open` 의 mmap 뒤에 있다. 방안 A 는 그 셋을 `PIM_IOC_LAUNCH` 로 바꾸는 것이고 (uapi `0x04` 예약해둠), **그 위는 아무것도 안 바뀐다** |

`pim_exec_open()` 이 여는 김에 **메모리 계층이 못 하던 검사**를 한다: 보드의 `MODE_CTRL`
이 libpim 이 들은 주소 맵과 같은지. 다르면 모든 가중치가 ROW 필드가 찾아갈 곳이 아닌
데 앉는데, 어디서도 에러가 안 난다.

## 9. 열린 항목

**설계 문서에 이미 있는 것**

* command 제출 경로 A/B (§9.1, §11) — 이게 정해지기 전에는 `runtime/` 이 더 자라지
  않는다. A 면 `uapi/pim_ioctl.h` 의 `0x04..` 에 `SUBMIT`/`WAIT` 이 생기고 driver 가
  ioremap 을 갖는다. B 면 driver 는 지금 모습 그대로다.
* 완료 통지 (폴링 vs interrupt), operand co-location 제약, tiled layout, `PIM_ALLOC_SMALL`

**구현하면서 새로 나온 것**

* **`vbase` 등록이 1:1 이 아니다.** 설계 §7.3 은 `ALLOC_CHUNKS` 한 번이 할당 하나에
  대응한다고 가정하는데, 2단 pool 에서는 아니다 — chunk 요청 하나가 나중의 여러 할당을
  먹이고, 할당 하나가 여러 요청의 chunk 를 쓴다. 그래서 `lib` 은 지금 `vbase=0`
  (미등록) 을 보낸다. ABI 필드와 driver 쪽 기록 코드는 남겨 뒀다(나중에 넣으면 ABI
  변경이므로). **어떻게 할지 정해야 한다** — 그대로 두고 `/proc/pim` 에서 vbase 열을
  빼거나, `pim_free` 시점에 vbase 만 지우는 호출을 하나 더 두거나.
* **죽은 보드를 조기에 못 잡는다.** 구버전 runtime 은 상태 레지스터가 all-ones 인지
  봐서 보드가 죽은 걸 알아냈다. BAR 를 안 매핑하기로 했으므로 이 스택은 못 한다 —
  QDMA 의 EIO 가 유일한 신호다. 실제로 문제가 되면 `drv` 에 "레지스터 하나 읽기"
  ioctl 을 두는 게 제일 작은 해결책이고, 그건 §9.1 방안 A 의 일부다.
* **`PIM_ACC_DUAL` 은 `allow_t_latch` 를 켜야 쓸 수 있다.** 기본값은 off 이고
  `pim_prog_verify` 가 T=1 을 거부한다 — ch2 에서만 검증됐고, 틀리면 에러가 아니라
  숫자이기 때문이다. ch1/ch4 에서 쓰려면 거기서 `tlatch_test` 를 먼저 돌려야 한다.
* **`ISR[35]` 이 ch1/ch4 에서도 디코드되는지 안 물어봤다.** ch2 에서만 확인했고,
  `pim_exec_config.allow_t_latch` 는 그래서 기본값이 off 다. 다른 이미지에서 쓰려면
  거기서 `tlatch_test` 를 먼저 돌려야 한다 — 틀리면 에러가 아니라 숫자다.
* **`CH_MASK` fan-out 은 이제 확인됐다.** 2채널 모두 poison 을 잃고 각자 다른 답을
  냈으므로 `fetch_decode.v` 의 `// TODO(CH_MASK)` 와 무관하게 실제로 동작한다.
  4채널에서는 아직 안 봤다.
* **성능을 안 쟀다.** 지금은 정확성만 봤다. MC 업로드 대역폭과 런치 지연은 별도.
