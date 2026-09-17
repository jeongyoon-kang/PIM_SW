# ISR opcode별 필드 사용 가이드

작성 2026-08-10. 대상 = **ISR 프로그램을 만드는 소프트웨어**.

목적: opcode 하나를 고르면 256b 워드의 **모든 필드에 무엇을 넣어야 하는지**를 한 자리에서 보게 하는 것.

## 이 문서의 위치

| 문서 | 무엇의 정본인가 |
|---|---|
| `src/front_end/emulator_controller/rtl/pim_isr_defs.vh` | **비트 위치와 opcode 값** — RTL·테스트벤치·이 문서가 전부 여기서 읽는다 |
| `docs/pim_isr_routing_spec_v2.5.md` | **규칙** — 파생 규칙(§4), 유효성 규칙 V1~V4(§5), 발행 파이프라인(§6) |
| `docs/pim_physical_timing_지침_v4.md` | 물리 타이밍 모델 전체 |
| **이 문서** | **opcode별 필드 채우는 법** (위 둘의 사용자용 요약 + 현행 RTL 확인 결과) |

숫자가 어긋나면 `pim_isr_defs.vh`가 이긴다. 규칙이 어긋나면 spec v2.5가 이긴다.

---

## 1. 공통 레이아웃 (256b, v2.7 배치)

| 비트 | 필드 | 폭 | 의미 |
|---|---|---|---|
| [5:0] | COL | 6 | row 안의 시작 beat 오프셋 (0~63) |
| [22:6] | ROW / GPR_ADDR | 17 | opcode가 해석을 정한다: MAC·EWMUL·COPY = DRAM row, WRVEC·RD_MAC = GPR 워드 주소 |
| [26:23] | BK / AF_IDX | 4 | 채널 안의 뱅크 번호. **pu_mask==0 일 때만** 쓰인다 |
| [34:27] | CH_MASK | 8 | 목적지 채널 비트마스크 |
| [35] | T | 1 | 누산기 latch(thread) 선택 |
| [48:36] | RESERVED | 13 | 전부 0 |
| [58:49] | OPSIZE | 10 | beat 개수 (최대 1023) |
| [63:59] | OPCODE | 5 | 명령 종류 |
| [77:64] | reserved | 14 | 0 |
| [93:78] | pu_mask | 16 | 연산을 실행하는 뱅크 집합 |
| [95:94] | reserved | 2 | 0 |
| [175:96] | route[16] | 80 | 뱅크 n의 출력 목적지, 5b씩. `0~15`=그 뱅크, `16`=GB, `31`=없음 |
| [191:176] | gb_mc_mask | 16 | GB가 소스일 때 받는 뱅크 집합. MAC에서는 **소싱 방식 선택자**를 겸한다 |
| [255:192] | 여유 | 64 | 0 |

주소 계산: 뱅크 내부 바이트 오프셋 = `(ROW << 11) | (COL << 5)` = row×2048 + col×32
([emulator_controller.v:1518-1520](src/front_end/emulator_controller/rtl/emulator_controller.v#L1518-L1520)).

## 2. opcode 목록

| 값 | 이름 | 확정 여부 | 뱅크 트래픽 | 한 줄 |
|---|---|---|---|---|
| 0x0C | ISR_MAC | **확정** (slide 11 all-bank MAC) | 있음 | 활성 뱅크 PU에서 MAC 누산 |
| 0x0D | ISR_EWMUL | 잠정 | 있음 | 원소별 곱, 결과를 제3 뱅크에 기록 |
| 0x0E | ISR_COPY | 잠정 | 있음 | 뱅크↔GB↔뱅크 복사 |
| 0x0F | ISR_WRVEC | 잠정 | **없음** | GPR의 벡터를 GB에 적재 |
| 0x10 | ISR_RDMAC | 잠정 | 레지스터 평면만 | 전 뱅크 누산 결과를 GPR에 회수 |
| 0x11 | ISR_EOS | 잠정 | **없음** | 커널 종료 표시 |

0x0C 외에는 잠정값이다. 값이 바뀌면 `pim_isr_defs.vh` 한 파일만 고치면 되고, RTL은 숫자를 직접 비교하지 않는다.

---

## 3. ISR_MAC (0x0C)

수리되는 형태는 **네 가지뿐**이다. 그 외 마스크는 spec §5 V3 위반이고, 하드웨어가 자동으로 쪼개주지 않는다 — 필요하면 소프트웨어가 아래 네 형태의 ISR을 여러 발로 나눠 쏜다.

**소싱 방식 선택자는 gb_mc_mask다**: `≠0` 이면 GB에서 벡터를 받고(route는 무시됨), `==0` 이면 옆 뱅크에서 받는다.

| 필드 | 단일 뱅크 | 4뱅크 (stride-4) | 16뱅크 (GB) | 16뱅크 (옆 뱅크 소싱) |
|---|---|---|---|---|
| OPCODE | 0x0C | 0x0C | 0x0C | 0x0C |
| OPSIZE | MAC 컬럼 수 | 〃 | 〃 | 〃 |
| T | thread 0/1 | 〃 | 〃 | 〃 |
| CH_MASK | 대상 채널 | 〃 | 〃 | 〃 |
| BK | 미사용 (pu_mask≠0이면) | 〃 | 〃 | 〃 |
| ROW | weight row | 〃 (4뱅크 공유) | 〃 (16뱅크 공유) | 공유 row (W·V 같은 오프셋 — 소프트웨어 약속) |
| COL | 시작 컬럼 | 〃 | 〃 | 〃 |
| pu_mask | 1-hot (16종) | `0x1111<<b`, b=0~3 | 0xFFFF | 0x5555 (짝수 8개가 실행) |
| gb_mc_mask | **pu_mask와 같은 값** | **pu_mask와 같은 값** | 0xFFFF | **0x0000** |
| route[16] | 무시됨 (NULL 권장) | 무시됨 | 무시됨 | route[2k+1]=2k (홀수가 짝수에 먹임) |
| 나머지 | 0 | 0 | 0 | 0 |

파생 결과:

| | 단일 | 4뱅크 | 16뱅크 GB | 16뱅크 peer |
|---|---|---|---|---|
| rd_mask / wr_mask | 0x0001 / 0 | 0x1111 / 0 | 0xFFFF / 0 | 0xFFFF / 0 |
| 뱅크 명령 | BK0=MAC | BK0,4,8,12=MAC | 전 뱅크 MAC | 짝수 8=MAC, 홀수 8=READ |
| rank gate 분류 | single (tFAW 슬롯 소비) | broadcast (tRRD만) | broadcast | broadcast |

**발행**: 마디 하나마다 대상 뱅크에 MAC 명령을 한 사이클 동시 fan-out. GB 소싱이면 ISR 시작 시점에 GB 읽기 포인터를 되감아(rewind) 같은 벡터를 다시 공급한다.

**주의**
- `pu_mask == 0` 으로 두면 `1<<BK` 로 정규화된다 (AiM 호환 경로). v2 확장 소프트웨어는 항상 명시해서 쓴다.
- GB에서 소싱하면 벡터 길이가 **64 beat 이하**여야 한다 (V4, GB 용량).
- 옆 뱅크 소싱은 **16뱅크 형태에서만** 허용된다 (V3). 인접쌍을 권장하지만 비인접도 동작한다.

---

## 4. ISR_EWMUL (0x0D)

**한 번에 한 조만** 수리된다 (`popcount(pu_mask) == 1`). 여러 조가 필요하면 조마다 ISR을 따로 쏜다.

한 조 = 실행 뱅크 1개 + 먹이는 뱅크 1개 + 결과를 받는 뱅크 1개. 조 구성은 route에서 유도된다:
조 = { 실행 i, 먹임 s (route[s]=i), 목적지 d (route[i]=d) }.

| 필드 | 값 | 예 (실행=BK3, 먹임=BK2, 목적지=BK1) |
|---|---|---|
| OPCODE | 0x0D | 0x0D |
| OPSIZE | 곱할 컬럼 수 | |
| T | 미사용 (누산 없음) | 0 |
| CH_MASK | 대상 채널 | |
| BK | 미사용 | 0 |
| ROW | 공유 row (W·V·목적지 같은 오프셋 — 소프트웨어 약속) | |
| COL | 시작 컬럼 | |
| pu_mask | 실행 뱅크 1개만 | 0x0008 |
| gb_mc_mask | **0x0000** | 0x0000 |
| route[16] | 먹임→실행, 실행→목적지 두 개. 나머지 31(없음) | route[2]=3, route[3]=1 |
| 나머지 | 0 | 0 |

파생: rd_mask=0x000C, wr_mask=0x0002 → BK3=EWMUL, BK2=READ, BK1=WRITE.

**발행 순서 (현행 RTL)**: 슬롯 0 = 먹임 뱅크 READ → 슬롯 1 = 목적지 뱅크 WRITE → 슬롯 2 = 실행 뱅크 EWMUL. 슬롯 사이는 tRRD 간격이고 각 슬롯이 독립된 rank gate 이벤트다
([emulator_controller.v:1476-1484](src/front_end/emulator_controller/rtl/emulator_controller.v#L1476-L1484)).

**주의**: 목적지 뱅크는 pu_mask에 들어가면 안 된다 (그러면 결과 기록이 아니라 먹이가 된다). `route[n]=n` (자기 자신)은 금지다.

---

## 5. ISR_COPY (0x0E)

모양을 정하는 별도 필드가 없다 — **route와 gb_mc_mask의 모양이 곧 종류**다. pu_mask는 항상 0.

| 필드 | read-half (뱅크→GB) | write-half (GB→뱅크) | fused (뱅크→뱅크) |
|---|---|---|---|
| OPCODE | 0x0E | 0x0E | 0x0E |
| OPSIZE | 복사 beat 수 | 복사 beat 수 (적재한 개수와 일치) | 복사 beat 수 |
| T | 0 | 0 | 0 |
| CH_MASK | 대상 채널 | 대상 채널 | 대상 채널 |
| BK | 0 | 0 | 0 |
| ROW | **소스** row | **목적지** row | 공유 row (소스=목적지 오프셋) |
| COL | 소스 컬럼 | 목적지 컬럼 | 공유 컬럼 |
| pu_mask | 0x0000 | 0x0000 | 0x0000 |
| gb_mc_mask | 0x0000 | 받을 뱅크 집합 (여러 비트면 broadcast) | 0x0000 |
| route[16] | route[소스]=16(GB), 나머지 31 | 전부 31 | route[소스]=목적지, 나머지 31 |

파생: read-half는 소스 뱅크가 READ, write-half는 목적지 뱅크(들)가 WRITE, fused는 소스=READ + 목적지=WRITE.

**주의**
- fused는 **소스와 목적지 오프셋이 같을 때만** 가능하다. 다른 row로 복사하거나 같은 뱅크 안에서 복사하려면 read-half + write-half **두 발**로 나눈다.
- GB는 순서를 보존하는 FIFO다. 순서를 바꾸는 복사는 불가능하고, 적재한 개수와 꺼내는 개수가 맞아야 한다 (V4).

---

## 6. ISR_WRVEC (0x0F)

GPR에 미리 올려둔 벡터를 GB에 적재한다. **뱅크 명령이 하나도 나가지 않는다.**

| 필드 | 값 |
|---|---|
| OPCODE | 0x0F |
| OPSIZE | GPR에서 읽을 워드 개수 (= GB에 채울 beat 수) |
| **GPR_ADDR [22:6]** | **GPR 시작 워드 번호.** 엔트리 하나 = 256b = 32B = beat 1개. 17b × 32B = 4 MiB |
| COL | 불필요, 0 |
| T | 불필요, 0 |
| CH_MASK | 대상 채널 |
| BK | 0 |
| pu_mask / gb_mc_mask / route | 전부 0 / 0 / 31(없음) |

**발행**: 디스패처가 이 ISR을 MC로 내보내는 **그 사이클에** 자기 안의 `gpr_read_bridge`를 시작한다 — 시작 주소는 [22:6], 개수는 OPSIZE. MC는 GPR 주소를 받지도 돌려주지도 않고 스트림만 받는다. MC 쪽은 GB를 비우고(flush) OPSIZE beat를 채운다.

**주의**
- GB 용량 때문에 **OPSIZE ≤ 64**를 지켜야 뒤이은 MAC이 성립한다 (V4).
- OPSIZE == 0 이면 bridge가 아예 시작하지 않는다 (무한 대기가 아니라 아무 일도 없음).
- 호스트가 **미리** GPR을 채워둬야 한다. WRVEC는 GPR을 읽기만 한다.

---

## 7. ISR_RDMAC (0x10)

전 뱅크의 누산 결과를 한 워드로 모아 GPR에 기록한다. **DRAM 트래픽이 없다** — 뱅크와 GB 사이 256b 컬럼 버스를 쓰지 않고, 별도의 16b 레지스터 평면으로 올라온다. rank gate도 소비하지 않는다.

| 필드 | 값 | 왜 |
|---|---|---|
| OPCODE | 0x10 | |
| OPSIZE | **0 고정** (V3) | 결과는 항상 집계 1 beat (16×16b) |
| T | 드레인할 thread | ※ 아래 "현행 RTL 차이" 참조 |
| CH_MASK | **1-hot 필수** (V3) | 여러 채널이 같은 GPR 주소에 겹쳐 쓰는 것을 막는다 |
| **GPR_ADDR [22:6]** | 결과가 착지할 GPR 엔트리 | WRVEC와 대칭 |
| COL | 0 | |
| BK | 0 | |
| pu_mask | **0x0000 고정** (V3) | 전 뱅크 broadcast가 이 opcode 고유의 의미다 |
| gb_mc_mask | 0x0000 | |
| route[16] | 전부 31(없음) | RES→호스트 경로는 opcode에 내포되어 있고, route에는 표현할 자리가 없다 |

**발행**: ① 16뱅크 전부에 RDMAC 명령 broadcast → ② 각 뱅크가 누산값(BF16)을 자기 lane에 캡처, 이 핸드셰이크가 곧 read-clear → ③ 16 lane이 다 차면 256b 한 beat로 MC에 올림 → ④ MC가 그 워드를 디스패처로 올리고 디스패처가 **변환 없이** GPR[GPR_ADDR]에 통째로 기록 (lane i = 뱅크 i).

**주의 — 소프트웨어 약속**
- **전 뱅크의 해당 latch가 읽을 수 있는 상태여야 한다.** MAC을 한 번도 안 돌린 뱅크가 하나라도 있으면 그 뱅크에서 멈춘다(hang). 검출 수단이 없다.
- 읽으면 지워진다(read-clear). 중간에 RDMAC을 넣으면 부분합 체크포인트 읽기가 되고, 그 시점에 누산이 리셋된다.
- 착지 주소는 결과와 함께 돌아오지 않는다. 디스패처가 이 ISR을 내보낼 때 [22:6]을 래치해 둔 값을 쓴다. ISR은 하나씩만 처리되므로(single outstanding) 래치가 다른 ISR 것일 수 없다.

---

## 8. ISR_EOS (0x11)

커널의 마지막에 놓는다. 데이터 경로도, 뱅크 명령도, rank gate 요청도 없다.

| 필드 | 값 |
|---|---|
| OPCODE | 0x11 |
| 그 외 전부 | 0 |

**존재 이유**: 이 워드가 **받아들여졌다는 사실 자체**가 "앞 ISR이 완전히 끝났다"는 신호다. MC가 이걸 받으려면 대기 상태에 도달해야 하고, ISR이 하나씩만 처리되므로 그건 직전 ISR(보통 RDMAC)이 완전히 끝났다는 뜻이다. 그래서 디스패처의 fetch 완료 신호가 커널 종료 시점으로 쓸 수 있게 된다.

**EOS를 빼먹으면** 제어권이 ISR 하나 이른 시점에 호스트로 돌아간다.

---

## 9. 필드별 — 누가 읽는가

| 필드 | 디스패처 | MC | 비고 |
|---|---|---|---|
| OPCODE | WRVEC인지만 비교 | 전부 해석 | |
| ROW / GPR_ADDR | **읽음** (결과 착지 주소 래치 + WRVEC 시작 주소) | 읽음 (뱅크 오프셋 계산) | 유일하게 양쪽이 다 읽는 필드 |
| OPSIZE | **읽음** (WRVEC 길이) | 읽음 (마디 분할) | |
| COL | — | 읽음 | |
| BK | — | 읽음 (pu_mask==0 정규화 때만) | |
| pu_mask / route / gb_mc_mask | — | 읽음 | 디스패처는 통과만 시킨다 |
| CH_MASK | — | — | **아직 아무도 안 읽는다** (단일 채널) |
| T | — | 사실상 안 읽음 | 아래 참조 |

---

## 10. 프로그램을 만들 때 소프트웨어가 지켜야 할 것

2026-07-29에 유효성 검사 회로를 fetch 경로에서 뺐다 (순수 조합 논리 2,379 LUT가 IMEM 읽기 준비 신호로 되돌아가 한 사이클 안에 고리를 닫았고, 200MHz에서 53단 18.503ns / WNS −14.136ns로 설계 전체 최악 경로였다). **지금 하드웨어에는 이걸 검사하는 주체가 없다.** 규칙을 어긴 ISR은 조용히 틀린 결과를 내거나 멈추고, 둘을 구분할 방법이 없다.

권고가 아니라 전제다:

| # | 규칙 |
|---|---|
| V1 | 각 뱅크는 최대 한 스트림의 목적지 — route 값이 겹치면 안 되고, route 값과 gb_mc_mask가 겹쳐도 안 된다 |
| V2 | 비-실행 뱅크는 입력이거나 출력이거나 둘 중 하나. 실행 뱅크는 입력 1개 이하, 출력 1개 이하. `route[n]=n` 금지 |
| V3 | opcode별 모양 — §3~§8의 각 표 |
| V4 | GB 소싱 길이 ≤ 64 beat, GB에 넣은 개수와 꺼내는 개수 일치 |
| — | CH_MASK ≠ 0 (RDMAC은 1-hot) |
| — | 프로그램 마지막은 EOS |

검사 규칙의 정본은 계류된 RTL `src/front_end/command_dispatcher/validity_gate/rtl/validity_gate.v` 이고, 되살릴 때 볼 것은 같은 디렉터리 `README.md`에 있다.

---

## 11. 현행 RTL과 문서가 어긋나는 지점

이 문서를 쓰면서 확인한 것들. **고치지 않고 기록만 한다.**

**① T 필드가 실제로는 뱅크에 도달하지 않는다.**
MC가 뱅크 명령을 조립할 때 latch 선택을 `1'b0`으로 고정한다
([emulator_controller.v:1854](src/front_end/emulator_controller/rtl/emulator_controller.v#L1854), 주석 "single thread").
따라서 MAC의 T도, RDMAC의 T도 지금은 항상 thread 0에 걸린다. 문서(spec §3a·§3e)는 T가 latch를 고른다고 되어 있다.

**② `s_isr_data[32]` — v2.7 이전 T 위치가 남아 있다.**
[emulator_controller.v:1744](src/front_end/emulator_controller/rtl/emulator_controller.v#L1744) 가 `r_rdmac_thread <= s_isr_data[32]` 로 T를 읽는데, v2.7에서 T는 [35]로 옮겨갔다. 비트 32는 지금 CH_MASK 안이다. 다만 `r_rdmac_thread`는 **어디서도 읽히지 않는** 레지스터라(선언·리셋·대입 세 곳이 전부) 동작에는 영향이 없다. `pim_isr_defs.vh` 헤더가 경고하던 "비트 숫자를 직접 적는 방식"의 잔재다.

**③ EWMUL 슬롯 순서 — 주석과 코드가 반대다.**
코드는 슬롯 0=먹임(READ), 1=목적지(WRITE), 2=실행(EWMUL) 순이다
([emulator_controller.v:1476-1484](src/front_end/emulator_controller/rtl/emulator_controller.v#L1476-L1484)).
같은 파일 위쪽 주석([1449-1466](src/front_end/emulator_controller/rtl/emulator_controller.v#L1449-L1466))은 "슬롯 1=실행, 슬롯 2=목적지"라고 적고 왜 목적지가 마지막이어야 하는지를 길게 설명한다. 코드는 그 설명과 반대로 되어 있고, 오히려 spec v2.5 §6d의 "목적지 AW 선발행"과 일치한다. **주석 쪽이 틀렸다.** 이 문서 §4는 코드를 따랐다.

**④ CH_MASK는 아직 아무도 읽지 않는다.**
단일 채널 구성이라 디스패처의 fan-out이 미구현이다 ([fetch_decode.v](src/front_end/command_dispatcher/fetch_decode/rtl/fetch_decode.v) 헤더 `TODO(CH_MASK)`). 그래도 프로그램에는 채워 넣어야 한다 — 다채널 확장 시 그대로 의미를 갖는다.

**⑤ 디스패처 헤더 주석의 `ISR[19:6]`.**
[dispatcher_top.v:42](src/front_end/command_dispatcher/dispatcher_top/rtl/dispatcher_top.v#L42), [219](src/front_end/command_dispatcher/dispatcher_top/rtl/dispatcher_top.v#L219) 가 아직 v2.5 시절 14b 폭으로 적혀 있다. 코드는 헤더 심볼로 [22:6]을 쓰므로 동작은 맞고 주석만 낡았다.
