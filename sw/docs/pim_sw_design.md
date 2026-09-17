# PIM Emulator 메모리 관리 SW 스택 설계

> FPGA 기반 PIM emulator를 위한 메모리 할당·전송·연산 SW 스택.
> user 공간에는 연속된 가상 PIM 메모리를 제공하고, 물리(카드) 주소의 비연속성은
> 중간 변환 계층이 흡수하는 구조를 설계한다.

---

## 1. 목표와 요구사항

만들고자 하는 것은 PIM emulator 위에서 동작하는 메모리 관리 SW 스택이다. 요구사항은 다음 문장으로 요약된다.

**"user process는 연속된 가상 PIM 메모리를 할당받아 쓰지만, 실제 PIM 물리 메모리에서는 연속이 아닐 수 있다. 이 간극을 변환 계층이 흡수하고, 그 위에서 PIM 연산 code를 생성할 수 있어야 한다. PIM 메모리를 쓰는 process는 여러 개일 수 있다."**

즉 kernel 영역에서 PIM 메모리를 관리하고, user 공간은 page 단위로 할당받는다. 이는 일반 OS의 가상 메모리 모델과 정확히 같은 문제 구조이므로, 먼저 일반 OS가 이 문제를 어떻게 푸는지 확인하고 그 구조를 PIM에 이식한다.

---

## 2. 배경: 일반 OS의 메모리 할당 구조

### 2.1 malloc — kernel이 아니라 user 라이브러리

malloc은 kernel API가 아니라 user 공간 라이브러리(glibc ptmalloc)다. kernel에서 큰 덩어리(heap arena)를 미리 받아두고, 작은 요청은 user 공간에서 자체적으로 쪼개서 반환한다. arena가 부족하면 `brk()`로 heap을 늘리거나, 큰 요청(기본 128KiB 이상, `MMAP_THRESHOLD`)은 내부적으로 `mmap()`을 호출한다. 즉 malloc은 "kernel 할당 위에 얹힌 sub-allocator"다.

### 2.2 mmap — kernel의 할당 단위는 page

kernel의 할당 단위는 **page(4KiB)** 다. mmap 요청은 page 단위로 올림되고 반환 주소도 page-aligned다. 중요한 사실은 mmap 시점에 kernel이 하는 일이 물리 메모리 할당이 아니라 **VMA(가상 주소 구간) 등록뿐**이라는 점이다. 물리 page는 process가 그 주소를 처음 touch해서 page fault가 났을 때 4KiB 단위로 붙는다(demand paging). 이때 물리 frame은 buddy allocator에서 오며, 가상적으로 연속인 구간도 물리적으로는 흩어져 있다.

### 2.3 교훈

| 계층 | 일반 OS | 역할 |
|------|---------|------|
| 인터페이스 | malloc (byte 단위) | user 편의 |
| 관리 단위 | page (4KiB) | kernel의 할당·보호 단위 |
| 변환 | page table + MMU | 연속 VA ↔ 비연속 PA |
| 물리 할당 | buddy allocator | 비연속 frame 공급 |

**인터페이스는 편한 단위로, 관리는 고정 크기 page로, 비연속성은 변환 계층이 흡수한다.** 이 3층 구조를 PIM 스택이 그대로 따른다.

---

## 3. 하드웨어 모델과 주소 체계

### 3.1 시스템 구성

PIM emulator는 FPGA 카드 위에 있고 host와 PCIe로 연결된다. 데이터 이동은 **AMD/Xilinx QDMA**를 사용한다. 따라서 이 시스템은 "host DRAM에 붙은 PIM"이 아니라 **discrete accelerator(GPU와 같은 부류)** 모델이며, 이 선택이 스택 전체의 형태를 결정한다.

이 모델에서는 서로 다른 세 개의 주소 공간이 등장하므로 항상 구분해야 한다.

| 주소 공간 | 누가 발급 | 용도 |
|-----------|----------|------|
| host 가상 주소 | OS (mmap/malloc) | user 코드가 쓰는 주소 |
| host 물리 주소 | OS (buddy) | host DRAM — QDMA driver만 다룸 |
| **카드 AXI 주소** | **우리 PIM driver** | **PIM 메모리의 "물리 주소"** |

이후 "PIM 물리 주소"는 전부 카드 AXI 주소를 의미한다. 카드 메모리는 host 주소 공간에 없으므로(BAR로 노출하지 않는 한) CPU가 직접 load/store할 수 없다.

### 3.2 주소 매핑: RoChBaCo와 broadcast unit

단일 채널에서는 RoBaCo(Row–Bank–Column) 매핑에 채널 단위 브로드캐스트 명령을 쓴다. 멀티 채널에서는 **RoChBaCo** 매핑을 적용한다. 비트 배치는 다음과 같다.

```
 MSB                                        LSB
 ┌────────┬─────────┬──────────┬───────────┐
 │  Row   │ Channel │   Bank   │  Column   │
 └────────┴─────────┴──────────┴───────────┘
              bit 15~   bit 11~14   bit 0~10 (row 2KiB 기준)
```

Column + Bank가 차지하는 폭이 32KiB다(예: row buffer 2KiB × 16 bank). 따라서:

* 연속 물리 주소 **32KiB** = 한 채널의 전 bank에 걸친 같은 row index 묶음 = **all-bank 연산 1회의 footprint**
* 그 위에 채널 비트가 있으므로, 채널 브로드캐스트까지 포함한 전체 연산 footprint는 **C × 32KiB** — 2채널이면 64KiB, 4채널이면 128KiB

이 C × 32KiB를 **broadcast unit**이라 부르고, SW 스택의 최소 할당 단위로 삼는다.

---

## 4. 할당 단위: 왜 2MiB chunk인가

kernel(driver)이 PIM 메모리를 나눠주는 단위는 **2MiB chunk**로 정한다. 근거는 세 가지다.

**첫째, 주소 비트 보존.** bank 비트는 bit 11~14, 채널 비트는 bit 15~16쯤에 있다. 일반 4KiB page는 offset이 bit 0~11까지만 보존되므로, buffer가 어느 bank/채널에 앉는지 통제할 수 없다. 반면 2MiB 정렬된 연속 chunk는 bit 0~20의 물리 구조를 그대로 보존하므로, **chunk 시작 주소 하나만 알면 내부 어느 offset이든 (Ro, Ch, Ba, Co) 좌표가 완전히 결정**된다. 이 성질이 user 공간에서의 주소 변환과 command 생성을 가능하게 한다.

**둘째, chunk의 완전 동질성.** 채널/bank 비트가 전부 2MiB 아래에 있으므로 어느 chunk든 전 채널 × 전 bank를 균등하게 품는다. 즉 chunk들은 완전히 fungible하고, allocator에 "이 chunk는 어느 채널" 같은 placement 지능이 전혀 필요 없다. (주의: 매핑을 바꿔 채널 비트를 2MiB 위로 올리는 실험을 하면 이 동질성이 깨지므로 allocator를 재검토해야 한다.)

**셋째, 고정 크기 = paging의 교훈.** 요청 크기에 딱 맞는 연속 공간을 주려는 방식(segmentation)은 외부 단편화 때문에, 전체 여유는 충분한데 연속 구간이 없어 할당이 실패하는 문제를 피할 수 없다. 고정 크기 단위로 관리하면 외부 단편화가 원천적으로 사라지고 **free chunk 수만 충분하면 할당은 항상 성공**한다. 흩어짐은 감수하고 상위 변환 계층이 흡수한다.

### 4.1 핵심 불변식

> **broadcast unit(64~128KiB)은 chunk(2MiB)의 약수이고, chunk는 2MiB로 정렬된다.**

이것이 보장되면 어떤 all-bank/broadcast 연산도 chunk 경계를 걸치지 않는다. 따라서 chunk들이 카드 주소 공간에서 아무리 흩어져 있어도 **연산 입장에서는 비연속성이 아예 보이지 않으며**, 비연속성이 영향을 주는 곳은 데이터 전송(pwrite 호출 횟수)뿐이다. 4채널 기준 2MiB = 128KiB × 16으로 자연히 성립한다. 채널을 더 늘려 unit이 chunk보다 커지는 구성이 나오면 chunk 크기를 함께 키워야 한다.

---

## 5. 전체 스택 구조

```
┌────────────────────────────────────────────────────────────┐
│ Application                                                │
│   pim_alloc / pim_free / pim_memcpy / pim_op / pim_sync    │
├────────────────────────────────────────────────────────────┤
│ libpim (user-space runtime)                                │
│   · 가상 PIM 주소 발급 (user에겐 연속으로 보임)              │
│   · chunk pool + unit 단위 sub-allocator (fast path)       │
│   · unit 변환 테이블: 가상 PIM 주소 → 카드 주소             │
│   · (Ro,Ch,Ba,Co) 계산 → PIM command 생성                  │
├──────────────┬──────────────────┬──────────────────────────┤
│ /dev/pim     │ /dev/qdma-MM-<q> │ command 제출 경로 (§9.1)  │
│ (자체 driver) │ (기존 QDMA drv)  │ ioctl 대행 or memory-ring │
│ chunk 장부    │ 데이터 이동       │ ※ BAR를 user에 직접       │
│ (slow path)  │ (pin+SG 자동)    │   매핑하지 않음 (결정)     │
├──────────────┴──────────────────┴──────────────────────────┤
│ FPGA: QDMA IP ── AXI ── PIM emulator + 메모리 컨트롤러      │
│        (메모리 컨트롤러가 선형 AXI 주소를 RoChBaCo로 해석)   │
└────────────────────────────────────────────────────────────┘
```

역할 분담의 원칙: **kernel driver는 "2MiB chunk를 나눠주는 장부 기계"일 뿐, broadcast unit이나 RoChBaCo 같은 PIM geometry를 전혀 모른다.** geometry 해석은 전부 libpim의 몫이다. 이 관심사 분리 덕분에 채널 수나 매핑 정책을 바꾸는 실험을 할 때 driver는 그대로 두고 `GET_INFO` 파라미터와 libpim의 계산식만 바꾸면 된다.

일반 OS와의 대응 관계는 다음과 같다.

| 일반 OS | PIM 스택 |
|---------|---------|
| malloc (byte) | pim_alloc (unit 올림) |
| page 4KiB | broadcast unit 64~128KiB |
| buddy allocator | driver의 chunk 비트맵 |
| page table + MMU | libpim의 unit 변환 테이블 |
| swap 가능한 page | 카드 메모리 (주소 불변, pinning 불필요) |

---

## 6. Kernel driver (/dev/pim)

### 6.1 하는 일과 하지 않는 일

QDMA 모델에서 이 driver의 일은 정확히 하나다: **카드 AXI 주소 공간의 2MiB chunk allocator + per-process 장부 + 종료 시 회수.** 순수한 장부 관리라 코드가 매우 얇다. 이 역할을 user-space daemon으로 뺄 수도 있으나, process가 crash해도 kernel의 `release()` 콜백으로 확실히 회수된다는 점에서 kernel module이 낫다.

**설계 대안 기록 — mmap 직접 매핑 모델은 어디로 갔는가.** 논의 초기에 이 스택은 host-attached 모델로 출발했다: driver의 mmap 핸들러가 `remap_pfn_range()`로 PIM 물리 page들을 process page table에 써넣고, CPU가 load/store로 PIM 메모리에 직접 접근하는 구조다. 그 모델에서 mmap은 세 가지 역할을 맡고 있었다.

| mmap의 역할 (host-attached 모델) | QDMA 모델에서의 대체물 |
|----------------------------------|------------------------|
| CPU load/store 직접 접근 — 매핑된 주소에 쓰는 행위가 곧 offloading | QDMA pread/pwrite (`pim_memcpy`) — **SW memcpy 폴백 없이 처음부터 QDMA 경로만 사용** (결정사항) |
| page table을 통한 VA→PA 변환 확립 | libpim의 unit 변환 테이블 (§7.2) — 가상 PIM 주소는 libpim이 발급 |
| `pa = chunk.pa + (va − chunk.va)` 역산의 근거 | `ALLOC_CHUNKS`가 반환하는 카드 주소 base (§6.4) |

카드 메모리가 host 주소 공간에 없다는 사실(§3.1)이 첫 번째 역할을 지웠고, 남을 수 있었던 마지막 소비자인 control BAR 접근도 **"BAR를 user 공간에 직접 매핑하지 않는다"는 결정**으로 소멸했다. 따라서 `.mmap`은 구현하지 않으며, 이는 생략이 아니라 세 역할 모두에 명시적 대체물이 있다는 확인의 결과다. 부활 조건도 명확하다: 훗날 카드 메모리 일부를 BAR 데이터 윈도로 노출하기로 결정하면, 그때 `io_remap_pfn_range` 기반 `.mmap`과 초기 모델의 논의가 그대로 되살아난다.

### 6.2 multi-process 지원

character device는 여러 process가 동시에 열 수 있다. process마다 별도의 `struct file`이 생기고, driver는 `open()` 콜백에서 `file->private_data`에 per-process context(할당 chunk 목록)를 달아둔다. 이후 ioctl마다 어느 process의 요청인지 자연히 알 수 있고, fd가 닫히면(정상 종료든 crash든) `release()`에서 그 process의 chunk를 전부 회수한다 — leak이 구조적으로 불가능하다.

주의점 두 가지:

1. **동시성.** syscall은 "그 process가 자기 문맥으로 kernel 코드를 실행"하는 것이므로, 여러 CPU가 driver 함수를 동시에 실행할 수 있다. 공유 자원인 chunk 비트맵은 mutex로 보호해야 한다. ioctl/open/release는 모두 process context라 sleep 가능한 mutex가 정답이며, spinlock은 (나중에 interrupt 처리가 생기기 전까지) 불필요하다. lock이 사실상 하나뿐이라 deadlock 여지도 없다.
2. **fork.** fd와 매핑이 자식에게 상속되면 private_data 하나를 두 process가 참조하게 된다. "fork 후에는 직접 다시 open하라"를 API 규칙으로 삼는다.

### 6.3 할당 알고리즘: 비트맵 + 2단 first-fit

세 개념의 역할을 명확히 구분한다.

* **비트맵 = 장부.** chunk별 free/used 기록. 어떤 정책을 쓰든 필요한 기본 전제이며, 그 자체는 최적화가 아니다. (free-list로도 장부를 만들 수 있으나 인접성 탐색이 불가능해 비트맵을 선택.)
* **first-fit = 탐색의 본질.** 앞에서부터 훑다가 처음 맞는 것을 취한다.
* **연속 run 우선 = 성능 정책.** 2MiB 조각 수만큼 QDMA 호출이 늘어나므로, 이를 줄이기 위해 낱개를 줍기 전에 "붙어 있는 n칸"을 먼저 찾는다.

```c
static DEFINE_MUTEX(pim_lock);
static unsigned long chunk_bitmap[BITS_TO_LONGS(NR_CHUNKS)];

static int pim_alloc_chunks(unsigned int n, u32 *out_idx)
{
    mutex_lock(&pim_lock);

    /* 1단계: 연속으로 빈 n칸 run을 first-fit으로 (성공 시 extent 1개) */
    start = bitmap_find_next_zero_area(chunk_bitmap, NR_CHUNKS, 0, n, 0);
    if (start < NR_CHUNKS) { bitmap_set(chunk_bitmap, start, n); ... }

    /* 2단계: 없으면 흩어진 빈 칸을 낱개 first-fit으로 폴백 */
    else { for_each_clear_bit(...) { ... } }

    mutex_unlock(&pim_lock);
}
```

두 단계의 차이는 자료구조도 탐색 순서도 아니고, **"연속 조건을 요구하느냐 버리느냐"** 하나다. 1단계는 성능(extent 1개 = pwrite 1회)을 위한 시도, 2단계는 가용성(총량만 있으면 무조건 성공)의 보장이다. 규모 감각: PIM 1GiB = chunk 512개 = 비트맵 64바이트이므로 선형 스캔 비용은 무시할 수준이며, 할당은 어차피 slow path에서만 발생한다. v1에서는 2단계만 구현해도 무방하다 — 1단계는 순수 성능 최적화이고, 인터페이스를 extent 배열로 잡아두면 상위 계층은 아무것도 바뀌지 않는다.

### 6.4 ioctl 인터페이스

```c
ioctl(fd, PIM_IOC_GET_INFO,     &info);   /* 채널/bank 수, row 크기 등 geometry */
ioctl(fd, PIM_IOC_ALLOC_CHUNKS, &req);    /* in: n, vbase(§7.3) → out: {start, count} extent 배열 */
ioctl(fd, PIM_IOC_FREE_CHUNKS,  &req);
```

`ALLOC_CHUNKS`의 입력에는 libpim이 예약한 vbase(§7.3)를 실을 수 있는 필드가 있다(선택, 0 = 미등록). baseline에서 kernel은 이 값을 기록만 하며, 용도와 우선순위는 §7.3 참조.

**geometry의 출처 (결정: module parameter).** 현재 HW에는 채널 수 등을 알려주는 info register가 없으므로, geometry는 conf 파일 → 적재 스크립트 → **module parameter** 경로로 driver에 주입한다.

```c
module_param(pim_channels, uint, 0444);   /* insmod pim.ko pim_channels=4 ... */
module_param(pim_banks,    uint, 0444);   /* 또는 /etc/modprobe.d/pim.conf의 options 줄 */
module_param(pim_row_size, uint, 0444);   /* /sys/module/pim/parameters/에서 확인 가능 */
```

driver는 init 때 이 값으로 unit 크기 계산과 §4.1 불변식(unit이 chunk의 약수) 검증을 수행하고, libpim은 GET_INFO로만 값을 받는다 — **libpim은 conf 파일의 존재를 모른다.** 이로써 진실의 출처가 insmod 인자 한 곳으로 수렴하여, driver와 libpim이 서로 다른 구성으로 빌드되는 불일치 사고가 구조적으로 불가능해지고, 양쪽 모두 재컴파일 없이 채널 구성 실험이 가능하다. 훗날 HW가 info register(또는 카드 메모리 고정 주소에 bitstream이 박아두는 geometry 블록 — conf와 실제 bitstream의 불일치를 적재 시점에 잡는 안전판)를 제공하면 driver init의 값 획득부만 교체하며, GET_INFO 인터페이스와 libpim은 변경되지 않는다.

`ALLOC_CHUNKS`가 chunk의 **카드 주소**(= `pim_base + idx * 2MiB`)를 반환하는 것이 핵심이다. §4의 주소 비트 보존 덕분에 libpim은 이 base 주소만으로 이후 모든 주소 변환을 kernel 왕복 없이 수행한다. 반환 형식을 extent 배열로 하면 1단계 성공 시 결과가 extent 1개로 압축된다.

운영 편의로 chunk별 소유자 배열(`u16 owner[NR_CHUNKS]`, leak 디버깅과 `/proc/pim` 상태 조회용)과 fd당 chunk quota(선택)를 둘 수 있다.

### 6.5 file_operations 구성

위 설계를 fops로 옮기면 다음이 전부다.

```c
static const struct file_operations pim_fops = {
    .owner          = THIS_MODULE,
    .open           = pim_open,        /* per-fd ctx 생성 (§6.2) */
    .release        = pim_release,     /* chunk 전량 회수 — crash 포함 보장 (§6.2) */
    .unlocked_ioctl = pim_ioctl,       /* GET_INFO / ALLOC / FREE dispatch (§6.4) */
};
```

`.release`가 이 driver에서 가장 중요한 함수다. fd가 닫히는 모든 경로(정상 close, exit, crash, kill -9)에서 kernel이 호출을 보장하므로, leak 불가능 보장이 여기서 나온다. ioctl 핸들러에서 user 포인터는 반드시 `copy_from_user`/`copy_to_user`로만 접근하고, 구조체 필드를 8바이트 정렬로 맞춰두면 32-bit user 지원(`.compat_ioctl`)이 필요해져도 같은 핸들러를 재사용할 수 있다.

**의도적으로 구현하지 않는 것들**이 오히려 정보량이 크다. `.read`/`.write`는 데이터가 전부 QDMA char device로 흐르므로 불필요하고, `.mmap`은 §6.1의 대안 기록대로 소비자가 소멸했으며, `.llseek`도 의미가 없다. 향후 command 제출을 ioctl 대행(§9.1의 방안 A)으로 확정하면 `SUBMIT`/`WAIT` ioctl이 §6.4에 추가되고, 완료 통지를 interrupt 기반으로 가면 `.poll`이 추가 후보가 된다.

등록은 `register_chrdev` + class/device 생성의 구식 절차 대신 **`misc_register`(miscdevice)** 를 쓴다. 구조체 하나로 `/dev/pim` 노드 생성, minor 할당, udev 연동까지 자동이며, 단일 노드 장부형 driver의 표준 선택이다.

---

## 7. libpim (user-space runtime)

### 7.1 2단 구조: malloc의 구조를 그대로

pim_alloc이 매번 kernel에 가면 syscall 비용이 크므로, malloc과 동일한 2단 구조를 쓴다.

* **fast path**: pool에 여유가 있으면 user 공간에서 즉시 반환. syscall 없음.
* **slow path**: 부족할 때만 `ioctl(ALLOC_CHUNKS)`로 chunk를 받아 pool에 등록.

driver와의 소통은 pool을 채우고 비우는 순간에만 발생한다.

### 7.2 자료구조

kernel의 구조가 한 층 아래에서 반복된다: kernel이 (PIM 영역 → chunk) 비트맵이라면, libpim은 (chunk → unit) 비트맵이다. 4채널 기준 chunk 하나 = 128KiB unit 16개이므로 비트맵이 `u16` 하나에 들어간다.

```c
struct pim_chunk {
    uint64_t card_addr;        /* driver가 준 카드 주소 base */
    uint16_t unit_bitmap;      /* 16 unit의 free/used */
    uint8_t  free_units;
};

struct pim_pool {
    struct pim_chunk *chunks;  /* 동적 배열 */
    pthread_mutex_t   lock;    /* 멀티스레드 app 대비 */
};

struct pim_allocation {        /* pim_alloc 반환값의 실체 */
    void    *vbase;            /* PROT_NONE 예약 구간의 시작 주소 (§7.3) */
    uint32_t nr_units;
    uint64_t unit_card_addr[]; /* unit별 카드 주소 — 사실상 page table */
};
```

**가상 PIM 주소의 정체**: 카드 메모리는 CPU로 역참조할 수 없으므로(cudaMalloc의 device pointer와 같은 처지), pim_alloc이 주는 주소는 접근 가능한 메모리를 가리키지 않는 순수 좌표다. 그 발급 방식은 §7.3에서 정의한다. user에게는 처음 요구사항 그대로 "연속된 PIM 메모리"로 보이고, 내부에서 `unit_card_addr[]` 테이블이 변환을 맡는다. **일반 OS에서 MMU + page table이 하던 역할을, 이 스택에서는 libpim의 unit 테이블이 맡는 것이다** — unit(128KiB)이 page 크기인 page table과 동형이다. 변환은 `unit_idx = offset >> 17` 후 배열 조회로 O(1)이다.

### 7.3 pim_alloc의 반환 형식 — 주소 공간 예약 (PROT_NONE)

pim_alloc은 무엇을 반환해야 하는가. 후보는 셋이다.

| 후보 | 형식 | 문제 |
|---|---|---|
| A | `typedef uint64_t pim_ptr` (CUDA 초기 방식) | 지어낸 값이 **진짜 host 주소와 겹칠 수 있다.** 실수로 역참조/`memcpy`하면 segfault가 아니라 자기 heap을 **조용히 오염** — 최악의 실패 모드 |
| B | `typedef struct { uint64_t v; } pim_ptr` 강타입 | 컴파일 타임에 host 포인터와의 혼용을 차단하지만, 산술마다 헬퍼 함수 강제 |
| **C** | **host 주소 공간 예약 (채택)** | 아래 참조 |

**채택안: 지어내는 대신, process 주소 공간에서 접근 불가능한 구간을 실제로 예약해 그 주소를 쓴다.**

```c
/* slow path에서 chunk group을 받을 때 */
void *va = mmap(NULL, group_size, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
```

**핵심 이해: 이것은 host "메모리" 할당이 아니라 host "주소 공간" 할당이다.** 일반 anonymous mmap의 lazy allocation은 물리 page 붙이기를 첫 touch까지 미루는 것이지만, PROT_NONE은 touch 시 fault의 결말이 다르다 — kernel이 물리 page를 붙이는 대신 SIGSEGV를 보낸다. 즉 demand paging이 성공할 경로 자체가 없어, 물리 메모리 소비는 지연되는 것이 아니라 **0으로 못박혀 있다.** 실제 소비 자원은 물리 메모리 0, page table entry 0(성공하는 fault가 없으므로 PTE가 만들어지지 않음), kernel VMA 트리의 노드 하나(수십 바이트, `vm.max_map_count` 한 칸)가 전부다. mmap을 메모리 할당기가 아니라 **주소 공간 할당기**로 쓰는 이 관용구는 guard page, CUDA UVA, ASan shadow 예약과 같은 부류의 표준 기법이다.

이 선택 하나가 필요한 성질 네 가지를 동시에 준다.

1. **평범한 `void*` 반환.** 배열 인덱싱, 포인터 산술, `%p` 출력이 자연스럽고, 실패 시 NULL + errno라는 malloc과 동일한 계약을 쓴다. `float *a = pim_alloc(n * sizeof(float));`가 그대로 성립한다.
2. **pim 주소끼리 절대 겹치지 않는다.** mmap은 시작 주소 하나가 아니라 **[va, va+size) 구간 전체를 VMA로 등록**한다(`/proc/self/maps`에 구간 한 줄로 표시). 다음 mmap은 기존 구간의 시작점이 아니라 **구간 전체**를 피해 배정되므로, "이전 va + 4MiB 안쪽에 새 va가 배정되는" 일은 VMA 관리자 수준에서 불가능하다. 유일성 관리를, 그 일이 본업인 kernel VMA 관리자에게 위임한 것이다. 스레드 동시 호출도 mmap_lock이 직렬화한다.
3. **host 주소와도 겹치지 않는다.** pim 주소가 그 자체로 이 process의 VMA이므로, heap/stack/라이브러리와의 충돌이 구조적으로 배제된다 — 후보 A의 치명적 약점이 해소된다.
4. **오용이 즉시, 시끄럽게 실패한다.** 실수로 역참조하면 그 자리에서 SIGSEGV — 조용한 오염 대신 정확한 위치의 조기 실패를 얻는다.

**정렬 보정.** mmap은 page(4KiB) 정렬만 보장하므로, `group_size + 2MiB`를 예약한 뒤 앞뒤 자투리를 munmap으로 쳐내 vbase를 2MiB 정렬로 만든다(남긴 구간은 여전히 VMA이므로 보호 유지). 그러면 va의 하위 21비트 = 카드 주소의 하위 21비트가 되어, 디버깅 시 va만 보고 (Ch, Ba, Co) 좌표를 읽을 수 있다.

**kernel 등록 (선택 사항).** `ALLOC_CHUNKS` 구조체에 vbase 입력 필드 자리를 두되, baseline에서 값 전달은 **선택**이다(0 = 미등록). libpim이 카드 주소를 직접 아는 baseline에서 kernel 장부의 vbase는 어떤 동작에도 쓰이지 않으므로 기능적 필요성이 없고, 격리 전환 시점에 lib·driver를 함께 바꾸며 필수로 승격하면 된다(소급 가능한 종류의 준비 — 별도 문서 §7-1). 지금 채워 넣을 때의 실익은 디버깅 가시성(`/proc/pim`이 va↔chunk 대응을 application 포인터와 바로 대조되는 형태로 출력)과 libpim 버그 조기 검출(같은 fd 내 vbase 구간 겹침 감지) 두 가지다.

**process 간 동일 va.** 서로 다른 process가 우연히 같은 va 값을 가질 수 있으나 문제가 아니다 — 장부가 per-fd이므로 **모든 조회는 호출한 fd의 장부 안에서만** 일어나고, 실질 키는 (process, va) 쌍이다. CPU MMU가 같은 VA를 현재 process의 page table(CR3)로 해석하는 것과 동형 구조이며, "fd = process" 등식은 §6.2의 "fork 후 재open" 규칙 위에 서 있다. user가 이상한 vbase를 신고해도 자기 장부 안의 이름표가 꼬일 뿐 남의 chunk를 지칭할 수단이 되지 않는다(남의 장부는 조회 대상이 아님).

**해제 순서.** `pim_free`는 반드시 libpim 테이블(및 kernel 장부) 엔트리 제거 → munmap 순서를 지킨다. munmap 이후 그 va 구간은 다른 mmap(malloc 내부 포함)이 재사용할 수 있는데(malloc의 주소 재사용과 같은 정상 시맨틱), 순서가 뒤집히면 재사용된 va가 옛 엔트리로 변환되는 stale 조회 창이 생긴다. 참고로 훗날 `pim_realloc`이 필요해지면 "새 구간 예약 + 테이블 이사"로 구현한다 — 카드 쪽 chunk는 움직일 필요가 없으므로 **데이터 복사 없이** va만 갈아끼우면 된다.

### 7.4 할당 정책: fill-partial-first

예시: 4MiB(chunk 2개, unit 32개)를 받아 3MiB만 pim_alloc했다면 chunk B에 unit 8개(1MiB)가 남는다. 다음 요청이 1MiB 이하면 새로 받아올 필요 없이 이 자투리에서 준다.

```
chunk A [1111 1111 1111 1111]   ← 3MiB 할당이 A를 채우고
chunk B [1111 1111 0000 0000]   ← B의 절반만 사용, 8 unit 여유
        → 다음 512KiB 요청: B의 빈 unit 4개로 즉시 처리 (ioctl 없음)
```

탐색 순서는 kernel과 같은 "연속 run 우선 → 낱개 폴백"이되, **정책이 하나 추가된다.** chunk는 unit이 하나라도 쓰이면 kernel에 반납할 수 없으므로, 새 할당을 빈 chunk에 뿌리지 말고 **이미 부분 사용 중인 chunk부터 채운다(fill-partial-first).** 완전히 빈 chunk를 최대한 확보해 반납 가능성을 높이는 정책이다. 전체 순서:

1. 부분 사용 chunk들에서 연속 unit run 탐색 (재사용 + 인접성)
2. 없으면 부분 사용 chunk들의 낱개 unit 폴백
3. 그래도 모자라면 ioctl로 새 chunk (잔여량을 chunk 수로 올림)

**반납에는 hysteresis를 둔다.** chunk가 완전히 비는 즉시 반납하면 alloc/free 반복 워크로드에서 ioctl이 널뛴다. malloc의 trim threshold처럼 "빈 chunk N개(2~4개)까지 캐싱, 초과분만 반납"으로 한다. 종료 시엔 driver release()가 회수하므로 leak 걱정은 없다.

**unit보다 작은 요청은 unit으로 올림한다.** all-bank 연산은 broadcast unit 전체를 건드리므로, 한 unit에 서로 다른 buffer가 섞이면 연산을 걸 수 없다. 올림은 내부 단편화를 감수하는 대신 안전을 보장한다(GPU의 CUDA VMM이 2MiB granularity를 강제하는 것과 같은 논리). 작은 buffer가 정말 많아지면 "unit을 공유하되 all-bank 연산 금지"인 `PIM_ALLOC_SMALL` pool을 후순위로 추가한다.

---

## 8. 데이터 이동 (pim_memcpy)

### 8.1 두 종류의 비연속성, 두 명의 담당자

host buffer → PIM 전송에는 비연속성이 양쪽에 존재하는데, 담당자가 다르다.

**host 쪽 (source): QDMA driver가 알아서 처리한다.** `pwrite(qdma_fd, host_buf, size, card_addr)`에 user 가상 주소를 그대로 넘기면, QDMA kernel driver가 내부에서 page들을 pin하고 흩어진 host 물리 page들로 scatter-gather descriptor 체인을 만들어 하드웨어에 건다. host 물리가 512조각이어도 pwrite는 1회다. pinning/SG/staging을 우리가 구현할 필요가 전혀 없다.

**카드 쪽 (destination): libpim이 extent 단위로 쪼갠다.** pwrite의 offset은 단일 시작 주소이므로, 한 번의 호출은 카드 주소가 연속인 구간 하나만 커버한다. 할당이 흩어진 chunk/unit으로 구성됐다면 libpim이 `unit_card_addr[]`에서 연속 구간(extent)끼리 합쳐 extent 개수만큼 pwrite한다.

각자 자기 쪽 비연속성만 책임지는 대칭 구조다.

### 8.2 예시: 흩어진 4MiB 전송

`pim_alloc(4MiB)`이 비인접 chunk 2개(카드 주소 `0x0020_0000`, `0x00A0_0000`)로 채워졌다면:

```
가상 PIM 주소            카드 주소
[P + 0MiB, P + 2MiB) → 0x0020_0000   (chunk 1)
[P + 2MiB, P + 4MiB) → 0x00A0_0000   (chunk 5)

pim_memcpy_to(P, host_buf, 4MiB) 의 내부:
  pwrite(qdma_fd, host_buf,        2MiB, 0x00200000);
  pwrite(qdma_fd, host_buf + 2MiB, 2MiB, 0x00A00000);
```

### 8.3 비용과 최적화

쪼개기의 비용은 작다. 조각 하나가 최소 2MiB(chunk 단위 흩어짐) 또는 128KiB(unit 단위 흩어짐)인데, PCIe Gen3~4에서 2MiB 전송은 수백 µs, pwrite 1회의 syscall + descriptor 세팅은 수 µs다. 즉 **오버헤드는 전송 시간의 ~1% 수준**이며, 이것이 "연속성은 correctness가 아니라 성능 힌트"라는 설계 철학의 정량적 근거다. 더 줄이려면 조각들을 순차가 아니라 aio(`io_submit`)로 동시에 걸어 PCIe에서 겹치게 한다. compaction(데이터를 옮겨 연속으로 만들기)은 이 수치상 할 이유가 없다.

### 8.4 API 경계

`pim_memcpy`의 내부는 **처음부터 QDMA 경로만 사용한다** — mmap 직접 쓰기나 SW memcpy 폴백은 두지 않는다(결정사항, §6.1 대안 기록 참조). user에게는 인터페이스만 고정 노출하고, 내부의 진화(동기 단일 queue → 다중 queue 병렬 제출 등)는 캡슐화한다.

```c
pim_memcpy(dst, src, size, PIM_TO_DEV, /* flags: 현재 0만 허용, 확장 예약 */ 0);
```

### 8.5 multi-process와 queue

QDMA는 physical function당 수백~2048개 queue를 지원한다. process(libpim 인스턴스)마다 전용 MM queue를 배정하면 전송 경로에서 process 간 간섭이 없다.

---

## 9. PIM 연산 경로 (pim_op)

### 9.1 command 생성

연산 요청이 가상 PIM 주소로 들어오면 kernel 왕복 없이 user 공간에서 command가 완성된다.

```
pim_op(dst_va, src_va, ...)
  → unit 테이블 조회: card_addr = unit_card_addr[offset >> 17] + (offset & MASK)
  → 카드 주소의 비트 슬라이스: (Ro, Ch, Ba, Co)
  → PIM command 시퀀스 생성
  → command 제출 경로로 발행 (아래 두 방안 중 택일)
```

§4.1의 불변식 덕분에 어떤 broadcast unit도 chunk 경계에 걸치지 않으므로, 할당이 물리적으로 흩어져 있어도 연산 correctness에는 영향이 없다. 큰 할당 전체에 연산을 걸면 어차피 unit 개수만큼의 command 시퀀스로 풀리며, 흩어진 경우 각 unit의 row 주소가 달라질 뿐 command 개수는 동일하다.

**command 제출 경로 — control BAR를 user에 직접 매핑하지 않기로 결정**했으므로(임의 process의 레지스터 접근 차단, multi-process 중재의 kernel 집중), 제출은 다음 두 방안 중 하나를 거친다.

**방안 A: kernel 대행 ioctl.** driver가 init 때 control BAR를 kernel 공간에만 `ioremap`해 두고, user는 `PIM_IOC_SUBMIT`으로 command를 건네면 driver가 레지스터에 대신 쓴다. 이때 command는 driver에게 **불투명한 바이트열**로 유지한다 — (Ro,Ch,Ba,Co) 계산과 인코딩은 여전히 libpim이 하며, "kernel은 geometry를 모른다"는 §5의 원칙이 지켜진다. syscall이 제출마다 붙으므로 SUBMIT은 단건이 아니라 **배열 단위 배치 제출**로 설계한다(큰 할당의 연산은 어차피 unit 개수만큼의 command 시퀀스라 배치가 자연스럽다). 함정 하나: 해당 PCIe function에는 이미 QDMA driver가 바인딩되어 있어 pim driver가 정식 PCI driver로 붙을 수 없다 — `pci_get_device()` + `pci_resource_start()`로 BAR 주소를 얻어 필요한 구간만 ioremap하는 우회(emulator 용도로 충분)나, FPGA에서 control용 PF/VF를 분리하는 정석 중 택일한다.

**방안 B: memory-ring — 레지스터를 아예 쓰지 않는다.** command ring buffer를 카드 메모리의 고정 영역에 두고, host가 QDMA pwrite로 command와 tail을 밀어넣으면 emulator가 tail을 폴링해 command를 집어간다. 완료 상태도 카드 메모리의 status 위치를 host가 QDMA pread로 폴링한다. BAR 접근이 host 어디에도 없으므로 pim driver는 순수 장부(§6.5의 fops 3개)로 유지되고 FPGA 쪽도 레지스터 인터페이스가 불필요하다. 대가는 제출·폴링이 QDMA 왕복이라는 지연인데, emulator의 목적이 절대 지연이 아니라면 단순성의 이득이 크다. **emulator 단계의 기본 후보는 방안 B**이며, 지연이 문제 되면 방안 A의 SUBMIT ioctl을 추가하는 진화 경로가 열려 있다(§11).

### 9.2 데이터 레이아웃 주의

연속 카드 주소 32KiB는 bank0의 row 2KiB → bank1의 row 2KiB → … 순서로 깔린다. 따라서 all-bank 연산이 "같은 column offset을 전 bank에서 동시에" 처리할 때, user buffer 관점에서는 **2KiB stride로 떨어진 원소들이 한 SIMD lane 묶음**이 된다. user가 배열을 자연스럽게 연속으로 채우면 병렬 처리 순서가 원소 순서와 어긋난다. 해법은 bank-interleave 변환 복사(`pim_memcpy_tiled`)를 제공하거나, 레이아웃을 그대로 노출하고 문서화하는 것 — 연산 시맨틱 설계와 함께 결정한다.

### 9.3 일관성 (pim_sync)

초기 mmap 모델에서는 "CPU가 cacheable 매핑에 써둔 데이터가 캐시에만 있는 상태에서 카드가 메모리를 읽는" stale 위험이 있었고, 이것이 `pim_sync()`(cache flush)의 원래 존재 이유였다. 그러나 **mmap 배제 + QDMA 전용 결정으로 이 위험은 구조적으로 소멸한다** — CPU가 카드 메모리를 직접 쓰는 경로 자체가 없으므로 host 캐시에 카드 데이터의 사본이 존재할 수 없고, host buffer와 DMA 간 일관성은 QDMA driver의 DMA 매핑이 처리한다(x86은 DMA 캐시 일관).

남는 것은 캐시가 아니라 **순서 보장**이다: 전송 완료 이전에 그 영역에 대한 연산이 제출되면 안 된다. 동기 `pim_memcpy`가 모든 조각의 완료를 기다린 후 반환하는 구조가 이 happens-before를 담당한다. `pim_sync()`는 cache flush가 아니라 fence(제출된 전송·연산의 완료 대기) 의미로 재정의해 예약해 두면, 향후 비동기 API나 BAR 데이터 윈도가 도입될 때 그대로 쓸 수 있다.

---

### 9.4 GPR — 두 번째 device memory pool

emulator에는 장치 전역 4MiB의 GPR 영역이 FPGA에 있다. host와 채널 내부 구조 사이의 staging 메모리로, dataflow에서 두 방향으로 쓰인다: WRVEC 명령이 GPR의 값을 각 채널의 global buffer(GB)에 multicast하여 shared operand(GEMV의 벡터 등)를 공급하고, RDMAC 명령이 각 채널의 MAC 결과를 unicast로 읽어와 GPR에 기록한다. AiM SDK의 SW 스택에서 "GPR in FPGA"가 device memory의 한 축으로 allocator 아래 놓이는 것과 같은 구도다.

GPR은 스크래치가 아니라 메모리다 — 그래서 관리가 필요하다. vector 등의 operand는 한 번 올려두고 수많은 WRVEC이 재사용하는 상주 데이터이고, result 버퍼도 호출자가 잡아둔 곳에 쌓인다. 즉 GPR 안에는 서로 다른 시점에 생기고 죽는 여러 객체가 동시에 살아 있으며 수명을 아는 주체는 application이다 — malloc의 문제 정의 그대로이므로, GPR은 DRAM과 나란한 두 번째 device memory pool로 관리한다.

| | DRAM pool (§6–7) | GPR pool |
|---|---|---|
| 크기 | 수 GiB | 4MiB |
| 담는 것 | weight matrix류 대형 데이터 | vector, result 버퍼 등 소형 객체 |
| 물리 배치 | RoChBaCo interleave | 선형 (interleave 없음) |
| kernel 관리 단위 | 2MiB chunk | 4KiB page (비트맵 1024칸) |
| libpim 하한 | 128KiB broadcast unit | 4KiB로 올림 (v1) |

**allocator: 같은 패턴, 다른 상수.** kernel은 §6.3의 코드 형태(비트맵 + 2단 first-fit + per-fd 장부 + release 회수)를 상수만 바꿔 GPR용으로 인스턴스화한다. GPR은 interleave가 없어 연속 run의 이득(전송 조각 감소)도 동일하게 성립한다. libpim은 v1에서 4KiB page 올림으로 단순화한다 — vector/result의 크기 분포가 아직 알려져 있지 않으므로 분포에 대한 가정(size-class SLAB 등)을 설계에 굽지 않고, 프로파일이 소형 객체 압력을 보여줄 때 sub-page 층을 추가한다(§11). bias류는 현 단계에서 비고려.

**주소 체계는 DRAM pool과 완전히 통합된다.** PROT_NONE 예약(§7.3)을 GPR 할당에도 동일하게 적용하고, libpim 변환 테이블 엔트리에 region 태그를 더해 `pim_translate(va)`가 (DRAM, 카드 주소) 또는 (GPR, offset)을 반환하게 한다. 전면 API는 위치를 명시적으로 받는다:

```c
void *w = pim_alloc(sz, PIM_MEM_DRAM);   /* weight matrix */
void *v = pim_alloc(sz, PIM_MEM_GPR);    /* vector, result 버퍼 */
```

host 전송도 별도 API 없이 기존 `pim_memcpy`가 GPR va를 그대로 받는다 — 선형 영역이라 extent 계산이 자명할 뿐, 같은 QDMA 경로다. 카드 주소 지도에는 `gpr_base`/`gpr_size`가 module parameter(§6.4)로 추가되어 chunk allocator의 data 영역과 분리된다.

**수명·동시성 계약.** GPR 내용은 메모리처럼 persist하며 수명은 alloc/free가 지배한다(launch 경계 무효화 같은 조항 없음). process/stream이 각자 자기 operand·result 버퍼를 할당하므로 GPR 수준에서는 서로 침범하지 않는다. 진짜 공유 가변 상태는 채널당 GB인데, GB는 host 직접 접근 경로가 없고 오직 WRVEC/RDMAC 명령으로만 읽고 쓰이므로(확정), allocator의 대상이 아니라 command 스케줄링의 관할이다 — "WRVEC으로 GB에 올린 operand를 어느 연산들이 소비하고 언제 다음 WRVEC이 덮어써도 되는가"는 codegen/스케줄러의 의존성 문제로 다룬다.

---

## 10. 설계 원칙 요약

1. **인터페이스는 연속, 관리는 고정 단위, 비연속은 변환 계층이 흡수** — 일반 OS 가상 메모리의 구조를 그대로 이식했다.
2. **고정 크기 2단 계층**: kernel은 (영역 → 2MiB chunk), libpim은 (chunk → broadcast unit). 두 층 모두 "비트맵 장부 + first-fit + 연속 run 우선 폴백"이라는 같은 알고리즘을 쓰고, 정책만 다르다(kernel: 연속성 우선 / libpim: 재사용 우선 + 반납 hysteresis).
3. **불변식 하나가 스택을 지탱한다**: unit | chunk(약수 + 정렬) → 물리 비연속성은 연산에 보이지 않고, 전송의 호출 횟수에만 영향을 준다. 그리고 그 비용은 ~1%로 정량적으로 무시 가능하다.
4. **kernel은 geometry를 모른다**: chunk 장부 기계일 뿐. RoChBaCo 해석·command 생성은 전부 libpim. 하드웨어 구성 실험 시 driver 재컴파일이 불필요하다.
5. **이미 있는 것은 다시 만들지 않는다**: host 쪽 pinning/SG는 QDMA driver가, process 격리와 자원 회수는 kernel의 fd 수명주기가 공짜로 제공한다.

---

## 11. 열린 논의사항

* **command 제출 경로 확정**: kernel 대행 ioctl(방안 A) vs memory-ring(방안 B, §9.1). emulator controller가 command를 받는 쪽 구조(레지스터 파일 존재 여부, 메모리 폴링 가능 여부)가 사실상 결정한다.
  * **잠정 결정 (2026-08-21, 되돌릴 수 있음)**: GEMV를 끝까지 돌려보기 위해 `sw/runtime/pim_exec.c`가 **CFR 페이지(32 KiB)를 user 공간에 mmap**한다. §6.1의 "BAR를 user에 매핑하지 않는다"를 그만큼 되돌린 것이다. 벌크는 전부 QDMA 그대로 — IMEM에 프로그램, GPR에 벡터·결과. 지금 받아들일 수 있는 이유: §6.1이 든 근거 둘 중 프로세스 간 중재는 단일 프로세스 전제 하에서 무의미하고, 임의 프로세스의 레지스터 접근 차단은 §11의 격리 비고려 결정에 이미 포함된다. **대가**: 다른 프로세스가 도어벨을 누를 수 있고 아무것도 막지 않는다. **되돌리는 법**: MMIO는 전부 `cfr_rd`/`cfr_wr`과 `pim_exec_open`의 mmap 뒤에 있다. 방안 A는 그 셋을 `PIM_IOC_LAUNCH`로 바꾸는 것이고(uapi 0x04 예약됨), 그 위는 아무것도 안 바뀐다.
* **연산 완료 통지**: 방안 B의 status 폴링(단순) vs interrupt 기반(`.poll` 추가). 이 선택이 `pim_op`를 동기로 할지 비동기(submit + wait)로 할지를 결정한다.
* **operand co-location 제약**: 연산이 두 operand를 요구할 때(A+B→C) 세 buffer가 같은 bank/채널에 있어야 하는 제약이 있는지. 있다면 allocator에 placement API("같은 bank에 co-locate")가 필요해져 인터페이스에 영향이 크다.
* **tiled layout API**: §9.2의 2KiB stride 문제를 runtime이 숨길지(변환 복사) user에 노출할지.
* **PIM_ALLOC_SMALL pool**: unit 미만 buffer가 많은 워크로드가 실재하는지 확인 후 도입.
* **GPR sub-page 층 (§9.4)**: v1은 4KiB page 올림이다. 프로파일이 소형 객체 압력을 보여주면 size-class 층을 추가한다 — 크기 분포를 관측하기 전에는 굽지 않는다.
* **단일 프로세스 전제 (2026-08-21 결정)**: 실행 엔진(dispatcher·IMEM·도어벨·누산기 latch)은 싱글턴이고 프로세스를 넘어 상태가 살아남는다. 당분간 PIM은 한 프로세스가 점유하는 것으로 보고 엔진 중재를 만들지 않는다. **메모리 쪽은 이 전제에 영향받지 않는다** — per-fd 장부는 crash 회수 수단이지 멀티프로세스 기능이 아니라서 그대로 둔다.
* **매핑 정책 실험**: 채널 비트를 2MiB 위로 올리는 구성은 chunk 동질성을 깨므로 allocator 재설계가 필요하다는 점을 실험 계획에 표시해 둘 것.
* **geometry 출처 2단계**: HW info register 또는 카드 메모리 고정 주소의 geometry 블록이 제공되면 driver init의 값 획득부만 교체(§6.4). 후자는 conf–bitstream 불일치 검출 안전판으로서 HW 비용 대비 효과가 크므로 FPGA 측에 제안해 둘 것.
* **process 간 격리 — 당분간 비고려 (결정)**: 본 문서의 설계는 카드 주소를 user에 노출하는 신뢰 기반이며, 격리 전환은 스택 상당 부분의 재작성(LOAD/LAUNCH/COPY ioctl, kernel queue pool, relocation, allocator 제약)을 수반하는 대공사이므로 현 단계에서는 고려하지 않는다. 전환 설계는 별도 문서 `pim_isolation_design.md`에 기록만 유지한다. 단, command 주소 필드를 선형 카드 주소로 두고 RoChBaCo 디코드를 HW에 두는 선택(그 문서 §3.2)은 격리와 무관하게 kernel/libpim을 geometry 실험에서 자유롭게 하므로 baseline에서도 유지한다.
