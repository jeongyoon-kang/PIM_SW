// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_gemv — run one GEMV tile on the device and check every lane.
//
//     ./emu_gemv                        # baseline: pu_mask 0xFFFF, L=1, row 100
//     ./emu_gemv --pu 0x0001            # bank 0 only
//     ./emu_gemv --pu 0x8000            # bank 15 only
//     ./emu_gemv --pu 0xFFFF --l 64     # K = 1024
//     ./emu_gemv --base 100             # change the operand values
//
// LAYER 3.  Needs layers 1 and 2 to have passed: the CFR must answer, the GPR must
// round-trip, and the direct HBM aperture must hold what the host puts there.
//
// THE PROGRAM (HANDOFF §3.6)
//   word 0  WRVEC   OPSIZE=L  ROW=vec_word      GPR -> GB.  NO MASK: WRVEC only
//                                               fills the GB.
//   word 1  MAC     OPSIZE=L  ROW=row  COL=0    GB -> the banks gb_mc_mask names,
//                   pu_mask=M gb_mc_mask=M      and those banks compute.
//   word 2  RD_MAC  OPSIZE=0  ROW=dst_word      16 BF16 lanes land in one GPR word
//   word 3  EOS
//
// ROW MEANS DIFFERENT THINGS IN DIFFERENT ISRs, from the same bits [22:6]:
//   WRVEC / RD_MAC -> a GPR word index
//   MAC            -> a DRAM row; bank offset = ROW*2048 + COL*32
//
// THE ARITHMETIC CONTRACT  [measured on ch1 2026-08-10, emu_chain --test order/acc]
//   BF16 inputs, exact products, accumulation in a format WIDER than BF16, one
//   round-to-nearest-even back to BF16 at RD_MAC.
//
//   The summation ORDER is STRICTLY SEQUENTIAL in lane order — not a reduction
//   tree.  This was settled by feeding vectors built so the candidate models give
//   answers that are not close (eps at the lanes a tree pairs first, +-A elsewhere
//   so every other pair cancels exactly): the device returned the sequential answer
//   and BOTH standard 16-lane adder-tree wirings, (2i,2i+1) and (i,i+n/2), were
//   refuted.  A control with the +-A lanes zeroed confirmed eps is not being
//   flushed, so the null results mean what they say.
//
//   On realistic data the distinction is nearly invisible — a sequential and a tree
//   reference part on about 1 lane in 4000 — which is exactly why it had to be
//   asked with a constructed vector rather than random data.
//
// WHY THE TEST VECTOR IS WHAT IT IS
//   v[k] = 1.0 and A_b[k] = base + b makes lane b exactly K*(base+b), which for
//   the default sizes is representable in BF16 with no rounding at all.  A
//   mismatch is then a defect, not a tolerance argument.  It also gives all 16
//   banks DIFFERENT answers, so a lane holding another bank's value is visible.
//
// WAITING: STATUS[31] IS NOT TRUSTED, AND THE WAIT IS IN TWO PARTS
//   done is a HELD level and it rises when the fetcher accepts the last ISR, not
//   when the kernel finishes.  It can therefore read 1 left over from the previous
//   run.  So the destination GPR word is poisoned before the doorbell and success
//   means the poison is gone — an observation, not an assumption.
//
//   STATUS is polled over MMIO because it is AXI-Lite and a poll is free there.
//   The result word is a 32 B GPR beat, so it comes back over C2H — and this
//   driver printk()s once per transfer, so it is read only AFTER done has risen
//   rather than on every poll.  If done never rises the reads are attempted
//   anyway: a result that landed is still worth reporting.
//
// Exit: 0 pass   1 fail   2 usage
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "emu_regs.h"
#include "pim_platform.h"

// ---- platform ------------------------------------------------------------------
// The topology (channel count, bank stride, bank window, MC base) is NOT a compile
// time constant: it is read at startup from platform/*.conf, the same file the
// shell scripts read.  g_ch is the channel this tool is addressing right now —
// tools that walk every channel set it in their loop, tools that take --ch set it
// once.  Every tool prints pim_platform_banner(), so a run made against the wrong
// topology says so in its own output.
static unsigned g_ch;
// The set of channels this run drives.  --ch N gives a 1-hot mask; --chs 0,1 gives
// a multicast one, which is the interesting case: ONE WRVEC and ONE MAC covering
// 16*n outputs instead of 16.  g_sel lists them in order so a result word index
// maps back to a channel.
static uint32_t g_chmask = 0;
static unsigned g_sel[PIM_NCH], g_nsel = 1;
static const char *g_chs_arg;


#define BDF_DEFAULT "0000:01:00.0"
#define H2C_DEFAULT "/dev/qdma01000-MM-0"
#define C2H_DEFAULT "/dev/qdma01000-MM-1"
#define CHUNK       (4u << 20)
#define POISON_LANE 0x7FC1u          // a quiet NaN payload nothing here generates

// --fill copy sources the vector from this bank's DRAM.  Fixed at 0 on purpose:
// COPY carries pu_mask = 0, and the ISR guide says BK is what normalises a zero
// pu_mask (1 << BK).  Whether that normalisation applies to COPY as well as to MAC
// is not stated.  With BK = 0 and the source bank = 0 the two readings agree, so
// the ambiguity cannot change the result.  Moving the source is how it gets
// settled, and that is a later experiment.
#define COPY_SRC_BANK 0u

static volatile uint8_t *g_bar;

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static uint32_t cfr_rd(uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off);
    __sync_synchronize();
    return v;
}
static void cfr_wr(uint32_t off, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off) = v;
    __sync_synchronize();
}
static uint32_t viol_rd(uint32_t off)
{
    __sync_synchronize();
    // The violation CSR is PER CHANNEL (OFF_VIOL + ch*0x1000).  Reading channel
    // 0's block while the kernel ran on channel 1 would report another channel's
    // counters as this run's — a quiet way to call a bad run clean.
    uint32_t v = *(volatile uint32_t *)
        (g_bar + PIM_OFF_VIOL + (size_t)g_ch * 0x1000u + off);
    __sync_synchronize();
    return v;
}

// violation 카운터는 누적된다.  실행 전에 지우지 않으면 로그의 cnt/max 가 이번
// 실행의 값이 아니라 마지막으로 지워진 이후의 합이 되어, 캡처한 파형 한 창과
// 대조가 성립하지 않는다.  VIOL_CTRL bit0 = clrstats, self-clearing.
static void viol_clear_all(void)
{
    for (unsigned ch = 0; ch < PIM_NCH; ch++) {
        __sync_synchronize();
        *(volatile uint32_t *)(g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + VIOL_CTRL) = 1u;
        __sync_synchronize();
    }
}

// 실제로 쓴 채널만, 16 뱅크 전부 찍는다.  ANY 비트가 0인 뱅크도 값을 보여야
// "위반 0 건" 과 "안 읽었음" 이 눈으로 구분된다.
static uint32_t viol_rd_ch(unsigned ch, uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)
        (g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + off);
    __sync_synchronize();
    return v;
}
// 위반 표를 볼 때 "무슨 예산에서 나온 수치인가" 가 같이 있어야 해석이 된다.
// 이 값들은 ./emu_timing --scale N --keep 이 걸어둔 것이고, 커널은 건드리지 않는다.
static void timing_print_now(void)
{
    printf("    timing : faw=%u rrd=%u rcd=%u ccd=%u rtp=%u rp=%u wr=%u ras=%u\n",
           cfr_rd(CFR_T_FAW), cfr_rd(CFR_T_RRD), cfr_rd(CFR_T_RCD), cfr_rd(CFR_T_CCD),
           cfr_rd(CFR_T_RTP), cfr_rd(CFR_T_RP),  cfr_rd(CFR_T_WR),  cfr_rd(CFR_T_RAS));
}

static void viol_report_ch(unsigned ch)
{
    uint32_t any = viol_rd_ch(ch, VIOL_ANY);
    printf("\n  violation CSR ch%u ANY: 0x%04x%s\n", ch, any,
           any ? "" : "   (no violations — every row below should read 0)");
    timing_print_now();
    // No REC_WR / drop column: GEMV never writes DRAM.  WRVEC and COPY land in the
    // vector register, MAC and RD_MAC are reads, so write recovery cannot move and the
    // two columns would be guaranteed zeros.  emu_mc --write and emu_ewmul print them.
    printf("    bank |   RCD_RD    |   CCD_RD    |   RCD_WR    |   CCD_WR    |\n");
    printf("         | cnt    max  | cnt    max  | cnt    max  | cnt    max  |\n");
    printf("    -----+-------------+-------------+-------------+-------------+\n");
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        uint32_t ca = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_CNT_A);
        uint32_t ma = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_MAX_A);
        printf("    %4u | %3u    %3u  | %3u    %3u  | %3u    %3u  | %3u    %3u  |%s\n",
               b,
               ca         & 0xffu, ma         & 0xffu,
               (ca >>  8) & 0xffu, (ma >>  8) & 0xffu,
               (ca >> 16) & 0xffu, (ma >> 16) & 0xffu,
               (ca >> 24) & 0xffu, (ma >> 24) & 0xffu,
               ((any >> b) & 1u) ? "  <-- ANY" : "");
    }
}
// ---- the 256-bit slaves: DMA ONLY, both directions ---------------------------
// MMIO cannot WRITE these.  [measured 2026-07-31] the host splits a 32 B MMIO
// store into <=16 B TLPs — memcpy and a single AVX 32 B store behave identically
// — and because the GPR and IMEM slaves have NO WSTRB port, every fragment is
// expanded to a full 32 B word write.  The second fragment therefore wins and the
// bytes it did not carry come through as zero:
//     wrote b0..bf c0..cf   read back 00..00 c0..cf
//
// Reads do not go over MMIO either.  emu_regs.h's transport policy gives each
// window ONE master, so the GPR is read back the same way it was written and no
// second address view has to be shown to agree before a result can be believed.
// It also means there is no torn read to rule out: a 32 B MMIO read arrives as two
// 16 B halves and can catch RD_MAC mid-flight, while a C2H transfer delivers the
// beat whole.
//
// The CFR and the violation CSR are AXI-Lite and are reached over MMIO: 32 bit
// registers, one TLP each, and the doorbell must not wait on a queue.
static int g_h2c = -1;
static int g_c2h = -1;

static int axi_write(uint64_t axi, const void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    const uint8_t *p = buf;
    size_t done = 0;
    while (done < len) {
        size_t want = len - done;
        if (want > CHUNK) want = CHUNK;
        ssize_t n = pwrite(g_h2c, p + done, want, (off_t)(axi + done));
        if (n <= 0) {
            fprintf(stderr, "pwrite(0x%011" PRIx64 " +0x%zx): %s\n", axi, done,
                    n < 0 ? strerror(errno) : "no progress");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int gpr_write(uint32_t word, const void *src, uint32_t n)
{
    return axi_write(EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, src,
                     (size_t)n * EMU_WORD_BYTES);
}
static int axi_read(uint64_t axi, void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    uint8_t *p = buf;
    size_t done = 0;
    while (done < len) {
        size_t want = len - done;
        if (want > CHUNK) want = CHUNK;
        ssize_t n = pread(g_c2h, p + done, want, (off_t)(axi + done));
        if (n <= 0) {
            fprintf(stderr, "pread(0x%011" PRIx64 " +0x%zx): %s\n", axi, done,
                    n < 0 ? strerror(errno) : "no progress");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static int gpr_read(uint32_t word, void *dst, uint32_t n)
{
    return axi_read(EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, dst,
                    (size_t)n * EMU_WORD_BYTES);
}
static int imem_write(uint32_t word, const struct emu_isr *isr, uint32_t n)
{
    // The ISR words are already contiguous 32 B records in memory.
    return axi_write(EMU_AXI_IMEM + (uint64_t)word * EMU_WORD_BYTES, isr,
                     (size_t)n * EMU_WORD_BYTES);
}


// The dispatcher must be idle before we touch ANYTHING it owns.
//
// Not just before the doorbell.  The IMEM command port is shared between the
// host write path and the fetch read path with FETCH PRIORITY:
//     wire imem_wr_ready = imem_cmd_ready & ~fd_cmd_valid;
//     assign s_axi_wready = (ws == WS_DATA) & imem_wr_ready;
// so while a fetcher is running, a host IMEM write has WREADY held low.  The DMA
// then stalls, and this driver gives a request 10 s (req->timeout_ms = 10*1000)
// before returning EIO — and one EIO latches the H2C engine, after which every
// write fails regardless of address.
//
// The GPR has the same shape: emu_gpr_wrap arbitrates its single command port by
// pure priority with the dispatcher first, so a host load contends with the MC's
// WRVEC fill while a kernel is live.
//
// Idle means done=1 (a program finished and the FSM is parked) or state=0 (fresh
// from reset).  Anything else is a kernel in progress.
static bool require_idle(const char *when, uint32_t timeout_ms)
{
    uint64_t t = now_us();
    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t > (uint64_t)timeout_ms * 1000ull) {
            fprintf(stderr,
                "REFUSING to %s: the dispatcher is still busy after %u ms "
                "(STATUS=0x%08x, done=0, state=%u).\n"
                "A previous kernel has not finished.  Writing IMEM now would stall on "
                "WREADY until the DMA times out, and that EIO latches the H2C engine.\n"
                "  sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n",
                when, timeout_ms, st, CFR_STATUS_STATE(st));
            return false;
        }
    }
}

// The device's summation order: strictly left to right in lane order, in fp32.
// See THE ARITHMETIC CONTRACT above — this was measured, not assumed.  For this
// tool's own operands every partial sum is exact so any order would agree; it is
// written correctly anyway, because this is the function anyone reaches for when
// they want to know what the hardware does.
static float device_sum(const float *v, unsigned n)
{
    float s = 0.0f;
    for (unsigned i = 0; i < n; i++) s += v[i];
    return s;
}

// ---- one doorbell: load IMEM, poison, ring, wait, fetch the 16 lanes ---------
// Everything that differs between the two fill paths is in the program itself, so
// this is shared.  It is also the only place that rings the doorbell, which keeps
// the irreversible step in one function.
struct run_result {
    // One GPR word per PARTICIPATING CHANNEL.  RD_MAC must be 1-hot (two channels
    // would otherwise write the same word), so a multi-channel program ends in one
    // RD_MAC per channel into consecutive words.
    uint16_t lane[PIM_NCH][EMU_LANES_PER_WORD];
    uint32_t st_before, st_after, polls, reads;
    uint64_t us;
    bool     landed, seen_done;
};

static int run_program(const struct emu_isr *prog, unsigned n, uint32_t dst_word,
                       unsigned ndst, struct run_result *r)
{
    memset(r, 0, sizeof *r);

    if (!require_idle("load the device", 1000)) return -1;
    if (imem_write(0, prog, n) < 0) return -1;

    uint16_t p[PIM_NCH][EMU_LANES_PER_WORD];
    for (unsigned d = 0; d < ndst; d++)
        for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++) p[d][i] = POISON_LANE;
    if (gpr_write(dst_word, p, ndst) < 0) return -1;
    // The poison is not verified.  Reading it back would be another DMA-write /
    // DMA-read pair, subject to exactly the visibility question it is meant to
    // resolve — so it would decide nothing.  What decides is the RESULT.

    if (!require_idle("ring the doorbell", 1000)) return -1;
    r->st_before = cfr_rd(CFR_STATUS);
    cfr_wr(CFR_PROG_LEN, n);
    if (cfr_rd(CFR_PROG_LEN) != n) { fprintf(stderr, "PROG_LEN readback failed\n"); return -1; }

    // Final ordering point.  This read is non-posted, so the CPU's earlier posted
    // MMIO writes — PROG_LEN in particular — are at the device before the doorbell.
    (void)cfr_rd(CFR_PROG_LEN);

    uint64_t t0 = now_us();
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);

    // STATUS over MMIO: AXI-Lite, so a poll costs one 32 bit read and nothing in
    // the kernel log.  done rising means the fetcher accepted EOS, and EOS is last.
    while (now_us() - t0 < 2000000ull) {
        r->polls++;
        if (cfr_rd(CFR_STATUS) & CFR_STATUS_DONE) { r->seen_done = true; break; }
    }
    // The result word over C2H, only after done — this driver printk()s once per
    // transfer.  done is a held level and can be stale, so it decides nothing: the
    // POISON does.  Reads continue even if done never rose.
    for (; r->reads < 64 && now_us() - t0 < 2000000ull; ) {
        r->reads++;
        if (gpr_read(dst_word, r->lane, ndst) < 0) return -1;
        // EVERY word must have lost its poison.  A channel that never executed
        // leaves its own word untouched, which is exactly the failure a multicast
        // program has to be able to show.
        bool all = true;
        for (unsigned d = 0; d < ndst && all; d++)
            for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++)
                if (r->lane[d][i] == POISON_LANE) { all = false; break; }
        if (all) { r->landed = true; break; }
        if (r->reads >= 4) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
    }
    r->us = now_us() - t0;
    r->st_after = cfr_rd(CFR_STATUS);
    return 0;
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--ch N | --chs LIST] [--fill wrvec|copy|both]\n"
"       [--pu MASK]\n"
"       [--l N] [--row N] [--base N] ...\n"
"\n"
"Runs one GEMV tile and checks all 16 lanes.  The vector always reaches the PUs\n"
"through the GLOBAL BUFFER; --fill picks how the GB is filled:\n"
"\n"
"  --fill wrvec   host -> GPR -> WRVEC -> GB          (no bank traffic)\n"
"  --fill copy    host -> bank %u DRAM -> COPY -> GB   (read-half, route[%u]=GB)\n"
"  --fill both    run both and compare them lane by lane (default)\n"
"\n"
"  --ch N         which channel to run on (default 0).  Stages operands in\n"
"                 that channel AND sets the ISR CH_MASK to 1<<N.\n"
"  --chs LIST     span SEVERAL channels in ONE program, e.g. --chs 0,1.\n"
"                 The fill and the MAC go out with a multicast CH_MASK,\n"
"                 so one WRVEC feeds every channel's GB and one MAC\n"
"                 covers 16*n outputs.  RD_MAC stays 1-hot per channel\n"
"                 into consecutive GPR words (two channels writing one\n"
"                 word would lose one silently).  Weights are staged as\n"
"                 base + 16*ch + bank, so every lane is distinct and a\n"
"                 channel that never ran cannot pass by matching.\n"
"\n"
"  This tool reaches the banks through the DIRECT aperture and never through the MC\n"
"  port, so --ch names a PHYSICAL channel: it verifies one emulator channel on its\n"
"  own.  That makes it independent of the channel address map — ChRoBaCo or\n"
"  RoChBaCo only changes how an MC-space address is split up, and nothing here is an\n"
"  MC-space address.  (emu_mc is the one that is: its --ch is a place in MC space,\n"
"  which is why that one is ChRoBaCo-only.)\n"
"\n"
"  --pu MASK      pu_mask = gb_mc_mask, 16 bits (default 0xFFFF)\n"
"  --l N          beats, 1..64; K = 16*N (default 1)\n"
"  --row N        DRAM row holding the weights (default 100)\n"
"  --vec-row N    DRAM row holding the vector, --fill copy only (default 200)\n"
"  --base N       operand value base: A_b[k] = base + 16*ch + b (default 1)\n"
"  --vec V        every vector element (default 1.0).  Use a value the\n"
"                 PREVIOUS run did not use to rule out a stale GB: the\n"
"                 answer must scale with it, and content left in a\n"
"                 channel's GB by an earlier program cannot.\n"
"  --vec-word N   GPR word holding the vector, --fill wrvec only (default 0)\n"
"  --dst-word N   GPR word the 16 lanes land in (default 1041)\n"
"  --h2c PATH     H2C MM queue (default %s)\n"
"  --c2h PATH     C2H MM queue (default %s)\n"
"\n"
"Both queues are required: IMEM and the GPR are DMA-only.  MMIO is used for the\n"
"AXI-Lite blocks only — the CFR and the violation CSR.\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p, COPY_SRC_BANK, COPY_SRC_BANK,
        H2C_DEFAULT, C2H_DEFAULT);
}

int main(int argc, char **argv)
{
    const char *bdf = BDF_DEFAULT, *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    unsigned pu = 0xFFFF, L = 1, row = 100, base = 1, vec_word = 0, dst_word = 1041;
    float    vecval = 1.0f;
    unsigned vec_row = 200;
    bool do_wrvec = true, do_copy = true;

    static const struct option lo[] = {
        { "ch",       required_argument, NULL, 'C' },
        { "chs",      required_argument, NULL, 'M' },
        { "vec",      required_argument, NULL, 'V' },
        { "pu",       required_argument, NULL, 1 }, { "l",        required_argument, NULL, 2 },
        { "row",      required_argument, NULL, 3 }, { "base",     required_argument, NULL, 4 },
        { "vec-word", required_argument, NULL, 5 }, { "dst-word", required_argument, NULL, 6 },
        { "bdf",      required_argument, NULL, 7 }, { "h2c",      required_argument, NULL, 8 },
        { "c2h",      required_argument, NULL, 9 }, { "fill",     required_argument, NULL, 10 },
        { "vec-row",  required_argument, NULL, 11 },
        { "help",     no_argument,       NULL, 'h' }, { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: pu = (unsigned)strtoul(optarg, NULL, 0); break;
        case 2: L  = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: row = (unsigned)strtoul(optarg, NULL, 0); break;
        case 4: base = (unsigned)strtoul(optarg, NULL, 0); break;
        case 5: vec_word = (unsigned)strtoul(optarg, NULL, 0); break;
        case 6: dst_word = (unsigned)strtoul(optarg, NULL, 0); break;
        case 7: bdf = optarg; break;
        case 8: h2c_path = optarg; break;
        case 9: c2h_path = optarg; break;
        case 10:
            if      (!strcmp(optarg, "wrvec")) { do_wrvec = true;  do_copy = false; }
            else if (!strcmp(optarg, "copy"))  { do_wrvec = false; do_copy = true;  }
            else if (!strcmp(optarg, "both"))  { do_wrvec = true;  do_copy = true;  }
            else { fprintf(stderr, "--fill must be wrvec, copy or both\n"); return 2; }
            break;
        case 11: vec_row = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'C': g_ch = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'M': g_chs_arg = optarg; break;
        case 'V': vecval = (float)atof(optarg); break;

        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 2;
        }
    }

    // The topology is compiled in.  What is checked is that this build matches the
    // platform currently SELECTED — choosing one and forgetting to rebuild is the
    // mistake that survives, since the board cannot report its own channel count.
    {
        const char *e = pim_platform_check();
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 2; }
        pim_platform_banner();
    }
    // --chs 0,1 spans channels; --ch N is the 1-hot case of the same thing, so the
    // program below is built once and the mask decides how wide it reaches.
    if (g_chs_arg) {
        char buf[64];
        snprintf(buf, sizeof buf, "%s", g_chs_arg);
        g_chmask = 0; g_nsel = 0;
        for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
            unsigned c = (unsigned)strtoul(t, NULL, 0);
            if (c >= PIM_NCH) {
                fprintf(stderr, "ERROR: --chs names channel %u but %s has %u\n",
                        c, PIM_PLATFORM_NAME, PIM_NCH);
                return 2;
            }
            if (g_chmask & (1u << c)) continue;
            g_chmask |= 1u << c;
            g_sel[g_nsel++] = c;
        }
        if (!g_nsel) { fprintf(stderr, "ERROR: --chs selected nothing\n"); return 2; }
        g_ch = g_sel[0];                 // where per-channel CSR reads point
    } else {
        if (g_ch >= PIM_NCH) {
            fprintf(stderr, "ERROR: --ch %u but %s has %u channel(s)\n",
                    g_ch, PIM_PLATFORM_NAME, PIM_NCH);
            return 2;
        }
        g_chmask = 1u << g_ch;
        g_sel[0] = g_ch;
        g_nsel   = 1;
    }
    const unsigned K = L * EMU_LANES_PER_WORD;
    if (dst_word >= vec_word && dst_word < vec_word + L) {
        fprintf(stderr, "dst-word %u is inside the vector [%u, %u) — the result would "
                        "overwrite it\n", dst_word, vec_word, vec_word + L);
        return 2;
    }
    // --fill copy puts the vector in the SOURCE BANK's DRAM.  If it shared a row
    // with the weights it would overwrite bank COPY_SRC_BANK's operands and that
    // bank alone would come out wrong — a confusing failure with an obvious cause.
    if (do_copy && vec_row == row) {
        fprintf(stderr, "--vec-row %u collides with the weight row: the vector would "
                        "overwrite bank %u's operands\n", vec_row, COPY_SRC_BANK);
        return 2;
    }

    // ---------------- open ----------------
    // BAR2 is mapped for the AXI-Lite blocks ONLY — the CFR (timing, PROG_LEN,
    // doorbell, STATUS) and the violation CSR.  IMEM and the GPR are 256 b slaves
    // and are reached exclusively through the queues below.
    char path[256];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource2", bdf);
    int bfd = open(path, O_RDWR | O_SYNC);
    if (bfd < 0) { fprintf(stderr, "ERROR: open(%s): %s\n", path, strerror(errno)); return 1; }
    void *m = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bfd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "ERROR: mmap: %s\n", strerror(errno)); return 1; }
    g_bar = m;
    g_h2c = open(h2c_path, O_WRONLY);
    g_c2h = open(c2h_path, O_RDONLY);
    if (g_h2c < 0 || g_c2h < 0) {
        fprintf(stderr, "ERROR: cannot open the MM queues (%s / %s): %s\n"
                        "       Both are required: IMEM and the GPR are DMA-only.\n"
                        "       sudo ./qdma_queues.sh setup\n",
                h2c_path, c2h_path, strerror(errno));
        return 1;
    }

    printf("GEMV tile — vector reaches the PUs through the GLOBAL BUFFER\n");
    printf("  fill    %s\n",
           do_wrvec && do_copy ? "wrvec AND copy, compared against each other"
                               : do_wrvec ? "wrvec (GPR -> GB)" : "copy (bank DRAM -> GB)");
    printf("  pu_mask = gb_mc_mask  0x%04X   (%u banks participate)\n", pu, __builtin_popcount(pu));
    printf("  L = %u beats,  K = %u,  weights at DRAM row %u,  A_c,b[k] = %u + 16c + b,  v[k] = %g\n",
           L, K, row, base, (double)vecval);
    if (do_wrvec) printf("  wrvec   vector at GPR word %u..%u\n", vec_word, vec_word + L - 1);
    if (do_copy) {
        printf("  copy    vector at bank %u DRAM row %u, route[%u]=%u (GB)\n",
               COPY_SRC_BANK, vec_row, COPY_SRC_BANK, GB_PORT);
        // A multicast COPY reads EACH channel's own bank: the ISR carries one ROW
        // and the channels each fetch from it locally.  So a shared vector has to be
        // REPLICATED into every selected channel before the launch, which is work
        // WRVEC does not need — it writes the GPR once and CH_MASK fans it out.
        if (g_nsel > 1)
            printf("          NOTE: %u channels, so the vector is written to bank %u "
                   "row %u of EACH of them.\n"
                   "          A multicast COPY makes every channel read its OWN bank, "
                   "so a shared vector\n"
                   "          must be replicated first.  WRVEC needs one GPR write "
                   "and no replication.\n",
                   g_nsel, COPY_SRC_BANK, vec_row);
    }
    printf("  result  GPR word %u\n\n", dst_word);

    // ---------------- build both programs FIRST, so a bad spec costs nothing --
    // The two differ ONLY in word 0.  Everything after it is identical, which is
    // what makes the lane-by-lane comparison a test of the fill path alone.
    // 1 fill + 1 MAC + one RD_MAC PER CHANNEL + EOS.
    static const char *rname_prog[2] = { "WRVEC", "COPY" };
    const unsigned NPROG = 3 + g_nsel;
    struct emu_isr prog_wrvec[3 + PIM_NCH], prog_copy[3 + PIM_NCH];
    struct emu_isr_spec s;
    const char *e;

    s = emu_isr_default_ch(ISR_OP_WRVEC, g_chmask); s.opsize = L; s.row = vec_word;
    if ((e = emu_isr_build(&prog_wrvec[0], &s))) { fprintf(stderr, "WRVEC refused: %s\n", e); return 2; }

    // COPY read-half: bank -> GB.  pu_mask and gb_mc_mask are both 0 — the shape is
    // carried entirely by route.  ROW is the SOURCE row.
    s = emu_isr_default_ch(ISR_OP_COPY, g_chmask);  s.opsize = L; s.row = vec_row; s.col = 0;
    s.pu_mask = 0; s.gb_mc_mask = 0; s.route[COPY_SRC_BANK] = GB_PORT;
    if ((e = emu_isr_build(&prog_copy[0], &s)))  { fprintf(stderr, "COPY refused: %s\n", e); return 2; }

    s = emu_isr_default_ch(ISR_OP_MAC, g_chmask);   s.opsize = L; s.row = row; s.col = 0;
    s.pu_mask = pu; s.gb_mc_mask = pu;
    if ((e = emu_isr_build(&prog_wrvec[1], &s))) { fprintf(stderr, "MAC refused: %s\n", e); return 2; }

    // RD_MAC is 1-HOT PER CHANNEL and the destinations are consecutive, so the
    // whole result comes back in one pread and word i belongs to g_sel[i].
    // emu_isr_check rejects a multicast RD_MAC — two channels would write the same
    // GPR word and one of them would be lost with nothing to show for it.
    for (unsigned i = 0; i < g_nsel; i++) {
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_sel[i]);
        s.opsize = 0; s.row = dst_word + i;
        if ((e = emu_isr_build(&prog_wrvec[2 + i], &s))) {
            fprintf(stderr, "RD_MAC ch%u refused: %s\n", g_sel[i], e); return 2; }
    }
    s = emu_isr_default_ch(ISR_OP_EOS, g_chmask);
    if ((e = emu_isr_build(&prog_wrvec[2 + g_nsel], &s))) { fprintf(stderr, "EOS refused: %s\n", e); return 2; }
    for (unsigned i = 1; i < NPROG; i++) prog_copy[i] = prog_wrvec[i];

    // The last ISR must not be one that produces a result: done rises when the
    // fetcher ACCEPTS the last ISR, so that result would still be in flight.
    if (emu_isr_produces_result((uint32_t)emu_isr_get(&prog_wrvec[NPROG - 1], ISR_F_OPCODE))) {
        fprintf(stderr, "the last ISR produces a result — append an EOS\n");
        return 2;
    }

    // TWO PROGRAMS, TWO DOORBELLS — printed in full, separately, because printing
    // one and calling the other "a different word 0" reads as though COPY were an
    // extra ISR inside the GEMV.  It is not: a GEMV needs exactly one GB fill, and
    // these are two ways of doing it.  What differs between them is where the vector
    // physically comes from, and that is a difference in the PROGRAM.
    {
        const struct emu_isr *pp[2] = { prog_wrvec, prog_copy };
        const char *what[2] = {
            "vector from the GPR — one host write, multicast to every channel by CH_MASK",
            "vector from bank DRAM — ONE COPY PER CHANNEL, see step 3b",
        };
        bool want[2] = { do_wrvec, do_copy };

        printf("  programs — one per fill path.  TWO SEPARATE DOORBELLS, not one\n"
               "  program: a GEMV needs exactly one GB fill and these are two ways\n"
               "  of doing it.  Everything after word 0 is identical, which is what\n"
               "  makes the lane comparison a test of the fill path alone.\n");
        for (int m = 0; m < 2; m++) {
            if (!want[m]) continue;
            printf("\n    [%s]  %s\n", rname_prog[m], what[m]);
            for (unsigned i = 0; i < NPROG; i++) {
                char nm[24];
                if      (i == 0)         snprintf(nm, sizeof nm, "%s", rname_prog[m]);
                else if (i == 1)         snprintf(nm, sizeof nm, "MAC");
                else if (i == NPROG - 1) snprintf(nm, sizeof nm, "EOS");
                else snprintf(nm, sizeof nm, "RD_MAC ch%u", g_sel[i - 2]);
                printf("      %u  %-11s %016" PRIx64 "%016" PRIx64 "%016" PRIx64
                       "%016" PRIx64 "\n", i, nm,
                       pp[m][i].w[3], pp[m][i].w[2], pp[m][i].w[1], pp[m][i].w[0]);
            }
        }
    }
    printf("\n");

    // ---------------- 0. the dispatcher must be idle before we load anything --
    if (!require_idle("load the device", 1000)) return 1;
    printf("  0. dispatcher idle (STATUS 0x%08x)\n", cfr_rd(CFR_STATUS));

    // ---------------- 1. timing ----------------
    // 타이밍 레지스터는 여기서 쓰지 않는다.  보드에 걸려 있는 값을 그대로 쓰고,
    // 무엇이었는지만 기록한다.  예전에는 여기서 8개를 하드코딩 값(faw 30 rrd 6
    // rcd 4 ccd 2 rtp 3 rp 3 wr 4 ras 6)으로 덮어썼는데, 그러면
    // ./emu_timing --scale N --keep 으로 걸어둔 설정을 이 커널이 되돌려 놓아서
    // 타이밍을 바꿔가며 비교하는 실험이 성립하지 않았다.
    printf("  1. timing (set with ./emu_timing --scale N --keep, this kernel does not write it)\n");
    timing_print_now();

    // ---------------- 2. operands into HBM, one row per bank ----------------
    uint16_t *a = calloc(L, EMU_WORD_BYTES);
    float    *acc = malloc(sizeof(float) * K);
    uint16_t expect[PIM_NCH][EMU_NBANKS];
    uint16_t vec[64 * EMU_LANES_PER_WORD];
    for (unsigned k = 0; k < K; k++) vec[k] = emu_f32_to_bf16(vecval);

    // A_c,b[k] = base + 16c + b.  EVERY (channel, bank) gets a DIFFERENT value, so
    // a channel that did not execute, or one whose answer landed in another
    // channel's word, names itself instead of matching by luck.  With one value
    // for all channels, a run where CH_MASK was ignored would pass.
    for (unsigned i = 0; i < g_nsel; i++) {
        const unsigned c = g_sel[i];
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            float av = (float)(base + 16u * c + b);
            for (unsigned k = 0; k < K; k++) a[k] = emu_f32_to_bf16(av);
            uint64_t axi = pim_bank(c, b) + (uint64_t)row * EMU_ROW_BYTES;
            if (axi_write(axi, a, (size_t)L * EMU_WORD_BYTES) < 0) return 1;
            for (unsigned k = 0; k < K; k++)
                acc[k] = emu_bf16_to_f32(vec[k]) * emu_bf16_to_f32(a[k]);  // exact products
            expect[c][b] = emu_f32_to_bf16(device_sum(acc, K));            // fp32, one RNE
        }
    }
    printf("  2. operands written: %u channel(s) x 16 banks x %u B at row %u "
           "(bank offset 0x%x)\n", g_nsel, L * EMU_WORD_BYTES, row, row * EMU_ROW_BYTES);

    // ---------------- 3. the vector, once per fill path ----------------------
    // Both paths carry the SAME bytes; only the place the hardware fetches them
    // from differs.  That is the whole point — the MAC downstream cannot tell.
    if (do_wrvec) {
        if (gpr_write(vec_word, vec, L) < 0) return 1;
        printf("  3a. vector -> GPR word %u..%u        (for WRVEC)\n",
               vec_word, vec_word + L - 1);
    }
    if (do_copy) {
        // One copy of the vector per channel, at the SAME (bank, row) in each.  That
        // is what makes a single multicast COPY legal: ISR carries one ROW, so the
        // channels must agree on where the vector is.
        for (unsigned i = 0; i < g_nsel; i++) {
            uint64_t vaxi = pim_bank(g_sel[i], COPY_SRC_BANK)
                          + (uint64_t)vec_row * EMU_ROW_BYTES;
            if (axi_write(vaxi, vec, (size_t)L * EMU_WORD_BYTES) < 0) return 1;
            printf("  3b. vector -> ch%u bank %u DRAM row %u (AXI 0x%011" PRIx64 ")  (for COPY)\n",
                   g_sel[i], COPY_SRC_BANK, vec_row, vaxi);
        }
    }

    // ---------------- 4. run each selected fill path -------------------------
    uint32_t viol_before = viol_rd(VIOL_ANY);
    viol_clear_all();   // 이번 실행분만 세도록
    struct run_result res[2];
    const char *rname[2] = { "WRVEC", "COPY" };
    bool ran[2] = { false, false };
    unsigned bad = 0;

    for (int mode = 0; mode < 2; mode++) {
        const struct emu_isr *prog = mode ? prog_copy : prog_wrvec;
        if (mode == 0 && !do_wrvec) continue;
        if (mode == 1 && !do_copy)  continue;

        printf("\n  === fill = %s ===\n", rname[mode]);
        if (run_program(prog, NPROG, dst_word, g_nsel, &res[mode]) < 0) return 1;
        ran[mode] = true;
        const struct run_result *r = &res[mode];

        if (!r->seen_done)
            printf("  !! done never rose within 2 s — the program did not finish "
                   "issuing.  The next run would restart the fetcher underneath it.\n");
        printf("  ran: STATUS 0x%08x -> 0x%08x (done=%u state=%u), %" PRIu64 " us, "
               "%u STATUS polls, %u GPR reads\n",
               r->st_before, r->st_after, (r->st_after & CFR_STATUS_DONE) ? 1u : 0u,
               CFR_STATUS_STATE(r->st_after), r->us, r->polls, r->reads);

        if (!r->landed) {
            printf("\nFAIL (%s): the result word never left its poison value in 2 s.\n",
                   rname[mode]);
            printf("      done was %sseen.  A PROG_LEN that overruns the program, a\n"
                   "      length mismatch between the GB fill and the MAC, or OPSIZE > 64\n"
                   "      all hang here with no diagnostic from the hardware.\n",
                   r->seen_done ? "" : "never ");
            return 1;
        }

        printf("\n   ch | bank | part | expected        | got             |\n");
        printf("  ----+------+------+-----------------+-----------------+------\n");
        for (unsigned i = 0; i < g_nsel; i++) {
            const unsigned c = g_sel[i];
            for (unsigned b = 0; b < EMU_NBANKS; b++) {
                bool part = (pu >> b) & 1u;
                uint16_t g = r->lane[i][b];
                printf("   %2u |  %2u  |  %s  | ", c, b, part ? "y" : "-");
                if (part) printf("%8.0f 0x%04x | ",
                                 (double)emu_bf16_to_f32(expect[c][b]), expect[c][b]);
                else      printf("%8s %6s | ", "-", "-");
                printf("%8.0f 0x%04x | ", (double)emu_bf16_to_f32(g), g);
                if (!part) {
                    printf("not asked to compute — value recorded, not judged\n");
                } else if (emu_bf16_same(g, expect[c][b])) {
                    printf("ok\n");
                } else {
                    printf("WRONG");
                    // Name where it DID come from.  Across channels this separates
                    // "the multicast never reached ch1" from "ch1 ran but its result
                    // landed in ch0's word".
                    bool named = false;
                    for (unsigned j = 0; j < g_nsel && !named; j++)
                        for (unsigned o = 0; o < EMU_NBANKS; o++)
                            if ((g_sel[j] != c || o != b) &&
                                emu_bf16_same(g, expect[g_sel[j]][o])) {
                                printf(" — this is ch%u bank %u's answer", g_sel[j], o);
                                named = true; break;
                            }
                    if (!named && emu_bf16_to_f32(g) == 0.0f)
                        printf(" — ZERO: this channel's GB was never filled");
                    printf("\n");
                    bad++;
                }
            }
        }
    }

    // ---------------- 5. the two fill paths against EACH OTHER ---------------
    // Both programs are identical from word 1 on, so any lane that differs is the
    // fill path and nothing else.  This also probes a question the guide leaves
    // open: WRVEC is documented to FLUSH the GB before filling it, COPY is not.
    // WRVEC ran first here, so leftover GB content would show up as a COPY-only
    // discrepancy.
    if (ran[0] && ran[1]) {
        unsigned diff = 0;
        for (unsigned i = 0; i < g_nsel; i++)
            for (unsigned b = 0; b < EMU_NBANKS; b++)
                if (!emu_bf16_same(res[0].lane[i][b], res[1].lane[i][b])) diff++;
        printf("\n  WRVEC vs COPY: ");
        const unsigned nlane = g_nsel * EMU_NBANKS;
        if (!diff) {
            printf("all %u lanes identical — the two GB fill paths agree", nlane);
            if (g_nsel > 1)
                printf(",\n                 across all %u channels", g_nsel);
            printf(".\n");
        } else {
            printf("%u of %u lanes DIFFER.\n", diff, nlane);
            printf("   ch | bank | WRVEC           | COPY            |\n");
            printf("  ----+------+-----------------+-----------------+\n");
            for (unsigned i = 0; i < g_nsel; i++)
                for (unsigned b = 0; b < EMU_NBANKS; b++) {
                    if (emu_bf16_same(res[0].lane[i][b], res[1].lane[i][b])) continue;
                    printf("   %2u |  %2u  | %8.0f 0x%04x | %8.0f 0x%04x |\n",
                           g_sel[i], b,
                           (double)emu_bf16_to_f32(res[0].lane[i][b]), res[0].lane[i][b],
                           (double)emu_bf16_to_f32(res[1].lane[i][b]), res[1].lane[i][b]);
                }
            bad += diff;
        }
    }

    // ---------------- 8. violations ----------------
    // viol_before 는 clear 이전 잔재라 이번 실행과 무관하다. 참고로만 찍는다.
    printf("\n  violation counters were CLEARED before this run "
           "(stale ANY from earlier runs was 0x%04x)\n", viol_before);
    for (unsigned i = 0; i < g_nsel; i++) viol_report_ch(g_sel[i]);

    printf("\n%s\n", bad ? "FAIL" : "PASS");
    free(a); free(acc); close(g_h2c); close(g_c2h);
    munmap(m, EMU_BAR2_SIZE); close(bfd);
    return bad ? 1 : 0;
}
