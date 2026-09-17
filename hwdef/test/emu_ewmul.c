// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_ewmul — run one ISR_EWMUL group and check every element.
//
//     ./emu_ewmul                          # feed 2, exec 3, dst 1 (the guide's group)
//     ./emu_ewmul --feed 0 --exec 1 --dst 2
//     ./emu_ewmul --l 4                    # 4 beats = 64 elements
//     ./emu_ewmul --sweep                  # walk the group across all 16 banks
//
// LAYER 3.  Needs layer 1 (the CFR answers) and the direct HBM aperture to hold
// what the host puts there.
//
// WHY THIS TOOL EXISTS SEPARATELY FROM emu_gemv
// EWMUL is the one compute opcode nothing has exercised.  It differs from MAC in
// every way that matters to a test:
//
//   MAC     accumulates into a PU latch; the result comes back through RD_MAC as
//           16 BF16 lanes in a GPR word.  One number per bank.
//   EWMUL   does NOT accumulate.  It writes a full elementwise product back into
//           a THIRD BANK'S DRAM.  OPSIZE beats of 16 BF16 lanes each.
//
// So the result never touches the GPR and RD_MAC is not part of the program.  The
// check is a direct-aperture read-back of the destination bank.
//
// THE GROUP (ISR guide §4)
// EWMUL fixes exactly one group per ISR — popcount(pu_mask) == 1.  The group is
// carried entirely by route:
//
//   feed s   route[s] = i     reads its DRAM row and streams it to the executor
//   exec i   route[i] = d     reads its OWN row, multiplies, streams the product on
//   dst  d                    writes the product into its DRAM row
//
//   pu_mask    = 1 << i       only the executor computes
//   gb_mc_mask = 0            the operand comes from a peer, not the GB
//   ROW/COL                   ONE field, so all three banks use the SAME offset
//
// The destination must NOT be in pu_mask: a bank that computes is a consumer, and
// naming it as the destination turns the product back into an input.
//
// WHY POISON THE DESTINATION FIRST
// Without it an EWMUL that silently did nothing could pass, because a previous run
// may have left the very values this one expects sitting in DRAM.  After poisoning,
// any element that comes back correct was put there by THIS run.
//
// WHY THE OPERANDS ARE WHAT THEY ARE
//   V[k] = 2.0 and W[k] = k + 1 makes element k exactly 2(k+1) — every product is
//   a small integer, exact in BF16, so a mismatch is a defect and not a rounding
//   argument.  Every element differs from every other, so a shifted or swapped
//   lane names itself instead of hiding.
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


#define BDF_DEFAULT "0000:01:00.0"
#define H2C_DEFAULT "/dev/qdma01000-MM-0"
#define C2H_DEFAULT "/dev/qdma01000-MM-1"
#define POISON_LANE 0x7FC1u          // a quiet NaN payload nothing here generates

static volatile uint8_t *g_bar;
static int g_h2c = -1, g_c2h = -1;

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
    printf("    bank |   RCD_RD    |   CCD_RD    |   RCD_WR    |   CCD_WR    |   REC_WR    | drop |\n");
    printf("         | cnt    max  | cnt    max  | cnt    max  | cnt    max  | cnt    max  |  cnt |\n");
    printf("    -----+-------------+-------------+-------------+-------------+-------------+------+\n");
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        uint32_t ca = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_CNT_A);
        uint32_t ma = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_MAX_A);
        uint32_t cb = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_CNT_B);
        uint32_t mb = viol_rd_ch(ch, VIOL_BANK_BASE(b) + VIOL_MAX_B);
        printf("    %4u | %3u    %3u  | %3u    %3u  | %3u    %3u  | %3u    %3u  | %3u    %3u  | %4u |%s\n",
               b,
               ca         & 0xffu, ma         & 0xffu,
               (ca >>  8) & 0xffu, (ma >>  8) & 0xffu,
               (ca >> 16) & 0xffu, (ma >> 16) & 0xffu,
               (ca >> 24) & 0xffu, (ma >> 24) & 0xffu,
               cb         & 0xffu, mb         & 0xffu,
               (cb >>  8) & 0xffu,
               ((any >> b) & 1u) ? "  <-- ANY" : "");
    }
}

static int axi_rw(uint64_t axi, void *buf, size_t len, bool writing)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    uint8_t *p = buf;
    size_t done = 0;
    while (done < len) {
        ssize_t n = writing ? pwrite(g_h2c, p + done, len - done, (off_t)(axi + done))
                            : pread (g_c2h, p + done, len - done, (off_t)(axi + done));
        if (n <= 0) {
            fprintf(stderr, "%s(0x%011" PRIx64 " +0x%zx): %s\n", writing ? "pwrite" : "pread",
                    axi, done, n < 0 ? strerror(errno) : "no progress");
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

// The dispatcher must be idle before anything it owns is touched.  A host IMEM
// write while a kernel runs has WREADY held low, the DMA times out, and one EIO
// latches the H2C engine — after which every write fails regardless of address.
static bool require_idle(const char *when, uint32_t timeout_ms)
{
    uint64_t t = now_us();
    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t > (uint64_t)timeout_ms * 1000ull) {
            fprintf(stderr,
                "REFUSING to %s: the dispatcher is still busy after %u ms "
                "(STATUS=0x%08x).\n"
                "  sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n",
                when, timeout_ms, st);
            return false;
        }
    }
}

// One group, one doorbell.  Returns 0 on a clean run, -1 on a host-side failure.
// *landed says whether the destination left its poison.
static int run_group(unsigned feed, unsigned exec, unsigned dst,
                     unsigned L, unsigned row, unsigned col,
                     const uint16_t *V, const uint16_t *W, uint16_t *out,
                     bool *landed, uint64_t *us, bool verbose)
{
    const size_t bytes = (size_t)L * EMU_WORD_BYTES;
    const uint64_t off = (uint64_t)row * EMU_ROW_BYTES + (uint64_t)col * EMU_WORD_BYTES;
    *landed = false;

    // ---- the program: EWMUL then EOS.  No RD_MAC — nothing accumulates. ----
    struct emu_isr prog[2];
    struct emu_isr_spec s;
    const char *e;

    s = emu_isr_default_ch(ISR_OP_EWMUL, 1u << g_ch);
    s.opsize = L; s.row = row; s.col = col;
    s.pu_mask = 1u << exec;          // exactly one executor
    s.gb_mc_mask = 0;                // peer sourced, not GB sourced
    s.route[feed] = (uint8_t)exec;   // feeder -> executor
    s.route[exec] = (uint8_t)dst;    // executor -> destination
    if ((e = emu_isr_build(&prog[0], &s))) { fprintf(stderr, "EWMUL refused: %s\n", e); return -1; }
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[1], &s))) { fprintf(stderr, "EOS refused: %s\n", e); return -1; }

    if (verbose) {
        printf("  program:\n");
        printf("    EWMUL   %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
               prog[0].w[3], prog[0].w[2], prog[0].w[1], prog[0].w[0]);
        printf("    EOS     %016" PRIx64 "%016" PRIx64 "%016" PRIx64 "%016" PRIx64 "\n",
               prog[1].w[3], prog[1].w[2], prog[1].w[1], prog[1].w[0]);
    }

    // ---- operands and poison, all through the direct aperture ----
    if (axi_rw(pim_bank(g_ch, feed) + off, (void *)V, bytes, true) < 0) return -1;
    if (axi_rw(pim_bank(g_ch, exec) + off, (void *)W, bytes, true) < 0) return -1;
    uint16_t *poison = malloc(bytes);
    if (!poison) return -1;
    for (size_t i = 0; i < bytes / 2; i++) poison[i] = POISON_LANE;
    if (axi_rw(pim_bank(g_ch, dst) + off, poison, bytes, true) < 0) { free(poison); return -1; }
    free(poison);

    if (!require_idle("load the device", 1000)) return -1;
    if (axi_rw(EMU_AXI_IMEM, prog, sizeof prog, true) < 0) return -1;

    if (!require_idle("ring the doorbell", 1000)) return -1;
    cfr_wr(CFR_PROG_LEN, 2);
    if (cfr_rd(CFR_PROG_LEN) != 2) { fprintf(stderr, "PROG_LEN readback failed\n"); return -1; }
    (void)cfr_rd(CFR_PROG_LEN);      // non-posted: order PROG_LEN before the doorbell

    uint64_t t0 = now_us();
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);

    // STATUS first (AXI-Lite, free), then read the destination back over C2H.
    // done is a held level and can be stale, so it decides nothing — the POISON
    // does.  EWMUL writes DRAM, so the read-back is the direct aperture.
    while (now_us() - t0 < 2000000ull)
        if (cfr_rd(CFR_STATUS) & CFR_STATUS_DONE) break;

    for (unsigned tries = 0; tries < 64 && now_us() - t0 < 2000000ull; tries++) {
        if (axi_rw(pim_bank(g_ch, dst) + off, out, bytes, false) < 0) return -1;
        bool all = true;
        for (unsigned k = 0; k < L * EMU_LANES_PER_WORD; k++)
            if (out[k] == POISON_LANE) { all = false; break; }
        if (all) { *landed = true; break; }
        if (tries >= 4) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
    }
    *us = now_us() - t0;
    return 0;
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--ch N] [--feed N] [--exec N] [--dst N] [--l N]\n""       [--row N] [--col N] [--sweep]\n"
"\n"
"Runs one ISR_EWMUL group and checks every element of the product.\n"
"\n"
"  --ch N         which channel to run on (default 0).  Stages operands in\n"
"                 that channel AND sets the ISR CH_MASK to 1<<N.\n"
"\n"
"  This tool reaches the banks through the DIRECT aperture and never through the MC\n"
"  port, so --ch names a PHYSICAL channel: it verifies one emulator channel on its\n"
"  own.  That makes it independent of the channel address map — ChRoBaCo or\n"
"  RoChBaCo only changes how an MC-space address is split up, and nothing here is an\n"
"  MC-space address.  (emu_mc is the one that is: its --ch is a place in MC space,\n"
"  which is why that one is ChRoBaCo-only.)\n"
"  --feed N    bank that streams the V operand        (default 2)\n"
"  --exec N    bank that multiplies                   (default 3)\n"
"  --dst  N    bank the product is written into       (default 1)\n"
"  --l N       beats, 1..64; elements = 16*N          (default 1)\n"
"  --row N     shared DRAM row for all three banks    (default 300)\n"
"  --col N     shared start beat inside the row       (default 0)\n"
"  --sweep     16 groups, rotating (feed,exec,dst) across every bank\n"
"\n"
"All three banks are DISTINCT and use the SAME (row, col): the ISR carries one\n"
"ROW field, so the offset is shared by construction.  The result is written to\n"
"the destination bank's DRAM, so it is checked by reading the direct aperture —\n"
"nothing lands in the GPR and there is no RD_MAC.\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p);
}

int main(int argc, char **argv)
{
    const char *bdf = BDF_DEFAULT, *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    unsigned feed = 2, exec = 3, dst = 1, L = 1, row = 300, col = 0;
    bool sweep = false;

    static const struct option lo[] = {
        { "ch",       required_argument, NULL, 'C' },
        { "feed", required_argument, NULL, 1 }, { "exec", required_argument, NULL, 2 },
        { "dst",  required_argument, NULL, 3 }, { "l",    required_argument, NULL, 4 },
        { "row",  required_argument, NULL, 5 }, { "col",  required_argument, NULL, 6 },
        { "sweep", no_argument,      NULL, 7 }, { "bdf",  required_argument, NULL, 8 },
        { "h2c",  required_argument, NULL, 9 }, { "c2h",  required_argument, NULL, 10 },
        { "help", no_argument,       NULL, 'h' }, { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: feed = (unsigned)strtoul(optarg, NULL, 0); break;
        case 2: exec = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: dst  = (unsigned)strtoul(optarg, NULL, 0); break;
        case 4: L    = (unsigned)strtoul(optarg, NULL, 0); break;
        case 5: row  = (unsigned)strtoul(optarg, NULL, 0); break;
        case 6: col  = (unsigned)strtoul(optarg, NULL, 0); break;
        case 7: sweep = true; break;
        case 8: bdf = optarg; break;
        case 9: h2c_path = optarg; break;
        case 10: c2h_path = optarg; break;
        case 'C': g_ch = (unsigned)strtoul(optarg, NULL, 0); break;

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
    if (g_ch >= PIM_NCH) {
        fprintf(stderr, "ERROR: --ch %u but %s has %u channel(s)\n",
                g_ch, PIM_PLATFORM_NAME, PIM_NCH);
        return 2;
    }
    if (feed >= EMU_NBANKS || exec >= EMU_NBANKS || dst >= EMU_NBANKS) {
        fprintf(stderr, "--feed/--exec/--dst must be 0..15\n"); return 2; }
    if (feed == exec || exec == dst || feed == dst) {
        fprintf(stderr, "the three banks must be distinct: a bank cannot feed itself, "
                        "and the destination must not be the executor\n"); return 2; }
    if (L < 1 || L > EMU_MAX_OPSIZE) { fprintf(stderr, "--l must be 1..64\n"); return 2; }
    if (col + L > EMU_BEATS_PER_ROW) { fprintf(stderr, "--col + --l must be <= 64\n"); return 2; }

    const unsigned N = L * EMU_LANES_PER_WORD;

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
                        "       sudo ./qdma_queues.sh setup\n",
                h2c_path, c2h_path, strerror(errno));
        return 1;
    }

    // ---- operands.  Products are small integers, exact in BF16. ----
    uint16_t *V = calloc(N, 2), *W = calloc(N, 2), *out = calloc(N, 2);
    uint16_t *expect = calloc(N, 2);
    if (!V || !W || !out || !expect) { fprintf(stderr, "out of memory\n"); return 1; }
    for (unsigned k = 0; k < N; k++) {
        V[k] = emu_f32_to_bf16(2.0f);
        W[k] = emu_f32_to_bf16((float)(k + 1));
        expect[k] = emu_f32_to_bf16(emu_bf16_to_f32(V[k]) * emu_bf16_to_f32(W[k]));
    }

    printf("ISR_EWMUL — elementwise product, result lands in a THIRD bank's DRAM\n");
    printf("  L = %u beats, %u elements   shared offset row %u col %u (bank +0x%llx)\n",
           L, N, row, col,
           (unsigned long long)((uint64_t)row * EMU_ROW_BYTES + (uint64_t)col * EMU_WORD_BYTES));
    printf("  operands  V[k] = 2.0 (feeder)   W[k] = k+1 (executor)   expect 2(k+1)\n\n");

    uint32_t viol_before = viol_rd(VIOL_ANY);
    viol_clear_all();   // 이번 실행분만 세도록
    unsigned groups = sweep ? EMU_NBANKS : 1, failed = 0;

    for (unsigned g = 0; g < groups; g++) {
        unsigned f = feed, x = exec, d = dst;
        if (sweep) { f = g; x = (g + 1) % EMU_NBANKS; d = (g + 2) % EMU_NBANKS; }

        printf("  group %2u: feed bank %2u -> exec bank %2u -> dst bank %2u\n", g, f, x, d);
        printf("            pu_mask 0x%04X  gb_mc_mask 0x0000  route[%u]=%u route[%u]=%u\n",
               1u << x, f, x, x, d);

        bool landed; uint64_t us;
        if (run_group(f, x, d, L, row, col, V, W, out, &landed, &us, !sweep) < 0) return 1;

        if (!landed) {
            printf("            FAIL: the destination never left its poison in 2 s.\n");
            printf("            EWMUL issues three bank slots (feed READ, dst WRITE,\n"
                   "            exec EWMUL); a shape the hardware does not accept stalls\n"
                   "            here with no diagnostic.\n");
            failed++;
            continue;
        }

        unsigned bad = 0, shown = 0;
        for (unsigned k = 0; k < N; k++) {
            if (emu_bf16_same(out[k], expect[k])) continue;
            bad++;
            if (shown < 6) {
                printf("            elem %-4u want %8.0f 0x%04x  got %8.0f 0x%04x",
                       k, (double)emu_bf16_to_f32(expect[k]), expect[k],
                       (double)emu_bf16_to_f32(out[k]), out[k]);
                if (out[k] == POISON_LANE) printf("   still POISON");
                else for (unsigned o = 0; o < N; o++)
                    if (o != k && emu_bf16_same(out[k], expect[o])) {
                        printf("   this is element %u's product", o); break; }
                printf("\n");
                shown++;
            }
        }
        if (bad) {
            printf("            FAIL: %u of %u elements wrong", bad, N);
            if (shown < bad) printf(" (%u more not shown)", bad - shown);
            printf("\n");
            failed++;
        } else {
            printf("            PASS: all %u elements are the product, %" PRIu64 " us\n", N, us);
        }
    }

    // ---- violations ----
    // viol_before 는 clear 이전 잔재라 이번 실행과 무관하다. 참고로만 찍는다.
    printf("\n  violation counters were CLEARED before this run "
           "(stale ANY from earlier runs was 0x%04x)\n", viol_before);
    viol_report_ch(g_ch);
    printf("    EWMUL writes DRAM, so RCD_WR / RECOVERY_WR are the interesting ones\n"
           "    here — MAC never exercises the write side of the timing model.\n");

    printf("\n%s\n", failed ? "FAIL" : "PASS");
    free(V); free(W); free(out); free(expect);
    close(g_h2c); close(g_c2h); munmap(m, EMU_BAR2_SIZE); close(bfd);
    return failed ? 1 : 0;
}
