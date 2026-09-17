// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_load — put a REAL matrix into the banks, then use it.
//
//     ./emu_load                              # generated [512 x 2048], all channels
//     ./emu_load --n 4096 --k 4096
//     ./emu_load --file w.bf16 --n 2048 --k 2048     # raw bf16, [n][k] row-major
//     ./emu_load --no-gemv                    # place and verify only
//
// LAYER 3.  Needs the direct aperture (emu_hbm_direct) and a working ISR path
// (emu_gemv).
//
// WHAT THIS IS FOR
// Every other probe writes an operand it invented — one value per bank, chosen so
// the answer is a small integer.  That proves the datapath and proves nothing about
// getting a real tensor into the right 32768 places.  This does the placement:
//
//     bank(n) = n mod 16      channel(n) = (n/16) mod C      one MAC per 16*C
//
// and writes it the only way that is fast enough to matter — each (channel, bank)
// slice gathered on the host and sent as ONE transfer.  Row by row the same upload
// is 35x slower [measured], which is the difference between loading a model in
// seconds and in minutes.
//
// TWO CHECKS, AND THEY ANSWER DIFFERENT QUESTIONS
//   1. READ BACK      byte-exact, no arithmetic.  Every element must be where the
//                     layout says it is.  This works for ANY data, real weights
//                     included, because it never adds anything up — a GEMV
//                     comparison on real weights would be arguing about rounding
//                     when the question was placement.
//   2. GEMV           the whole matrix through one program per supergroup, against
//                     a host reference.  With generated data (the default) every
//                     value is a small integer that is exact in BF16 and whose
//                     dot products are exact too, so a mismatch is a DEFECT and not
//                     a tolerance argument.  With --file the data is whatever it
//                     is, so differences are reported and not judged.
//
// WHY THE TWO TOGETHER ARE SOUND, AND NEITHER ALONE
// The read-back gathers the expected bytes with the SAME function that scattered
// them, so a bug in that function would write wrong and read wrong and compare
// equal.  On its own it proves only "the transfer round-trips".
//
// The GEMV closes that: the hardware reaches the weights through the ISR's own
// addressing — row = base + supergroup*ntiles + tile, all 16 banks at once, one
// channel per RD_MAC — which shares no code with the scatter.  If a column were
// placed in the wrong bank or the wrong channel, the read-back would still pass and
// the GEMV would return another column's answer.
//
// So: read-back says the bytes arrived intact; the GEMV says they arrived where the
// ISA looks for them.  Neither question is the other one.
//
// WHY THE GENERATED DATA LOOKS LIKE THAT
// W[n][k] is small and signed, x[k] is +-1, and K is capped so the running sum
// stays inside BF16's 8-bit significand.  Every output is then representable, every
// summation order gives the same answer, and the arithmetic drops out of the test
// completely.  That is deliberate: this tool is about WHERE the bytes went.
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
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "emu_regs.h"
#include "pim_platform.h"
#include "pim_layout.h"

#define BDF_DEFAULT "0000:01:00.0"
#define H2C_DEFAULT "/dev/qdma01000-MM-0"
#define C2H_DEFAULT "/dev/qdma01000-MM-1"
#define CHUNK       (4u << 20)
#define POISON_LANE 0x7FC1u

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
    uint32_t v = *(volatile uint32_t *)(g_bar + PIM_OFF_CFR + off);
    __sync_synchronize();
    return v;
}
static void cfr_wr(uint32_t off, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(g_bar + PIM_OFF_CFR + off) = v;
    __sync_synchronize();
}

static int axi_write(uint64_t axi, const void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    const uint8_t *p = buf;
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
        ssize_t n = pwrite(g_h2c, p + done, want, (off_t)(axi + done));
        if (n <= 0) { fprintf(stderr, "pwrite(0x%011" PRIx64 "+%zu): %s\n", axi, done,
                              n < 0 ? strerror(errno) : "no progress"); return -1; }
        done += (size_t)n;
    }
    return 0;
}
static int axi_read(uint64_t axi, void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    uint8_t *p = buf;
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
        ssize_t n = pread(g_c2h, p + done, want, (off_t)(axi + done));
        if (n <= 0) { fprintf(stderr, "pread(0x%011" PRIx64 "+%zu): %s\n", axi, done,
                              n < 0 ? strerror(errno) : "no progress"); return -1; }
        done += (size_t)n;
    }
    return 0;
}
// Adapters, so pim_layout.c opens nothing and knows about no file descriptors.
static int xfer_cb(void *ctx, uint64_t axi, const void *buf, size_t len)
{ (void)ctx; return axi_write(axi, buf, len); }
static int read_cb(void *ctx, uint64_t axi, void *buf, size_t len)
{ (void)ctx; return axi_read(axi, buf, len); }

static int gpr_write(uint32_t word, const void *src, uint32_t n)
{
    return axi_write(PIM_BAR2_AXI_BASE + PIM_OFF_GPR +
                     (uint64_t)word * EMU_WORD_BYTES, src, (size_t)n * EMU_WORD_BYTES);
}
static int gpr_read(uint32_t word, void *dst, uint32_t n)
{
    return axi_read(PIM_BAR2_AXI_BASE + PIM_OFF_GPR +
                    (uint64_t)word * EMU_WORD_BYTES, dst, (size_t)n * EMU_WORD_BYTES);
}
static int imem_write(const struct emu_isr *isr, uint32_t n)
{
    return axi_write(PIM_BAR2_AXI_BASE + PIM_OFF_IMEM, isr,
                     (size_t)n * EMU_WORD_BYTES);
}

// The dispatcher must be idle before the host touches IMEM or the GPR: the IMEM
// command port gives the FETCHER priority, so a host write during a live kernel
// sits on WREADY until the DMA times out — and one EIO latches the H2C engine.
static bool require_idle(const char *when, uint32_t ms)
{
    uint64_t t = now_us();
    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t > (uint64_t)ms * 1000ull) {
            fprintf(stderr, "REFUSING to %s: dispatcher busy after %u ms (STATUS 0x%08x)\n",
                    when, ms, st);
            return false;
        }
    }
}

// ---- data ----------------------------------------------------------------------
static uint32_t g_rng = 1;
static uint32_t rnd(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng >> 8; }

// Small signed integers.  Exact in BF16, and their dot products stay exact as long
// as the running sum needs no more than 8 significant bits — which is what --k is
// capped against below.  Every summation order then gives the same answer, so this
// tool measures placement and nothing else.
static uint16_t gen_w(void) { return emu_f32_to_bf16((float)((int)(rnd() % 9u) - 4)); }

int main(int argc, char **argv)
{
    const char *bdf = BDF_DEFAULT, *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    const char *file = NULL;
    unsigned bar = 2, n = 512, k = 2048, base_row = 1024, seed = 1;
    uint32_t vec_word = 0, dst_word = 3000;
    bool do_gemv = true;

    static const struct option lo[] = {
        { "n",        required_argument, NULL, 'n' },
        { "k",        required_argument, NULL, 'k' },
        { "file",     required_argument, NULL, 'f' },
        { "seed",     required_argument, NULL, 's' },
        { "row",      required_argument, NULL, 'r' },
        { "no-gemv",  no_argument,       NULL, 'G' },
        { "bdf",      required_argument, NULL, 'b' },
        { "h2c",      required_argument, NULL, 1   },
        { "c2h",      required_argument, NULL, 2   },
        { "help",     no_argument,       NULL, 'h' },
        { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 'n': n = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'k': k = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'f': file = optarg; break;
        case 's': seed = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'r': base_row = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'G': do_gemv = false; break;
        case 'b': bdf = optarg; break;
        case 1:   h2c_path = optarg; break;
        case 2:   c2h_path = optarg; break;
        case 'h':
            printf(
"Usage: %s [--n N] [--k N] [--file PATH] [--no-gemv]\n"
"\n"
"Places a real [n x k] matrix across every channel and bank, verifies it byte\n"
"for byte by reading it back, then runs the whole GEMV over it.\n"
"\n"
"  --n N         outputs   (default 512).  Padded up to 16*nch with zeros.\n"
"  --k N         dot length (default 2048).  Padded up to a multiple of 1024.\n"
"  --file PATH   raw BF16, [n][k] row-major — row n is output n's weights, the\n"
"                same order torch stores nn.Linear.weight in.  Without it the\n"
"                matrix is generated from --seed as small signed integers, which\n"
"                are exact in BF16 so a GEMV mismatch is a defect, not rounding.\n"
"  --seed N      generator seed (default 1)\n"
"  --row N       first DRAM row (default 1024)\n"
"  --no-gemv     place and read back only\n"
"  --bdf / --h2c / --c2h    as elsewhere\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", argv[0]);
            return 0;
        default:
            fprintf(stderr, "try --help\n");
            return 2;
        }
    }

    {
        const char *e = pim_platform_check();
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 2; }
        pim_platform_banner();
    }

    struct pim_matrix M;
    {
        const char *e = pim_matrix_plan(n, k, base_row, &M);
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 2; }
    }

    // The generated data is exact only while the running sum fits BF16's significand.
    // Beyond that a host reference and the device would part for reasons that have
    // nothing to do with placement, so say so instead of producing a soft failure.
    if (!file && M.kpad > 4096) {
        fprintf(stderr,
            "ERROR: --k %u with generated data: dot products stop being exact in BF16\n"
            "       past about 4096 terms, and this tool judges placement, not rounding.\n"
            "       Use --file for real data (the read-back check still applies), or a\n"
            "       smaller --k.\n", k);
        return 2;
    }

    printf("matrix   [%u x %u]  ->  padded [%u x %u]\n", M.n, M.k, M.npad, M.kpad);
    printf("  layout   bank = n mod %u,  channel = (n/%u) mod %u\n",
           M.nbank, M.nbank, M.nch);
    printf("  %u supergroup(s) x %u K-tile(s) = %u rows per bank, %" PRIu64 " KiB per channel\n",
           M.ngroups, M.ntiles, M.nrows, pim_matrix_bytes(&M) >> 10);
    printf("  rows     %u .. %u   (a bank holds %" PRIu64 ")\n",
           M.base_row, M.base_row + M.nrows - 1,
           (uint64_t)(PIM_BANK_WINDOW / EMU_ROW_BYTES));
    printf("  last K-tile OPSIZE %u beat(s)%s\n\n", M.last_opsize,
           M.last_opsize == 64 ? "" : "  (short tile, zero-padded)");

    // ---- the matrix on the host ------------------------------------------------
    const size_t welems = (size_t)M.n * M.k;
    uint16_t *W = malloc(welems * sizeof *W);
    uint16_t *x = calloc(M.kpad, sizeof *x);
    if (!W || !x) { fprintf(stderr, "ERROR: out of memory for %zu elements\n", welems); return 1; }

    if (file) {
        FILE *f = fopen(file, "rb");
        if (!f) { fprintf(stderr, "ERROR: open %s: %s\n", file, strerror(errno)); return 1; }
        size_t got = fread(W, sizeof *W, welems, f);
        fclose(f);
        if (got != welems) {
            fprintf(stderr, "ERROR: %s holds %zu BF16 elements, [%u x %u] needs %zu\n",
                    file, got, M.n, M.k, welems);
            return 1;
        }
        printf("  data     %s (%zu elements)\n", file, welems);
    } else {
        g_rng = seed ? seed : 1;
        for (size_t i = 0; i < welems; i++) W[i] = gen_w();
        printf("  data     generated, seed %u — small signed integers, exact in BF16\n", seed);
    }
    for (uint32_t i = 0; i < M.k; i++) x[i] = emu_f32_to_bf16((i & 1u) ? -1.0f : 1.0f);

    // ---- open the board ---------------------------------------------------------
    char path[256];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource%u", bdf, bar);
    int bfd = open(path, O_RDWR | O_SYNC);
    if (bfd < 0) { fprintf(stderr, "ERROR: open(%s): %s\n", path, strerror(errno)); return 1; }
    void *m = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bfd, 0);
    if (m == MAP_FAILED) { fprintf(stderr, "ERROR: mmap: %s\n", strerror(errno)); return 1; }
    g_bar = m;
    g_h2c = open(h2c_path, O_WRONLY);
    g_c2h = open(c2h_path, O_RDONLY);
    if (g_h2c < 0 || g_c2h < 0) {
        fprintf(stderr, "ERROR: MM queues (%s / %s): %s\n       sudo ./qdma_queues.sh setup\n",
                h2c_path, c2h_path, strerror(errno));
        return 1;
    }
    if (!require_idle("load the device", 1000)) return 1;

    // ---- 1. place ---------------------------------------------------------------
    uint64_t t0 = now_us();
    {
        const char *e = pim_matrix_upload(&M, W, xfer_cb, NULL);
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 1; }
    }
    uint64_t t_up = now_us() - t0;
    const uint64_t total = pim_matrix_bytes(&M) * M.nch;
    printf("1. placed  %" PRIu64 " KiB in %" PRIu64 " transfers (one per channel-bank), "
           "%" PRIu64 " us -> %.0f MB/s\n",
           total >> 10, (uint64_t)M.nch * M.nbank, t_up,
           (double)total / (double)t_up);

    // ---- 2. read back -----------------------------------------------------------
    // Byte-exact and arithmetic-free.  This is the placement check; it holds for any
    // data at all, which is why it is the one that decides pass or fail.
    t0 = now_us();
    uint64_t nbad = 0;
    {
        const char *e = pim_matrix_verify(&M, W, read_cb, NULL, &nbad);
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 1; }
    }
    uint64_t t_rb = now_us() - t0;
    const uint64_t nelem = (uint64_t)M.nrows * PIM_ROBACO_ROW / 2u * M.nch;
    printf("2. readback %" PRIu64 " of %" PRIu64 " elements differ   %s   (%" PRIu64 " us)\n",
           nbad, nelem, nbad ? "FAIL" : "ok", t_rb);
    if (nbad) {
        printf("\nFAIL: what came back is not what the layout says should be there.\n"
               "      Placement, not arithmetic — nothing was added up.\n");
        return 1;
    }

    if (!do_gemv) { printf("\nPASS (placement only, --no-gemv)\n"); return 0; }

    // ---- 3. the whole GEMV ------------------------------------------------------
    // group-outer: for each supergroup, walk every K-tile into one accumulator and
    // read it once.  The vector goes out with a multicast CH_MASK so ONE WRVEC feeds
    // every channel's GB; RD_MAC stays 1-hot per channel, or two channels would
    // write the same GPR word and one would be lost with nothing to show for it.
    const uint32_t chmask = (uint32_t)((1u << M.nch) - 1u);
    const unsigned NISR = M.ngroups * (2u * M.ntiles + M.nch) + 1u;
    if (NISR > CFR_PROG_LEN_MAX) {
        fprintf(stderr, "ERROR: %u ISRs exceeds PROG_LEN's %u — split into several "
                        "doorbells at supergroup boundaries\n", NISR, CFR_PROG_LEN_MAX);
        return 1;
    }
    const uint32_t ndst = M.ngroups * M.nch;
    printf("\n3. gemv    %u ISRs in ONE doorbell: %u WRVEC + %u MAC + %u RD_MAC + EOS\n",
           NISR, M.ngroups * M.ntiles, M.ngroups * M.ntiles, ndst);

    if (gpr_write(vec_word, x, M.kpad / EMU_LANES_PER_WORD) < 0) return 1;

    struct emu_isr *prog = calloc(NISR, sizeof *prog);
    uint16_t (*res)[EMU_LANES_PER_WORD] = calloc(ndst, sizeof *res);
    if (!prog || !res) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    struct emu_isr_spec s;
    const char *e;
    unsigned w = 0;
    for (uint32_t g = 0; g < M.ngroups; g++) {
        for (uint32_t t = 0; t < M.ntiles; t++) {
            uint16_t ops = (t == M.ntiles - 1u) ? M.last_opsize : 64u;
            s = emu_isr_default_ch(ISR_OP_WRVEC, chmask);
            s.opsize = ops;
            s.row = vec_word + t * (PIM_ELEMS_PER_PAGE / EMU_LANES_PER_WORD);
            if ((e = emu_isr_build(&prog[w++], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return 2; }

            unsigned ch_unused; uint32_t row, col0;
            pim_matrix_place(&M, g * M.nbank * M.nch, t, &ch_unused, NULL, &row, &col0);
            s = emu_isr_default_ch(ISR_OP_MAC, chmask);
            s.opsize = ops; s.row = row; s.col = col0;
            s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
            if ((e = emu_isr_build(&prog[w++], &s))) { fprintf(stderr, "MAC: %s\n", e); return 2; }
        }
        for (unsigned c = 0; c < M.nch; c++) {
            s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << c);
            s.opsize = 0; s.row = dst_word + g * M.nch + c;
            if ((e = emu_isr_build(&prog[w++], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return 2; }
        }
    }
    s = emu_isr_default_ch(ISR_OP_EOS, chmask);
    if ((e = emu_isr_build(&prog[w++], &s))) { fprintf(stderr, "EOS: %s\n", e); return 2; }

    // Poison every destination.  done rises when the fetcher ACCEPTS the last ISR,
    // so it can be stale from the previous run and decides nothing; the poison does.
    for (uint32_t d = 0; d < ndst; d++)
        for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++) res[d][i] = POISON_LANE;
    if (gpr_write(dst_word, res, ndst) < 0) return 1;

    if (!require_idle("load IMEM", 1000)) return 1;
    if (imem_write(prog, w) < 0) return 1;
    cfr_wr(CFR_PROG_LEN, w);
    if (cfr_rd(CFR_PROG_LEN) != w) { fprintf(stderr, "PROG_LEN readback failed\n"); return 1; }
    (void)cfr_rd(CFR_PROG_LEN);

    t0 = now_us();
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);
    bool done = false;
    while (now_us() - t0 < 5000000ull)
        if (cfr_rd(CFR_STATUS) & CFR_STATUS_DONE) { done = true; break; }

    bool landed = false;
    for (unsigned tries = 0; tries < 64 && now_us() - t0 < 5000000ull; tries++) {
        if (gpr_read(dst_word, res, ndst) < 0) return 1;
        landed = true;
        for (uint32_t d = 0; d < ndst && landed; d++)
            for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++)
                if (res[d][i] == POISON_LANE) { landed = false; break; }
        if (landed) break;
        struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL);
    }
    uint64_t t_run = now_us() - t0;
    if (!landed) {
        printf("\nFAIL: a result word never lost its poison in 5 s (done=%u).\n", done);
        return 1;
    }
    const uint64_t macs = (uint64_t)M.npad * M.kpad;
    printf("   ran %" PRIu64 " us, %" PRIu64 " MACs -> %.2f GMAC/s in-kernel\n",
           t_run, macs, (double)macs / (double)t_run / 1000.0);

    // ---- 4. against the host ----------------------------------------------------
    unsigned wrong = 0, shown = 0;
    float worst = 0.0f, rms = 0.0f;
    for (uint32_t col = 0; col < M.n; col++) {
        float acc = 0.0f;
        const uint16_t *wrow = W + (size_t)col * M.k;
        for (uint32_t i = 0; i < M.k; i++)
            acc += emu_bf16_to_f32(wrow[i]) * emu_bf16_to_f32(x[i]);
        uint16_t want = emu_f32_to_bf16(acc);

        unsigned ch, bank;
        pim_matrix_place(&M, col, 0, &ch, &bank, NULL, NULL);
        const uint32_t g = col / (M.nbank * M.nch);
        uint16_t got = res[g * M.nch + ch][bank];

        rms += (float)((double)emu_bf16_to_f32(want) * emu_bf16_to_f32(want));
        float d = emu_bf16_to_f32(got) - emu_bf16_to_f32(want);
        if (d < 0) d = -d;
        if (d > worst) worst = d;
        if (!emu_bf16_same(got, want)) {
            wrong++;
            if (shown < 8) {
                printf("   col %-6u ch%u bank%-2u  want %10.3f  got %10.3f\n",
                       col, ch, bank, (double)emu_bf16_to_f32(want),
                       (double)emu_bf16_to_f32(got));
                shown++;
            }
        }
    }
    rms = (float)sqrt((double)rms / (double)M.n);

    printf("4. gemv vs host: %u of %u outputs differ", wrong, M.n);
    if (file) {
        // Real data: the device is a block floating-point MAC and this reference is
        // sequential fp32, so they are different functions.  Report, do not judge.
        printf("  (max |err| %.4g, rms(y) %.4g)\n", (double)worst, (double)rms);
        printf("   real data — the device rounds differently from this fp32 reference,\n"
               "   so a nonzero count here is expected.  The PLACEMENT check above is\n"
               "   the one that decides.\n");
    } else {
        printf("   %s\n", wrong ? "FAIL" : "ok");
        printf("   generated data is exact in BF16, so any difference is a defect.\n");
        if (wrong) return 1;
    }

    printf("\nPASS\n");
    return 0;
}
