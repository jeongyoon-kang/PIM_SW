// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_chain — what COMPOSES inside a single ISR program.
//
//     ./emu_chain                    # run every probe
//     ./emu_chain --test chain       # does the accumulator survive MAC -> MAC?
//     ./emu_chain --test rewind      # does one WRVEC feed more than one MAC?
//     ./emu_chain --test kchunk      # WRVEC/MAC/WRVEC/MAC — a dot product split in two
//     ./emu_chain --test gemv        # WRVEC once, then G x {MAC, RD_MAC} — the real kernel
//
// LAYER 3, same as emu_gemv.  emu_gemv proves ONE tile is right.  This proves the
// four composition rules a compiler has to rely on to emit a whole GEMV — or a
// whole transformer layer — as ONE IMEM image behind ONE doorbell.
//
// WHY THESE FOUR
//   Every one of them is load bearing for a program longer than four words, and
//   none of them is observable from emu_gemv, which issues exactly one MAC and one
//   RD_MAC.  They are asked separately so that a failure names its own cause.
//
//   chain   MAC accumulates into a per-bank latch.  If the latch survives from one
//           MAC ISR to the next, a dot product longer than the 64-beat GB limit is
//           just several MACs in a row.  If it does NOT, K > 1024 has to come back
//           to the host as partial sums and the whole cost model changes.
//
//   rewind  the ISR guide says a GB-sourced MAC rewinds the GB read pointer when it
//           starts.  If true, one WRVEC serves every MAC that wants the same vector,
//           and a GEMV over N outputs costs 1 vector load, not N/16 of them.  The
//           probe reads the SAME weights twice with one WRVEC between them and a
//           read-clearing RD_MAC in the middle: both answers must be the 1x value.
//           A second answer of zero would mean the GB was drained and not rewound.
//
//   kchunk  the previous two combined, which is what K > 1024 actually looks like:
//           two different weight rows, two different vectors, one accumulator.  The
//           second vector is 2.0 rather than 1.0 so that a GB that was NOT refilled
//           is a wrong number and not a coincidence.
//
//   gemv    the kernel a compiler emits: one WRVEC, then a MAC/RD_MAC pair per group
//           of 16 outputs.  Every group has different weights and lands in its own
//           GPR word, so a group whose result leaked into its neighbour is visible.
//
// WHAT COUNTS AS AN ANSWER
//   Every destination GPR word is poisoned before the doorbell and the result is
//   believed only once the poison is gone — the same rule emu_gemv uses, for the
//   same reason: STATUS[31] rises when the fetcher ACCEPTS the last ISR, so it can
//   be stale from the previous run and decides nothing.
//
// Exit: 0 all probes pass   1 a probe failed   2 usage
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
#include <math.h>
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
#define CHUNK       (4u << 20)
#define POISON_LANE 0x7FC1u          // a quiet NaN payload nothing here generates
#define MAX_PROG    (EMU_IMEM_WORDS)     // a program may fill IMEM
#define MAX_DST     4096u                // result words, always a contiguous run

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

static int axi_write(uint64_t axi, const void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    const uint8_t *p = buf;
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
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
static int axi_read(uint64_t axi, void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    uint8_t *p = buf;
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
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
static int gpr_write(uint32_t word, const void *src, uint32_t n)
{
    return axi_write(EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, src,
                     (size_t)n * EMU_WORD_BYTES);
}
static int gpr_read(uint32_t word, void *dst, uint32_t n)
{
    return axi_read(EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, dst,
                    (size_t)n * EMU_WORD_BYTES);
}

// The dispatcher must be idle before the host touches IMEM or the GPR: the IMEM
// command port gives the FETCHER priority, so a host write during a live kernel
// sits on WREADY until the DMA times out — and one EIO latches the H2C engine.
static bool require_idle(const char *when, uint32_t timeout_ms)
{
    uint64_t t = now_us();
    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t > (uint64_t)timeout_ms * 1000ull) {
            fprintf(stderr,
                "REFUSING to %s: dispatcher still busy after %u ms (STATUS=0x%08x).\n"
                "  sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n",
                when, timeout_ms, st);
            return false;
        }
    }
}

static float tree_sum(float *v, unsigned n)
{
    while (n > 1) {
        unsigned m = 0;
        for (unsigned i = 0; i + 1 < n; i += 2) v[m++] = v[i] + v[i + 1];
        if (n & 1) v[m++] = v[n - 1];
        n = m;
    }
    return n ? v[0] : 0.0f;
}

// ---- a program with any number of result words --------------------------------
// The result words of a program are ALWAYS a contiguous run: RD_MAC number g lands
// in dst_base+g.  That is not an accident of these probes, it is what a compiler
// wants — it makes the whole result of a GEMV one C2H transfer instead of N/16 of
// them, and at 909 MB/s read a per-word transfer would cost more than the compute.
struct run_result {
    uint16_t (*lane)[EMU_LANES_PER_WORD];   // [ndst][16], owned by the caller
    uint32_t st_after, polls, reads;
    uint64_t us, load_us, drain_us;
    bool     landed, seen_done;
};

static int run_program(const struct emu_isr *prog, unsigned n,
                       uint32_t dst_base, unsigned ndst, struct run_result *r)
{
    if (n == 0 || n > MAX_PROG || ndst == 0 || ndst > MAX_DST) {
        fprintf(stderr, "program length %u / %u result words out of range\n", n, ndst);
        return -1;
    }
    r->st_after = r->polls = r->reads = 0;
    r->us = r->load_us = r->drain_us = 0;
    r->landed = r->seen_done = false;

    uint64_t tl = now_us();
    if (!require_idle("load the device", 1000)) return -1;
    if (axi_write(EMU_AXI_IMEM, prog, (size_t)n * EMU_WORD_BYTES) < 0) return -1;

    // One poison write for the whole result range.
    for (unsigned d = 0; d < ndst; d++)
        for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++) r->lane[d][i] = POISON_LANE;
    if (gpr_write(dst_base, r->lane, ndst) < 0) return -1;
    r->load_us = now_us() - tl;

    if (!require_idle("ring the doorbell", 1000)) return -1;
    cfr_wr(CFR_PROG_LEN, n);
    if (cfr_rd(CFR_PROG_LEN) != n) { fprintf(stderr, "PROG_LEN readback failed\n"); return -1; }
    (void)cfr_rd(CFR_PROG_LEN);      // non-posted: orders the posted writes ahead of the doorbell

    uint64_t t0 = now_us();
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);

    while (now_us() - t0 < 10000000ull) {
        r->polls++;
        if (cfr_rd(CFR_STATUS) & CFR_STATUS_DONE) { r->seen_done = true; break; }
    }
    // The whole result range in one C2H transfer.  done can be stale, so what
    // decides is the poison: the LAST word to be written is the last RD_MAC's, so
    // checking every word catches a program that stopped part way.
    uint64_t td = now_us();
    for (; r->reads < 64 && now_us() - t0 < 10000000ull; ) {
        r->reads++;
        if (gpr_read(dst_base, r->lane, ndst) < 0) return -1;
        bool all = true;
        for (unsigned d = 0; d < ndst && all; d++)
            for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++)
                if (r->lane[d][i] == POISON_LANE) { all = false; break; }
        if (all) { r->landed = true; break; }
        if (r->reads >= 4) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
    }
    r->drain_us = now_us() - td;
    r->us = now_us() - t0;
    r->st_after = cfr_rd(CFR_STATUS);
    return 0;
}

// Every probe uses the same result buffer; sized once for the largest run.
static uint16_t (*g_lane)[EMU_LANES_PER_WORD];

// ---- operand staging ----------------------------------------------------------
// Weight row `row` gets A_b[k] = val + b in every one of the 16 banks, so each bank
// has a DIFFERENT answer and a lane holding its neighbour's value is visible.
static int put_weights(unsigned row, unsigned val, unsigned L)
{
    uint16_t a[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    const unsigned K = L * EMU_LANES_PER_WORD;
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        for (unsigned k = 0; k < K; k++) a[k] = emu_f32_to_bf16((float)(val + b));
        if (axi_write(pim_bank(g_ch, b) + (uint64_t)row * EMU_ROW_BYTES, a,
                      (size_t)L * EMU_WORD_BYTES) < 0) return -1;
    }
    return 0;
}
// One MAC of L beats against a row holding (val + b), with a vector of `vv`:
// lane b = sum over K of vv*(val+b), summed as the device does it.
static uint16_t expect_one(unsigned b, unsigned val, float vv, unsigned L)
{
    const unsigned K = L * EMU_LANES_PER_WORD;
    float acc[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    for (unsigned k = 0; k < K; k++)
        acc[k] = vv * emu_bf16_to_f32(emu_f32_to_bf16((float)(val + b)));
    return emu_f32_to_bf16(tree_sum(acc, K));
}

// G groups x C chunks = G*C rows, staged the way an allocator MUST stage them:
// bank b's whole slice is CONTIGUOUS in the direct aperture, so it is ONE pwrite.
// Row by row it would be G*C*16 transfers of 2 KB and the upload would dominate
// everything — [measured] 64 rows cost 13.9 ms that way and 0.4 ms this way.  A
// weight layout that does not keep each bank's slice contiguous cannot be loaded
// at speed.
static int put_weight_block_c(unsigned row, unsigned val, unsigned L, unsigned G, unsigned C)
{
    const size_t nrow = (size_t)G * C;
    uint8_t *blk = calloc(nrow, EMU_ROW_BYTES);
    if (!blk) return -1;
    const unsigned K = L * EMU_LANES_PER_WORD;
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        memset(blk, 0, nrow * EMU_ROW_BYTES);
        for (unsigned g = 0; g < G; g++)
            for (unsigned c = 0; c < C; c++) {
                uint16_t *r = (uint16_t *)(blk + ((size_t)g * C + c) * EMU_ROW_BYTES);
                uint16_t w = emu_f32_to_bf16((float)(val + 16 * g + b));
                for (unsigned k = 0; k < K; k++) r[k] = w;
            }
        if (axi_write(pim_bank(g_ch, b) + (uint64_t)row * EMU_ROW_BYTES, blk,
                      nrow * EMU_ROW_BYTES) < 0) { free(blk); return -1; }
    }
    free(blk);
    return 0;
}

// C chunks accumulate into one fp32 latch, rounded once at RD_MAC.
static uint16_t expect_c(unsigned b, unsigned val, unsigned L, unsigned C)
{
    const unsigned K = L * EMU_LANES_PER_WORD;
    static float acc[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    for (unsigned k = 0; k < K; k++) acc[k] = emu_bf16_to_f32(emu_f32_to_bf16((float)(val + b)));
    return emu_f32_to_bf16((float)C * tree_sum(acc, K));
}

static int put_vector(uint32_t word, float v, unsigned L)
{
    uint16_t vec[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    for (unsigned k = 0; k < L * EMU_LANES_PER_WORD; k++) vec[k] = emu_f32_to_bf16(v);
    return gpr_write(word, vec, L);
}

// ---- reporting ----------------------------------------------------------------
// Prints all 16 lanes and returns the number that are wrong.  On a mismatch it
// says what the value WOULD have been under the alternative hypothesis, because
// that is the whole diagnostic: "you got the 1x answer" means no accumulation,
// "you got zero" means the GB ran dry.
static unsigned judge(const char *what, const uint16_t *got, const uint16_t *want,
                      const uint16_t *alt, const char *alt_name)
{
    unsigned bad = 0;
    printf("    lane | expected        | got             |\n");
    printf("    -----+-----------------+-----------------+------\n");
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        printf("     %2u  | %8.0f 0x%04x | %8.0f 0x%04x | ", b,
               (double)emu_bf16_to_f32(want[b]), want[b],
               (double)emu_bf16_to_f32(got[b]),  got[b]);
        if (emu_bf16_same(got[b], want[b])) {
            printf("ok\n");
        } else {
            bad++;
            printf("WRONG");
            if (alt && emu_bf16_same(got[b], alt[b])) printf(" — this is the %s value", alt_name);
            else if (emu_bf16_to_f32(got[b]) == 0.0f)  printf(" — zero");
            printf("\n");
        }
    }
    printf("    %s: %s\n", what, bad ? "FAIL" : "PASS");
    return bad;
}

// ============================== the probes =====================================
// Every probe returns the number of wrong lanes.  ~0u means it could not run.

// A. Does the accumulator survive from one MAC ISR to the next?
//    WRVEC(v=1) | MAC(row,L) | MAC(row,L) | RD_MAC | EOS   ->  2x the one-MAC value.
//    If the latch is cleared between ISRs the answer is the 1x value instead, and
//    `alt` names it in the report.
static unsigned probe_chain(unsigned row, unsigned L, unsigned val,
                            uint32_t vec_word, uint32_t dst_word)
{
    printf("\n=== chain — does the accumulator survive MAC -> MAC? ===\n");
    printf("    WRVEC(L=%u) | MAC(row %u) | MAC(row %u) | RD_MAC | EOS\n", L, row, row);
    printf("    expecting 2x the single-MAC value.  1x would mean the latch is cleared\n"
           "    between ISRs, and K > 1024 could not be chained on the device.\n");

    struct emu_isr prog[5];
    struct emu_isr_spec s;
    const char *e;
    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch);  s.opsize = L; s.row = vec_word;
    if ((e = emu_isr_build(&prog[0], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch);    s.opsize = L; s.row = row; s.col = 0;
    s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
    if ((e = emu_isr_build(&prog[1], &s))) { fprintf(stderr, "MAC: %s\n", e); return ~0u; }
    prog[2] = prog[1];
    s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word;
    if ((e = emu_isr_build(&prog[3], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[4], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

    if (put_weights(row, val, L) < 0) return ~0u;
    if (put_vector(vec_word, 1.0f, L) < 0) return ~0u;

    struct run_result r = { .lane = g_lane };
    if (run_program(prog, 5, dst_word, 1, &r) < 0) return ~0u;
    printf("    ran %" PRIu64 " us, %u polls, %u reads, done=%u\n",
           r.us, r.polls, r.reads, r.seen_done);
    if (!r.landed) { printf("    FAIL: the result word never lost its poison in 2 s\n"); return ~0u; }

    uint16_t want[EMU_NBANKS], alt[EMU_NBANKS];
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        uint16_t one = expect_one(b, val, 1.0f, L);
        alt[b]  = one;
        want[b] = emu_f32_to_bf16(2.0f * emu_bf16_to_f32(one));
    }
    return judge("chain", r.lane[0], want, alt, "un-accumulated 1x");
}

// B. Does one WRVEC feed more than one MAC?
//    WRVEC(v=1) | MAC | RD_MAC(w0) | MAC | RD_MAC(w1) | EOS  ->  both words the 1x value.
//    RD_MAC is read-clear, so the second MAC starts from zero; if the GB rewinds it
//    sees the same vector and w1 == w0.  A zero w1 means the GB was drained once.
static unsigned probe_rewind(unsigned row, unsigned L, unsigned val,
                             uint32_t vec_word, uint32_t dst_word)
{
    printf("\n=== rewind — does ONE WRVEC feed more than one MAC? ===\n");
    printf("    WRVEC(L=%u) | MAC | RD_MAC(w%u) | MAC | RD_MAC(w%u) | EOS\n",
           L, dst_word, dst_word + 1);
    printf("    expecting BOTH words to hold the same 1x value.  A zero second word\n"
           "    would mean the GB is drained by the first MAC and not rewound — every\n"
           "    MAC would then need its own WRVEC.\n");

    struct emu_isr prog[6];
    struct emu_isr_spec s;
    const char *e;
    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch);  s.opsize = L; s.row = vec_word;
    if ((e = emu_isr_build(&prog[0], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch);    s.opsize = L; s.row = row; s.col = 0;
    s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
    if ((e = emu_isr_build(&prog[1], &s))) { fprintf(stderr, "MAC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word;
    if ((e = emu_isr_build(&prog[2], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
    prog[3] = prog[1];
    s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word + 1;
    if ((e = emu_isr_build(&prog[4], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[5], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

    if (put_weights(row, val, L) < 0) return ~0u;
    if (put_vector(vec_word, 1.0f, L) < 0) return ~0u;

    struct run_result r = { .lane = g_lane };
    if (run_program(prog, 6, dst_word, 2, &r) < 0) return ~0u;
    printf("    ran %" PRIu64 " us, %u polls, %u reads, done=%u\n",
           r.us, r.polls, r.reads, r.seen_done);
    if (!r.landed) { printf("    FAIL: a result word never lost its poison in 2 s\n"); return ~0u; }

    uint16_t want[EMU_NBANKS];
    for (unsigned b = 0; b < EMU_NBANKS; b++) want[b] = expect_one(b, val, 1.0f, L);
    printf("  first MAC:\n");
    unsigned bad = judge("rewind (1st)", r.lane[0], want, NULL, NULL);
    printf("  second MAC — the one that needs the GB to have rewound:\n");
    bad += judge("rewind (2nd)", r.lane[1], want, NULL, NULL);
    return bad;
}

// C. A dot product split across two MACs with a DIFFERENT vector in each half.
//    WRVEC(v=1) | MAC(rowA) | WRVEC(v=2) | MAC(rowB) | RD_MAC | EOS
//    lane b = K*(valA+b)*1 + K*(valB+b)*2.  The second vector is 2.0 so that a GB
//    which was not refilled gives a different number rather than the right one.
static unsigned probe_kchunk(unsigned rowA, unsigned rowB, unsigned L,
                             unsigned valA, unsigned valB,
                             uint32_t vw0, uint32_t vw1, uint32_t dst_word)
{
    printf("\n=== kchunk — a dot product split in two, one vector per half ===\n");
    printf("    WRVEC(v=1.0) | MAC(row %u) | WRVEC(v=2.0) | MAC(row %u) | RD_MAC | EOS\n",
           rowA, rowB);
    printf("    this is what K > 1024 looks like.  The second vector is 2.0, so a GB\n"
           "    that was not refilled produces a wrong number, not a lucky one.\n");

    struct emu_isr prog[6];
    struct emu_isr_spec s;
    const char *e;
    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch);  s.opsize = L; s.row = vw0;
    if ((e = emu_isr_build(&prog[0], &s))) { fprintf(stderr, "WRVEC0: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch);    s.opsize = L; s.row = rowA; s.col = 0;
    s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
    if ((e = emu_isr_build(&prog[1], &s))) { fprintf(stderr, "MAC A: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch);  s.opsize = L; s.row = vw1;
    if ((e = emu_isr_build(&prog[2], &s))) { fprintf(stderr, "WRVEC1: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch);    s.opsize = L; s.row = rowB; s.col = 0;
    s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
    if ((e = emu_isr_build(&prog[3], &s))) { fprintf(stderr, "MAC B: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word;
    if ((e = emu_isr_build(&prog[4], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[5], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

    if (put_weights(rowA, valA, L) < 0) return ~0u;
    if (put_weights(rowB, valB, L) < 0) return ~0u;
    if (put_vector(vw0, 1.0f, L) < 0) return ~0u;
    if (put_vector(vw1, 2.0f, L) < 0) return ~0u;

    struct run_result r = { .lane = g_lane };
    if (run_program(prog, 6, dst_word, 1, &r) < 0) return ~0u;
    printf("    ran %" PRIu64 " us, %u polls, %u reads, done=%u\n",
           r.us, r.polls, r.reads, r.seen_done);
    if (!r.landed) { printf("    FAIL: the result word never lost its poison in 2 s\n"); return ~0u; }

    uint16_t want[EMU_NBANKS], alt[EMU_NBANKS];
    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        float ha = emu_bf16_to_f32(expect_one(b, valA, 1.0f, L));
        float hb = emu_bf16_to_f32(expect_one(b, valB, 2.0f, L));
        want[b] = emu_f32_to_bf16(ha + hb);
        // if the second WRVEC had not taken effect the second half would use v=1.0
        alt[b]  = emu_f32_to_bf16(ha + emu_bf16_to_f32(expect_one(b, valB, 1.0f, L)));
    }
    return judge("kchunk", r.lane[0], want, alt, "GB-not-refilled");
}

// D. The kernel a compiler would emit: one vector load, then a MAC/RD_MAC pair per
//    group of 16 outputs.  Group g reads row (row+g), whose weights are (val+16g+b),
//    and lands in its own GPR word — so a group whose answer leaked into another
//    group's word shows up as a named mismatch.
// C = K-chunks per output. Each group issues C {WRVEC, MAC} pairs into ONE
// accumulator and then a single RD_MAC — which is exactly a GEMV with K = C*16*L.
// The WRVEC has to be repeated per chunk because the accumulator holds only one
// output group, so groups must be the OUTER loop and the vector chunk the inner.
// ORDER: which loop is outer.
//   group-outer  for g { for c { WRVEC(c); MAC(g,c) } RD_MAC(g) }
//        one fp32 accumulation per output, rounded once — the accurate shape.
//        Costs G*C vector loads, because the single accumulator latch forces the
//        group to be the outer loop and the vector chunk therefore repeats.
//   chunk-outer  for c { WRVEC(c); for g { MAC(g,c); RD_MAC(g,c) } }
//        C vector loads instead of G*C, but G*C partial sums instead of G, each
//        rounded to BF16 on its way out, and the host adds the C partials.
// The trade is vector-load time against RD_MAC time AND one rounding against C.
// Both are measured here rather than argued about.
enum order { ORDER_GROUP, ORDER_CHUNK };

static unsigned probe_gemv(unsigned row, unsigned L, unsigned val, unsigned G,
                           uint32_t vec_word, uint32_t dst_word, unsigned C,
                           enum order ord)
{
    const bool chunk_outer = (ord == ORDER_CHUNK) && C > 1;
    const unsigned NDST = chunk_outer ? G * C : G;
    const unsigned NWRVEC = chunk_outer ? C : (C == 1 ? 1 : G * C);
    const unsigned NISR = NWRVEC + G * C + NDST + 1;
    printf("\n=== gemv (%s-outer) — N = %u, K = %u (%u x %u) ===\n",
           chunk_outer ? "chunk" : "group", G * EMU_NBANKS,
           C * L * EMU_LANES_PER_WORD, C, L * EMU_LANES_PER_WORD);
    printf("    %u ISRs in ONE doorbell: %u WRVEC + %u MAC + %u RD_MAC + EOS\n",
           NISR, NWRVEC, G * C, NDST);

    if (G == 0 || C == 0 || NISR > MAX_PROG || NDST > MAX_DST) {
        printf("    G/C out of range (%u ISRs, %u result words)\n", NISR, NDST); return ~0u; }

    struct emu_isr *prog = calloc(NISR, sizeof *prog);
    if (!prog) { printf("    out of memory\n"); return ~0u; }
    struct emu_isr_spec s;
    const char *e;
    unsigned n = 0;
#define BAIL(...) do { fprintf(stderr, __VA_ARGS__); free(prog); return ~0u; } while (0)

    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word;
#define EMIT_WRVEC(c) do { \
        s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word + (c) * L; \
        if ((e = emu_isr_build(&prog[n++], &s))) BAIL("WRVEC: %s\n", e); } while (0)
#define EMIT_MAC(g, c) do { \
        s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch); s.opsize = L; s.row = row + (g) * C + (c); s.col = 0; \
        s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF; \
        if ((e = emu_isr_build(&prog[n++], &s))) BAIL("MAC: %s\n", e); } while (0)
#define EMIT_RDMAC(w) do { \
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = (w); \
        if ((e = emu_isr_build(&prog[n++], &s))) BAIL("RD_MAC: %s\n", e); } while (0)

    if (chunk_outer) {
        for (unsigned c = 0; c < C; c++) {
            EMIT_WRVEC(c);
            for (unsigned g = 0; g < G; g++) { EMIT_MAC(g, c); EMIT_RDMAC(dst_word + c * G + g); }
        }
    } else if (C == 1) {
        EMIT_WRVEC(0);                       // one shared load; the GB rewinds per MAC
        for (unsigned g = 0; g < G; g++) { EMIT_MAC(g, 0); EMIT_RDMAC(dst_word + g); }
    } else {
        for (unsigned g = 0; g < G; g++) {
            for (unsigned c = 0; c < C; c++) { EMIT_WRVEC(c); EMIT_MAC(g, c); }
            EMIT_RDMAC(dst_word + g);
        }
    }
#undef EMIT_WRVEC
#undef EMIT_MAC
#undef EMIT_RDMAC
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[n++], &s))) BAIL("EOS: %s\n", e);

    uint64_t tw = now_us();
    if (put_weight_block_c(row, val, L, G, C) < 0) BAIL("weight upload failed\n");
    uint64_t weight_us = now_us() - tw;
    for (unsigned c = 0; c < C; c++)
        if (put_vector(vec_word + c * L, 1.0f, L) < 0) BAIL("vector upload failed\n");

    struct run_result r = { .lane = g_lane };
    if (run_program(prog, n, dst_word, NDST, &r) < 0) BAIL("run failed\n");
    free(prog);
#undef BAIL
    const uint64_t macs = (uint64_t)G * C * EMU_NBANKS * L * EMU_LANES_PER_WORD;
    printf("    ran %" PRIu64 " us for %u ISRs, %u polls, %u reads, done=%u\n",
           r.us, n, r.polls, r.reads, r.seen_done);
    printf("    %" PRIu64 " MAC ops -> %.2f GMAC/s in-kernel"
           "   (IMEM+poison load %" PRIu64 " us, result drain %" PRIu64 " us,"
           " weight upload %" PRIu64 " us)\n",
           macs, r.us ? (double)macs / (double)r.us / 1000.0 : 0.0,
           r.load_us, r.drain_us, weight_us);
    printf("    weight bytes %.2f MiB in %" PRIu64 " us = %.0f MB/s"
           "  (16 contiguous per-bank transfers, not %u row-sized ones)\n",
           (double)G * C * EMU_ROW_BYTES * EMU_NBANKS / 1048576.0, weight_us,
           weight_us ? (double)G * C * EMU_ROW_BYTES * EMU_NBANKS / (double)weight_us : 0.0,
           G * C * EMU_NBANKS);
    if (!r.landed) { printf("    FAIL: some result word never lost its poison\n"); return ~0u; }

    if (chunk_outer) {
        // every partial must be the one-chunk value, and the host sum must be exact
        unsigned pbad = 0, fbad = 0;
        for (unsigned g = 0; g < G; g++) {
            for (unsigned b = 0; b < EMU_NBANKS; b++) {
                float sum = 0.0f;
                for (unsigned c = 0; c < C; c++) {
                    uint16_t got = r.lane[c * G + g][b];
                    if (!emu_bf16_same(got, expect_c(b, val + 16 * g, L, 1))) pbad++;
                    sum += emu_bf16_to_f32(got);
                }
                if (!emu_bf16_same(emu_f32_to_bf16(sum), expect_c(b, val + 16 * g, L, C))) fbad++;
            }
        }
        printf("    partials: %u wrong of %u   host-summed outputs: %u wrong of %u\n",
               pbad, G * C * EMU_NBANKS, fbad, G * EMU_NBANKS);
        printf("    gemv: %s\n", (pbad || fbad) ? "FAIL" : "PASS");
        return pbad + fbad;
    }

    unsigned bad = 0;
    printf("    group | out range   | result | lanes\n");
    printf("    ------+-------------+--------+------------------------------------\n");
    for (unsigned g = 0; g < G; g++) {
        unsigned wrong = 0, culprit = 0;
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!emu_bf16_same(r.lane[g][b], expect_c(b, val + 16 * g, L, C))) { wrong++; culprit = b; }
        }
        if (G > 16 && g >= 4 && g + 4 < G) { if (g == 4) printf("     ...  | (%u groups elided) |\n", G - 8); 
            for (unsigned b = 0; b < EMU_NBANKS; b++)
                if (!emu_bf16_same(r.lane[g][b], expect_c(b, val + 16 * g, L, C))) bad++;
            continue; }
        printf("     %3u  | y[%4u..%4u] | w%-5u | ", g, g * 16, g * 16 + 15, dst_word + g);
        if (!wrong) {
            printf("all 16 ok  (%.0f .. %.0f)\n",
                   (double)emu_bf16_to_f32(r.lane[g][0]),
                   (double)emu_bf16_to_f32(r.lane[g][15]));
        } else {
            printf("%u WRONG, first at lane %u: got %.0f want %.0f", wrong, culprit,
                   (double)emu_bf16_to_f32(r.lane[g][culprit]),
                   (double)emu_bf16_to_f32(expect_c(culprit, val + 16 * g, L, C)));
            // did it get another GROUP's answer?  that names a leak between groups.
            for (unsigned o = 0; o < G; o++)
                if (o != g && emu_bf16_same(r.lane[g][culprit],
                                            expect_c(culprit, val + 16 * o, L, C))) {
                    printf(" — this is group %u's answer", o);
                    break;
                }
            printf("\n");
            bad += wrong;
        }
    }
    printf("    gemv: %s (%u wrong lanes of %u)\n", bad ? "FAIL" : "PASS", bad, G * EMU_NBANKS);
    return bad;
}

// D2. HOW WIDE IS THE ACCUMULATOR, on THIS device?
//     bank_controller_top answered this, but that DUT is ONE bank lifted out of the
//     emulator and run standalone through a CSR interface.  It is not this silicon
//     and it is not this command path, so the answer does not carry over.  Asked
//     again here, on ch1, over ISRs.
//
//     The trick is a vector whose terms are individually BELOW HALF AN ULP of the
//     running sum.  w[0] = 1.0 and every other w = 2^-9, against v = 1.0:
//       - a BF16 accumulator rounds each 2^-9 away as it arrives (half an ulp of
//         1.0 is 2^-8) and lands on exactly 1.0 = 0x3F80
//       - anything wider keeps them and lands on 1 + (K-1)*2^-9
//     Every one of those terms is a power of two, so the exact sum is representable
//     in fp32 and a TREE and a SEQUENTIAL order give the same answer.  That is
//     deliberate: this probe isolates WIDTH.  Order is a separate question (below).
static unsigned probe_acc(unsigned row, unsigned val_unused,
                          uint32_t vec_word, uint32_t dst_word)
{
    (void)val_unused;
    printf("\n=== acc — how wide is the accumulator, ON THIS DEVICE? ===\n");
    printf("    w[0]=1.0, w[k>0]=2^-9, v=1.0.  Each 2^-9 is below half an ulp of the\n"
           "    running sum, so a BF16 accumulator drops them all and returns exactly\n"
           "    1.0 (0x3F80).  A wider one returns 1 + (K-1)/512.\n");
    printf("    every term is a power of two, so tree and sequential agree here —\n"
           "    this measures WIDTH only.\n\n");

    const uint16_t ONE  = emu_f32_to_bf16(1.0f);
    const uint16_t TINY = emu_f32_to_bf16(1.0f / 512.0f);
    uint16_t a[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    uint16_t v[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    unsigned bad = 0;

    // ---- part A: inside ONE MAC -------------------------------------------
    printf("    A. inside one MAC\n");
    printf("       L  |    K | bf16 acc | wider acc | device   |\n");
    printf("       ---+------+----------+-----------+----------+------\n");
    static const unsigned LS[] = { 1, 4, 16, 64 };
    for (unsigned i = 0; i < 4; i++) {
        const unsigned L = LS[i], K = L * EMU_LANES_PER_WORD;
        for (unsigned k = 0; k < K; k++) { a[k] = TINY; v[k] = ONE; }
        a[0] = ONE;
        for (unsigned b = 0; b < EMU_NBANKS; b++)
            if (axi_write(pim_bank(g_ch, b) + (uint64_t)(row + i) * EMU_ROW_BYTES, a,
                          (size_t)L * EMU_WORD_BYTES) < 0) return ~0u;
        if (gpr_write(vec_word, v, L) < 0) return ~0u;

        struct emu_isr prog[4];
        struct emu_isr_spec s;
        const char *e;
        s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word;
        if ((e = emu_isr_build(&prog[0], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch);   s.opsize = L; s.row = row + i; s.col = 0;
        s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
        if ((e = emu_isr_build(&prog[1], &s))) { fprintf(stderr, "MAC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word;
        if ((e = emu_isr_build(&prog[2], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
        if ((e = emu_isr_build(&prog[3], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

        struct run_result r = { .lane = g_lane };
        if (run_program(prog, 4, dst_word, 1, &r) < 0) return ~0u;
        if (!r.landed) { printf("       L=%u: result never landed\n", L); return ~0u; }

        uint16_t wide = emu_f32_to_bf16(1.0f + (float)(K - 1) / 512.0f);
        uint16_t got  = r.lane[0][0];
        unsigned spread = 0;
        for (unsigned b = 1; b < EMU_NBANKS; b++)
            if (!emu_bf16_same(r.lane[0][b], got)) spread++;
        printf("       %2u | %4u |   0x%04x |    0x%04x | 0x%04x   | ", L, K, ONE, wide, got);
        if      (emu_bf16_same(got, wide)) printf("WIDER than bf16\n");
        else if (emu_bf16_same(got, ONE))  { printf("bf16 accumulator\n"); bad++; }
        else                               { printf("NEITHER — unexpected\n"); bad++; }
        if (spread) { printf("       !! %u of 16 lanes disagree with lane 0\n", spread); bad++; }
    }

    // ---- part B: ACROSS MAC ISRs ------------------------------------------
    // MAC(row_a) puts 1.0 in the latch.  Eight MAC(row_b) each add 2^-9.  If the
    // latch narrows to BF16 between ISRs, all eight are rounded away and the answer
    // is 1.0 — which is exactly the failure that would make chunked K unsafe.
    printf("\n    B. across MAC ISRs — 1 x MAC(1.0) then 8 x MAC(2^-9)\n");
    {
        const unsigned L = 1, K = EMU_LANES_PER_WORD;
        const unsigned ra = row + 8, rb = row + 9;
        for (unsigned k = 0; k < K; k++) { a[k] = 0; v[k] = ONE; }
        a[0] = ONE;
        for (unsigned b = 0; b < EMU_NBANKS; b++)
            if (axi_write(pim_bank(g_ch, b) + (uint64_t)ra * EMU_ROW_BYTES, a,
                          (size_t)L * EMU_WORD_BYTES) < 0) return ~0u;
        for (unsigned k = 0; k < K; k++) a[k] = 0;
        a[0] = TINY;
        for (unsigned b = 0; b < EMU_NBANKS; b++)
            if (axi_write(pim_bank(g_ch, b) + (uint64_t)rb * EMU_ROW_BYTES, a,
                          (size_t)L * EMU_WORD_BYTES) < 0) return ~0u;
        if (gpr_write(vec_word, v, L) < 0) return ~0u;

        struct emu_isr prog[12];
        struct emu_isr_spec s;
        const char *e;
        unsigned n = 0;
        s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word;
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch); s.opsize = L; s.row = ra; s.col = 0;
        s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "MAC a: %s\n", e); return ~0u; }
        s.row = rb;
        for (unsigned j = 0; j < 8; j++)
            if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "MAC b: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word;
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

        struct run_result r = { .lane = g_lane };
        if (run_program(prog, n, dst_word, 1, &r) < 0) return ~0u;
        if (!r.landed) { printf("       result never landed\n"); return ~0u; }

        uint16_t wide = emu_f32_to_bf16(1.0f + 8.0f / 512.0f);
        uint16_t got  = r.lane[0][0];
        printf("       bf16 latch would give 0x%04x (1.0), a wider one 0x%04x (%.6f)\n",
               ONE, wide, (double)emu_bf16_to_f32(wide));
        printf("       device 0x%04x (%.6f) — ", got, (double)emu_bf16_to_f32(got));
        if      (emu_bf16_same(got, wide)) printf("the latch STAYS WIDE across ISRs\n");
        else if (emu_bf16_same(got, ONE))  { printf("the latch NARROWS TO BF16 between ISRs\n"); bad++; }
        else                               { printf("NEITHER — unexpected\n"); bad++; }
    }
    printf("\n    acc: %s\n", bad ? "the accumulator is NOT what the schedule assumes" : "PASS");
    return bad;
}

// E. WHICH SUMMATION ORDER does the device use?
//    Every other probe here uses operands chosen so that every partial sum is
//    exact — which is right for finding defects and useless for this question,
//    because all orders agree when nothing rounds.  This one uses RANDOM BF16 and
//    asks which host model reproduces the device BIT FOR BIT.
//
//    It matters because a PIM software stack needs a CPU reference: to run long
//    studies without the board, and to tell "the device is wrong" from "the host
//    model is wrong".  A reference that is right 99.9% of the time is exactly the
//    kind that poisons a single-ULP investigation.
static uint32_t rng_s = 0x12345678u;
static uint32_t rng(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 17; rng_s ^= rng_s << 5; return rng_s; }
// Three distributions, because the QUESTION the probe can answer depends on them.
//   uniform  [-2,2).  Nothing here is realistic, it is just well conditioned.
//   normal   what a transformer actually holds: weights ~N(0, 0.02), activations
//            ~N(0, 1).  This is the distribution the golden model has to be right on.
//   wide     exponents spread over 2^-8..2^8 with mixed signs, so partial sums
//            cancel and an fp32 ordering difference can grow to a BF16 ulp.  This is
//            the only one with any power to tell the orders apart, and it is not
//            data any model produces.
//   cancel   NOT random.  A vector built to make the tree and a strict sequential
//            sum give ANSWERS THAT ARE NOT CLOSE, rather than answers a ulp apart.
//            Per beat: lanes 0,1 = eps; lanes 2..15 = +A,-A,+A,-A,... with A huge.
//              tree      (eps+eps) = 2*eps, and every (+A,-A) pair is exactly 0,
//                        so the beat reduces to 2*eps EXACTLY.  64 beats -> 128*eps.
//              sequential  eps, 2*eps, then +A drowns 2*eps (A is 2^29 above it, and
//                        fp32 keeps 24 bits), then -A returns to 0, and it ends at 0.
//            So the two models differ by EVERYTHING, not by one ulp — which is the
//            only way a BF16 output can tell them apart at all.
//   cancel-half  the same idea aimed at the HALVES tree: eps at lanes 0 and 8 (which
//            that tree pairs first), +A at 1..7 and -A at 9..15 so every other pair
//            is exactly 0.  Halves tree -> 2*eps per beat.  Sequential -> 0.
//   eps-only  THE CONTROL.  The same eps lanes with the +-A lanes set to ZERO, so
//            every model — sequential, either tree — predicts the same 128*eps.  If
//            the device does NOT return that, eps is being flushed somewhere and the
//            two cancel vectors prove nothing.  Run this FIRST.
enum dist { DIST_UNIFORM, DIST_NORMAL, DIST_WIDE, DIST_CANCEL, DIST_CANCEL_HALF, DIST_EPS };
static enum dist g_dist = DIST_NORMAL;

static float rnd_unit(void) { return (float)(rng() >> 8) / 8388608.0f; }   // [0,1)
static float rnd_normal(void)
{
    // Irwin-Hall: 12 uniforms minus 6.  Good enough to a few sigma, and this is a
    // conditioning test, not a statistics test.
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += rnd_unit();
    return s - 6.0f;
}
static uint16_t rand_bf16(float scale)
{
    float f;
    switch (g_dist) {
    case DIST_UNIFORM: f = rnd_unit() * 4.0f - 2.0f;               break;
    case DIST_NORMAL:  f = rnd_normal() * scale;                   break;
    default: {
        int e = (int)(rng() % 17) - 8;                              // 2^-8 .. 2^8
        f = ldexpf(1.0f + rnd_unit(), e);
        if (rng() & 1) f = -f;
        break; }
    }
    return emu_f32_to_bf16(f);
}

static float dot_seq(const uint16_t *w, const uint16_t *v, unsigned K)
{
    float s = 0.0f;
    for (unsigned k = 0; k < K; k++) s += emu_bf16_to_f32(w[k]) * emu_bf16_to_f32(v[k]);
    return s;
}
static float dot_tree(const uint16_t *w, const uint16_t *v, unsigned K)
{
    static float t[EMU_MAX_OPSIZE * EMU_LANES_PER_WORD];
    for (unsigned k = 0; k < K; k++) t[k] = emu_bf16_to_f32(w[k]) * emu_bf16_to_f32(v[k]);
    return tree_sum(t, K);
}
// The device's actual structure, as built: stage 1 is a row of MULTIPLIER LANES,
// their products fall into an ADDER TREE, and the tree's output goes into the MAC
// accumulator.  So the reduction is a TREE INSIDE A BEAT (16 lanes, 4 levels) and
// SEQUENTIAL ACROSS BEATS — the accumulator sees one tree result per beat.
//
// Neither of the two models the repo argues about is this.  A pairwise tree over
// the whole of K ignores the beat boundary; a strict left-to-right sum ignores the
// tree.  They only coincide with this one at L = 1, where K = 16 is a single beat —
// which is exactly emu_gemv's default, and why its tree_sum has never been caught.
// Two ways to wire a 16-input adder tree, and they are NOT the same function in
// floating point:
//   ADJACENT   level pairs (2i, 2i+1) — neighbours combine first
//   HALVES     level pairs (i, i+n/2) — the usual RTL form, `sum[j] = in[j] + in[j+n/2]`
// Same 4 levels, same latency, different association.  Which one is built decides
// what a bit-exact host reference has to do.
static float beat16(const float *p, bool halves)
{
    float t[EMU_LANES_PER_WORD];
    memcpy(t, p, sizeof t);
    for (unsigned n = EMU_LANES_PER_WORD; n > 1; n /= 2)
        for (unsigned i = 0; i < n / 2; i++)
            t[i] = halves ? t[i] + t[i + n / 2] : t[2 * i] + t[2 * i + 1];
    return t[0];
}

static float dot_beat_tree_ex(const uint16_t *w, const uint16_t *v, unsigned L, bool halves)
{
    float acc = 0.0f;
    for (unsigned b = 0; b < L; b++) {
        float t[EMU_LANES_PER_WORD];
        for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++) {
            unsigned k = b * EMU_LANES_PER_WORD + i;
            t[i] = emu_bf16_to_f32(w[k]) * emu_bf16_to_f32(v[k]);
        }
        acc += beat16(t, halves);                                 // into the latch
    }
    return acc;
}
static float dot_beat_tree(const uint16_t *w, const uint16_t *v, unsigned L)
{ return dot_beat_tree_ex(w, v, L, false); }
static float dot_beat_half(const uint16_t *w, const uint16_t *v, unsigned L)
{ return dot_beat_tree_ex(w, v, L, true); }

static double dot_f64(const uint16_t *w, const uint16_t *v, unsigned K)
{
    double s = 0.0;
    for (unsigned k = 0; k < K; k++) s += (double)emu_bf16_to_f32(w[k]) * (double)emu_bf16_to_f32(v[k]);
    return s;
}

static unsigned probe_order(unsigned row, unsigned L, unsigned G,
                            uint32_t vec_word, uint32_t dst_word)
{
    const unsigned K = L * EMU_LANES_PER_WORD;
    printf("\n=== order — which fp32 summation order is the device? ===\n");
    static const char *DN[] = { "uniform [-2,2)", "normal (w~N(0,0.02), x~N(0,1))",
                                "wide exponents, mixed signs",
                                "CONSTRUCTED for the ADJACENT-pair tree",
                                "CONSTRUCTED for the HALVES tree",
                                "CONTROL — eps lanes only, every model agrees" };
    printf("    K = %u, %u lanes sampled, distribution = %s.\n",
           K, G * EMU_NBANKS, DN[g_dist]);
    printf("    three host models compete: sequential left-to-right, pairwise tree,\n"
           "    and f64.  Only a model that matches EVERY lane is a golden — and a\n"
           "    model can only be TOLD APART where the models disagree with each other.\n");
    if (G == 0 || 2 * G + 2 > MAX_PROG || G > MAX_DST) { printf("    G out of range\n"); return ~0u; }

    uint16_t *W = malloc((size_t)G * EMU_NBANKS * K * sizeof *W);
    uint16_t *v = malloc((size_t)K * sizeof *v);
    uint8_t  *blk = calloc(G, EMU_ROW_BYTES);
    struct emu_isr *prog = calloc(2 * G + 2, sizeof *prog);
    if (!W || !v || !blk || !prog) { printf("    out of memory\n"); return ~0u; }

    if (g_dist >= DIST_CANCEL) {
        // eps/A ratio 2^26, just past fp32's 24 bits, with BOTH comfortably normal in
        // any float format — so a null result cannot be blamed on flush-to-zero.
        const uint16_t EPS = emu_f32_to_bf16(ldexpf(1.0f, -12));
        const uint16_t BIG = emu_f32_to_bf16(ldexpf(1.0f, 14));
        const uint16_t NEG = emu_f32_to_bf16(-ldexpf(1.0f, 14));
        const bool half = (g_dist == DIST_CANCEL_HALF);
        const bool ctl  = (g_dist == DIST_EPS);
        for (unsigned k = 0; k < K; k++) v[k] = emu_f32_to_bf16(1.0f);
        for (size_t i = 0; i < (size_t)G * EMU_NBANKS * K; i++) {
            unsigned j = (unsigned)(i % EMU_LANES_PER_WORD);       // lane within beat
            bool is_eps = half || ctl ? (j == 0 || j == 8) : (j < 2);
            if (is_eps)    W[i] = EPS;
            else if (ctl)  W[i] = 0;
            else if (half) W[i] = (j < 8) ? BIG : NEG;
            else           W[i] = (j & 1) ? NEG : BIG;
        }
    } else {
        // weights are the small ones in a transformer; activations are O(1)
        for (unsigned k = 0; k < K; k++) v[k] = rand_bf16(1.0f);
        for (size_t i = 0; i < (size_t)G * EMU_NBANKS * K; i++) W[i] = rand_bf16(0.02f);
    }

    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        memset(blk, 0, (size_t)G * EMU_ROW_BYTES);
        for (unsigned g = 0; g < G; g++)
            memcpy(blk + (size_t)g * EMU_ROW_BYTES,
                   W + ((size_t)g * EMU_NBANKS + b) * K, K * sizeof *W);
        if (axi_write(pim_bank(g_ch, b) + (uint64_t)row * EMU_ROW_BYTES, blk,
                      (size_t)G * EMU_ROW_BYTES) < 0) return ~0u;
    }
    if (gpr_write(vec_word, v, L) < 0) return ~0u;

    struct emu_isr_spec s;
    const char *e;
    unsigned n = 0;
    s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word;
    if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return ~0u; }
    for (unsigned g = 0; g < G; g++) {
        s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch); s.opsize = L; s.row = row + g; s.col = 0;
        s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "MAC: %s\n", e); return ~0u; }
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = dst_word + g;
        if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return ~0u; }
    }
    s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch);
    if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr, "EOS: %s\n", e); return ~0u; }

    struct run_result r = { .lane = g_lane };
    if (run_program(prog, n, dst_word, G, &r) < 0) return ~0u;
    if (!r.landed) { printf("    FAIL: results never landed\n"); return ~0u; }

    unsigned nl = 0, m_seq = 0, m_tree = 0, m_bt = 0, m_bh = 0, m_f64 = 0, agree = 0;
    for (unsigned g = 0; g < G; g++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            const uint16_t *w = W + ((size_t)g * EMU_NBANKS + b) * K;
            uint16_t got = r.lane[g][b];
            uint16_t sq = emu_f32_to_bf16(dot_seq(w, v, K));
            uint16_t tr = emu_f32_to_bf16(dot_tree(w, v, K));
            uint16_t bt = emu_f32_to_bf16(dot_beat_tree(w, v, L));
            uint16_t bh = emu_f32_to_bf16(dot_beat_half(w, v, L));
            uint16_t f6 = emu_f32_to_bf16((float)dot_f64(w, v, K));
            nl++;
            if (emu_bf16_same(got, sq)) m_seq++;
            if (emu_bf16_same(got, tr)) m_tree++;
            if (emu_bf16_same(got, bt)) m_bt++;
            if (emu_bf16_same(got, bh)) m_bh++;
            if (emu_bf16_same(got, f6)) m_f64++;
            if (emu_bf16_same(sq, tr) && emu_bf16_same(sq, bt) && emu_bf16_same(sq, bh)) agree++;
        }
    printf("    ran %" PRIu64 " us, %u lanes\n", r.us, nl);
    const unsigned split = nl - agree;
    printf("    the three fp32 models DISAGREE with each other on %u/%u lanes (%.3f%%)"
           " — only those discriminate\n", split, nl, 100.0 * split / nl);
    if (split == 0)
        printf("    >> zero discriminating lanes: this run cannot identify the order.\n"
               "    >> What it DOES establish is that on this data the order is not\n"
               "    >> observable in the BF16 result at all.\n");
    if (g_dist >= DIST_CANCEL) {
        const uint16_t *w0 = W;
        printf("    lane 0 predictions:  sequential %.9g   adjacent-tree %.9g"
               "   halves-tree %.9g\n",
               (double)dot_seq(w0, v, K), (double)dot_beat_tree(w0, v, L),
               (double)dot_beat_half(w0, v, L));
        printf("    device lane 0:       %.9g  (0x%04x)\n",
               (double)emu_bf16_to_f32(r.lane[0][0]), r.lane[0][0]);
    }
    printf("    model                     | matches device      | verdict\n");
    printf("    --------------------------+---------------------+---------\n");
    printf("    sequential fp32 (l-to-r)  | %6u / %-6u %5.2f%% | %s\n",
           m_seq, nl, 100.0 * m_seq / nl, m_seq == nl ? "EXACT" : "not a golden");
    printf("    pairwise tree over all K  | %6u / %-6u %5.2f%% | %s\n",
           m_tree, nl, 100.0 * m_tree / nl, m_tree == nl ? "EXACT" : "not a golden");
    printf("    beat tree ADJACENT (2i,2i+1)| %6u / %-6u %5.2f%% | %s\n",
           m_bt, nl, 100.0 * m_bt / nl, m_bt == nl ? "EXACT" : "not a golden");
    printf("    beat tree HALVES (i,i+n/2)| %6u / %-6u %5.2f%% | %s\n",
           m_bh, nl, 100.0 * m_bh / nl, m_bh == nl ? "EXACT" : "not a golden");
    printf("    f64 then round            | %6u / %-6u %5.2f%% | %s\n",
           m_f64, nl, 100.0 * m_f64 / nl, m_f64 == nl ? "EXACT" : "not a golden");
    if (split) {
        // Among the lanes where the models differ, which one did the device follow?
        unsigned ds = 0, dt = 0, db = 0, dh = 0;
        for (unsigned g = 0; g < G; g++)
            for (unsigned b = 0; b < EMU_NBANKS; b++) {
                const uint16_t *w = W + ((size_t)g * EMU_NBANKS + b) * K;
                uint16_t sq = emu_f32_to_bf16(dot_seq(w, v, K));
                uint16_t tr = emu_f32_to_bf16(dot_tree(w, v, K));
                uint16_t bt = emu_f32_to_bf16(dot_beat_tree(w, v, L));
                uint16_t bh = emu_f32_to_bf16(dot_beat_half(w, v, L));
                if (emu_bf16_same(sq, tr) && emu_bf16_same(sq, bt) && emu_bf16_same(sq, bh)) continue;
                if (emu_bf16_same(r.lane[g][b], sq)) ds++;
                if (emu_bf16_same(r.lane[g][b], tr)) dt++;
                if (emu_bf16_same(r.lane[g][b], bt)) db++;
                if (emu_bf16_same(r.lane[g][b], bh)) dh++;
            }
        printf("    on the %u discriminating lanes: sequential %u, all-K tree %u,"
               " adjacent %u, halves %u\n", split, ds, dt, db, dh);
    }
    free(W); free(v); free(blk); free(prog);
    // Not a pass/fail of the hardware — it is an identification.  Failing to
    // identify any model is itself the finding, so it never fails the run.
    return 0;
}

// F. WHAT DOES chunk-outer COST IN ACCURACY?
//    The schedule choice for K > 1024, measured instead of argued:
//      group-outer  chains C MAC ISRs into one accumulator and reads it ONCE, so the
//                   whole dot product rounds to BF16 exactly once
//      chunk-outer  reads the accumulator after every chunk, so there are C BF16
//                   values that the HOST then sums — C roundings instead of 1
//    chunk-outer is 1.6x faster (it loads the vector C times instead of G*C times).
//    This asks what that speed costs.  Random normal operands, because the answer
//    only exists where things actually round.
//
//    Reference = the device's own arithmetic at full width: sequential fp32 over the
//    whole of K, rounded once.  group-outer should equal it EXACTLY.
static unsigned bf16_ulps(uint16_t a, uint16_t b)
{
    if (emu_bf16_same(a, b)) return 0;
    // map to a monotone integer line so a difference is a ulp count
    int32_t ia = (a & 0x8000u) ? -(int32_t)(a & 0x7FFFu) : (int32_t)a;
    int32_t ib = (b & 0x8000u) ? -(int32_t)(b & 0x7FFFu) : (int32_t)b;
    int32_t d = ia - ib;
    return (unsigned)(d < 0 ? -d : d);
}

static unsigned probe_split(unsigned row, unsigned L, unsigned G, unsigned C,
                            uint32_t vec_word, uint32_t dst_word)
{
    const unsigned Kc = L * EMU_LANES_PER_WORD, K = C * Kc;
    printf("\n=== split — what does chunk-outer cost in accuracy? ===\n");
    printf("    N = %u, K = %u (%u x %u), random normal operands.\n",
           G * EMU_NBANKS, K, C, Kc);
    printf("    group-outer rounds once per output; chunk-outer rounds %u times and\n"
           "    the host adds the pieces.  Reference = sequential fp32 over all of K.\n",
           C);
    if (C < 2) { printf("    needs --kchunks >= 2\n"); return ~0u; }
    const unsigned nisr_g = G * (2 * C + 1) + 1, nisr_c = C * (1 + 2 * G) + 1;
    if (nisr_g > MAX_PROG || G * C > MAX_DST) { printf("    too big\n"); return ~0u; }

    uint16_t *W = malloc((size_t)G * EMU_NBANKS * K * sizeof *W);
    uint16_t *v = malloc((size_t)K * sizeof *v);
    uint8_t  *blk = calloc((size_t)G * C, EMU_ROW_BYTES);
    struct emu_isr *prog = calloc(nisr_g > nisr_c ? nisr_g : nisr_c, sizeof *prog);
    uint16_t *ref = malloc((size_t)G * EMU_NBANKS * sizeof *ref);
    uint16_t *got_g = malloc((size_t)G * EMU_NBANKS * sizeof *got_g);
    if (!W || !v || !blk || !prog || !ref || !got_g) { printf("    out of memory\n"); return ~0u; }

    for (unsigned k = 0; k < K; k++) v[k] = rand_bf16(1.0f);
    for (size_t i = 0; i < (size_t)G * EMU_NBANKS * K; i++) W[i] = rand_bf16(0.02f);

    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        memset(blk, 0, (size_t)G * C * EMU_ROW_BYTES);
        for (unsigned g = 0; g < G; g++)
            for (unsigned c = 0; c < C; c++)
                memcpy(blk + ((size_t)g * C + c) * EMU_ROW_BYTES,
                       W + ((size_t)g * EMU_NBANKS + b) * K + (size_t)c * Kc,
                       Kc * sizeof *W);
        if (axi_write(pim_bank(g_ch, b) + (uint64_t)row * EMU_ROW_BYTES, blk,
                      (size_t)G * C * EMU_ROW_BYTES) < 0) return ~0u;
    }
    if (gpr_write(vec_word, v, C * L) < 0) return ~0u;

    for (unsigned g = 0; g < G; g++)
        for (unsigned b = 0; b < EMU_NBANKS; b++)
            ref[g * EMU_NBANKS + b] =
                emu_f32_to_bf16(dot_seq(W + ((size_t)g * EMU_NBANKS + b) * K, v, K));

    struct emu_isr_spec s;
    const char *e;
    unsigned n;
#define EW(c) do { s = emu_isr_default_ch(ISR_OP_WRVEC, 1u << g_ch); s.opsize = L; s.row = vec_word + (c) * L; \
                   if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr,"%s\n",e); return ~0u; } } while (0)
#define EM(g, c) do { s = emu_isr_default_ch(ISR_OP_MAC, 1u << g_ch); s.opsize = L; s.row = row + (g) * C + (c); \
                   s.col = 0; s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF; \
                   if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr,"%s\n",e); return ~0u; } } while (0)
#define ER(w) do { s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << g_ch); s.opsize = 0; s.row = (w); \
                   if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr,"%s\n",e); return ~0u; } } while (0)
#define EE()  do { s = emu_isr_default_ch(ISR_OP_EOS, 1u << g_ch); \
                   if ((e = emu_isr_build(&prog[n++], &s))) { fprintf(stderr,"%s\n",e); return ~0u; } } while (0)

    // ---- group-outer: one rounding per output --------------------------------
    n = 0;
    for (unsigned g = 0; g < G; g++) {
        for (unsigned c = 0; c < C; c++) { EW(c); EM(g, c); }
        ER(dst_word + g);
    }
    EE();
    struct run_result rg = { .lane = g_lane };
    uint64_t tg = now_us();
    if (run_program(prog, n, dst_word, G, &rg) < 0) return ~0u;
    tg = now_us() - tg;
    if (!rg.landed) { printf("    group-outer never landed\n"); return ~0u; }
    for (unsigned g = 0; g < G; g++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) got_g[g * EMU_NBANKS + b] = rg.lane[g][b];

    // ---- chunk-outer: C partials, summed on the host -------------------------
    n = 0;
    for (unsigned c = 0; c < C; c++) {
        EW(c);
        for (unsigned g = 0; g < G; g++) { EM(g, c); ER(dst_word + c * G + g); }
    }
    EE();
    struct run_result rc = { .lane = g_lane };
    uint64_t tc = now_us();
    if (run_program(prog, n, dst_word, G * C, &rc) < 0) return ~0u;
    tc = now_us() - tc;
    if (!rc.landed) { printf("    chunk-outer never landed\n"); return ~0u; }
#undef EW
#undef EM
#undef ER
#undef EE

    unsigned nl = 0, bad_g = 0, bad_c = 0, max_g = 0, max_c = 0;
    unsigned long sum_c = 0;
    for (unsigned g = 0; g < G; g++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            uint16_t want = ref[g * EMU_NBANKS + b];
            unsigned ug = bf16_ulps(got_g[g * EMU_NBANKS + b], want);
            float acc = 0.0f;
            for (unsigned c = 0; c < C; c++) acc += emu_bf16_to_f32(rc.lane[c * G + g][b]);
            unsigned uc = bf16_ulps(emu_f32_to_bf16(acc), want);
            nl++;
            if (ug) { bad_g++; if (ug > max_g) max_g = ug; }
            if (uc) { bad_c++; if (uc > max_c) max_c = uc; sum_c += uc; }
        }
    printf("\n    schedule     | ISRs | time    | differs from reference | max err | mean err\n");
    printf("    -------------+------+---------+------------------------+---------+---------\n");
    printf("    group-outer  | %4u | %5" PRIu64 " us | %6u / %-6u %6.2f%% | %2u ulp  | %s\n",
           nisr_g, tg, bad_g, nl, 100.0 * bad_g / nl, max_g,
           bad_g ? "-" : "exact");
    printf("    chunk-outer  | %4u | %5" PRIu64 " us | %6u / %-6u %6.2f%% | %2u ulp  | %.4f ulp\n",
           nisr_c, tc, bad_c, nl, 100.0 * bad_c / nl, max_c, (double)sum_c / nl);
    printf("\n    chunk-outer is %.2fx %s and costs %.2f%% of outputs differing by <= %u ulp\n",
           tc ? (double)tg / (double)tc : 0.0, tg > tc ? "faster" : "slower",
           100.0 * bad_c / nl, max_c);
    if (bad_g)
        printf("    !! group-outer should be EXACT against a sequential fp32 reference.\n"
               "       It is not, so either the chain does not keep full width or the\n"
               "       reference is wrong.  This invalidates the comparison.\n");
    free(W); free(v); free(blk); free(prog); free(ref); free(got_g);
    return bad_g;
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--ch N] [--test chain|rewind|kchunk|gemv|all]\n"
"       [--l N] [--g N] [--row N] ...\n"
"\n"
"Probes the four composition rules a multi-ISR program depends on:\n"
"\n"
"  chain    MAC | MAC | RD_MAC        does the accumulator survive between ISRs?\n"
"  rewind   MAC | RD_MAC | MAC | ...  does ONE WRVEC feed more than one MAC?\n"
"  kchunk   WRVEC|MAC|WRVEC|MAC|...   a dot product split in two — what K>1024 needs\n"
"  gemv     WRVEC then G x {MAC,RD}   the whole kernel in one doorbell\n"
"  acc      terms below half an ulp     how WIDE is the accumulator, on THIS device?\n"
"  split    group-outer vs chunk-outer  what the 1.6x schedule costs in accuracy\n"
"  order    random bf16 operands       which fp32 summation order is the device?\n"
"  --dist uniform|normal|wide|cancel  operands for the order probe; cancel is built\n"
"                                     to separate tree from sequential, not random\n"
"\n"
"  --ch N         which channel to run on (default 0).  Stages operands in\n"
"                 that channel AND sets the ISR CH_MASK to 1<<N.  Each\n"
"                 composition rule has to be re-asked per channel: holding\n"
"                 on channel 0 does not make it hold on channel 1.\n"
"\n"
"  This tool reaches the banks through the DIRECT aperture and never through the MC\n"
"  port, so --ch names a PHYSICAL channel: it verifies one emulator channel on its\n"
"  own.  That makes it independent of the channel address map — ChRoBaCo or\n"
"  RoChBaCo only changes how an MC-space address is split up, and nothing here is an\n"
"  MC-space address.  (emu_mc is the one that is: its --ch is a place in MC space,\n"
"  which is why that one is ChRoBaCo-only.)\n"
"  --test WHICH   default all\n"
"  --l N          beats per MAC, 1..64; K = 16*N (default 1)\n"
"  --g N          groups for the gemv probe (default 8, so N = 128 outputs)\n"
"  --kchunks C    K-chunks per output for the gemv probe: K = C*16*L (default 1)\n"
"  --order WHICH  group (default, one rounding) or chunk (C vector loads, C partials)\n"
"  --row N        first DRAM weight row (default 300)\n"
"  --vec-word N   first GPR word for vectors (default 0)\n"
"  --dst-word N   first GPR result word (default 2000)\n"
"  --h2c PATH     H2C MM queue (default %s)\n"
"  --c2h PATH     C2H MM queue (default %s)\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p, H2C_DEFAULT, C2H_DEFAULT);
}

int main(int argc, char **argv)
{
    const char *bdf = BDF_DEFAULT, *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    const char *test = "all";
    unsigned L = 1, G = 8, row = 300, val = 1, C = 1;
    enum order ord = ORDER_GROUP;
    uint32_t vec_word = 0, dst_word = 2000;

    static const struct option lo[] = {
        { "ch",       required_argument, NULL, 'C' },
        { "test",     required_argument, NULL, 1 }, { "l",        required_argument, NULL, 2 },
        { "g",        required_argument, NULL, 3 }, { "row",      required_argument, NULL, 4 },
        { "vec-word", required_argument, NULL, 5 }, { "dst-word", required_argument, NULL, 6 },
        { "h2c",      required_argument, NULL, 7 }, { "c2h",      required_argument, NULL, 8 },
        { "bdf",      required_argument, NULL, 9 }, { "base",     required_argument, NULL, 10 },
        { "kchunks",  required_argument, NULL, 11 }, { "order", required_argument, NULL, 12 },
        { "dist",     required_argument, NULL, 13 },
        { "help",     no_argument,       NULL, 'h' }, { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: test = optarg; break;
        case 2: L = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: G = (unsigned)strtoul(optarg, NULL, 0); break;
        case 4: row = (unsigned)strtoul(optarg, NULL, 0); break;
        case 5: vec_word = (uint32_t)strtoul(optarg, NULL, 0); break;
        case 6: dst_word = (uint32_t)strtoul(optarg, NULL, 0); break;
        case 7: h2c_path = optarg; break;
        case 8: c2h_path = optarg; break;
        case 9: bdf = optarg; break;
        case 10: val = (unsigned)strtoul(optarg, NULL, 0); break;
        case 11: C = (unsigned)strtoul(optarg, NULL, 0); break;
        case 12:
            if      (!strcmp(optarg, "group")) ord = ORDER_GROUP;
            else if (!strcmp(optarg, "chunk")) ord = ORDER_CHUNK;
            else { fprintf(stderr, "--order must be group or chunk\n"); return 2; }
            break;
        case 13:
            if      (!strcmp(optarg, "uniform")) g_dist = DIST_UNIFORM;
            else if (!strcmp(optarg, "normal"))  g_dist = DIST_NORMAL;
            else if (!strcmp(optarg, "wide"))    g_dist = DIST_WIDE;
            else if (!strcmp(optarg, "cancel")) g_dist = DIST_CANCEL;
            else if (!strcmp(optarg, "cancel-half")) g_dist = DIST_CANCEL_HALF;
            else if (!strcmp(optarg, "eps-only")) g_dist = DIST_EPS;
            else { fprintf(stderr, "--dist must be uniform, normal, wide or cancel\n"); return 2; }
            break;
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
    if (L == 0 || L > EMU_MAX_OPSIZE) { fprintf(stderr, "--l must be 1..64\n"); return 2; }

    bool all    = !strcmp(test, "all");
    bool t_ch   = all || !strcmp(test, "chain");
    bool t_rw   = all || !strcmp(test, "rewind");
    bool t_kc   = all || !strcmp(test, "kchunk");
    bool t_gv   = all || !strcmp(test, "gemv");
    bool t_or   = all || !strcmp(test, "order");
    bool t_ac   = all || !strcmp(test, "acc");
    bool t_sp   = !strcmp(test, "split");           // opt-in: needs --kchunks >= 2
    if (!t_ch && !t_rw && !t_kc && !t_gv && !t_or && !t_ac && !t_sp) { usage(argv[0]); return 2; }

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

    g_lane = calloc(MAX_DST, sizeof *g_lane);
    if (!g_lane) { fprintf(stderr, "out of memory\n"); return 1; }

    printf("emu_chain — what composes inside one ISR program\n");
    printf("  L = %u beats, K = %u, weights from row %u, A_b[k] = base + b\n",
           L, L * EMU_LANES_PER_WORD, row);

    if (!require_idle("start", 1000)) return 1;
    const struct { uint32_t off; uint8_t v; } TM[] = {
        { CFR_T_FAW, 30 }, { CFR_T_RRD, 6 }, { CFR_T_RCD, 4 }, { CFR_T_CCD, 2 },
        { CFR_T_RTP, 3 },  { CFR_T_RP,  3 }, { CFR_T_WR,  4 }, { CFR_T_RAS, 6 },
    };
    for (unsigned i = 0; i < 8; i++) cfr_wr(TM[i].off, TM[i].v);
    for (unsigned i = 0; i < 8; i++)
        if (cfr_rd(TM[i].off) != TM[i].v) { fprintf(stderr, "timing readback failed\n"); return 1; }
    printf("  timing set and verified\n");

    unsigned fails = 0, ran = 0;
    unsigned rc;
    if (t_ch) { rc = probe_chain(row, L, val, vec_word, dst_word);
                ran++; if (rc) fails++; }
    if (t_rw) { rc = probe_rewind(row + 1, L, val, vec_word, dst_word + 8);
                ran++; if (rc) fails++; }
    if (t_kc) { rc = probe_kchunk(row + 2, row + 3, L, val, val + 32,
                                  vec_word, vec_word + EMU_MAX_OPSIZE, dst_word + 16);
                ran++; if (rc) fails++; }
    if (t_gv) { rc = probe_gemv(row + 16, L, val, G, vec_word, dst_word + 32, C, ord);
                ran++; if (rc) fails++; }

    if (t_sp) { rc = probe_split(row + 6144, L, G, C, vec_word, dst_word + 32);
                ran++; if (rc) fails++; }
    if (t_ac) { rc = probe_acc(row + 2048, val, vec_word, dst_word + 24);
                ran++; if (rc) fails++; }
    if (t_or) { rc = probe_order(row + 4096, L, G, vec_word, dst_word + 32);
                ran++; if (rc && rc != ~0u) fails++; }

    printf("\n================================================================\n");
    printf("%u of %u probes passed\n", ran - fails, ran);
    printf("%s\n", fails ? "FAIL" : "PASS");

    close(g_h2c); close(g_c2h); munmap(m, EMU_BAR2_SIZE); close(bfd);
    return fails ? 1 : 0;
}
