// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_timing — the DRAM timing registers, one argument each, and what they do.
//
//     ./emu_timing                              the base set, x1, and measure it
//     ./emu_timing --scale 2                    the base set doubled
//     ./emu_timing --rcd 30                     one value replaced, the rest base
//     ./emu_timing --sweep rcd 4,8,15,30        one knob, several values
//     ./emu_timing --sweep ccd 1,2,3 --allow-unsafe
//     ./emu_timing --reset                      base becomes EMU_TIMING_SIM
//     ./emu_timing --scale 4 --no-run --keep    set it and leave; no queues needed
//
// EVERY RUN WRITES ALL EIGHT.  The base is the set below — the HBM2 device numbers
// in controller cycles — and every register is written every time, so what is on the
// board after a run depends on this file and the arguments, never on what some
// earlier run left behind.  --scale multiplies the whole set; an explicitly named
// value replaces its base entry and is then scaled with the others, because the
// point of the knob is to move the memory's speed as a WHOLE.  IT IS A WHOLE
// NUMBER: these are cycle counts, and a fractional scale would land on whatever
// lround() did with each of the eight rather than on a set anyone chose.  To reach
// a value between two multiples, name it — --rcd 64 does what --scale 4.25 did.
//
//     faw 16   rrd 4   rcd 15   ccd 2   rtp 4   rp 17   wr 28   ras 34
//
// T_RTP IS T_RBTP.  The CFR spells the register T_RTP (dispatcher_top.v 0x18); the
// bank controller's parameter is RBTP, read-to-precharge WITHIN a bank.  --rbtp is
// accepted as the same option so neither name has to be translated by hand.
//
// EIGHT BITS EACH (dispatcher_top.v:84, TW=8).  A scale that takes a value past 255
// is REFUSED rather than truncated: the register would take the low byte and report
// a number nobody asked for, and the readback would agree with it.
//
// WHY IT LIVES HERE, AND FOR HOW LONG.  hwdef/test is the tier that reaches the
// hardware DIRECTLY over MMIO, because nothing owns the control plane yet.  These
// eight registers are AXI-Lite in the CFR, so `--no-run` needs the BAR and nothing
// else — usable on a board that was programmed thirty seconds ago, before any queue
// or module exists.
//
// THAT IS A STAGE, NOT A PRINCIPLE.  Control-plane state belongs to whatever ends up
// owning the control plane, and these registers go down there with the doorbell when
// that happens.  This probe would keep its job — asking what a value DOES — and stop
// being the thing that owns the write.
//
// WITHOUT --no-run IT ALSO MOVES TRAFFIC, because setting a register is the easy
// half.  What a timing value is worth is what it does to a real access, so every
// setting is followed by:
//
//     1. the violation CSR CLEARED, and checked clear
//     2. one QDMA READ per bank, at --row, THROUGH THE MC
//     3. the violation CSR read back and printed
//
// THE READS GO THROUGH THE MC, NOT THE DIRECT APERTURE.  A host read of
// 0x40_0000_0000 + b*stride goes through the NoC straight to HBM and never touches a
// bank controller (docs/bd-reference.md §5), so no counter can move however hard it
// is driven.  The MC normal path is the one that segments the access and hands it to
// bank_XX, which is where the timing model — and every counter this tool prints —
// lives.  That is the whole reason the sweep is addressed the way it is.
//
// THE CSR IS NOT READ-ON-CLEAR.  Reading it changes nothing; CTRL[0] at +0x400 is
// write-1 self-clearing and is the only thing that zeroes a counter
// (emu_viol_csr.v:183-196, 283-292).  The clear is therefore VERIFIED here, not
// assumed — a run that started with someone else's counters would otherwise read as
// this setting's own.
//
// A VIOLATION IS NOT AN ERROR.  It is the emulator reporting that its memory was
// SLOWER than the allowance it was configured with.  Raising a value lowers its
// counter.  Counts saturate at 0xFF and never wrap, so a saturated column means "at
// least 255" and the max-overrun column is what still carries information there.
//
// THE ONE VALUE THAT LIES.  T_CCD < 2 halves a MAC result and reports NOTHING: no
// counter, no status bit, no error.  acc_top is a two-stage feedback pipe with no
// forwarding (mac_top.sv:26-33), so two beats into one accumulator latch land only
// every OTHER beat.  It is refused unless --allow-unsafe.  Nothing in the read sweep
// can show this — only a kernel can, which is what --gemv is for.
//
// --gemv ADDS THE OLD WORKLOAD: a GEMV whose MACs walk --chunks different DRAM rows,
// so each one pays a row activate, with operands chosen to make a wrong lane
// visible — A_c,b[k] = 1 + 16c + b and v[k] = 1, so lane b of channel c is exactly
// K*(1+16c+b), an integer with no BF16 rounding and different in every lane.  Its
// `exact` column is the only thing here that judges an ANSWER rather than a rate.
//
// LEAVES THE BOARD AS IT FOUND IT unless --keep.
//
// Exit: 0 everything ran (violations are data, not failure)   1 a transfer or a
//       register or a --gemv answer failed   2 usage
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

#define CHUNK        (4u << 20)
#define POISON_LANE  0x7FC1u
#define VEC_WORD     0u
#define DST_WORD     2000u

static volatile uint8_t *g_bar;
static int g_h2c = -1, g_c2h = -1;

// The eight registers are contiguous 4-byte words from CFR_T_FAW, and struct
// emu_timing lists them in that order — so an index walks both.
static const char *const TNAME[8] =
    { "faw", "rrd", "rcd", "ccd", "rtp", "rp", "wr", "ras" };

// The base set: HBM2 device timing in controller cycles.  --scale multiplies THESE,
// and --faw/--rcd/... replace an entry before the multiply.  Field order is struct
// emu_timing's: faw, rrd, rcd, ccd, rtp, rp, wr, ras.
static const struct emu_timing TIMING_BASE = { 16, 4, 15, 2, 4, 17, 28, 34 };

#define T_REG_MAX 255u          // dispatcher_top.v:84 — TW = 8

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
static uint32_t viol_rd(unsigned ch, uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)
        (g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + off);
    __sync_synchronize();
    return v;
}
static void viol_wr_ctrl(unsigned ch, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + VIOL_CTRL) = v;
    __sync_synchronize();
}

// ---- the violation CSR ------------------------------------------------------
// NOT read-on-clear.  A read leaves every counter exactly as it was; CTRL[0] is
// write-1, self-clearing, and is the only path to zero (emu_viol_csr.v:183-196).
// So this writes it AND CHECKS — a counter that survived the clear would otherwise
// be attributed to the setting about to be measured.
static bool viol_clear(unsigned nch)
{
    bool ok = true;

    for (unsigned ch = 0; ch < nch; ch++) viol_wr_ctrl(ch, 1u);
    for (unsigned ch = 0; ch < nch; ch++) {
        uint32_t any = viol_rd(ch, VIOL_ANY);
        if (any) {
            fprintf(stderr, "ch%u: the violation CSR did not clear — ANY=0x%04x "
                            "after CTRL[0]=1\n", ch, any & 0xFFFFu);
            ok = false;
        }
    }
    return ok;
}

struct viol_sum {
    unsigned rcd_rd, ccd_rd, rcd_wr, ccd_wr, rec_wr, drop;
    unsigned worst_rcd_rd, worst_ccd_rd, worst_rcd_wr, worst_ccd_wr, worst_rec_wr;
    unsigned banks;                 // how many banks reported anything at all
    uint32_t any[PIM_NCH];
};

// Reads every bank of every channel and prints the ones that fired.  ANY (+0x404)
// first, because one read names the guilty bank instead of sixteen.
static void viol_read_print(unsigned nch, struct viol_sum *s)
{
    bool head = false;

    memset(s, 0, sizeof *s);
    for (unsigned ch = 0; ch < nch; ch++) s->any[ch] = viol_rd(ch, VIOL_ANY) & 0xFFFFu;

    for (unsigned ch = 0; ch < nch; ch++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            uint32_t st = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_STICKY);
            uint32_t a  = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_CNT_A);
            uint32_t bb = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_CNT_B);
            uint32_t ma = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_MAX_A);
            uint32_t mb = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_MAX_B);
            unsigned c[5] = { a & 0xFFu, (a >> 8) & 0xFFu, (a >> 16) & 0xFFu,
                              (a >> 24) & 0xFFu, bb & 0xFFu };
            unsigned m[5] = { ma & 0xFFu, (ma >> 8) & 0xFFu, (ma >> 16) & 0xFFu,
                              (ma >> 24) & 0xFFu, mb & 0xFFu };

            s->rcd_rd += c[0];  s->ccd_rd += c[1];  s->rcd_wr += c[2];
            s->ccd_wr += c[3];  s->rec_wr += c[4];  s->drop += (bb >> 8) & 0xFFu;
            if (m[0] > s->worst_rcd_rd) s->worst_rcd_rd = m[0];
            if (m[1] > s->worst_ccd_rd) s->worst_ccd_rd = m[1];
            if (m[2] > s->worst_rcd_wr) s->worst_rcd_wr = m[2];
            if (m[3] > s->worst_ccd_wr) s->worst_ccd_wr = m[3];
            if (m[4] > s->worst_rec_wr) s->worst_rec_wr = m[4];
            if (!st && !a && !bb) continue;
            s->banks++;
            if (!head) {
                printf("    ch bk  sticky      RCD_RD CCD_RD RCD_WR CCD_WR REC_WR"
                       " | worst cycles over\n");
                head = true;
            }
            printf("    %2u %2u  0x%08x %6u %6u %6u %6u %6u | %u %u %u %u %u\n",
                   ch, b, st, c[0], c[1], c[2], c[3], c[4],
                   m[0], m[1], m[2], m[3], m[4]);
        }

    printf("  violations   :");
    for (unsigned ch = 0; ch < nch; ch++) printf(" ch%u ANY=0x%04x", ch, s->any[ch]);
    printf("\n                 RCD_RD %u  CCD_RD %u  RCD_WR %u  CCD_WR %u  REC_WR %u"
           "%s   %s\n",
           s->rcd_rd, s->ccd_rd, s->rcd_wr, s->ccd_wr, s->rec_wr,
           s->drop ? "  ewmul_drop!" : "",
           s->banks ? "" : "(every bank clean)");
}

// ---- timing: the whole point of the tool ------------------------------------
static void timing_get(struct emu_timing *t)
{
    for (unsigned i = 0; i < 8; i++)
        ((uint8_t *)t)[i] = (uint8_t)cfr_rd(CFR_T_FAW + i * 4u);
}

// base * scale into `out`.  Refuses rather than truncates: the register is 8 bits,
// and a value that wrapped would read back as whatever the low byte is and look
// like it had been accepted.
static bool timing_scale(const struct emu_timing *base, unsigned s, struct emu_timing *out)
{
    for (unsigned i = 0; i < 8; i++) {
        unsigned v = (unsigned)((const uint8_t *)base)[i] * s;
        if (v > T_REG_MAX) {
            fprintf(stderr, "--scale %u makes T_%s %u, and the register is 8 bits "
                            "(dispatcher_top.v TW=8)\n", s, TNAME[i], v);
            return false;
        }
        ((uint8_t *)out)[i] = (uint8_t)v;
    }
    return true;
}

// Written, then READ BACK.  These are posted MMIO stores; a register that ignored
// one would otherwise be indistinguishable from one that took it.
static bool timing_set(const struct emu_timing *t, bool allow_unsafe)
{
    struct emu_timing got;

    if (t->ccd < 2 && !allow_unsafe) {
        fprintf(stderr,
            "REFUSING T_CCD = %u.  Below 2, two beats into one accumulator latch "
            "land only every\nOTHER beat — the answer is halved and no counter, "
            "status bit or error reports it.\nPass --allow-unsafe if you are "
            "measuring that.\n", t->ccd);
        return false;
    }
    for (unsigned i = 0; i < 8; i++)
        cfr_wr(CFR_T_FAW + i * 4u, ((const uint8_t *)t)[i]);
    timing_get(&got);
    for (unsigned i = 0; i < 8; i++)
        if (((uint8_t *)&got)[i] != ((const uint8_t *)t)[i]) {
            fprintf(stderr, "T_%s read back %u after writing %u\n",
                    TNAME[i], ((uint8_t *)&got)[i], ((const uint8_t *)t)[i]);
            return false;
        }
    return true;
}

static void timing_print(const char *tag, const struct emu_timing *t)
{
    printf("%s", tag);
    for (unsigned i = 0; i < 8; i++)
        printf(" %s=%-3u", TNAME[i], ((const uint8_t *)t)[i]);
    printf("\n");
}

static int tname_index(const char *s)
{
    for (int i = 0; i < 8; i++) if (!strcmp(s, TNAME[i])) return i;
    if (!strcmp(s, "rbtp")) return 4;      // the bank controller's name for T_RTP
    return -1;
}

// ---- transport: only reached when a workload is asked for -------------------
static int axi_write(uint64_t axi, const void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    const uint8_t *p = buf; size_t done = 0;
    while (done < len) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
        ssize_t n = pwrite(g_h2c, p + done, want, (off_t)(axi + done));
        if (n <= 0) { fprintf(stderr, "pwrite(0x%011" PRIx64 "): %s\n", axi,
                              n < 0 ? strerror(errno) : "no progress"); return -1; }
        done += (size_t)n;
    }
    return 0;
}
static int axi_read_ex(uint64_t axi, void *buf, size_t len, unsigned flags)
{
    const char *bad = pim_dma_check_ex(axi, len, flags);
    if (bad) { fprintf(stderr, "REFUSED DMA 0x%011" PRIx64 ": %s\n", axi, bad); return -1; }
    uint8_t *p = buf; size_t done = 0;
    while (done < len) {
        size_t want = len - done > CHUNK ? CHUNK : len - done;
        ssize_t n = pread(g_c2h, p + done, want, (off_t)(axi + done));
        if (n <= 0) { fprintf(stderr, "pread(0x%011" PRIx64 "): %s\n", axi,
                              n < 0 ? strerror(errno) : "no progress"); return -1; }
        done += (size_t)n;
    }
    return 0;
}
static int axi_read(uint64_t axi, void *buf, size_t len)
{ return axi_read_ex(axi, buf, len, 0u); }

static int gpr_write(uint32_t w, const void *s, uint32_t n)
{ return axi_write(EMU_AXI_GPR + (uint64_t)w * EMU_WORD_BYTES, s, (size_t)n * EMU_WORD_BYTES); }
static int gpr_read(uint32_t w, void *d, uint32_t n)
{ return axi_read(EMU_AXI_GPR + (uint64_t)w * EMU_WORD_BYTES, d, (size_t)n * EMU_WORD_BYTES); }

// ============================ the read sweep =================================
// MC-space offset of (channel, bank, row) — the inverse of pim_decode(), which the
// header gives but does not invert.  The map decides where the channel field is:
//   ChRoBaCo   ch*MC_CH_SPAN + row*32 KiB + bank*2048
//   RoChBaCo   row*NCH*32 KiB + ch*32 KiB + bank*2048
// pim_decode() is asked to agree with it below, so a map change that this missed
// stops the run instead of quietly reading another bank.
static uint64_t mc_row_off(unsigned ch, unsigned bank, unsigned row)
{
    uint64_t within = (uint64_t)bank * EMU_ROW_BYTES;

    if (PIM_ADDR_MAP == PIM_MAP_ROCHBACO)
        return (uint64_t)row * PIM_NCH * PIM_ROBACO_ROW_BYTES
             + (uint64_t)ch * PIM_ROBACO_ROW_BYTES + within;
    return (uint64_t)ch * PIM_MC_CH_SPAN
         + (uint64_t)row * PIM_ROBACO_ROW_BYTES + within;
}

static bool sweep_addr_check(unsigned nch, unsigned row)
{
    for (unsigned ch = 0; ch < nch; ch++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            unsigned gc, gb; uint64_t off;
            pim_decode(mc_row_off(ch, b, row), &gc, &gb, &off);
            if (gc != ch || gb != b || off != (uint64_t)row * EMU_ROW_BYTES) {
                fprintf(stderr,
                    "address model disagrees with pim_decode() under %s: asked for "
                    "ch%u bank%u row%u,\ndecoded ch%u bank%u offset 0x%" PRIx64 "\n",
                    PIM_ADDR_MAP_NAME, ch, b, row, gc, gb, off);
                return false;
            }
        }
    return true;
}

// One QDMA read per bank, at `row`, through the MC — the path that reaches a bank
// controller and therefore the timing model.  Returns -1 if a transfer failed.
static int sweep_run(unsigned nch, unsigned row, unsigned beats, uint8_t *buf,
                     uint64_t *us_out)
{
    size_t len = (size_t)beats * EMU_WORD_BYTES;
    uint64_t t0 = now_us();

    for (unsigned ch = 0; ch < nch; ch++)
        for (unsigned b = 0; b < EMU_NBANKS; b++)
            if (axi_read_ex(pim_mc_at(mc_row_off(ch, b, row)), buf, len,
                            PIM_DMA_ALLOW_MC) < 0)
                return -1;
    if (us_out) *us_out = now_us() - t0;
    return 0;
}

// Idle before IMEM, not just before the doorbell: the IMEM command port gives the
// FETCHER priority, so a host write during a live kernel sits on WREADY until the
// DMA times out — and one EIO latches the H2C engine.
static bool require_idle(const char *when, uint32_t timeout_ms)
{
    uint64_t t = now_us();
    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if (st == 0xFFFFFFFFu) {
            fprintf(stderr, "cannot %s: the control plane reads all-ones\n", when);
            return false;
        }
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t > (uint64_t)timeout_ms * 1000ull) {
            fprintf(stderr, "cannot %s: dispatcher busy after %u ms (STATUS=0x%08x)\n",
                    when, timeout_ms, st);
            return false;
        }
    }
}

// ============================ the kernel (--gemv) ============================
// A GEMV whose MACs walk SEVERAL DRAM ROWS, because a row activate is what T_RCD
// gates and a single-row kernel would answer almost nothing.
struct work {
    unsigned  chunks, L, nch, row0;
    unsigned  K;                        // chunks * L * 16
    struct emu_isr *prog;
    unsigned  nprog;
    uint16_t *expect;                   // [nch][16]
};

static bool work_build(struct work *w)
{
    const char *e;
    struct emu_isr_spec s;
    unsigned all_ch = (1u << w->nch) - 1u;
    unsigned i = 0;

    w->K     = w->chunks * w->L * EMU_LANES_PER_WORD;
    w->nprog = 2 * w->chunks + w->nch + 1;
    w->prog  = calloc(w->nprog, sizeof *w->prog);
    w->expect = calloc((size_t)w->nch * EMU_LANES_PER_WORD, sizeof *w->expect);
    if (!w->prog || !w->expect) { fprintf(stderr, "out of memory\n"); return false; }

    for (unsigned c = 0; c < w->chunks; c++) {
        // WRVEC: ROW is a GPR WORD here.  Chunk c reads its own L words, so a GB
        // that was not refilled is a wrong number rather than a coincidence.
        s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
        s.opsize = w->L;  s.row = VEC_WORD + c * w->L;
        if ((e = emu_isr_build(&w->prog[i++], &s))) { fprintf(stderr, "WRVEC: %s\n", e); return false; }

        // MAC: ROW is a DRAM row — the same bits, another meaning.  A different row
        // per chunk is the point: each one pays an activate.
        s = emu_isr_default_ch(ISR_OP_MAC, all_ch);
        s.opsize = w->L;  s.row = w->row0 + c;  s.col = 0;
        s.pu_mask = 0xFFFFu;  s.gb_mc_mask = 0xFFFFu;
        if ((e = emu_isr_build(&w->prog[i++], &s))) { fprintf(stderr, "MAC: %s\n", e); return false; }
    }
    for (unsigned ch = 0; ch < w->nch; ch++) {
        // 1-hot: emu_isr_check refuses a multicast RD_MAC, because two channels
        // would write the same GPR word and one result would vanish.
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << ch);
        s.opsize = 0;  s.row = DST_WORD + ch;
        if ((e = emu_isr_build(&w->prog[i++], &s))) { fprintf(stderr, "RD_MAC: %s\n", e); return false; }
    }
    // done rises when the fetcher ACCEPTS the last ISR, so a program ending in
    // RD_MAC reports finished while its own result is still in flight.
    s = emu_isr_default_ch(ISR_OP_EOS, all_ch);
    if ((e = emu_isr_build(&w->prog[i++], &s))) { fprintf(stderr, "EOS: %s\n", e); return false; }

    for (unsigned ch = 0; ch < w->nch; ch++)
        for (unsigned b = 0; b < EMU_LANES_PER_WORD; b++)
            w->expect[ch * EMU_LANES_PER_WORD + b] =
                emu_f32_to_bf16((float)w->K * (float)(1u + 16u * ch + b));
    return true;
}

// Operands go down ONCE — they do not depend on the timing.
static bool work_load(const struct work *w)
{
    size_t beats = (size_t)w->L * EMU_LANES_PER_WORD;
    uint16_t *buf = malloc(beats * sizeof *buf);
    uint16_t *vec = malloc((size_t)w->chunks * beats * sizeof *vec);
    bool ok = true;

    if (!buf || !vec) { free(buf); free(vec); return false; }

    for (size_t i = 0; i < (size_t)w->chunks * beats; i++) vec[i] = emu_f32_to_bf16(1.0f);
    if (gpr_write(VEC_WORD, vec, w->chunks * w->L) < 0) ok = false;

    for (unsigned ch = 0; ch < w->nch && ok; ch++)
        for (unsigned b = 0; b < EMU_NBANKS && ok; b++) {
            for (size_t i = 0; i < beats; i++)
                buf[i] = emu_f32_to_bf16((float)(1u + 16u * ch + b));
            for (unsigned c = 0; c < w->chunks && ok; c++) {
                uint64_t axi = pim_bank(ch, b)
                             + (uint64_t)(w->row0 + c) * EMU_ROW_BYTES;
                if (axi_write(axi, buf, beats * sizeof *buf) < 0) ok = false;
            }
        }
    free(buf); free(vec);
    return ok;
}

// One launch.  Returns wrong-lane count, or -1 if it could not run.
static int work_run(const struct work *w, uint64_t *us_out)
{
    uint16_t got[PIM_NCH][EMU_LANES_PER_WORD];
    uint16_t poison[PIM_NCH][EMU_LANES_PER_WORD];
    uint64_t t0;
    int wrong = 0;

    for (unsigned ch = 0; ch < w->nch; ch++)
        for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++) poison[ch][i] = POISON_LANE;

    if (!require_idle("load the program", 1000)) return -1;
    if (axi_write(EMU_AXI_IMEM, w->prog, (size_t)w->nprog * EMU_WORD_BYTES) < 0) return -1;
    if (gpr_write(DST_WORD, poison, w->nch) < 0) return -1;
    if (!require_idle("ring the doorbell", 1000)) return -1;

    cfr_wr(CFR_PROG_LEN, w->nprog);
    if (cfr_rd(CFR_PROG_LEN) != w->nprog) {
        fprintf(stderr, "PROG_LEN readback failed\n"); return -1;
    }
    (void)cfr_rd(CFR_PROG_LEN);        // ordering point: posted write lands first

    t0 = now_us();
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);
    while (now_us() - t0 < 2000000ull)
        if (cfr_rd(CFR_STATUS) & CFR_STATUS_DONE) break;
    if (us_out) *us_out = now_us() - t0;

    // done is a held level and can be stale, so it decides nothing.  The POISON
    // does: every word must have lost it.
    for (unsigned tries = 0; tries < 64; tries++) {
        bool all = true;
        if (gpr_read(DST_WORD, got, w->nch) < 0) return -1;
        for (unsigned ch = 0; ch < w->nch && all; ch++)
            for (unsigned i = 0; i < EMU_LANES_PER_WORD; i++)
                if (got[ch][i] == POISON_LANE) { all = false; break; }
        if (all) break;
    }
    for (unsigned ch = 0; ch < w->nch; ch++)
        for (unsigned b = 0; b < EMU_LANES_PER_WORD; b++)
            if (got[ch][b] != w->expect[ch * EMU_LANES_PER_WORD + b]) wrong++;
    return wrong;
}

// ================================ main ======================================
static void usage(const char *p)
{
    printf(
"%s — the DRAM timing registers, one argument each.\n"
"\n"
"  EVERY RUN WRITES ALL EIGHT.  The base set is\n"
"    faw %u  rrd %u  rcd %u  ccd %u  rtp %u  rp %u  wr %u  ras %u\n"
"  and what goes into the register is base * scale, refused above %u\n"
"  (the register is 8 bits).\n"
"\n"
"    --scale N                 multiply the whole set by a whole number (default 1)\n"
"    --faw N  --rrd N  --rcd N  --ccd N  --rtp N  --rp N  --wr N  --ras N\n"
"                              replace one base value; it is scaled with the rest\n"
"    --rbtp N                  the bank controller's name for --rtp\n"
"    --sweep <name> v1,v2,..   one knob, several base values, one run each\n"
"    --reset                   base becomes EMU_TIMING_SIM (emu_regs.h)\n"
"    --show                    print what the board holds and exit, writing nothing\n"
"    --allow-unsafe            permit T_CCD < 2 (see below)\n"
"\n"
"  what each setting is measured with\n"
"    the violation CSR is cleared and checked clear, then ONE QDMA READ PER BANK\n"
"    at --row goes THROUGH THE MC, then the CSR is read back and printed.  The\n"
"    direct aperture is not used: it reaches HBM without a bank controller, so no\n"
"    counter could move (docs/bd-reference.md §5).\n"
"    --row N      the row every bank is read at        (default 300)\n"
"    --beats N    beats read per bank, 1..%u          (default %u = a whole row)\n"
"    --reps N     sweeps per setting                   (default 10)\n"
"    --no-run     set the registers and stop.  NEEDS NO QUEUES.\n"
"    --keep       leave the last setting on the board\n"
"    --h2c/--c2h PATH\n"
"\n"
"  --gemv        also run the old kernel and check the ANSWER\n"
"    --chunks N   MACs, each against its own DRAM row  (default 4)\n"
"    --l N        beats per MAC, 1..%u                 (default %u)\n"
"\n"
"  T_CCD < 2 HALVES A MAC RESULT AND REPORTS NOTHING — no counter, no status bit,\n"
"  no error.  It is refused unless --allow-unsafe, and only --gemv can show it.\n"
"\n"
"  A violation counter is the emulator saying its memory was SLOWER than the\n"
"  allowance it was given, not that anything is wrong.  Counts saturate at 255.\n"
"\n"
"Exit: 0 everything ran, 1 a transfer/register/--gemv answer failed, 2 usage\n",
    p,
    TIMING_BASE.faw, TIMING_BASE.rrd, TIMING_BASE.rcd, TIMING_BASE.ccd,
    TIMING_BASE.rtp, TIMING_BASE.rp,  TIMING_BASE.wr,  TIMING_BASE.ras,
    T_REG_MAX, EMU_BEATS_PER_ROW, EMU_BEATS_PER_ROW, EMU_MAX_OPSIZE, EMU_MAX_OPSIZE);
}

// One setting: apply, drive traffic, report.  Returns 0 if everything ran, -1 if a
// register or a transfer failed, and the wrong-lane count if --gemv disagreed.
static int one(const struct emu_timing *base, unsigned scale, const struct work *w,
               unsigned reps, unsigned row, unsigned beats, uint8_t *buf,
               bool allow_unsafe, bool run, bool gemv)
{
    struct emu_timing t;
    struct viol_sum s;
    uint64_t us = 0;
    int wrong = 0;

    if (!timing_scale(base, scale, &t)) return -1;
    printf("\n");
    timing_print("timing         :", &t);
    if (scale != 1u) timing_print("  from base    :", base);

    if (!timing_set(&t, allow_unsafe)) return -1;
    if (!run) { printf("  set, not measured (--no-run)\n"); return 0; }

    if (!viol_clear(w->nch)) return -1;
    for (unsigned r = 0; r < reps; r++) {
        uint64_t u = 0;
        if (sweep_run(w->nch, row, beats, buf, &u) < 0) return -1;
        us += u;
    }
    printf("  read sweep   : %u rep(s) x %u ch x %u bank(s) x %u B at row %u"
           " -> %.1f us/sweep, %.1f MB/s\n",
           reps, w->nch, EMU_NBANKS, beats * EMU_WORD_BYTES, row,
           (double)us / reps,
           us ? (double)reps * w->nch * EMU_NBANKS * beats * EMU_WORD_BYTES
                / (double)us : 0.0);
    viol_read_print(w->nch, &s);

    if (gemv) {
        if (!viol_clear(w->nch)) return -1;
        us = 0;
        for (unsigned r = 0; r < reps; r++) {
            uint64_t u = 0;
            int bad = work_run(w, &u);
            if (bad < 0) return -1;
            us += u;
            if (r == reps - 1) wrong = bad;
        }
        printf("  gemv         : %.1f us/launch, %u/%u lanes  %s\n",
               (double)us / reps, w->nch * EMU_LANES_PER_WORD - wrong,
               w->nch * EMU_LANES_PER_WORD, wrong ? "WRONG" : "exact");
        viol_read_print(w->nch, &s);
    }
    return wrong;
}

int main(int argc, char **argv)
{
    const char *bdf = NULL, *h2c = NULL, *c2h = NULL;
    const char *sweep_name = NULL, *sweep_vals = NULL;
    struct work w = { .chunks = 4, .L = EMU_MAX_OPSIZE, .nch = PIM_NCH, .row0 = 300 };
    unsigned reps = 10, row = 300, beats = EMU_BEATS_PER_ROW;
    unsigned scale = 1;
    bool run = true, keep = false, unsafe = false, do_reset = false, gemv = false;
    bool show = false;
    bool set_field[8] = { false };
    uint8_t set_val[8] = { 0 };
    struct emu_timing before, base;
    uint8_t *buf = NULL;
    int failed = 0;

    pim_platform_banner();
    { const char *bad = pim_platform_check(); if (bad) { fprintf(stderr, "%s\n", bad); return 2; } }

    static struct option lo[] = {
        {"faw",1,0,'0'},{"rrd",1,0,'1'},{"rcd",1,0,'2'},{"ccd",1,0,'3'},
        {"rtp",1,0,'4'},{"rp",1,0,'5'},{"wr",1,0,'6'},{"ras",1,0,'7'},
        {"rbtp",1,0,'4'},
        {"scale",1,0,'s'},
        {"sweep",1,0,'S'},{"reset",0,0,'R'},{"allow-unsafe",0,0,'U'},{"show",0,0,'w'},
        {"row",1,0,'W'},{"beats",1,0,'b'},{"reps",1,0,'N'},
        {"gemv",0,0,'g'},{"chunks",1,0,'C'},{"l",1,0,'L'},
        {"no-run",0,0,'n'},{"keep",0,0,'K'},
        {"bdf",1,0,'B'},{"h2c",1,0,'H'},{"c2h",1,0,'D'},{"help",0,0,'h'},{0,0,0,0}
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        if (o >= '0' && o <= '7') {
            set_field[o - '0'] = true;
            set_val[o - '0'] = (uint8_t)strtoul(optarg, NULL, 0);
            continue;
        }
        switch (o) {
        case 's': {
            char *end = NULL;
            unsigned long v = strtoul(optarg, &end, 0);
            // A whole number, and said so: strtoul would take "4.25" as 4 and run
            // a setting nobody asked for.
            if (!v || *end) {
                fprintf(stderr, "--scale takes a whole number >= 1 ('%s'); to reach a "
                                "value between two multiples name it, e.g. --rcd 64\n",
                        optarg);
                return 2;
            }
            scale = (unsigned)v;
            break;
        }
        case 'S':
            // --sweep takes TWO words; getopt gives one, so the value list is the
            // next argv.  Checked here so a missing list is a usage error and not
            // a sweep over nothing.
            sweep_name = optarg;
            if (tname_index(sweep_name) < 0) {
                fprintf(stderr, "--sweep: '%s' is not a timing parameter\n", sweep_name);
                return 2;
            }
            if (optind >= argc) { fprintf(stderr, "--sweep %s: no value list\n", sweep_name); return 2; }
            sweep_vals = argv[optind++];
            break;
        case 'R': do_reset = true; break;
        case 'w': show = true; break;
        case 'U': unsafe = true; break;
        case 'g': gemv = true; break;
        case 'C': w.chunks = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'L': w.L      = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'b': beats    = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'W': row      = (unsigned)strtoul(optarg, NULL, 0); w.row0 = row; break;
        case 'N': reps     = (unsigned)strtoul(optarg, NULL, 0); break;
        case 'n': run = false; break;
        case 'K': keep = true; break;
        case 'B': bdf = optarg; break;
        case 'H': h2c = optarg; break;
        case 'D': c2h = optarg; break;
        default: usage(argv[0]); return o == 'h' ? 0 : 2;
        }
    }
    if (!w.chunks || !w.L || w.L > EMU_MAX_OPSIZE) { usage(argv[0]); return 2; }
    if (!beats || beats > EMU_BEATS_PER_ROW) { usage(argv[0]); return 2; }
    if (!reps) reps = 1;

    // ---- the BAR, and nothing else unless a workload was asked for ----------
    {
        char path[256];
        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource2",
                 bdf ? bdf : "0000:01:00.0");
        int fd = open(path, O_RDWR | O_SYNC);
        if (fd < 0) { fprintf(stderr, "open(%s): %s\n", path, strerror(errno)); return 1; }
        void *m = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { fprintf(stderr, "mmap BAR2: %s\n", strerror(errno)); return 1; }
        g_bar = m;
    }
    if (cfr_rd(CFR_STATUS) == 0xFFFFFFFFu) {
        fprintf(stderr, "the CFR reads all-ones: the device is not answering.\n"
                        "Either the BAR mapping is stale or the PDI was reprogrammed "
                        "underneath this process.\n");
        return 1;
    }

    timing_get(&before);
    timing_print("board now      :", &before);

    base = TIMING_BASE;
    if (do_reset) { struct emu_timing sim = EMU_TIMING_SIM; base = sim; }
    for (unsigned i = 0; i < 8; i++) if (set_field[i]) ((uint8_t *)&base)[i] = set_val[i];
    timing_print("base           :", &base);
    printf("scale          : x%u\n", scale);
    // The only path that writes NOTHING.  Every other run writes all eight, so
    // without this there is no way to ask the board what it is holding.
    if (show) {
        struct emu_timing t;
        if (timing_scale(&base, scale, &t)) timing_print("would write    :", &t);
        return 0;
    }

    if (run) {
        // The sweep reads; only --gemv writes.  Opening H2C for a read-only run
        // would demand a queue this tool is not going to use.
        g_c2h = open(c2h ? c2h : "/dev/qdma01000-MM-1", O_RDONLY);
        if (g_c2h < 0 || (gemv && (g_h2c = open(h2c ? h2c : "/dev/qdma01000-MM-0",
                                                O_WRONLY)) < 0)) {
            fprintf(stderr, "cannot open the MM queues: %s\n"
                            "  sudo ./qdma_queues.sh setup\n"
                            "  (or --no-run to set the registers without measuring)\n",
                    strerror(errno));
            return 1;
        }
        if (!sweep_addr_check(w.nch, row)) return 1;
        buf = malloc((size_t)beats * EMU_WORD_BYTES);
        if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
        printf("\nread sweep     : %u channel(s) x %u banks, MC path %s, row %u,\n"
               "                 %u beat(s) = %u B per bank, %u rep(s) per setting\n",
               w.nch, EMU_NBANKS, PIM_ADDR_MAP_NAME, row,
               beats, beats * EMU_WORD_BYTES, reps);
        if (gemv) {
            if (!work_build(&w)) return 1;
            printf("gemv           : %u chunk(s) x %u beat(s) = K %u, DRAM rows %u..%u,"
                   " %u ISRs\n", w.chunks, w.L, w.K, w.row0, w.row0 + w.chunks - 1,
                   w.nprog);
            if (!work_load(&w)) return 1;
        }
    }

    if (sweep_name) {
        int f = tname_index(sweep_name);
        char *copy = strdup(sweep_vals), *tok, *save = NULL;
        for (tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
            struct emu_timing b = base;
            ((uint8_t *)&b)[f] = (uint8_t)strtoul(tok, NULL, 0);
            if (one(&b, scale, &w, reps, row, beats, buf, unsafe, run, gemv) != 0)
                failed++;
        }
        free(copy);
    } else {
        if (one(&base, scale, &w, reps, row, beats, buf, unsafe, run, gemv) != 0)
            failed++;
    }

    if (keep) {
        printf("\nleft on the board as set (--keep)\n");
    } else {
        // Restoring needs the same permission that reading it did not: the board may
        // have arrived with an unsafe value.
        if (!timing_set(&before, true)) fprintf(stderr, "could not restore the timing\n");
        else timing_print("\nrestored       :", &before);
    }

    printf("\n%s\n", failed ? "FAIL — a setting could not be applied or measured" : "PASS");
    return failed ? 1 : 0;
}
