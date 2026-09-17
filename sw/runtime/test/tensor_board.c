// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// tensor_board.c — pim_tensor's RED_MAJOR half, on the card, and the one question
// only the card can answer.
//
// lib/test/tensor_test proves PLACEMENT against fake_drv.h: every element of an
// append lands where pim_tensor_offset says, the lazy clear covers exactly the
// chunks it should, truncate restores what append cannot.  All of that is address
// arithmetic against a simulated backing store, and it would pass identically on a
// machine with no PCIe in it.
//
// THREE THINGS IT CANNOT SAY.
//
//   1  that PIM_ALLOC_F_ZERO clears REAL card memory.  The fake's memcpy is a host
//      memcpy; the real one is a pwrite through a QDMA queue into DRAM behind a
//      memory controller, and "the DMA reported success" is not "the bytes are
//      zero".  Poison, free, re-allocate the same granules with the flag, read back.
//
//   2  that a RED_MAJOR append — the scatter, one two-byte piece per output at
//      row_bytes apart — arrives intact.  This is the transfer pattern the whole KV
//      design rests on and it is the one the transport is worst at.
//
//   3  AND THE REAL PRIZE: whether the zero invariant is LOAD-BEARING.
//
// On (3).  The tail between the frontier and its beat boundary is multiplied by a
// vector lane the host zeroed, and the question is whether that zero is enough.
// This file was written believing it was not, for a reason that turned out to be
// wrong: that the datapath is a block float and one large-exponent lane would set
// the exponent for its beat and shift the real values beside it into nothing.
//
// The sweep below says otherwise, and the real answer is narrower and more useful.
// A tail lane of 1.5e18, or even the largest finite BF16, changes nothing — so the
// exponent is taken AFTER the multiply.  What a zero multiplier does not survive is
// the IEEE special: 0 * Inf is NaN, and one NaN poisons its bank's whole
// accumulation.  Six values are tried for exactly that reason; one would have
// answered "fine" and been believed.
//
// Then the tail is cleared and the same MAC must come out exact, which is the
// regression test whichever way the sweep falls.
//
//     ./tensor_board [--nkv N]      # needs pim.ko and the card
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim/pim.h"
#include "pim/pim_addr.h"
#include "pimrt/pim_exec.h"
#include "pimrt/pim_gemv.h"
#include "pimrt/pim_tensor.h"

#include "emu_regs.h"

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float    b2f(uint16_t h){ uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t rng = 2463534242u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

// One launch: WRVEC the vector, one MAC over `beats` beats at COL 0, drain, EOS.
// Hand-assembled for the same reason col_probe's is — there is no attention builder
// yet, and this is what one will emit.
static const char *run_sv(pim_exec *e, const pim_geometry *g, uint32_t row,
                          uint32_t beats, uint32_t xword, uint32_t yword)
{
    pim_isr  storage[8];
    pim_prog p;
    struct emu_isr_spec s;
    struct emu_isr      isr;
    const char *bad;

    pim_prog_init(&p, storage, 8);

    s = emu_isr_default_ch(ISR_OP_WRVEC, (1u << g->nch) - 1u);
    s.opsize = beats;  s.row = xword;
    if ((bad = emu_isr_build(&isr, &s))) return bad;
    if ((bad = pim_prog_push(&p, (const pim_isr *)&isr))) return bad;

    s = emu_isr_default_ch(ISR_OP_MAC, (1u << g->nch) - 1u);
    s.opsize     = beats;
    s.row        = row;
    s.col        = 0;
    s.pu_mask    = (1u << g->nbank) - 1u;
    s.gb_mc_mask = s.pu_mask;
    if ((bad = emu_isr_build(&isr, &s))) return bad;
    if ((bad = pim_prog_push(&p, (const pim_isr *)&isr))) return bad;

    for (uint32_t ch = 0; ch < g->nch; ch++) {
        s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << ch);
        s.opsize = 0;  s.row = yword + ch;
        if ((bad = emu_isr_build(&isr, &s))) return bad;
        if ((bad = pim_prog_push(&p, (const pim_isr *)&isr))) return bad;
    }

    s = emu_isr_default_ch(ISR_OP_EOS, (1u << g->nch) - 1u);
    if ((bad = emu_isr_build(&isr, &s))) return bad;
    if ((bad = pim_prog_push(&p, (const pim_isr *)&isr))) return bad;

    return pim_exec_run(e, &p, NULL);
}

int main(int argc, char **argv)
{
    pim_ctx  *c = NULL;
    pim_exec *e = NULL;
    const pim_geometry *g;
    const char *bad;
    uint32_t    nkv = 100;              /* deliberately not a multiple of 16 */
    pim_tensor  V;
    uint32_t    nout, beats, npadded;
    uint16_t   *Vh, *S, *got, *golden, *col;
    void       *xg, *yg;
    size_t      ybytes;
    uint32_t    xword, xn, yword, yn, row;
    pim_unit    un;

    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--nkv") && i + 1 < argc) nkv = (uint32_t)atoi(argv[++i]);

    printf("pim_tensor RED_MAJOR, on the board\n");
    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }
    {
        pim_exec_config ec = { 0 };
        if ((bad = pim_exec_open(c, &ec, &e))) {
            printf("SKIPPED: %s\n", bad); pim_close(c); return 0;
        }
    }
    pim_exec_clear_violations(e);
    g = pim_geom_ctx(c);

    nout    = g->nch * g->nbank;                    /* exactly one supergroup */
    beats   = (nkv + 15u) / 16u;
    npadded = beats * 16u;
    printf("  %u ch x %u bank -> %u outputs, S_kv = %u -> %u beats covering %u\n",
           g->nch, g->nbank, nout, nkv, beats, npadded);

    // ---- 1. PIM_ALLOC_F_ZERO against real card memory ----------------------
    printf("\n  1. PIM_ALLOC_F_ZERO on real DRAM\n");
    {
        size_t   n = 256u << 10;
        uint16_t *back = malloc(n);
        unsigned char *poison = malloc(n);
        void    *p;
        pim_loc  l0, l1;
        size_t   run, nonzero = 0, kept = 0;

        memset(poison, 0xA5, n);
        p = pim_alloc_ex_ctx(c, n, PIM_MEM_DRAM, 0);
        CHECK(p != NULL, "plain alloc: %s", pim_last_error_ctx(c));
        if (p) {
            CHECK(!pim_memcpy_ctx(c, p, poison, n, PIM_TO_DEV, 0), "poison");
            CHECK(!pim_resolve_ctx(c, p, n, &l0, &run), "resolve");
            pim_free_ctx(c, p);
        }
        p = pim_alloc_ex_ctx(c, n, PIM_MEM_DRAM, PIM_ALLOC_F_ZERO);
        CHECK(p != NULL, "F_ZERO alloc: %s", pim_last_error_ctx(c));
        if (p) {
            CHECK(!pim_resolve_ctx(c, p, n, &l1, &run), "resolve");
            CHECK(l1.axi == l0.axi, "different granules (%#llx vs %#llx), so nothing "
                  "had poisoned what F_ZERO cleared",
                  (unsigned long long)l1.axi, (unsigned long long)l0.axi);
            CHECK(!pim_memcpy_ctx(c, back, p, n, PIM_FROM_DEV, 0), "readback");
            for (size_t i = 0; i < n / 2; i++) if (back[i]) nonzero++;
            for (size_t i = 0; i < n / 2; i++) if (back[i] == 0xA5A5) kept++;
            CHECK(nonzero == 0, "%zu of %zu elements are not zero (%zu still poison)",
                  nonzero, n / 2, kept);
            printf("     %zu KiB at %#llx: poisoned, freed, re-taken with F_ZERO -> "
                   "%zu/%zu nonzero\n", n >> 10, (unsigned long long)l1.axi,
                   nonzero, n / 2);
            pim_free_ctx(c, p);
        }
        free(back); free(poison);
    }

    // ---- 2. a RED_MAJOR append lands, one scattered piece per output --------
    printf("\n  2. RED_MAJOR append: %u tokens x %u outputs, two bytes a piece\n",
           nkv, nout);
    if ((bad = pim_tensor_alloc(c, PIM_LAYOUT_RED_MAJOR, nout, 512, 0, &V))) {
        printf("  alloc: %s\n", bad); return 1;
    }
    Vh     = malloc((size_t)npadded * nout * 2);    /* [red][out], host order */
    S      = malloc((size_t)npadded * 2);
    got    = NULL;
    golden = malloc((size_t)nout * 2);
    col    = malloc((size_t)npadded * 2);
    memset(Vh, 0, (size_t)npadded * nout * 2);
    memset(S,  0, (size_t)npadded * 2);

    for (uint32_t s = 0; s < nkv; s++) {
        for (uint32_t d = 0; d < nout; d++) Vh[(size_t)s * nout + d] = rnd_bf16();
        S[s] = rnd_bf16();
        if ((bad = pim_tensor_append(c, &V, s, 1, Vh + (size_t)s * nout))) {
            printf("  append %u: %s\n", s, bad); return 1;
        }
    }
    CHECK(V.frontier == nkv, "frontier is %u after %u appends", V.frontier, nkv);
    {
        size_t    nb = pim_tensor_bytes(g, &V);
        uint16_t *snap = malloc(nb);
        uint32_t  misplaced = 0, tail_bad = 0;

        CHECK(!pim_memcpy_ctx(c, snap, V.base, nb, PIM_FROM_DEV, 0), "snapshot");
        for (uint32_t s = 0; s < nkv; s++)
            for (uint32_t d = 0; d < nout; d++)
                if (snap[pim_tensor_offset(g, &V, d, s) / 2] != Vh[(size_t)s * nout + d])
                    misplaced++;
        for (uint32_t s = nkv; s < npadded; s++)
            for (uint32_t d = 0; d < nout; d++)
                if (snap[pim_tensor_offset(g, &V, d, s) / 2]) tail_bad++;
        CHECK(misplaced == 0, "%u of %u appended elements landed elsewhere",
              misplaced, nkv * nout);
        CHECK(tail_bad == 0, "%u element(s) in the frontier's own beat are not zero",
              tail_bad);
        printf("     %u elements placed, [%u, %u) clear on the card\n",
               nkv * nout, nkv, npadded);
        free(snap);
    }

    // ---- 3. the GPR side, and the golden answer ----------------------------
    xg = pim_alloc_ctx(c, (size_t)npadded * 2, PIM_MEM_GPR);
    ybytes = (size_t)g->nch * 32;
    yg = pim_alloc_ctx(c, ybytes, PIM_MEM_GPR);
    if (!xg || !yg) { printf("  GPR: %s\n", pim_last_error_ctx(c)); return 1; }
    got = malloc(ybytes);
    if ((bad = pim_memcpy_ctx(c, xg, S, (size_t)npadded * 2, PIM_TO_DEV, 0)) ||
        (bad = pim_addr_gpr_words_ctx(c, xg, (size_t)npadded * 2, &xword, &xn)) ||
        (bad = pim_addr_gpr_words_ctx(c, yg, ybytes, &yword, &yn)) ||
        (bad = pim_addr_unit_ctx(c, V.base, pim_tensor_bytes(g, &V), 0, &un))) {
        printf("  setup: %s\n", bad); return 1;
    }
    row = (uint32_t)un.row;

    // out[d] = sum_s S[s] * V[s][d], over the PADDED length, with the tail zeroed on
    // both sides.  pim_mac_exact is the transcribed RTL, not an fp32 model.
    for (uint32_t d = 0; d < nout; d++) {
        for (uint32_t s = 0; s < npadded; s++) col[s] = Vh[(size_t)s * nout + d];
        golden[d] = pim_mac_exact(col, S, npadded);
    }

    // The accumulator latch outlives processes.
    pim_exec_scrub(e, yword);

    // ---- 4. is the zero invariant load-bearing? ----------------------------
    printf("\n  4. the tail between the frontier and its beat boundary\n");
    {
        uint32_t nbad = 0, first = 0;

        // (a) clean, which is what append left behind
        memset(got, 0, ybytes);
        CHECK(!pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0), "poison y");
        if ((bad = run_sv(e, g, row, beats, xword, yword))) {
            printf("     clean tail: %s\n", bad); fail++;
        } else {
            CHECK(!pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0), "readback");
            for (uint32_t d = 0; d < nout; d++) {
                uint16_t hw = got[(d / g->nbank) * 16 + (d % g->nbank)];
                if (hw != golden[d]) { if (!nbad) first = d; nbad++; }
            }
            CHECK(nbad == 0, "clean tail: %u of %u differ, first at %u (%g vs %g)",
                  nbad, nout, first, b2f(got[(first / g->nbank) * 16 + first % g->nbank]),
                  b2f(golden[first]));
            if (!nbad) printf("     invariant kept   : %u/%u exact against "
                              "pim_mac_exact\n", nout, nout);
        }

        // (b) THE SWEEP.  One value would not characterise this.  Uninitialised
        // DRAM decodes as whatever is there, and BF16's exponent field makes three
        // genuinely different cases out of that: merely enormous, infinite, and not
        // a number.  A block float could plausibly survive the first and not the
        // others — the exponent maximum is arithmetic, NaN propagation is not.
        if (npadded > nkv) {
            static const struct { uint16_t v; const char *what; } poison[] = {
                { 0x5DA6u, "1.5e18, large but finite" },
                { 0x7F7Fu, "3.4e38, the largest finite BF16" },
                { 0x7F80u, "+Inf" },
                { 0xFF80u, "-Inf" },
                { 0x7FC0u, "NaN" },
                { 0xA5A5u, "0xA5A5, what an unwritten page reads as" },
            };
            uint32_t broke = 0;

            for (unsigned pi = 0; pi < sizeof poison / sizeof poison[0]; pi++) {
                uint32_t dirty = 0, dfirst = 0;

                for (uint32_t s = nkv; s < npadded; s++)
                    for (uint32_t d = 0; d < nout; d++) {
                        uint16_t v = poison[pi].v;
                        CHECK(!pim_memcpy_ctx(c, (char *)V.base +
                                              pim_tensor_offset(g, &V, d, s),
                                              &v, 2, PIM_TO_DEV, 0), "dirty the tail");
                    }

                pim_exec_scrub(e, yword);
                memset(got, 0, ybytes);
                CHECK(!pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0), "poison y");
                if ((bad = run_sv(e, g, row, beats, xword, yword))) {
                    printf("     %-34s launch failed: %s\n", poison[pi].what, bad);
                    fail++;
                    continue;
                }
                CHECK(!pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0), "readback");
                for (uint32_t d = 0; d < nout; d++) {
                    uint16_t hw = got[(d / g->nbank) * 16 + (d % g->nbank)];
                    if (hw != golden[d]) { if (!dirty) dfirst = d; dirty++; }
                }
                printf("     %-34s %2u/%u outputs differ", poison[pi].what, dirty, nout);
                if (dirty) {
                    printf("   first %u: %g vs %g", dfirst,
                           b2f(got[(dfirst / g->nbank) * 16 + dfirst % g->nbank]),
                           b2f(golden[dfirst]));
                    broke++;
                }
                printf("\n");
            }

            if (broke)
                printf("\n       ==> THE INVARIANT IS LOAD-BEARING for %u of %zu tail\n"
                       "           values.  A zeroed vector lane does NOT neutralise\n"
                       "           every DRAM lane, so V's tail must be cleared.\n",
                       broke, sizeof poison / sizeof poison[0]);
            else
                printf("\n       ==> A ZERO MULTIPLIER IS ENOUGH, for every tail value\n"
                       "           tried including Inf and NaN.  The block exponent is\n"
                       "           taken after the multiply, not before.  Clearing V's\n"
                       "           tail is belt-and-braces; keep it, it is nearly free,\n"
                       "           but the S vector's zero padding is what carries the\n"
                       "           contract and THAT one is not optional.\n");

        }
    }

    if (npadded > nkv) {
        // (c) AND RESTORING IT MUST BRING THE EXACT ANSWER BACK.  This is not
        // decoration: section 5 below assumes the matrix tail is zero, and without
        // this it inherits whatever the sweep left there — which made the mirror
        // measure a product of two poisons and read as a finding.
        CHECK(!pim_tensor_truncate(c, &V, nkv), "truncate to the frontier");
        {
            uint16_t z = 0;
            for (uint32_t s2 = nkv; s2 < npadded; s2++)
                for (uint32_t d = 0; d < nout; d++)
                    CHECK(!pim_memcpy_ctx(c, (char *)V.base +
                                          pim_tensor_offset(g, &V, d, s2),
                                          &z, 2, PIM_TO_DEV, 0), "re-clear");
        }
        pim_exec_scrub(e, yword);
        memset(got, 0, ybytes);
        CHECK(!pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0), "poison y");
        if (!(bad = run_sv(e, g, row, beats, xword, yword))) {
            uint32_t back_bad = 0;
            CHECK(!pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0), "readback");
            for (uint32_t d = 0; d < nout; d++)
                if (got[(d / g->nbank) * 16 + (d % g->nbank)] != golden[d]) back_bad++;
            CHECK(back_bad == 0, "after re-clearing the matrix tail, %u of %u differ",
                  back_bad, nout);
            if (!back_bad)
                printf("     matrix tail cleared again: %u/%u exact\n", nout, nout);
        }
    }

    // ---- 5. the MIRROR: garbage in the VECTOR's tail, zero in the matrix's ----
    //
    // pim_gemv.h says the same thing about the vector side that pim_tensor.h said
    // about this one — that a stale lane "can annihilate every real lane beside it"
    // through the beat's exponent maximum.  Section 4 showed that is not what
    // happens when the garbage is in DRAM.  The multiply has two operands and there
    // is no reason to assume it treats them alike, so the mirror is measured rather
    // than argued from symmetry.
    //
    // The matrix tail is zero here (section 4 restored it), and the VECTOR tail
    // carries the poison.
    printf("\n  5. the mirror: poison in the VECTOR tail, matrix tail zero\n");
    if (npadded > nkv) {
        static const struct { uint16_t v; const char *what; } poison[] = {
            { 0x5DA6u, "1.5e18, large but finite" },
            { 0x7F7Fu, "3.4e38, the largest finite BF16" },
            { 0x7F80u, "+Inf" },
            { 0x7FC0u, "NaN" },
        };
        uint32_t broke = 0;

        for (unsigned pi = 0; pi < sizeof poison / sizeof poison[0]; pi++) {
            uint32_t dirty = 0, dfirst = 0;

            for (uint32_t t = nkv; t < npadded; t++) S[t] = poison[pi].v;
            CHECK(!pim_memcpy_ctx(c, xg, S, (size_t)npadded * 2, PIM_TO_DEV, 0),
                  "dirty the vector");
            pim_exec_scrub(e, yword);
            memset(got, 0, ybytes);
            CHECK(!pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0), "poison y");
            if ((bad = run_sv(e, g, row, beats, xword, yword))) {
                printf("     %-34s launch failed: %s\n", poison[pi].what, bad);
                fail++; continue;
            }
            CHECK(!pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0), "readback");
            for (uint32_t d = 0; d < nout; d++) {
                uint16_t hw = got[(d / g->nbank) * 16 + (d % g->nbank)];
                if (hw != golden[d]) { if (!dirty) dfirst = d; dirty++; }
            }
            printf("     %-34s %2u/%u outputs differ", poison[pi].what, dirty, nout);
            if (dirty) {
                printf("   first %u: %g vs %g", dfirst,
                       b2f(got[(dfirst / g->nbank) * 16 + dfirst % g->nbank]),
                       b2f(golden[dfirst]));
                broke++;
            }
            printf("\n");
        }
        printf("       ==> %s\n", broke
               ? "the vector side behaves like the matrix side: only the\n"
                 "           IEEE specials get through a zero multiplier."
               : "nothing gets through, not even Inf.  The two operands are\n"
                 "           NOT symmetric, and the vector's tail is the safer one.");

        for (uint32_t t = nkv; t < npadded; t++) S[t] = 0;
        CHECK(!pim_memcpy_ctx(c, xg, S, (size_t)npadded * 2, PIM_TO_DEV, 0), "re-zero");
        pim_exec_scrub(e, yword);
        memset(got, 0, ybytes);
        CHECK(!pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0), "poison y");
        if (!(bad = run_sv(e, g, row, beats, xword, yword))) {
            uint32_t back_bad = 0;
            CHECK(!pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0), "readback");
            for (uint32_t d = 0; d < nout; d++)
                if (got[(d / g->nbank) * 16 + (d % g->nbank)] != golden[d]) back_bad++;
            CHECK(back_bad == 0, "after re-zeroing the vector tail, %u of %u differ",
                  back_bad, nout);
            if (!back_bad) printf("     vector tail zeroed again: %u/%u exact\n",
                                  nout, nout);
        }
    }

    {
        // THE CONDITIONS THIS RUN WAS MEASURED UNDER.  Nothing here sets them —
        // hwdef/test/emu_timing does — so a result is only comparable to another
        // result taken at the same eight numbers, and printing them is what makes
        // that checkable instead of remembered.
        const pim_timing *t = pim_exec_timing_at_open(e);
        printf("  timing (set by emu_timing, read here): faw %u rrd %u rcd %u "
               "ccd %u rtp %u rp %u wr %u ras %u\n",
               t->faw, t->rrd, t->rcd, t->ccd, t->rtp, t->rp, t->wr, t->ras);
    }
    for (unsigned ch = 0; ch < g->nch; ch++) {
        pim_viol v;
        if (!pim_exec_violation_detail(e, ch, &v))
            printf("  ch%u  rcd_rd %u  ccd_rd %u  ccd_wr %u  recovery_wr %u\n",
                   ch, v.rcd_rd, v.ccd_rd, v.ccd_wr, v.recovery_wr);
    }

    pim_free_ctx(c, xg); pim_free_ctx(c, yg);
    pim_tensor_free(c, &V);
    free(Vh); free(S); free(got); free(golden); free(col);
    pim_exec_close(e); pim_close(c);

    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail ? 1 : 0;
}
