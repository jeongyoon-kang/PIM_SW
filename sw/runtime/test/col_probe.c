// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// col_probe.c — what does a MAC's COL field offset?
//
// THE QUESTION.  COL is a beat index into the DRAM row and it has never been used:
// every MAC this stack has ever issued carried COL = 0.  A MAC has TWO operands
// though — the DRAM row and the global buffer a WRVEC filled — and the ISA
// documents COL only for the DRAM side.  So:
//
//     MAC(COL = 8, OPSIZE = 8)   reads DRAM row beats 8..15.   And the GB?
//
//   case A   the GB is read from beat 0 regardless.  Then a row can hold several
//            independent operands and COL picks one, while the vector for it is
//            just WRVEC'd normally.  A K cache stops wasting (1024 / D) of its
//            memory, which for D = 128 is 8x.
//   case B   the GB is offset too.  Then the vector has to be padded to the same
//            place in the GB, and whatever the row packing saved, the vector loses.
//
// WHY IT PROBABLY IS A, AND WHY THAT IS NOT ENOUGH.  The GB has no addressing at
// all: a WRVEC fills it, a MAC rewinds it and consumes OPSIZE beats [measured,
// emu_chain --test rewind].  There is nothing for COL to index.  And pim_exec.h's
// rule 1 says a MAC must consume exactly as many beats as the WRVEC put in, which
// case B would break for any COL != 0.  But this hardware answers questions like
// this with a NUMBER rather than an error, so an argument is not an answer.
//
// HOW IT IS ASKED.  One supergroup of outputs, a full 1024-element row of weights,
// and a vector of only 128 elements (8 beats) in the GPR.  Then two launches that
// differ in ONE field:
//
//     control   MAC(COL = 0, OPSIZE = 8)  ->  must be dot(W[j][  0..127], x)
//     probe     MAC(COL = 8, OPSIZE = 8)  ->  case A: dot(W[j][128..255], x)
//
// The control is what makes the probe readable: if it fails, the harness is wrong
// and the probe says nothing.  If the control passes and the probe matches
// W[128..255], COL offsets the DRAM side only.  If the probe instead matches
// W[0..127], COL was ignored.  Anything else is case B or worse.
//
//     ./col_probe            # needs pim.ko and the card
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim/pim.h"
#include "pim/pim_addr.h"
#include "pimrt/pim_exec.h"
#include "pimrt/pim_gemv.h"

#include "emu_regs.h"

#define KFULL     1024u            /* one whole DRAM row of weights   */
#define SLICE      128u            /* elements a probe MAC reads       */
#define BEATS      (SLICE / 16u)   /* 8                                */

static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static uint32_t rng = 12345u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

static int fail;

// One launch: WRVEC the 8-beat vector, one MAC at `col`, drain, EOS.
static const char *run_at(pim_exec *e, const pim_geometry *g, uint32_t col,
                          uint32_t row, uint32_t xword, uint32_t yword)
{
    pim_isr  storage[8];
    pim_prog p;
    struct emu_isr_spec s;
    struct emu_isr      isr;
    const char *bad;

    pim_prog_init(&p, storage, 8);

    s = emu_isr_default_ch(ISR_OP_WRVEC, (1u << g->nch) - 1u);
    s.opsize = BEATS;  s.row = xword;
    if ((bad = emu_isr_build(&isr, &s))) return bad;
    if ((bad = pim_prog_push(&p, (const pim_isr *)&isr))) return bad;

    s = emu_isr_default_ch(ISR_OP_MAC, (1u << g->nch) - 1u);
    s.opsize     = BEATS;
    s.row        = row;
    s.col        = col;                    /* <-- the whole question */
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

int main(void)
{
    pim_ctx    *c = NULL;
    pim_exec   *e = NULL;
    const pim_geometry *g;
    pim_tensor  w;
    const char *bad;
    uint16_t   *W, x[SLICE], *got;
    void       *xg, *yg;
    uint32_t    n, xword, xn, yword, yn, urow;
    pim_unit    un;
    size_t      ybytes;

    printf("what does a MAC's COL offset?\n\n");

    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }
    {
        pim_exec_config ec = { .set_timing = true };
        if ((bad = pim_exec_open(c, &ec, &e))) {
            printf("SKIPPED: %s\n", bad); pim_close(c); return 0;
        }
    }
    g = pim_geom_ctx(c);
    n = g->nch * g->nbank;                  /* exactly one supergroup */
    printf("  %u ch x %u bank, one supergroup = %u outputs, row = %u elements\n",
           g->nch, g->nbank, n, KFULL);

    // The counters are sticky and survive processes, so a reading taken without
    // clearing first is whatever ran on this board before.
    pim_exec_clear_violations(e);

    // Scrub: the latch survives across processes and a MAC accumulates into it.
    {
        void *sg = pim_alloc_ctx(c, 4096, PIM_MEM_GPR);
        uint32_t word, nw;
        if (sg && !pim_addr_gpr_words_ctx(c, sg, g->nch * 32, &word, &nw))
            pim_exec_scrub(e, word);
        if (sg) pim_free_ctx(c, sg);
    }

    // Weights: a full row per output, so both slices exist in the same row.
    if ((bad = pim_gemv_alloc(c, n, KFULL, &w))) { printf("  %s\n", bad); return 1; }
    W = malloc((size_t)n * KFULL * 2);
    for (size_t i = 0; i < (size_t)n * KFULL; i++) W[i] = rnd_bf16();
    if ((bad = pim_gemv_upload(c, &w, W))) { printf("  upload: %s\n", bad); return 1; }

    // The vector: ONLY 8 beats.  That is the point — if the GB were offset by COL,
    // beats 8..15 would be whatever was left there, and nothing put anything there.
    for (uint32_t i = 0; i < SLICE; i++) x[i] = rnd_bf16();
    xg = pim_alloc_ctx(c, SLICE * 2, PIM_MEM_GPR);
    ybytes = (size_t)g->nch * 32;
    yg = pim_alloc_ctx(c, ybytes, PIM_MEM_GPR);
    if (!xg || !yg) { printf("  GPR: %s\n", pim_last_error_ctx(c)); return 1; }
    if ((bad = pim_memcpy_ctx(c, xg, x, SLICE * 2, PIM_TO_DEV, 0))) {
        printf("  vector: %s\n", bad); return 1;
    }
    if ((bad = pim_addr_gpr_words_ctx(c, xg, SLICE * 2, &xword, &xn)) ||
        (bad = pim_addr_gpr_words_ctx(c, yg, ybytes, &yword, &yn))) {
        printf("  words: %s\n", bad); return 1;
    }
    if ((bad = pim_addr_unit_ctx(c, w.base, pim_tensor_bytes(g, &w), 0, &un))) {
        printf("  unit: %s\n", bad); return 1;
    }
    urow = (uint32_t)un.row;
    got  = malloc(ybytes);

    // ------------------------------------------------------------------------
    for (int pass = 0; pass < 2; pass++) {
        uint32_t col   = pass ? BEATS : 0u;
        uint32_t first = pass ? SLICE : 0u;       /* which slice case A predicts */
        uint32_t nbad_a = 0, nbad_ctrl = 0;

        memset(got, 0, ybytes);
        if ((bad = pim_memcpy_ctx(c, yg, got, ybytes, PIM_TO_DEV, 0))) {
            printf("  poison: %s\n", bad); return 1;
        }
        if ((bad = run_at(e, g, col, urow, xword, yword))) {
            printf("  COL=%u launch: %s\n", col, bad); fail++; break;
        }
        if ((bad = pim_memcpy_ctx(c, got, yg, ybytes, PIM_FROM_DEV, 0))) {
            printf("  readback: %s\n", bad); return 1;
        }

        for (uint32_t j = 0; j < n; j++) {
            uint32_t ch   = (j / g->nbank) % g->nch;
            uint32_t bank = j % g->nbank;
            uint16_t hw   = got[ch * 16 + bank];
            uint16_t a    = pim_mac_exact(&W[(size_t)j * KFULL + first], x, SLICE);
            uint16_t ctrl = pim_mac_exact(&W[(size_t)j * KFULL],         x, SLICE);
            if (hw != a)    nbad_a++;
            if (hw != ctrl) nbad_ctrl++;
        }

        printf("\n  %s  MAC(COL=%u, OPSIZE=%u)\n",
               pass ? "probe  " : "control", col, BEATS);
        printf("     vs dot(W[%u..%u], x) : %u of %u differ\n",
               first, first + SLICE - 1, nbad_a, n);
        if (pass)
            printf("     vs dot(W[0..%u], x)   : %u of %u differ   (COL ignored?)\n",
                   SLICE - 1, nbad_ctrl, n);

        if (!pass) {
            if (nbad_a) { printf("     -> the harness is wrong; the probe says nothing\n"); fail++; }
            else          printf("     -> harness good\n");
        } else {
            if (!nbad_a)
                printf("     ==> CASE A: COL offsets the DRAM row only.  The GB is read\n"
                       "         from beat 0, so a row can pack several operands.\n");
            else if (!nbad_ctrl)
                printf("     ==> COL WAS IGNORED: the MAC read the row from beat 0.\n");
            else {
                printf("     ==> NEITHER: COL moved the GB too, or something else.\n");
                fail++;
            }
        }
        if (fail) break;
    }

    printf("\n");
    for (unsigned ch = 0; ch < g->nch; ch++) {
        pim_viol v;
        if (pim_exec_violation_detail(e, ch, &v)) continue;
        printf("  ch%u  rcd_rd %u (worst %u cy)  ccd_rd %u  ccd_wr %u  "
               "recovery_wr %u  ewmul_drop-sticky %#x\n",
               ch, v.rcd_rd, v.worst_rcd_rd, v.ccd_rd, v.ccd_wr,
               v.recovery_wr, v.sticky);
    }

    pim_free_ctx(c, xg); pim_free_ctx(c, yg); pim_tensor_free(c, &w);
    free(W); free(got);
    pim_exec_close(e); pim_close(c);
    printf("\n%s\n", fail ? "INCONCLUSIVE / FAILED" : "answered");
    return fail ? 1 : 0;
}
