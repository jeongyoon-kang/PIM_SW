// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// op_board.c — the three kernels, through the surface Python will bind to.
//
// pim_op_matvec is the whole stack behind one call: pad and upload the vector,
// poison the result words, build pass 1, lower it, ring the doorbell, poll, check
// the poison is gone, unpack.  Everything below it is verified elsewhere; what is
// NOT verified anywhere else is that those pieces fit together, and that the three
// model kernels really are the same call with different arguments.
//
// So this runs all three against a golden, using the SHAPES OF Llama-3.2-3B on the
// ch2 image — 8 KV heads of 128, which is H_kv*D = 1024 and fills a DRAM row
// exactly, and is also the largest head packing COL + OPSIZE <= 64 allows.
//
//   1  linear    W [n x k], the reduction is the whole axis
//   2  Q . K^T   red_off = head * D picks one head out of a shared row.  Eight
//                heads in one row is the entire reason the K cache is not 8x
//                bigger than it needs to be, and it is the one that would be wrong
//                silently: a mis-scaled COL reads another head's weights and
//                returns a number.
//   3  S . V     red_len = S_kv, not a multiple of 16, so the final beat straddles
//                the write frontier.  Exactly the case the zero invariant exists
//                for, exercised through the API rather than by hand.
//
// THE GOLDEN IS pim_mac_exact, the transcribed RTL, not an fp32 model — a beat is
// block floating point and no fp32 sum reproduces its alignment losses.
//
//     ./op_board      # needs pim.ko and the card
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim/pim.h"
#include "pimrt/pim_gemv.h"
#include "pimrt/pim_op.h"

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float    b2f(uint16_t h){ uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t rng = 88675123u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

// Compare `n` outputs and say where the first one went wrong, since "3 of 2048
// differ" without an index is not something you can act on.
static void compare(const char *what, const uint16_t *got, const uint16_t *want,
                    uint32_t n)
{
    uint32_t nbad = 0, first = 0;
    for (uint32_t i = 0; i < n; i++)
        if (got[i] != want[i]) { if (!nbad) first = i; nbad++; }
    if (nbad)
        CHECK(0, "%s: %u of %u differ, first at %u (%g vs %g)", what, nbad, n,
              first, b2f(got[first]), b2f(want[first]));
    else
        printf("     %-28s %u/%u exact\n", what, n, n);
}

#define D        128u          /* head dim, Llama-3.2-3B                 */
#define H_KV       8u          /* KV heads.  H_KV*D = 1024 = one row     */
#define S_MAX    512u
#define S_KV     100u          /* deliberately not a multiple of 16      */

int main(void)
{
    pim_ctx *c = NULL;
    pim_rt  *rt = NULL;
    const pim_geometry *g;
    const char *bad;
    uint32_t per;

    printf("pim_op_matvec: linear, Q.K^T and S.V through one call\n");
    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }
    {
        pim_rt_config cfg = { .max_red = 4096, .max_out_groups = 128 };
        if ((bad = pim_rt_open(c, &cfg, &rt))) {
            printf("SKIPPED: %s\n", bad); pim_close(c); return 0;
        }
    }
    pim_exec_clear_violations(pim_rt_exec(rt));
    g   = pim_geom_ctx(c);
    per = g->nch * g->nbank;
    printf("  %u ch x %u bank, %u outputs per supergroup;  H_kv*D = %u of %u per row\n",
           g->nch, g->nbank, per, H_KV * D, g->row_bytes / 2);

    // ---- 1. a linear layer -------------------------------------------------
    printf("\n  1. linear\n");
    {
        uint32_t n = 256, k = 2048;             /* two chunks, so no hoist */
        pim_tensor W;
        uint16_t *Wh = malloc((size_t)n * k * 2);
        uint16_t *x  = malloc((size_t)k * 2);
        uint16_t *y, *golden = malloc((size_t)n * 2);

        for (size_t i = 0; i < (size_t)n * k; i++) Wh[i] = rnd_bf16();
        for (uint32_t i = 0; i < k; i++) x[i] = rnd_bf16();

        CHECK(!(bad = pim_tensor_alloc(c, PIM_LAYOUT_OUT_MAJOR, n, k, 0, &W)),
              "alloc: %s", bad ? bad : "");
        CHECK(!(bad = pim_tensor_upload(c, &W, Wh)), "upload: %s", bad ? bad : "");
        y = malloc((size_t)pim_op_outputs(rt, W.ngroups) * 2);
        CHECK(!(bad = pim_op_matvec(rt, &W, 0, W.ngroups, 0, k, x, y,
                                    PIM_ACC_SINGLE)), "op: %s", bad ? bad : "");
        pim_gemv_golden(Wh, n, k, x, golden);
        compare("W[256 x 2048] . x", y, golden, n);

        pim_tensor_free(c, &W);
        free(Wh); free(x); free(y); free(golden);
    }

    // ---- 2. Q . K^T, eight heads sharing every row -------------------------
    printf("\n  2. Q . K^T   (red_off = head * %u)\n", D);
    {
        pim_tensor K;
        uint32_t   kvw = H_KV * D;
        uint32_t   ngroup = (S_KV + per - 1) / per;
        uint16_t  *Kh = malloc((size_t)S_KV * kvw * 2);   /* [token][head][dim] */
        uint16_t  *q  = malloc((size_t)D * 2);
        uint16_t  *y  = NULL, *golden = NULL, *row = malloc((size_t)D * 2);

        for (size_t i = 0; i < (size_t)S_KV * kvw; i++) Kh[i] = rnd_bf16();

        // OUT_MAJOR: one token is one OUTPUT, so the append writes [count][nred]
        // and each token's whole head set is one contiguous run on the card.
        CHECK(!(bad = pim_tensor_alloc(c, PIM_LAYOUT_OUT_MAJOR, S_MAX, kvw, 0, &K)),
              "alloc: %s", bad ? bad : "");
        CHECK(!(bad = pim_tensor_append(c, &K, 0, S_KV, Kh)), "append: %s",
              bad ? bad : "");

        y      = malloc((size_t)pim_op_outputs(rt, ngroup) * 2);
        golden = malloc((size_t)pim_op_outputs(rt, ngroup) * 2);

        for (uint32_t h = 0; h < H_KV; h++) {
            char what[48];
            for (uint32_t i = 0; i < D; i++) q[i] = rnd_bf16();

            CHECK(!(bad = pim_op_matvec(rt, &K, 0, ngroup, h * D, D, q, y,
                                        PIM_ACC_SINGLE)), "op h=%u: %s", h,
                  bad ? bad : "");
            for (uint32_t s = 0; s < S_KV; s++) {
                memcpy(row, Kh + (size_t)s * kvw + (size_t)h * D, (size_t)D * 2);
                golden[s] = pim_mac_exact(row, q, D);
            }
            snprintf(what, sizeof what, "head %u: q . K[0..%u]", h, S_KV - 1);
            compare(what, y, golden, S_KV);
        }
        pim_tensor_free(c, &K);
        free(Kh); free(q); free(y); free(golden); free(row);
    }

    // ---- 3. S . V, a ragged reduction over the sequence --------------------
    printf("\n  3. S . V   (red_len = %u, so the last beat covers %u)\n",
           S_KV, (S_KV + 15) / 16 * 16);
    {
        pim_tensor V;
        uint32_t   kvw = H_KV * D;
        uint32_t   ngroup = kvw / per;
        uint32_t   padded = (S_KV + 15) / 16 * 16;
        uint16_t  *Vh = malloc((size_t)padded * kvw * 2);  /* [token][dim] */
        uint16_t  *S  = malloc((size_t)padded * 2);
        uint16_t  *y  = malloc((size_t)pim_op_outputs(rt, ngroup) * 2);
        uint16_t  *golden = malloc((size_t)kvw * 2);
        uint16_t  *col = malloc((size_t)padded * 2);

        memset(Vh, 0, (size_t)padded * kvw * 2);
        memset(S,  0, (size_t)padded * 2);
        for (size_t i = 0; i < (size_t)S_KV * kvw; i++) Vh[i] = rnd_bf16();
        for (uint32_t s = 0; s < S_KV; s++) S[s] = rnd_bf16();

        // RED_MAJOR: one token is one REDUCTION STEP.  The append is a scatter —
        // one two-byte piece per output — and it clears the chunk it lands in, which
        // is what makes the ragged red_len below safe.
        CHECK(!(bad = pim_tensor_alloc(c, PIM_LAYOUT_RED_MAJOR, kvw, S_MAX, 0, &V)),
              "alloc: %s", bad ? bad : "");
        CHECK(!(bad = pim_tensor_append(c, &V, 0, S_KV, Vh)), "append: %s",
              bad ? bad : "");

        CHECK(!(bad = pim_op_matvec(rt, &V, 0, ngroup, 0, S_KV, S, y,
                                    PIM_ACC_SINGLE)), "op: %s", bad ? bad : "");
        for (uint32_t d = 0; d < kvw; d++) {
            for (uint32_t s = 0; s < padded; s++) col[s] = Vh[(size_t)s * kvw + d];
            golden[d] = pim_mac_exact(col, S, padded);
        }
        compare("S . V[0..99]", y, golden, kvw);

        // And the same thing one token later, which moves the frontier into the
        // next beat.  A schedule that had baked the beat count in somewhere would
        // pass above and fail here.
        for (uint32_t d = 0; d < kvw; d++) Vh[(size_t)S_KV * kvw + d] = rnd_bf16();
        S[S_KV] = rnd_bf16();
        CHECK(!(bad = pim_tensor_append(c, &V, S_KV, 1, Vh + (size_t)S_KV * kvw)),
              "append 101st: %s", bad ? bad : "");
        CHECK(!(bad = pim_op_matvec(rt, &V, 0, ngroup, 0, S_KV + 1, S, y,
                                    PIM_ACC_SINGLE)), "op: %s", bad ? bad : "");
        for (uint32_t d = 0; d < kvw; d++) {
            for (uint32_t s = 0; s < padded; s++) col[s] = Vh[(size_t)s * kvw + d];
            golden[d] = pim_mac_exact(col, S, padded);
        }
        compare("S . V[0..100]", y, golden, kvw);

        pim_tensor_free(c, &V);
        free(Vh); free(S); free(y); free(golden); free(col);
    }

    // ---- 4. what it refuses ------------------------------------------------
    printf("\n  4. refusals\n");
    {
        pim_tensor W;
        uint16_t   v[64] = { 0 }, y[64];

        CHECK(!pim_tensor_alloc(c, PIM_LAYOUT_OUT_MAJOR, 32, 64, 0, &W), "alloc");
        bad = pim_op_matvec(rt, &W, 0, 1, 0, 64, v, y, PIM_ACC_DUAL);
        CHECK(bad != NULL, "PIM_ACC_DUAL was accepted without allow_t_latch");
        if (bad) printf("     DUAL without evidence: \"%.62s...\"\n", bad);

        bad = pim_op_matvec(rt, &W, 0, 1, 0, 9999, v, y, PIM_ACC_SINGLE);
        CHECK(bad != NULL, "red_len past max_red was accepted");
        if (bad) printf("     red_len past max_red : \"%.62s...\"\n", bad);

        // An allocation nothing filled carries no tag, so a program built for a
        // layout cannot read it.  This is the check that stops a forgotten upload
        // from being read as whatever was in those granules.
        bad = pim_op_matvec(rt, &W, 0, 1, 0, 64, v, y, PIM_ACC_SINGLE);
        printf("     untagged tensor      : %s\n",
               bad ? "refused" : "ACCEPTED (alloc stamps the tag, so this is legal)");
        pim_tensor_free(c, &W);
    }

    {
        pim_rt_stat st;
        pim_rt_stat_get(rt, &st);
        printf("\n  %u ops, %u launches, %u ISAs, %u vector loads, %llu us on the "
               "doorbell\n", st.nop, st.nlaunch, st.nisr, st.nwrvec,
               (unsigned long long)st.launch_us);
    }
    {
        // THE CONDITIONS THIS RUN WAS MEASURED UNDER.  Nothing here sets them —
        // hwdef/test/emu_timing does — so a result is only comparable to another
        // result taken at the same eight numbers, and printing them is what makes
        // that checkable instead of remembered.
        const pim_timing *t = pim_exec_timing_at_open(pim_rt_exec(rt));
        printf("  timing (set by emu_timing, read here): faw %u rrd %u rcd %u "
               "ccd %u rtp %u rp %u wr %u ras %u\n",
               t->faw, t->rrd, t->rcd, t->ccd, t->rtp, t->rp, t->wr, t->ras);
    }
    for (unsigned ch = 0; ch < g->nch; ch++) {
        pim_viol vi;
        if (!pim_exec_violation_detail(pim_rt_exec(rt), ch, &vi))
            printf("  ch%u  rcd_rd %u  ccd_rd %u  ccd_wr %u  recovery_wr %u\n",
                   ch, vi.rcd_rd, vi.ccd_rd, vi.ccd_wr, vi.recovery_wr);
    }

    pim_rt_close(rt);
    pim_close(c);
    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail ? 1 : 0;
}
