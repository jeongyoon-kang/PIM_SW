// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// eager_board.c — the eager upload puts the same bytes on the card as the
// permuting one, and a GEMV built on it is bit-exact.
//
// lib/test/eager_test proves the ARITHMETIC: for k <= one row buffer, the device
// offset of every element equals its linear index in a [noutpad][1024] host array.
// That is a statement about pim_tensor_offset(), and pim_tensor_offset() is also what
// the permuting upload uses — so a bug shared by both would pass it.
//
// This asks the card instead, and in two independent ways:
//
//   1. UPLOAD BOTH WAYS, READ BOTH BACK, COMPARE.  Two allocations, the same
//      matrix, one sent through the host permutation and one sent as a single
//      memcpy.  If a single byte differs the fast path is wrong.  This does not
//      involve the compute at all, so it isolates placement from everything else.
//
//   2. RUN A GEMV ON THE EAGER ONE.  Byte-identical weights could still be wrong
//      weights if the layout itself were mistaken; only the arithmetic says
//      otherwise.  Compared against pim_gemv_golden(), which is the transcribed
//      RTL rather than an fp32 model.
//
//     ./eager_board [--n N] [--k K]        # needs pim.ko and the card
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pim/pim.h"
#include "pim/pim_addr.h"
#include "pimrt/pim_exec.h"
#include "pimrt/pim_gemv.h"

static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float    b2f(uint16_t h){ uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t rng = 987654321u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

static int fail;

static uint64_t now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000ull + (uint64_t)t.tv_nsec / 1000ull;
}

int main(int argc, char **argv)
{
    pim_ctx  *c = NULL;
    pim_exec *e = NULL;
    const pim_geometry *g;
    pim_tensor  wa, wb, wc;
    const char *bad;
    uint32_t    n = 2048, k = 1024;
    uint16_t   *W, *padded, *rowbuf, *x, *y, *golden, *ba, *bb, *bc;
    uint64_t    t0, t_perm, t_eager, t_rows;
    void       *xg, *yg;
    size_t      xb, yb, nbytes;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n") && i + 1 < argc) n = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i + 1 < argc) k = (uint32_t)atoi(argv[++i]);
    }

    printf("eager all-bank upload, on the board\n");
    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }
    {
        pim_exec_config ec = { 0 };
        if ((bad = pim_exec_open(c, &ec, &e))) {
            printf("SKIPPED: %s\n", bad); pim_close(c); return 0;
        }
    }
    pim_exec_clear_violations(e);
    g = pim_geom_ctx(c);

    pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, n, k, &wa);
    if (!pim_gemv_is_eager(&wa)) {
        printf("  k=%u needs %u chunks; the eager form wants k <= %u\n",
               k, wa.nchunks, pim_gemv_host_stride(&wa));
        return 1;
    }
    wb = wa;  wc = wa;
    nbytes = pim_tensor_bytes(g, &wa);
    printf("  %u ch x %u bank   n=%u k=%u -> noutpad %u, OPSIZE %u, %zu KiB each\n",
           g->nch, g->nbank, n, k, wa.noutpad, wa.last_beats, nbytes >> 10);

    // Scrub before anything: the latch outlives processes.
    {
        void *sg = pim_alloc_ctx(c, 4096, PIM_MEM_GPR);
        uint32_t word, nw;
        if (sg && !pim_addr_gpr_words_ctx(c, sg, g->nch * 32, &word, &nw))
            pim_exec_scrub(e, word);
        if (sg) pim_free_ctx(c, sg);
    }

    W      = malloc((size_t)n * k * 2);
    padded = malloc(nbytes);
    for (size_t i = 0; i < (size_t)n * k; i++) W[i] = rnd_bf16();
    pim_gemv_pack_eager(&wa, W, padded);

    // beat-padded but NOT row-padded: what the row form wants
    rowbuf = calloc((size_t)wa.nout * wa.nredpad, 2);
    for (uint32_t j = 0; j < wa.nout; j++)
        memcpy(rowbuf + (size_t)j * wa.nredpad, W + (size_t)j * k, (size_t)k * 2);

    wa.base = pim_alloc_ctx(c, nbytes, PIM_MEM_DRAM);
    wb.base = pim_alloc_ctx(c, nbytes, PIM_MEM_DRAM);
    wc.base = pim_alloc_ctx(c, nbytes, PIM_MEM_DRAM);
    if (!wa.base || !wb.base || !wc.base) { printf("  alloc: %s\n", pim_last_error_ctx(c)); return 1; }

    // The row form does not write rows n..noutpad-1, so poison the whole allocation
    // first: if a MAC ever read one of them the answer would change, and this is
    // what makes "never read" a claim the test can back rather than assume.
    memset(padded, 0xA5, nbytes);
    if ((bad = pim_memcpy_ctx(c, wc.base, padded, nbytes, PIM_TO_DEV, 0))) {
        printf("  poison: %s\n", bad); return 1;
    }
    pim_gemv_pack_eager(&wa, W, padded);

    // ---- 1. the three uploads, timed -----------------------------------------
    t0 = now_us();
    if ((bad = pim_gemv_upload(c, &wa, W)))            { printf("  permuting: %s\n", bad); return 1; }
    t_perm = now_us() - t0;
    t0 = now_us();
    if ((bad = pim_gemv_upload_eager(c, &wb, padded))) { printf("  eager: %s\n", bad); return 1; }
    t_eager = now_us() - t0;
    t0 = now_us();
    if ((bad = pim_gemv_upload_rows(c, &wc, rowbuf)))  { printf("  rows: %s\n", bad); return 1; }
    t_rows = now_us() - t0;

    printf("\n  upload cost\n");
    printf("     permuting   host permute + 1 pwrite of %8zu B   %6llu us\n",
           nbytes, (unsigned long long)t_perm);
    printf("     one DMA     1 pwrite of %8zu B                  %6llu us\n",
           nbytes, (unsigned long long)t_eager);
    printf("     per row     %u pwrites of %6zu B                %6llu us\n",
           wa.nout, (size_t)wa.nredpad * 2, (unsigned long long)t_rows);

    ba = malloc(nbytes);  bb = malloc(nbytes);  bc = malloc(nbytes);
    if ((bad = pim_memcpy_ctx(c, ba, wa.base, nbytes, PIM_FROM_DEV, 0)) ||
        (bad = pim_memcpy_ctx(c, bb, wb.base, nbytes, PIM_FROM_DEV, 0)) ||
        (bad = pim_memcpy_ctx(c, bc, wc.base, nbytes, PIM_FROM_DEV, 0))) {
        printf("  readback: %s\n", bad); return 1;
    }
    printf("\n  1. what actually landed, %zu B read back from each\n", nbytes);
    {
        size_t nbad = 0, first = 0;
        for (size_t i = 0; i < nbytes / 2; i++)
            if (ba[i] != bb[i]) { if (!nbad) first = i; nbad++; }
        if (nbad) { printf("     FAIL permuting vs one-DMA: %zu differ, first at %zu "
                           "(%04x vs %04x)\n", nbad, first, ba[first], bb[first]); fail++; }
        else       printf("     permuting == one DMA   : all %zu elements\n", nbytes / 2);
    }
    {   // The row form is compared only where it WROTE: rows 0..n-1, elements
        // 0..nredpad-1.  Everywhere else it deliberately left the poison, and saying
        // so is the point rather than a caveat.
        size_t nbad = 0, npois = 0;
        uint32_t stride_el = g->row_bytes / 2;
        for (uint32_t j = 0; j < wa.nout; j++)
            for (uint32_t i = 0; i < wa.nredpad; i++)
                if (bc[(size_t)j * stride_el + i] != ba[(size_t)j * stride_el + i]) nbad++;
        for (uint32_t j = 0; j < wa.nout; j++)
            for (uint32_t i = wa.nredpad; i < stride_el; i++)
                if (bc[(size_t)j * stride_el + i] == 0xA5A5) npois++;
        if (nbad) { printf("     FAIL per-row: %zu of the written elements differ\n", nbad); fail++; }
        else       printf("     per row   == permuting : all %zu written elements\n",
                          (size_t)wa.nout * wa.nredpad);
        printf("     and %zu untouched element(s) still hold the poison, as intended\n",
               npois);
    }

    // ---- 2. and a GEMV on the eager one is bit-exact ------------------------
    x  = malloc((size_t)wa.nredpad * 2);
    y  = malloc((size_t)n * 2);
    golden = malloc((size_t)n * 2);
    memset(x, 0, (size_t)wa.nredpad * 2);
    for (uint32_t i = 0; i < k; i++) x[i] = rnd_bf16();

    pim_gemv_gpr_bytes(&wb, PIM_ACC_SINGLE, pim_exec_max_isrs(e), &xb, &yb);
    xg = pim_alloc_ctx(c, xb, PIM_MEM_GPR);
    yg = pim_alloc_ctx(c, yb, PIM_MEM_GPR);
    if (!xg || !yg) { printf("  GPR: %s\n", pim_last_error_ctx(c)); return 1; }

    pim_gemv_golden(W, n, k, x, golden);
    printf("\n  2. a GEMV on each\n");
    {
        pim_tensor *which[2] = { &wb, &wc };
        const char *name[2]  = { "one DMA", "per row " };
        for (int t = 0; t < 2; t++) {
            uint32_t nbad = 0, first = 0;
            if ((bad = pim_gemv_ex(c, e, which[t], x, xg, yg, y, PIM_ACC_SINGLE, NULL))) {
                printf("     %s: %s\n", name[t], bad); fail++; continue;
            }
            for (uint32_t j = 0; j < n; j++)
                if (y[j] != golden[j]) { if (!nbad) first = j; nbad++; }
            if (nbad) {
                printf("     %s FAIL: %u of %u differ, first at %u (%g vs %g)\n",
                       name[t], nbad, n, first, b2f(y[first]), b2f(golden[first]));
                fail++;
            } else {
                printf("     %s  %u/%u exact against pim_gemv_golden()\n", name[t], n, n);
            }
        }
    }

    for (unsigned ch = 0; ch < g->nch; ch++) {
        pim_viol v;
        if (!pim_exec_violation_detail(e, ch, &v))
            printf("  ch%u  rcd_rd %u  ccd_rd %u  ccd_wr %u  recovery_wr %u\n",
                   ch, v.rcd_rd, v.ccd_rd, v.ccd_wr, v.recovery_wr);
    }

    pim_free_ctx(c, xg); pim_free_ctx(c, yg);
    pim_tensor_free(c, &wa); pim_tensor_free(c, &wb); pim_tensor_free(c, &wc);
    free(W); free(padded); free(rowbuf); free(ba); free(bb); free(bc); free(x); free(y); free(golden);
    pim_exec_close(e); pim_close(c);

    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail ? 1 : 0;
}
