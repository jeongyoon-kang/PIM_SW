// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// eager_test.c — the eager all-bank layout is the host's own array.
//
// THE CLAIM.  For k <= one row buffer, the device offset of W[j][i] is exactly its
// linear index in a [npad][1024] host array:
//
//     pim_gemv_offset(g, w, j, i)  ==  (j * 1024 + i) * 2
//
// If that holds for every (j, i) then pim_gemv_upload_eager()'s single pim_memcpy
// puts every element where a MAC will look for it, and the permutation
// pim_gemv_upload() performs is doing nothing that the addressing did not already
// do.  If it holds for MOST (j, i) the upload is worse than useless: it would place
// most of a matrix correctly and a few elements somewhere else, which is the shape
// of bug this hardware reports as a slightly wrong number.
//
// SO IT IS CHECKED EXHAUSTIVELY, not sampled — every element of every shape below.
// It costs milliseconds and it is the entire justification for the fast path.
//
// AND THE BOUNDARY IS CHECKED TOO.  k > 1024 must NOT be linear, because chunks of
// different outputs interleave; a test that only confirmed the good cases would not
// notice if pim_gemv_upload_eager stopped refusing them.
//
//     make run        # needs no board and no module
//////////////////////////////////////////////////////////////////////////////////
#include "fake_drv.h"

#include "pimrt/pim_gemv.h"

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

// Every element, not a sample.  Returns the first (j, i) that disagrees.
static int walk(const pim_geometry *g, const pim_gemv_w *w, uint32_t stride,
                uint32_t *bad_j, uint32_t *bad_i)
{
    for (uint32_t j = 0; j < w->npad; j++)
        for (uint32_t i = 0; i < w->k; i++) {
            uint64_t dev = pim_gemv_offset(g, w, j, i);
            uint64_t lin = ((uint64_t)j * stride + i) * 2u;
            if (dev != lin) { *bad_j = j; *bad_i = i; return 0; }
        }
    return 1;
}

static void eager_case(pim_ctx *c, uint32_t n, uint32_t k, const char *what)
{
    const pim_geometry *g = pim_geom_ctx(c);
    pim_gemv_w w;
    uint32_t   stride, bj = 0, bi = 0;

    pim_gemv_plan(g, n, k, &w);
    stride = pim_gemv_host_stride(&w);

    printf("\n  %-22s n=%-5u k=%-5u -> npad %-5u nchunks %u  OPSIZE %u\n",
           what, n, k, w.npad, w.nchunks, w.last_beats);

    CHECK(pim_gemv_is_eager(&w), "k=%u should be eager-able", k);
    if (!pim_gemv_is_eager(&w)) return;

    if (walk(g, &w, stride, &bj, &bi))
        printf("     every element: device offset == linear index   (%llu checked)\n",
               (unsigned long long)w.npad * w.k);
    else
        CHECK(0, "W[%u][%u]: device %llu, linear %llu", bj, bi,
              (unsigned long long)pim_gemv_offset(g, &w, bj, bi),
              (unsigned long long)((uint64_t)bj * stride + bi) * 2u);

    // The host array and the allocation must also be the same number of bytes, or
    // the single memcpy would either truncate or run past the allocation.
    CHECK((size_t)w.npad * stride * 2 == pim_gemv_bytes(g, &w),
          "host %zu B vs allocation %zu B",
          (size_t)w.npad * stride * 2, pim_gemv_bytes(g, &w));
    printf("     host array == allocation == %zu B\n", pim_gemv_bytes(g, &w));
}

int main(void)
{
    pim_ctx *c = mkctx(2);
    const pim_geometry *g = pim_geom_ctx(c);

    printf("eager all-bank: the host array is the device layout\n");
    printf("%u ch x %u bank, row buffer %u B = %u BF16\n",
           g->nch, g->nbank, g->row_bytes, g->row_bytes / 2);

    eager_case(c, 32,   128,  "attention head");
    eager_case(c, 2048, 1024, "a full row");
    eager_case(c, 64,   256,  "quarter row");
    eager_case(c, 20,   100,  "n and k both ragged");
    eager_case(c, 1,    16,   "one output, one beat");

    // ---- the boundary ------------------------------------------------------
    printf("\n  what the eager form must refuse\n");
    {
        pim_gemv_w w;
        void *p;
        const char *bad;
        uint16_t dummy = 0;

        pim_gemv_plan(g, 32, 2048, &w);          /* two chunks */
        CHECK(!pim_gemv_is_eager(&w), "k=2048 was reported eager-able");
        p = pim_alloc_ctx(c, pim_gemv_bytes(g, &w), PIM_MEM_DRAM);
        w.w = p;
        bad = pim_gemv_upload_eager(c, &w, &dummy);
        CHECK(bad != NULL, "k=2048 was accepted by the eager upload");
        if (bad) printf("     k=2048 (2 chunks) -> refused\n        \"%.64s...\"\n", bad);

        // and it really is non-linear there, which is WHY it is refused
        {
            uint32_t bj = 0, bi = 0;
            CHECK(!walk(g, &w, pim_gemv_host_stride(&w), &bj, &bi),
                  "k=2048 looked linear, so the refusal is over-strict");
            if (bj || bi)
                printf("        first divergence at W[%u][%u]: device %llu, linear %llu\n",
                       bj, bi, (unsigned long long)pim_gemv_offset(g, &w, bj, bi),
                       (unsigned long long)((uint64_t)bj * 1024 + bi) * 2u);
        }
        if (p) pim_free_ctx(c, p);
    }

    // ---- packing zeroes what must be zero ----------------------------------
    printf("\n  pim_gemv_pack_eager\n");
    {
        pim_gemv_w w;
        uint16_t   tight[20 * 100], *padded;
        uint32_t   stride, nz = 0;

        pim_gemv_plan(g, 20, 100, &w);
        stride = pim_gemv_host_stride(&w);
        for (uint32_t i = 0; i < 20 * 100; i++) tight[i] = (uint16_t)(i + 1);
        padded = malloc((size_t)w.npad * stride * 2);
        memset(padded, 0xAB, (size_t)w.npad * stride * 2);      /* poison first */
        pim_gemv_pack_eager(&w, tight, padded);

        for (uint32_t j = 0; j < w.n; j++)
            for (uint32_t i = 0; i < w.k; i++)
                if (padded[(size_t)j * stride + i] != tight[(size_t)j * w.k + i]) nz++;
        CHECK(nz == 0, "%u element(s) of the real matrix were misplaced", nz);

        // The tail of the FINAL BEAT is the part that must be zero: k=100 -> beat 6
        // covers 96..111, so 100..111 are read and must not be poison.
        nz = 0;
        for (uint32_t j = 0; j < w.npad; j++)
            for (uint32_t i = w.k; i < w.kpad; i++)
                if (padded[(size_t)j * stride + i]) nz++;
        CHECK(nz == 0, "%u element(s) of the final beat's tail are not zero", nz);
        printf("     k=%u kpad=%u: elements %u..%u zeroed in all %u rows\n",
               w.k, w.kpad, w.k, w.kpad - 1, w.npad);
        free(padded);
    }

    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail ? 1 : 0;
}
