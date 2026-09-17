// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// tensor_test.c — a pim_tensor puts every element where it says it does, and a
// RED_MAJOR one keeps its zero invariant across append and truncate.
//
// WHY THIS IS NOT CHECKED AGAINST pim_tensor_offset ALONE.  A test that computed an
// expected address with the same function the code uses would agree with it by
// construction, including when both are wrong.  So everything below READS THE CARD
// BACK through fake_drv.h's backing store: the bytes were placed by
// pim_tensor_append walking its own arithmetic, and are found again by an
// independent index into a flat snapshot.
//
// AND THE CARD IS POISONED, NOT ZEROED.  That is what lets "this element is zero"
// mean "something cleared it" rather than "nothing has been here".  Every check
// below distinguishes three states — the value written, zero, and 0xA5A5 — and it
// is the third that makes the other two worth anything.
//
// THE SHAPES ARE THE REAL ONES.  Llama-3.2-3B on the ch2 image: 8 KV heads of 128,
// so H_kv*D = 1024, which fills a DRAM row exactly.  And 1B's 8 x 64 = 512, which
// fills half of one — the case where the padding is real and has to stay harmless.
//
//     make run        # needs no board and no module
//////////////////////////////////////////////////////////////////////////////////
#include "fake_drv.h"

#include "pimrt/pim_tensor.h"

#define ELEMS_PER_ROW  PIM_TENSOR_ELEMS_PER_ROW

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

#define POISON16  ((uint16_t)((FAKE_POISON << 8) | FAKE_POISON))

// The whole allocation, flat, as the card holds it right now.
static uint16_t *snapshot(pim_ctx *c, const pim_tensor *t)
{
    size_t    n = pim_tensor_bytes(pim_geom_ctx(c), t);
    uint16_t *b = malloc(n);
    const char *bad;

    if (!b) { printf("  FAIL: out of host memory for a %zu B snapshot\n", n); fail++; return NULL; }
    bad = pim_memcpy_ctx(c, b, t->base, n, PIM_FROM_DEV, 0);
    if (bad) { printf("  FAIL: snapshot: %s\n", bad); fail++; free(b); return NULL; }
    return b;
}

static uint16_t at(pim_ctx *c, const uint16_t *snap, const pim_tensor *t,
                   uint32_t out, uint32_t red)
{ return snap[pim_tensor_offset(pim_geom_ctx(c), t, out, red) / 2]; }

// A value that names its own coordinates, so a misplaced element says where it came
// from instead of merely differing.  Never 0 and never the poison.
static uint16_t mark(uint32_t out, uint32_t red)
{ uint16_t v = (uint16_t)(((out & 0xFFu) << 8) | (red & 0xFFu)); return v ? v : 0x0101u; }

//////////////////////////////////////////////////////////////////////////////////
// 1.  The addressing matches an independent reading of the rule.
//
// Until the port, this compared pim_tensor_offset against pim_gemv_offset, which
// was the point: the two had to agree before one could replace the other.  They do,
// and now only one of them exists — so comparing them would be comparing a function
// with itself, and a check that cannot fail is worse than no check because it reads
// like one that can.
//
// What replaces it is a SECOND DERIVATION of the rule pim_tensor.h states, written
// from the prose rather than from the code.  It differs in two places where the
// implementation could be wrong and still be self-consistent:
//
//   ch      (out - group*per)/nbank   rather than   (out/nbank) % nch
//   lane    red_in_row * 2            rather than   (in/16)*32 + (in%16)*2
//
// The second is the one worth having.  It is equal only because a beat is exactly
// 16 lanes of two bytes with nothing between them — so it pins "beats are densely
// packed", which the implementation's form would silently survive losing.
//////////////////////////////////////////////////////////////////////////////////
static uint64_t expected_offset(const pim_geometry *g, const pim_tensor *t,
                                uint32_t out, uint32_t red)
{
    uint32_t per   = g->nbank * g->nch;       /* outputs one all-bank MAC covers  */
    uint32_t group = out / per;
    uint32_t ch    = (out - group * per) / g->nbank;
    uint32_t bank  = out % g->nbank;
    uint32_t chunk = red / ELEMS_PER_ROW;
    uint32_t in    = red - chunk * ELEMS_PER_ROW;

    return ((uint64_t)group * t->nchunks + chunk) * g->unit_bytes
         + (uint64_t)ch * g->nbank * g->row_bytes
         + (uint64_t)bank * g->row_bytes
         + (uint64_t)in * 2u;
}

static void addressing_case(pim_ctx *c, uint32_t nout, uint32_t nred, const char *what)
{
    const pim_geometry *g = pim_geom_ctx(c);
    pim_tensor  t;
    uint32_t    bad_o = 0, bad_r = 0;
    int         ok = 1;

    pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, nout, nred, &t);

    // The last element of the last group's last chunk must be the last byte of the
    // allocation — one element further and every shape here would be over-sized
    // without any single offset being wrong.
    CHECK(pim_tensor_offset(g, &t, t.noutpad - 1, t.nredpad - 1) + 2
              <= pim_tensor_bytes(g, &t),
          "%s: the last element runs past the %zu byte allocation", what,
          pim_tensor_bytes(g, &t));

    for (uint32_t o = 0; o < t.noutpad && ok; o++)
        for (uint32_t r = 0; r < t.nredpad && ok; r++)
            if (pim_tensor_offset(g, &t, o, r) != expected_offset(g, &t, o, r)) {
                bad_o = o; bad_r = r; ok = 0;
            }
    CHECK(ok, "%s: [%u][%u] -> %llu, the rule says %llu", what, bad_o, bad_r,
          (unsigned long long)pim_tensor_offset(g, &t, bad_o, bad_r),
          (unsigned long long)expected_offset(g, &t, bad_o, bad_r));
    if (ok)
        printf("     %-22s nout=%-5u nred=%-5u  %llu offsets agree\n", what, nout, nred,
               (unsigned long long)t.noutpad * t.nredpad);
}

//////////////////////////////////////////////////////////////////////////////////
// 2.  upload places the real rectangle and zeroes the padding.
//////////////////////////////////////////////////////////////////////////////////
static void upload_case(pim_ctx *c, pim_layout layout, uint32_t nout, uint32_t nred,
                        const char *what)
{
    pim_tensor  t;
    uint16_t   *src, *snap;
    const char *bad;
    uint32_t    misplaced = 0, dirty_pad = 0;

    if ((bad = pim_tensor_alloc(c, layout, nout, nred, 0, &t))) {
        CHECK(0, "%s: alloc: %s", what, bad); return;
    }
    src = malloc((size_t)nout * nred * 2);
    for (uint32_t o = 0; o < nout; o++)
        for (uint32_t r = 0; r < nred; r++)
            src[layout == PIM_LAYOUT_OUT_MAJOR ? (size_t)o * nred + r
                                               : (size_t)r * nout + o] = mark(o, r);

    if ((bad = pim_tensor_upload(c, &t, src))) { CHECK(0, "%s: upload: %s", what, bad); goto out; }
    if (!(snap = snapshot(c, &t))) goto out;

    for (uint32_t o = 0; o < nout; o++)
        for (uint32_t r = 0; r < nred; r++)
            if (at(c, snap, &t, o, r) != mark(o, r)) misplaced++;
    // Padding is what the hardware reads and the model does not have.  It must be
    // zero, not merely "not the data" — see the block-float note in pim_tensor.h.
    for (uint32_t o = 0; o < t.noutpad; o++)
        for (uint32_t r = 0; r < t.nredpad; r++)
            if ((o >= nout || r >= nred) && at(c, snap, &t, o, r) != 0) dirty_pad++;

    CHECK(misplaced == 0, "%s: %u of %llu elements landed elsewhere", what, misplaced,
          (unsigned long long)nout * nred);
    CHECK(dirty_pad == 0, "%s: %u padding element(s) are not zero", what, dirty_pad);
    printf("     %-18s [%u x %u] -> padded [%u x %u], %llu placed, %llu padding zeroed\n",
           what, nout, nred, t.noutpad, t.nredpad, (unsigned long long)nout * nred,
           (unsigned long long)t.noutpad * t.nredpad - (unsigned long long)nout * nred);
    free(snap);
out:
    free(src);
    pim_tensor_free(c, &t);
}

//////////////////////////////////////////////////////////////////////////////////
// 3.  A KV cache, appended one token at a time, keeps its invariant.
//
// Both layouts run the same script so the asymmetry is visible rather than argued:
// K is OUT_MAJOR with the token on the output axis, V is RED_MAJOR with it on the
// reduction axis, and only V has a tail to keep clear.
//////////////////////////////////////////////////////////////////////////////////
static void kv_case(pim_ctx *c, pim_layout layout, uint32_t hkvd, uint32_t smax,
                    uint32_t ntok, const char *what)
{
    uint32_t    nout = (layout == PIM_LAYOUT_OUT_MAJOR) ? smax : hkvd;
    uint32_t    nred = (layout == PIM_LAYOUT_OUT_MAJOR) ? hkvd : smax;
    pim_tensor  t;
    uint16_t   *tok, *snap;
    const char *bad;
    uint32_t    misplaced = 0, tail_bad = 0, beyond_poison = 0, beyond_seen = 0;
    uint32_t    beat = PIM_TENSOR_ELEMS_PER_BEAT;

    if ((bad = pim_tensor_alloc(c, layout, nout, nred, 0, &t))) {
        CHECK(0, "%s: alloc: %s", what, bad); return;
    }
    // POISON THE WHOLE ALLOCATION FIRST, rather than trusting what the fake card
    // already holds.  The pool recycles granules between cases here, so a region
    // left zero by an earlier test would look exactly like a region this one
    // cleared — and the check below is precisely the one that would be fooled.
    {
        size_t n = pim_tensor_bytes(pim_geom_ctx(c), &t);
        unsigned char *p = malloc(n);
        memset(p, FAKE_POISON, n);
        CHECK(!pim_memcpy_ctx(c, t.base, p, n, PIM_TO_DEV, 0), "%s: poison", what);
        free(p);
    }
    tok = malloc((size_t)hkvd * 2);

    for (uint32_t s = 0; s < ntok; s++) {
        for (uint32_t d = 0; d < hkvd; d++) tok[d] = mark(s, d);
        if ((bad = pim_tensor_append(c, &t, s, 1, tok))) {
            CHECK(0, "%s: append %u: %s", what, s, bad); goto out;
        }
    }
    CHECK(t.frontier == ntok, "%s: frontier is %u after %u appends", what, t.frontier, ntok);
    if (!(snap = snapshot(c, &t))) goto out;

    // Every element of every token, found by its own coordinates.
    for (uint32_t s = 0; s < ntok; s++)
        for (uint32_t d = 0; d < hkvd; d++) {
            uint16_t got = (layout == PIM_LAYOUT_OUT_MAJOR) ? at(c, snap, &t, s, d)
                                                            : at(c, snap, &t, d, s);
            if (got != mark(s, d)) misplaced++;
        }

    // THE INVARIANT, which is what a launch at red_len = frontier actually reads
    // past: [frontier, roundup(frontier, 16)).  The implementation satisfies it by
    // clearing a whole reduction CHUNK on first entry rather than a beat, because
    // 32 transfers of 64 KiB once per 1024 tokens beats 1024 transfers of 30 bytes
    // once per token — so everything up to the cleared chunk boundary is zero, and
    // that is checked rather than just the beat.
    { uint32_t cleared = (ntok + ELEMS_PER_ROW - 1) / ELEMS_PER_ROW * ELEMS_PER_ROW;
      if (cleared > nred) cleared = nred;
      CHECK(cleared >= (ntok + beat - 1) / beat * beat || ntok == nred,
            "%s: the cleared region stops short of the frontier's own beat", what);
      for (uint32_t r = ntok; r < cleared; r++)
          for (uint32_t o = 0; o < t.noutpad; o++)
              if (at(c, snap, &t, o, r) != 0) tail_bad++;

      // And BEYOND the cleared chunks nothing is promised.  Checking that the
      // poison survived there is what proves the clear stayed lazy instead of being
      // quietly widened to the whole tensor — which would pass every other check
      // here and cost 630 ms of DMA at the start of every generation.
      for (uint32_t r = cleared; r < nred; r += 97)
          for (uint32_t o = 0; o < t.noutpad; o += 7) {
              beyond_seen++;
              if (at(c, snap, &t, o, r) == POISON16) beyond_poison++;
          }
    }

    CHECK(misplaced == 0, "%s: %u of %llu appended elements landed elsewhere", what,
          misplaced, (unsigned long long)ntok * hkvd);
    if (layout == PIM_LAYOUT_RED_MAJOR) {
        CHECK(tail_bad == 0, "%s: %u element(s) between the frontier and its beat "
              "boundary are not zero", what, tail_bad);
        CHECK(beyond_poison == beyond_seen,
              "%s: %u of %u sampled elements past the frontier's beat were cleared; "
              "the weak invariant should have left them alone",
              what, beyond_seen - beyond_poison, beyond_seen);
    }
    printf("     %-14s [%u x %u] %u token(s): %llu placed, cleared tail %s, "
           "%u/%u sampled beyond still poison\n", what, nout, nred, ntok,
           (unsigned long long)ntok * hkvd,
           layout == PIM_LAYOUT_RED_MAJOR ? (tail_bad ? "DIRTY" : "zero") : "n/a",
           beyond_poison, beyond_seen);
    free(snap);
out:
    free(tok);
    pim_tensor_free(c, &t);
}

//////////////////////////////////////////////////////////////////////////////////
int main(void)
{
    pim_ctx *c = mkctx(2);
    const pim_geometry *g = pim_geom_ctx(c);

    printf("pim_tensor: placement, and the zero invariant\n");
    printf("%u ch x %u bank, row buffer %u B = %u BF16, unit %llu KiB\n",
           g->nch, g->nbank, g->row_bytes, g->row_bytes / 2,
           (unsigned long long)(g->unit_bytes >> 10));

    printf("\n  1. addressing, against an independent reading of the rule\n");
    addressing_case(c, 2048, 1024, "a full row");
    addressing_case(c, 32,   128,  "attention head");
    addressing_case(c, 2048, 2048, "two chunks");
    addressing_case(c, 20,   100,  "both ragged");
    addressing_case(c, 1,    16,   "one output, one beat");
    addressing_case(c, 3072, 8192, "3B down_proj");
    addressing_case(c, 1024, 8192, "3B V cache at S_max 8K");

    printf("\n  2. upload places the rectangle and zeroes the rest\n");
    upload_case(c, PIM_LAYOUT_OUT_MAJOR, 64, 100, "OUT_MAJOR ragged");
    upload_case(c, PIM_LAYOUT_RED_MAJOR, 64, 100, "RED_MAJOR ragged");
    upload_case(c, PIM_LAYOUT_OUT_MAJOR, 96, 1024, "OUT_MAJOR full row");
    upload_case(c, PIM_LAYOUT_RED_MAJOR, 32, 2048, "RED_MAJOR 2 chunks");

    printf("\n  3. a KV cache appended token by token\n");
    kv_case(c, PIM_LAYOUT_OUT_MAJOR, 1024, 512, 100, "K 3B");
    kv_case(c, PIM_LAYOUT_RED_MAJOR, 1024, 2048, 100, "V 3B");
    kv_case(c, PIM_LAYOUT_OUT_MAJOR,  512, 512, 100, "K 1B");
    kv_case(c, PIM_LAYOUT_RED_MAJOR,  512, 2048, 100, "V 1B");
    kv_case(c, PIM_LAYOUT_RED_MAJOR,  512, 2048, 1040, "V past a chunk");

    // ---- 4. truncate restores what append cannot ---------------------------
    printf("\n  4. truncate\n");
    {
        pim_tensor t;
        uint16_t  *tok = malloc(512 * 2), *snap;
        const char *bad;
        uint32_t   stale = 0, cleared = 0;

        CHECK(!(bad = pim_tensor_alloc(c, PIM_LAYOUT_RED_MAJOR, 512, 2048, 0, &t)),
              "alloc: %s", bad ? bad : "");
        for (uint32_t s = 0; s < 100; s++) {
            for (uint32_t d = 0; d < 512; d++) tok[d] = mark(s, d);
            CHECK(!pim_tensor_append(c, &t, s, 1, tok), "append %u", s);
        }
        CHECK(!(bad = pim_tensor_truncate(c, &t, 50)), "truncate: %s", bad ? bad : "");
        CHECK(t.frontier == 50, "frontier is %u after truncate to 50", t.frontier);

        if ((snap = snapshot(c, &t))) {
            // EVERYTHING from 50 to the old frontier, not just to the next beat.
            // Clearing only [50, 64) would satisfy the invariant now and break it on
            // the very next append, which pushes the frontier into [64, 100).
            for (uint32_t r = 50; r < 100; r++)
                for (uint32_t o = 0; o < t.noutpad; o++)
                    if (at(c, snap, &t, o, r) != 0) stale++;
            for (uint32_t r = 0; r < 50; r++)
                for (uint32_t o = 0; o < 512; o++)
                    if (at(c, snap, &t, o, r) == mark(r, o)) cleared++;
            CHECK(stale == 0, "%u element(s) above the new frontier still hold the "
                  "previous generation's values", stale);
            CHECK(cleared == 50u * 512u, "truncate destroyed %u element(s) BELOW the "
                  "new frontier", 50u * 512u - cleared);
            printf("     to 50 of 100: [50,100) cleared across all %u outputs, "
                   "[0,50) intact\n", t.noutpad);
            free(snap);
        }

        // Appending again must not find the stale region.
        for (uint32_t s = 50; s < 70; s++) {
            for (uint32_t d = 0; d < 512; d++) tok[d] = mark(s, d);
            CHECK(!pim_tensor_append(c, &t, s, 1, tok), "re-append %u", s);
        }
        if ((snap = snapshot(c, &t))) {
            uint32_t tail_bad = 0;
            for (uint32_t r = 70; r < 80; r++)
                for (uint32_t o = 0; o < t.noutpad; o++)
                    if (at(c, snap, &t, o, r) != 0) tail_bad++;
            CHECK(tail_bad == 0, "after re-appending to 70, %u element(s) in [70,80) "
                  "are not zero", tail_bad);
            printf("     re-appended to 70: [70,80) still zero\n");
            free(snap);
        }

        // truncate(0) is the Cache.reset() path and is free; the next append must
        // still clear the chunk it touches.
        CHECK(!pim_tensor_truncate(c, &t, 0), "truncate to 0");
        CHECK(t.zeroed_chunks == 0, "truncate(0) left zeroed_chunks at %u",
              t.zeroed_chunks);
        for (uint32_t d = 0; d < 512; d++) tok[d] = mark(0, d);
        CHECK(!pim_tensor_append(c, &t, 0, 1, tok), "append after reset");
        if ((snap = snapshot(c, &t))) {
            uint32_t tail_bad = 0;
            for (uint32_t r = 1; r < 16; r++)
                for (uint32_t o = 0; o < t.noutpad; o++)
                    if (at(c, snap, &t, o, r) != 0) tail_bad++;
            CHECK(tail_bad == 0, "after reset and one append, %u element(s) in [1,16) "
                  "are not zero", tail_bad);
            printf("     truncate(0) then one append: [1,16) zero without a full "
                   "clear\n");
            free(snap);
        }
        free(tok);
        pim_tensor_free(c, &t);
    }

    // ---- 5. what it refuses ------------------------------------------------
    printf("\n  5. refusals\n");
    {
        pim_tensor t;
        uint16_t   tok[64] = { 1 };
        const char *bad;

        CHECK(!pim_tensor_alloc(c, PIM_LAYOUT_RED_MAJOR, 64, 256, 0, &t), "alloc");

        bad = pim_tensor_append(c, &t, 4, 1, tok);      /* frontier is 0 */
        CHECK(bad != NULL, "an append four steps past the frontier was accepted");
        if (bad) printf("     gap above the frontier: \"%.66s...\"\n", bad);

        bad = pim_tensor_append(c, &t, 250, 20, tok);   /* 270 > 256 */
        CHECK(bad != NULL, "an append past the reserved axis was accepted");
        if (bad) printf("     past the reservation:   \"%.66s...\"\n", bad);

        bad = pim_tensor_alloc(c, (pim_layout)7, 32, 32, 0, &t);
        CHECK(bad != NULL, "layout 7 was accepted");
        pim_tensor_free(c, &t);
    }

    // ---- 6. the tag separates what the addressing cannot --------------------
    printf("\n  6. the layout tag\n");
    {
        pim_tensor a, b;
        pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, 512, 512, &a);
        pim_tensor_plan(g, PIM_LAYOUT_RED_MAJOR, 512, 512, &b);
        CHECK(pim_tensor_bytes(g, &a) == pim_tensor_bytes(g, &b),
              "a square shape should occupy the same bytes either way");
        CHECK(a.tag != b.tag, "both layouts hash to %#llx, so a program built for "
              "one could read the other's transposed contents",
              (unsigned long long)a.tag);
        printf("     same shape, same %zu B, tags %#llx vs %#llx\n",
               pim_tensor_bytes(g, &a), (unsigned long long)a.tag,
               (unsigned long long)b.tag);
    }

    // ---- 7. nothing leaked -------------------------------------------------
    pim_pool_trim(c, PIM_MEM_DRAM, true);
    CHECK(out[0] == 0, "%u dram hugepages leaked", out[0]);
    printf("\n  7. after trim: %u dram hugepages out\n", out[0]);

    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail != 0;
}
