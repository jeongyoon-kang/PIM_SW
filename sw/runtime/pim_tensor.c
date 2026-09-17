// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_tensor.c — placing a matrix on the card, and keeping it placed.
//
// The header carries the design.  What is worth saying beside the code is how the
// transfers are shaped, because that is where the two layouts stop looking alike:
//
//   §1  arithmetic        plan / bytes / offset / tag.  No context, no board.
//   §2  the scatter       one walk, used by upload, append and both clears.
//   §3  allocation        alloc / free
//   §4  filling           upload / append / truncate
//
// ONE WALK FOR EVERYTHING.  upload, append, the lazy chunk clear and truncate all
// come down to "copy host bytes to (out, red) pairs", differing only in which
// rectangle of (out, red) they cover and where the host bytes are.  Writing that
// once and passing it a rectangle is not merely shorter: these four are exactly the
// functions that must agree with pim_tensor_offset, and four hand-rolled loops are
// four chances to agree with it three times.
//////////////////////////////////////////////////////////////////////////////////
#include "pimrt/pim_tensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ELEMS_PER_BEAT  PIM_TENSOR_ELEMS_PER_BEAT
#define ELEMS_PER_ROW   PIM_TENSOR_ELEMS_PER_ROW

static char te_err[256];

static uint32_t roundup_u32(uint32_t v, uint32_t to) { return (v + to - 1) / to * to; }

//////////////////////////////////////////////////////////////////////////////////
// §1  ARITHMETIC
//////////////////////////////////////////////////////////////////////////////////
void pim_tensor_plan(const pim_geometry *g, pim_layout layout,
                     uint32_t nout, uint32_t nred, pim_tensor *out)
{
    uint32_t per;

    if (!g || !out) return;
    memset(out, 0, sizeof *out);
    if (!nout || !nred) return;              /* left zeroed; bytes() gives 0 */

    per = g->nbank * g->nch;
    out->layout  = layout;
    out->nout    = nout;
    out->nred    = nred;
    out->noutpad = roundup_u32(nout, per);
    out->ngroups = out->noutpad / per;
    out->nredpad = roundup_u32(nred, ELEMS_PER_BEAT);
    out->nchunks = (out->nredpad + ELEMS_PER_ROW - 1) / ELEMS_PER_ROW;
    out->last_beats = (out->nredpad - (out->nchunks - 1) * ELEMS_PER_ROW)
                      / ELEMS_PER_BEAT;
    out->nch     = g->nch;
    out->nbank   = g->nbank;
    out->tag     = pim_tensor_tag(g, out);
}

size_t pim_tensor_bytes(const pim_geometry *g, const pim_tensor *t)
{
    if (!g || !t) return 0;
    return (size_t)t->ngroups * t->nchunks * g->unit_bytes;
}

uint64_t pim_tensor_offset(const pim_geometry *g, const pim_tensor *t,
                           uint32_t out, uint32_t red)
{
    uint32_t bank  = out % g->nbank;
    uint32_t ch    = (out / g->nbank) % g->nch;
    uint32_t group = out / (g->nbank * g->nch);
    uint32_t chunk = red / ELEMS_PER_ROW;
    uint32_t in    = red % ELEMS_PER_ROW;

    // The last two terms are the beat and the lane inside it.  They are NOT one
    // multiply: a beat is EMU_WORD_BYTES on the wire and 16 lanes of BF16, and the
    // two happen to agree at 32 bytes only because BF16 is two bytes wide.  Keeping
    // them separate is what makes this survive a different element width.
    return (uint64_t)pim_tensor_unit(t, group, chunk) * g->unit_bytes
         + (uint64_t)ch   * g->nbank * g->row_bytes
         + (uint64_t)bank * g->row_bytes
         + (uint64_t)(in / ELEMS_PER_BEAT) * (ELEMS_PER_BEAT * 2u)
         + (uint64_t)(in % ELEMS_PER_BEAT) * 2u;
}

uint64_t pim_tensor_tag(const pim_geometry *g, const pim_tensor *t)
{
    uint64_t h = 0xcbf29ce484222325ull;      /* FNV-1a, 64 bit */
#define MIX(v) do { h ^= (uint64_t)(v); h *= 0x100000001b3ull; } while (0)
    if (!g || !t) return 0;
    MIX(0x74656e73u);                        /* 'tens' — which kind of layout */
    MIX((uint32_t)t->layout);
    MIX(t->noutpad); MIX(t->nredpad); MIX(t->nchunks); MIX(t->ngroups);
    MIX(g->nch); MIX(g->nbank); MIX(g->row_bytes); MIX(g->unit_bytes);
#undef MIX
    return h ? h : 1;                        /* 0 is reserved for "unstamped" */
}

//////////////////////////////////////////////////////////////////////////////////
// §2  THE SCATTER
//
// Copy the rectangle [out0, out0+nout) x [red0, red0+nred) between the card and a
// host buffer.  `src` is NULL to write zeroes; otherwise it is the host side, laid
// out as `count` steps of the non-growth axis exactly as pim_tensor_append
// documents.
//
// THE LOOP NEST IS out-OUTER, red-INNER, AND THAT IS THE WHOLE PERFORMANCE STORY.
// Consecutive `red` at one `out` are consecutive bytes on the card (inside a chunk),
// so the inner loop turns into ONE transfer per (out, chunk).  Consecutive `out` are
// row_bytes apart and never merge.  So:
//
//   a rectangle one `out` wide   -> nchunks transfers.  K's per-token append.
//   a rectangle one `red` wide   -> nout transfers of two bytes.  V's.
//   both wide                    -> nout * nchunks transfers of 2*nred bytes,
//                                   which is why batching V's appends pays: the
//                                   piece grows and the count does not.
//
// WHEN THE HOST SIDE IS RED_MAJOR the inner run is strided on the HOST (stride
// nout) while contiguous on the card, so it is gathered through a staging buffer
// first.  That buffer is sized by the run, not by the rectangle.
//////////////////////////////////////////////////////////////////////////////////
struct rect { uint32_t out0, nout, red0, nred; };

static const char *scatter(pim_ctx *c, const pim_tensor *t, const struct rect *r,
                           const uint16_t *src, uint32_t src_stride)
{
    const pim_geometry *g = pim_geom_ctx(c);
    uint16_t   *stage = NULL;
    const char *bad = NULL;
    uint32_t    o;

    if (!g) return "pim_tensor: no geometry";
    if (!r->nout || !r->nred) return NULL;

    // A buffer is needed in two cases, and NOT only for long runs: a RED_MAJOR host
    // side is strided and has to be gathered however short the run is — appending a
    // single token is the common case and it is exactly r->nred == 1 — and a clear
    // needs somewhere to read zeroes from.  An OUT_MAJOR fill needs neither, since
    // its runs are already contiguous in `src`.
    if (!src || t->layout == PIM_LAYOUT_RED_MAJOR) {
        stage = calloc(r->nred, 2);
        if (!stage) return "out of host memory staging a tensor transfer";
    }

    for (o = 0; o < r->nout && !bad; o++) {
        uint32_t out = r->out0 + o;
        uint32_t done = 0;

        while (done < r->nred && !bad) {
            uint32_t red   = r->red0 + done;
            // A run stops at the end of the rectangle or at the chunk boundary,
            // whichever comes first — crossing a chunk jumps to another unit.
            uint32_t room  = ELEMS_PER_ROW - (red % ELEMS_PER_ROW);
            uint32_t run   = r->nred - done < room ? r->nred - done : room;
            const uint16_t *hp;

            if (!src) {
                hp = stage;                    /* calloc'd, and never written */
            } else if (t->layout == PIM_LAYOUT_OUT_MAJOR) {
                // src is [step][nred]; a step is one `out`, so the run is already
                // contiguous and goes straight out.
                hp = src + (size_t)o * src_stride + done;
            } else {
                // src is [step][nout]; a step is one `red`.  The run walks steps at
                // one output, so it is strided on the host.
                uint32_t i;
                for (i = 0; i < run; i++)
                    stage[i] = src[(size_t)(done + i) * src_stride + o];
                hp = stage;
            }

            bad = pim_memcpy_ctx(c, (char *)t->base + pim_tensor_offset(g, t, out, red),
                                 hp, (size_t)run * 2, PIM_TO_DEV, 0);
            done += run;
        }
    }
    free(stage);
    return bad;
}

//////////////////////////////////////////////////////////////////////////////////
// §3  ALLOCATION
//////////////////////////////////////////////////////////////////////////////////
const char *pim_tensor_alloc(pim_ctx *c, pim_layout layout,
                             uint32_t nout, uint32_t nred, unsigned aflags,
                             pim_tensor *out)
{
    const pim_geometry *g = pim_geom_ctx(c);
    size_t bytes;

    if (!c || !out) return "pim_tensor_alloc: null argument";
    if (!nout || !nred) return "pim_tensor_alloc: nout and nred must both be nonzero";
    if (layout != PIM_LAYOUT_OUT_MAJOR && layout != PIM_LAYOUT_RED_MAJOR)
        return "pim_tensor_alloc: not a layout";

    pim_tensor_plan(g, layout, nout, nred, out);
    bytes = pim_tensor_bytes(g, out);
    out->base = pim_alloc_ex_ctx(c, bytes, PIM_MEM_DRAM, aflags);
    if (!out->base) {
        snprintf(te_err, sizeof te_err,
                 "[%u x %u] (padded to [%u x %u] = %u group(s) x %u chunk(s) of "
                 "%llu KiB): %s", nout, nred, out->noutpad, out->nredpad,
                 out->ngroups, out->nchunks,
                 (unsigned long long)(g->unit_bytes >> 10), pim_last_error_ctx(c));
        return te_err;
    }
    // An eager clear IS the invariant, for every chunk at once.  Say so, or the
    // lazy path would clear chunk 0 a second time on the first append.
    if (aflags & PIM_ALLOC_F_ZERO)
        out->zeroed_chunks = out->nchunks;

    if (pim_tag_set_ctx(c, out->base, out->tag)) {
        pim_free_ctx(c, out->base);
        out->base = NULL;
        return "pim_tensor_alloc: could not stamp the layout tag";
    }
    return NULL;
}

void pim_tensor_free(pim_ctx *c, pim_tensor *t)
{
    if (!c || !t || !t->base) return;
    pim_free_ctx(c, t->base);
    t->base = NULL;
    t->frontier = t->zeroed_chunks = 0;
}

//////////////////////////////////////////////////////////////////////////////////
// §4  FILLING
//////////////////////////////////////////////////////////////////////////////////
const char *pim_tensor_upload(pim_ctx *c, pim_tensor *t, const void *src)
{
    struct rect r;
    const char *bad;

    if (!c || !t || !t->base || !src) return "pim_tensor_upload: null argument";

    // THE PADDING IS CLEARED, AND ONLY THE PADDING.  Zero-extending on the host
    // would need a staging buffer the size of the tensor, so the zeros are written
    // separately — but as the two rectangles that ARE padding, not as a pass over
    // the whole allocation followed by an overwrite.
    //
    // That distinction is most of the upload.  A transformer's weights have no
    // padding at all: every out_features is a multiple of the supergroup and every
    // in_features a multiple of a beat, so both rectangles below are empty and the
    // clear costs nothing.  Clearing everything first would have doubled the DMA for
    // six gigabytes of weights to zero bytes that are about to be overwritten.
    if (t->noutpad > t->nout) {                     /* whole padded output rows */
        r = (struct rect){ t->nout, t->noutpad - t->nout, 0, t->nredpad };
        if ((bad = scatter(c, t, &r, NULL, 0))) return bad;
    }
    if (t->nredpad > t->nred) {                     /* the beat tail of real rows */
        r = (struct rect){ 0, t->nout, t->nred, t->nredpad - t->nred };
        if ((bad = scatter(c, t, &r, NULL, 0))) return bad;
    }

    r = (struct rect){ 0, t->nout, 0, t->nred };
    bad = scatter(c, t, &r, (const uint16_t *)src,
                  t->layout == PIM_LAYOUT_OUT_MAJOR ? t->nred : t->nout);
    if (bad) return bad;

    t->zeroed_chunks = t->nchunks;
    t->frontier = (t->layout == PIM_LAYOUT_OUT_MAJOR) ? t->nout : t->nred;
    return NULL;
}

// Clear reduction chunks [from, to) across every output.  RED_MAJOR's lazy clear.
static const char *clear_chunks(pim_ctx *c, pim_tensor *t, uint32_t from, uint32_t to)
{
    struct rect r;

    if (from >= to) return NULL;
    r = (struct rect){ 0, t->noutpad,
                       from * ELEMS_PER_ROW, (to - from) * ELEMS_PER_ROW };
    if (r.red0 + r.nred > t->nredpad) r.nred = t->nredpad - r.red0;
    return scatter(c, t, &r, NULL, 0);
}

const char *pim_tensor_append(pim_ctx *c, pim_tensor *t, uint32_t first,
                              uint32_t count, const void *src)
{
    uint32_t grow_max, other;
    struct rect r;
    const char *bad;

    if (!c || !t || !t->base || !src) return "pim_tensor_append: null argument";
    if (!count) return NULL;

    grow_max = (t->layout == PIM_LAYOUT_OUT_MAJOR) ? t->nout  : t->nred;
    other    = (t->layout == PIM_LAYOUT_OUT_MAJOR) ? t->nred  : t->nout;

    if ((uint64_t)first + count > grow_max) {
        snprintf(te_err, sizeof te_err,
                 "pim_tensor_append: [%u, %u) runs past the growth axis, which this "
                 "tensor reserved %u of", first, first + count, grow_max);
        return te_err;
    }
    // A GAP WOULD BE SILENT.  Appending past the frontier leaves elements below it
    // that nothing wrote, and a launch whose red_len covers them reads whatever was
    // there.  Re-writing at or below the frontier is fine and is what a rejected
    // speculative token does.
    if (first > t->frontier) {
        snprintf(te_err, sizeof te_err,
                 "pim_tensor_append: starting at %u would leave [%u, %u) unwritten; "
                 "append from the frontier or truncate to it first",
                 first, t->frontier, first);
        return te_err;
    }

    // Establish the zero invariant one chunk ahead of where this write lands.  Only
    // RED_MAJOR needs it — OUT_MAJOR's reduction length is a fixed model dimension,
    // already beat-aligned, so no launch reads past what has been written.
    if (t->layout == PIM_LAYOUT_RED_MAJOR) {
        uint32_t want = (first + count + ELEMS_PER_ROW - 1) / ELEMS_PER_ROW;
        if (want > t->zeroed_chunks) {
            if ((bad = clear_chunks(c, t, t->zeroed_chunks, want))) return bad;
            t->zeroed_chunks = want;
        }
    }

    if (t->layout == PIM_LAYOUT_OUT_MAJOR)
        r = (struct rect){ first, count, 0, t->nred };
    else
        r = (struct rect){ 0, t->nout, first, count };

    if ((bad = scatter(c, t, &r, (const uint16_t *)src, other))) return bad;
    if (first + count > t->frontier) t->frontier = first + count;
    return NULL;
}

const char *pim_tensor_truncate(pim_ctx *c, pim_tensor *t, uint32_t pos)
{
    const char *bad;

    if (!c || !t || !t->base) return "pim_tensor_truncate: null argument";
    if (pos >= t->frontier) { t->frontier = pos; return NULL; }

    if (t->layout == PIM_LAYOUT_RED_MAJOR) {
        if (pos == 0) {
            // FREE, AND NOT A SHORTCUT AROUND THE INVARIANT.  Retracting the
            // lazy-clear bookkeeping makes the next append clear the chunk it
            // touches before writing, which is the same guarantee arrived at
            // without moving a byte now.  This is the Cache.reset() path, so it is
            // the one worth not charging half a gigabyte of DMA for.
            t->zeroed_chunks = 0;
        } else {
            // Everything from the new frontier to the old one holds the previous
            // generation's values.  Clearing only [pos, roundup(pos,16)) would
            // satisfy the invariant TODAY and break it on the next append, which
            // would push the frontier up into the stale region beyond that beat.
            struct rect r = { 0, t->noutpad, pos, t->frontier - pos };
            if ((bad = scatter(c, t, &r, NULL, 0))) return bad;
        }
    }
    t->frontier = pos;
    return NULL;
}
