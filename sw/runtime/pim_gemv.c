// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_gemv.c — one GEMV tile.  See pim_gemv.h for the shape and the constraints.
//
// THE PROGRAM, for nchunks = C and nch = N:
//
//     WRVEC  OPSIZE=L0  ROW=xword+0*64      GPR -> GB, every channel
//     MAC    OPSIZE=L0  ROW=row0  COL=0     pu_mask = gb_mc_mask = 0xFFFF
//     WRVEC  OPSIZE=L1  ROW=xword+1*64
//     MAC    OPSIZE=L1  ROW=row1  COL=0
//     ...
//     RD_MAC OPSIZE=0   ROW=yword+0        CH_MASK = 1<<0     1-HOT, one per channel
//     RD_MAC OPSIZE=0   ROW=yword+1        CH_MASK = 1<<1
//     ...
//     EOS
//
// ROW MEANS TWO DIFFERENT THINGS from the same bits [22:6]: a GPR word index in
// WRVEC and RD_MAC, a DRAM row in MAC.  Getting that backwards is not an error
// anywhere in the hardware — it is a wrong number.
//
// WHY THE MACs CHAIN.  The accumulator is fp32 and survives across ISRs, so all
// nchunks partial products of one output land in it and are rounded ONCE, at RD_MAC.
// The alternative (chunk-outer: one RD_MAC per chunk, host sums the partials) is
// 1.4-1.6x faster and rounds nchunks times; it is not implemented here because a
// first kernel that is bit-exact against a host reference is worth more than a fast
// one whose disagreements have to be argued about.
//
// WHY RD_MAC IS ONE PER CHANNEL.  emu_isr_check refuses a multicast RD_MAC: two
// channels would write the same GPR word and one result would be lost silently.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pimrt/pim_gemv.h"
#include "pim/pim_addr.h"

#include <stdbool.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "emu_regs.h"

#define BEATS_PER_ROW   EMU_MAX_OPSIZE                  // 64
#define ELEMS_PER_BEAT  EMU_LANES_PER_WORD              // 16
#define ELEMS_PER_ROW   (BEATS_PER_ROW * ELEMS_PER_BEAT) // 1024

static char gv_err[300];

// ---------------------------------------------------------- the placement ---
//
// Under RoChBaCo a linear offset inside one broadcast unit decomposes as
//
//     offset = ch * (nbank * row_bytes) + bank * row_bytes + beat * 32 + lane * 2
//
// and the unit index is (supergroup, K-chunk) flattened.  Everything about where a
// weight goes is those two lines; the rest of this file is program building and
// transport.
uint64_t pim_gemv_offset(const pim_geometry *g, const pim_gemv_w *w,
                         uint32_t j, uint32_t i)
{
    uint32_t bank  = j % g->nbank;
    uint32_t ch    = (j / g->nbank) % g->nch;
    uint32_t sg    = j / (g->nbank * g->nch);
    uint32_t chunk = i / ELEMS_PER_ROW;
    uint32_t in    = i % ELEMS_PER_ROW;

    return (uint64_t)pim_gemv_unit(w, sg, chunk) * g->unit_bytes
         + (uint64_t)ch   * g->nbank * g->row_bytes
         + (uint64_t)bank * g->row_bytes
         + (uint64_t)(in / ELEMS_PER_BEAT) * EMU_WORD_BYTES
         + (uint64_t)(in % ELEMS_PER_BEAT) * 2u;
}

uint64_t pim_gemv_tag(const pim_geometry *g, const pim_gemv_w *w)
{
    uint64_t h = 0xcbf29ce484222325ull;      /* FNV-1a, 64 bit */
#define MIX(v) do { h ^= (uint64_t)(v); h *= 0x100000001b3ull; } while (0)
    if (!g || !w) return 0;
    MIX(0x67656d76u);                        /* 'gemv' — which layout, not just which shape */
    MIX(w->npad); MIX(w->kpad); MIX(w->nchunks); MIX(w->ngroups);
    MIX(g->nch); MIX(g->nbank); MIX(g->row_bytes); MIX(g->unit_bytes);
#undef MIX
    return h ? h : 1;                        /* 0 is reserved for "unstamped" */
}

uint64_t pim_gemv_xtag(const pim_gemv_w *w)
{
    uint64_t h = 0xcbf29ce484222325ull;
#define MIX(v) do { h ^= (uint64_t)(v); h *= 0x100000001b3ull; } while (0)
    if (!w) return 0;
    MIX(0x67766563u);                        /* 'gvec' */
    MIX(w->kpad); MIX(BEATS_PER_ROW); MIX(ELEMS_PER_BEAT);
#undef MIX
    return h ? h : 1;
}

const char *pim_gemv_upload_x(pim_ctx *c, const pim_gemv_w *w, void *xgpr,
                              const uint16_t *x)
{
    uint16_t *buf;
    const char *bad;

    if (!c || !w || !xgpr || !x) return "pim_gemv_upload_x: null argument";

    // calloc, NOT malloc: the tail from k to kpad must be zero and this is the
    // only place that guarantees it.
    buf = calloc(w->kpad, 2);
    if (!buf) return "out of host memory staging the vector";
    memcpy(buf, x, (size_t)w->k * 2);
    bad = pim_memcpy_ctx(c, xgpr, buf, (size_t)w->kpad * 2, PIM_TO_DEV, 0);
    free(buf);
    if (bad) return bad;
    return pim_tag_set_ctx(c, xgpr, pim_gemv_xtag(w));
}

// ------------------------------------------------------------- allocation ---
void pim_gemv_plan(const pim_geometry *g, uint32_t n, uint32_t k, pim_gemv_w *out)
{
    uint32_t per;

    if (!g || !out) return;
    memset(out, 0, sizeof *out);
    if (!n || !k) return;                 /* left zeroed; pim_gemv_bytes gives 0 */

    per = g->nbank * g->nch;
    out->n = n;
    out->k = k;
    // PADDED TO A WHOLE SUPERGROUP.  One all-bank, all-channel MAC covers exactly
    // `per` outputs and there is no lane mask anywhere on the MAC path — so a
    // partial supergroup is not a thing the ISA can express.  The tail outputs get
    // zero weights (the staging buffer is calloc'd) and are dropped on the way back.
    out->npad    = (n + per - 1) / per * per;
    out->ngroups = out->npad / per;
    // Padded with ZEROS to a whole beat.  A zero weight contributes exactly 0.0f to
    // a sum in any order, so this does not perturb the result — it is not a
    // tolerance argument.  k is deliberately NOT padded to a whole 1024: the last
    // chunk carries a shorter OPSIZE instead, which is real work saved.
    out->kpad       = (k + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT * ELEMS_PER_BEAT;
    out->nchunks    = (out->kpad + ELEMS_PER_ROW - 1) / ELEMS_PER_ROW;
    out->last_beats = (out->kpad - (out->nchunks - 1) * ELEMS_PER_ROW)
                      / ELEMS_PER_BEAT;
    out->nch        = g->nch;
    out->nbank      = g->nbank;
}

size_t pim_gemv_bytes(const pim_geometry *g, const pim_gemv_w *w)
{
    if (!g || !w) return 0;
    return (size_t)w->ngroups * w->nchunks * g->unit_bytes;
}

const char *pim_gemv_alloc(pim_ctx *c, uint32_t n, uint32_t k, pim_gemv_w *out)
{
    const pim_geometry *g = pim_geom_ctx(c);

    if (!c || !out || !k) return "pim_gemv_alloc: null or empty argument";
    if (!n) return "pim_gemv_alloc: n is zero";

    pim_gemv_plan(g, n, k, out);
    out->w = pim_alloc_ctx(c, pim_gemv_bytes(g, out), PIM_MEM_DRAM);
    if (!out->w) {
        snprintf(gv_err, sizeof gv_err,
                 "weights for [%u x %u] (padded to [%u x %u] = %u group(s) x %u "
                 "chunk(s) of %llu KiB): %s", n, k, out->npad, out->kpad,
                 out->ngroups, out->nchunks,
                 (unsigned long long)(g->unit_bytes >> 10), pim_last_error_ctx(c));
        return gv_err;
    }
    return NULL;
}

void pim_gemv_free(pim_ctx *c, pim_gemv_w *w)
{
    if (!c || !w || !w->w) return;
    pim_free_ctx(c, w->w);
    w->w = NULL;
}

// Permute on the host, then ONE transfer.  See the header note on why this is not a
// scatter any more.
const char *pim_gemv_upload(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W)
{
    const pim_geometry *g = pim_geom_ctx(c);
    size_t nb = pim_gemv_bytes(g, w);
    uint16_t *buf;
    const char *bad;

    if (!c || !w || !W || !w->w) return "pim_gemv_upload: null argument";
    buf = calloc(1, nb);                  // zeroed: that IS the padding
    if (!buf) return "out of host memory staging the weights";

    // j < w->n, not w->npad: the padded supergroup's rows stay as calloc left them,
    // which IS the zero padding.  Same for the k tail of every row.
    for (uint32_t j = 0; j < w->n; j++)
        for (uint32_t i = 0; i < w->k; i++)
            buf[pim_gemv_offset(g, w, j, i) / 2] = W[(size_t)j * w->k + i];

    bad = pim_memcpy_ctx(c, w->w, buf, nb, PIM_TO_DEV, 0);
    free(buf);
    if (bad) return bad;
    // STAMP WHAT WE JUST WROTE.  Until this runs the allocation holds whatever was
    // there before, and a program built against it would be reading someone else's
    // arrangement.  See pim_tag_set().
    return pim_tag_set_ctx(c, w->w, pim_gemv_tag(g, w));
}

const char *pim_gemv_verify(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W,
                            uint64_t *n_bad)
{
    const pim_geometry *g = pim_geom_ctx(c);
    size_t nb = pim_gemv_bytes(g, w);
    uint16_t *got = malloc(nb);
    uint64_t bad_n = 0;
    const char *bad;

    if (!got) return "out of host memory reading the weights back";
    if ((bad = pim_memcpy_ctx(c, got, w->w, nb, PIM_FROM_DEV, 0))) { free(got); return bad; }

    for (uint32_t j = 0; j < w->n && bad_n == 0; j++)
        for (uint32_t i = 0; i < w->k; i++) {
            uint64_t off = pim_gemv_offset(g, w, j, i);
            if (got[off / 2] != W[(size_t)j * w->k + i]) {
                snprintf(gv_err, sizeof gv_err,
                         "weight [%u][%u] read back %04x, wrote %04x (allocation "
                         "offset %llu)", j, i, got[off / 2],
                         W[(size_t)j * w->k + i], (unsigned long long)off);
                bad_n++;
                break;
            }
        }
    free(got);
    if (n_bad) *n_bad = bad_n;
    return bad_n ? gv_err : NULL;
}

// ------------------------------------------------------- eager all-bank ---
uint32_t pim_gemv_host_stride(const pim_gemv_w *w)
{ (void)w; return ELEMS_PER_ROW; }

bool pim_gemv_is_eager(const pim_gemv_w *w)
{ return w && w->nchunks == 1; }

void pim_gemv_pack_eager(const pim_gemv_w *w, const uint16_t *W, uint16_t *out)
{
    if (!w || !W || !out) return;
    // calloc'd or memset by the caller?  Neither — do it here, because the ONE
    // region that must be zero (the final beat's tail) is easy to forget and a
    // stale element there does not fail, it changes the answer.
    memset(out, 0, (size_t)w->npad * ELEMS_PER_ROW * 2);
    for (uint32_t j = 0; j < w->n; j++)
        memcpy(out + (size_t)j * ELEMS_PER_ROW, W + (size_t)j * w->k, (size_t)w->k * 2);
}

const char *pim_gemv_upload_eager(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W)
{
    const pim_geometry *g = pim_geom_ctx(c);
    const char *bad;

    if (!c || !w || !W || !w->w) return "pim_gemv_upload_eager: null argument";
    if (!pim_gemv_is_eager(w)) {
        snprintf(gv_err, sizeof gv_err,
                 "k = %u needs %u chunk(s), and chunks of different outputs "
                 "interleave in the allocation — the eager form is linear only "
                 "while k fits one %u-element row.  Use pim_gemv_upload().",
                 w->k, w->nchunks, ELEMS_PER_ROW);
        return gv_err;
    }

    // THE WHOLE POINT: no permutation, no staging.  The host array already IS the
    // device layout, and the two are the same number of bytes.
    if ((bad = pim_memcpy_ctx(c, w->w, W, pim_gemv_bytes(g, w), PIM_TO_DEV, 0)))
        return bad;
    return pim_tag_set_ctx(c, w->w, pim_gemv_tag(g, w));
}

const char *pim_gemv_upload_rows(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W)
{
    const pim_geometry *g = pim_geom_ctx(c);
    const char *bad;

    if (!c || !w || !W || !w->w) return "pim_gemv_upload_rows: null argument";
    if (!pim_gemv_is_eager(w)) {
        snprintf(gv_err, sizeof gv_err,
                 "k = %u needs %u chunk(s); the row form is linear only while k "
                 "fits one %u-element row.  Use pim_gemv_upload().",
                 w->k, w->nchunks, ELEMS_PER_ROW);
        return gv_err;
    }

    // ROWS n..npad-1 ARE NOT SENT.  They back outputs nobody reads, and a bank
    // computes its own row into its own accumulator, so whatever is in them stays
    // there.  Skipping them is not an optimisation of a safe thing into a risky
    // one — it is declining to write memory that is never read.
    for (uint32_t j = 0; j < w->n; j++) {
        uint16_t *dst = (uint16_t *)((char *)w->w + (size_t)j * g->row_bytes);
        if ((bad = pim_memcpy_ctx(c, dst, W + (size_t)j * w->kpad,
                                  (size_t)w->kpad * 2, PIM_TO_DEV, 0)))
            return bad;
    }
    return pim_tag_set_ctx(c, w->w, pim_gemv_tag(g, w));
}

// --------------------------------------------------------- the program ---
//
// ONE PASS PER `nlatch` SUPERGROUPS.  Inside a pass the vector is loaded once per
// K-chunk and every group in the pass MACs off that one load, differing only in ROW
// and in T.  That is the whole point of the second latch: RD_MAC is read-clearing,
// so without T a second group cannot be interleaved without destroying the first
// group's running sum, and the vector has to be reloaded for it.
// ONE CHUNK IS THE CASE WHERE THE SECOND LATCH BUYS NOTHING, and saying so honestly
// is the difference between a measurement and an advertisement.  When K fits a
// single page there is no accumulator to protect between groups: the vector goes up
// once and every group takes a (MAC, RD_MAC) pair off it, because a GB-sourced MAC
// rewinds and RD_MAC does not touch the GB.  Both schedules hoist it, both issue one
// WRVEC, and DUAL's advantage is exactly zero.
static bool hoist_wrvec(const pim_gemv_w *w) { return w->nchunks == 1; }

uint32_t pim_gemv_nwrvec(const pim_gemv_w *w, uint32_t count, pim_acc_mode mode)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    if (hoist_wrvec(w)) return count ? 1u : 0u;
    return ((count + nlatch - 1) / nlatch) * w->nchunks;
}

uint32_t pim_gemv_nisr(const pim_gemv_w *w, uint32_t count, pim_acc_mode mode)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t n = 1 + (hoist_wrvec(w) ? 1u : 0u);        // EOS, and the hoisted WRVEC
    for (uint32_t g0 = 0; g0 < count; g0 += nlatch) {
        uint32_t ng = count - g0 < nlatch ? count - g0 : nlatch;
        n += w->nchunks * ng            // one MAC per group per chunk
           + ng * w->nch;               // one RD_MAC per group per channel
        if (!hoist_wrvec(w)) n += w->nchunks;
    }
    return n;
}

// HOW MUCH OF A GEMV FITS BEHIND ONE DOORBELL.  Rounded down to a whole number of
// PASSES, because a pass is the unit that shares a vector load — splitting one in
// half would reload the vector for its second group and undo the point of DUAL.
uint32_t pim_gemv_groups_per_launch(const pim_gemv_w *w, pim_acc_mode mode,
                                    uint32_t max_isrs)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t fixed  = 1u + (hoist_wrvec(w) ? 1u : 0u);
    uint32_t pass   = w->nchunks * nlatch + nlatch * w->nch
                    + (hoist_wrvec(w) ? 0u : w->nchunks);
    uint32_t passes;

    if (max_isrs <= fixed) return 0;
    passes = (max_isrs - fixed) / pass;
    if (!passes) return 0;
    if (passes * nlatch >= w->ngroups) return w->ngroups;
    return passes * nlatch;
}

void pim_gemv_gpr_bytes(const pim_gemv_w *w, pim_acc_mode mode, uint32_t max_isrs,
                        size_t *xbytes, size_t *ybytes)
{
    uint32_t gpl = pim_gemv_groups_per_launch(w, mode, max_isrs);

    if (xbytes) *xbytes = (size_t)w->kpad * 2u;
    if (ybytes) *ybytes = (size_t)(gpl ? gpl : w->ngroups) * w->nch * EMU_WORD_BYTES;
}

const char *pim_gemv_program(const pim_geometry *g, const pim_gemv_w *w,
                             uint32_t group_first, uint32_t group_count,
                             uint32_t xword, uint32_t yword,
                             const uint32_t *unit_row, pim_acc_mode mode,
                             pim_prog *out)
{
    uint32_t all_ch = (1u << g->nch) - 1u;
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    bool     hoist  = hoist_wrvec(w);
    const char *bad;

    if (!g || !w || !unit_row || !out) return "pim_gemv_program: null argument";
    if (group_first + group_count > w->ngroups)
        return "pim_gemv_program: the group range runs past the allocation";
    if (pim_gemv_nisr(w, group_count, mode) > out->cap) {
        snprintf(gv_err, sizeof gv_err,
                 "the program needs %u ISRs and the buffer holds %u",
                 pim_gemv_nisr(w, group_count, mode), out->cap);
        return gv_err;
    }

    if (hoist) {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
        struct emu_isr      isr;
        s.opsize = w->last_beats;       // the only chunk, so this is its size
        s.row    = xword;
        if ((bad = emu_isr_build(&isr, &s))) {
            snprintf(gv_err, sizeof gv_err, "WRVEC refused: %s", bad);
            return gv_err;
        }
        if ((bad = pim_prog_push(out, (const pim_isr *)&isr))) return bad;
    }

    for (uint32_t g0 = 0; g0 < group_count; g0 += nlatch) {
        uint32_t ng = group_count - g0 < nlatch ? group_count - g0 : nlatch;

        for (uint32_t ck = 0; ck < w->nchunks; ck++) {
            uint32_t beats = (ck == w->nchunks - 1) ? w->last_beats : BEATS_PER_ROW;
            struct emu_isr_spec s;
            struct emu_isr      isr;

            // WRVEC: GPR -> GB, on every channel.  ROW is a GPR WORD here.  Chunk ck
            // starts 64 words on from the last, because one chunk is 64 beats.
            // ISSUED ONCE FOR THE WHOLE PASS'S GROUPS — the MACs below rewind the
            // GB read pointer and see the same vector.  Skipped entirely when the
            // whole program shares one hoisted WRVEC (see hoist_wrvec).
            if (!hoist) {
                s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
                s.opsize = beats;
                s.row    = xword + ck * BEATS_PER_ROW;
                if ((bad = emu_isr_build(&isr, &s))) {
                    snprintf(gv_err, sizeof gv_err, "WRVEC refused: %s", bad);
                    return gv_err;
                }
                if ((bad = pim_prog_push(out, (const pim_isr *)&isr))) return bad;
            }

            for (uint32_t t = 0; t < ng; t++) {
                uint32_t row = unit_row[pim_gemv_unit(w, group_first + g0 + t, ck)];

                if (row > EMU_ISR_ROW_MAX) {
                    snprintf(gv_err, sizeof gv_err,
                             "group %u chunk %u is at DRAM row %u and the ISR's ROW "
                             "field holds 17 bits", group_first + g0 + t, ck, row);
                    return gv_err;
                }
                // MAC: the banks gb_mc_mask names receive the GB broadcast, the banks
                // pu_mask names compute.  Equal, or a bank gets the vector without
                // computing.  ROW is a DRAM row here — the same bits, another meaning.
                s = emu_isr_default_ch(ISR_OP_MAC, all_ch);
                s.opsize     = beats;
                s.row        = row;
                s.col        = 0;
                s.pu_mask    = (1u << g->nbank) - 1u;
                s.gb_mc_mask = s.pu_mask;
                if ((bad = emu_isr_build(&isr, &s))) {
                    snprintf(gv_err, sizeof gv_err, "MAC refused: %s", bad);
                    return gv_err;
                }
                // T IS SET AFTER emu_isr_build, NOT THROUGH IT.  emu_isr_check refuses
                // a nonzero T outright — "the MC hard-ties latch_sel" — and that is a
                // measured fact about an earlier image, not a law.  A schedule whose
                // purpose is to use the second latch has to be able to encode the
                // word; pim_prog_verify is where the gate lives instead, so the
                // decision is made once, explicitly, by whoever set allow_t_latch.
                if (t) emu_isr_set(&isr, ISR_F_T, 1u);
                if ((bad = pim_prog_push(out, (const pim_isr *)&isr))) return bad;
            }
        }

        // Drain the pass.  One RD_MAC per (group, channel), 1-hot on the channel: a
        // multicast would have two channels writing the same GPR word, and
        // emu_isr_check refuses it for that reason.  The word index is relative to
        // THIS launch's first group, so a split GEMV reuses the same result buffer
        // instead of needing one word per supergroup of the whole thing.
        for (uint32_t t = 0; t < ng; t++)
            for (uint32_t ch = 0; ch < g->nch; ch++) {
                struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << ch);
                struct emu_isr      isr;

                s.opsize = 0;
                s.row    = yword + (g0 + t) * g->nch + ch;
                if ((bad = emu_isr_build(&isr, &s))) {
                    snprintf(gv_err, sizeof gv_err, "RD_MAC ch%u refused: %s", ch, bad);
                    return gv_err;
                }
                if (t) emu_isr_set(&isr, ISR_F_T, 1u);
                if ((bad = pim_prog_push(out, (const pim_isr *)&isr))) return bad;
            }
    }

    // The trailing EOS is not decoration: done rises when the fetcher ACCEPTS the
    // last ISR, so a program ending in RD_MAC reports finished while its own result
    // is still in flight.
    {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_EOS, all_ch);
        struct emu_isr      isr;
        if ((bad = emu_isr_build(&isr, &s))) return bad;
        if ((bad = pim_prog_push(out, (const pim_isr *)&isr))) return bad;
    }
    return NULL;
}

// ------------------------------------------------------------- the launch ---
#define POISON_LANE 0x7FC1u     // a quiet NaN payload nothing here computes

const char *pim_gemv(pim_ctx *c, pim_exec *e, const pim_gemv_w *w,
                     const uint16_t *x, void *xgpr, void *ygpr, uint16_t *y)
{
    return pim_gemv_ex(c, e, w, x, xgpr, ygpr, y, PIM_ACC_SINGLE, NULL);
}

const char *pim_gemv_ex(pim_ctx *c, pim_exec *e, const pim_gemv_w *w,
                        const uint16_t *x, void *xgpr, void *ygpr, uint16_t *y,
                        pim_acc_mode mode, pim_gemv_stat *st)
{
    const pim_geometry *g = pim_geom_ctx(c);
    uint32_t nunits = w->ngroups * w->nchunks;
    uint32_t gpl, ywords, xword, xn, yword, yn;
    uint32_t *rows = NULL;
    pim_isr  *storage = NULL;
    uint16_t *ybuf = NULL;
    const char *bad = NULL;

    if (!c || !e || !w || !x || !xgpr || !ygpr || !y)
        return "pim_gemv: null argument";
    if (mode != PIM_ACC_SINGLE && mode != PIM_ACC_DUAL)
        return "pim_gemv_ex: mode is neither PIM_ACC_SINGLE nor PIM_ACC_DUAL";
    if (st) memset(st, 0, sizeof *st);

    // HOW MUCH FITS BEHIND ONE DOORBELL.  A model-sized GEMV does not fit in one:
    // IMEM holds 16383 ISRs.  Splitting is by SUPERGROUP because a supergroup owns
    // its accumulators from first MAC to RD_MAC — cutting anywhere else would strand
    // a running sum in a latch across a doorbell.
    gpl = pim_gemv_groups_per_launch(w, mode, pim_exec_max_isrs(e));
    if (!gpl) {
        snprintf(gv_err, sizeof gv_err,
                 "not even one supergroup fits in %u ISRs at k=%u (%u chunks); the "
                 "engine's program limit is too low for this shape",
                 pim_exec_max_isrs(e), w->k, w->nchunks);
        return gv_err;
    }
    ywords = gpl * w->nch;

    // ---- the GPR coordinates the ISA will name -------------------------
    if ((bad = pim_addr_gpr_words_ctx(c, xgpr, (size_t)w->kpad * 2, &xword, &xn)))
        return bad;
    if ((bad = pim_addr_gpr_words_ctx(c, ygpr, (size_t)ywords * EMU_WORD_BYTES,
                                &yword, &yn)))
        return bad;

    rows    = malloc((size_t)nunits * sizeof *rows);
    storage = malloc((size_t)pim_gemv_nisr(w, gpl, mode) * sizeof *storage);
    ybuf    = malloc((size_t)ywords * EMU_WORD_BYTES);
    if (!rows || !storage || !ybuf) { bad = "out of host memory"; goto out; }

    // ---- the DRAM row of every (supergroup, chunk) ---------------------
    // Read out of the allocation rather than assumed consecutive: the units are
    // broadcast units and the pool may have handed them out scattered.
    for (uint32_t u = 0; u < nunits; u++) {
        pim_unit un;
        if ((bad = pim_addr_unit_ctx(c, w->w, pim_gemv_bytes(g, w), u, &un))) goto out;
        rows[u] = (uint32_t)un.row;
    }

    // ---- tile the vector into the GPR, ONCE ----------------------------
    // One WRVEC reads OPSIZE beats from consecutive GPR words, and a beat is 16
    // BF16 — so the tiling is: pad to a whole beat with zeros, and chunk c starts at
    // word xword + c*64.  The zeros are why padding is free: zero times any weight
    // is exactly 0.0f in the accumulator.  Every launch re-reads the same buffer.
    if ((bad = pim_gemv_upload_x(c, w, xgpr, x))) goto out;

    for (uint32_t g0 = 0; g0 < w->ngroups; g0 += gpl) {
        uint32_t ng = w->ngroups - g0 < gpl ? w->ngroups - g0 : gpl;
        uint32_t nw = ng * w->nch;
        pim_prog prog;
        pim_launch info;

        // ---- poison this launch's result words -------------------------
        // Not verified by reading it back: that would be another write/read pair
        // subject to exactly the visibility question it is meant to settle.  What
        // decides is the RESULT losing it.
        for (size_t i = 0; i < (size_t)nw * ELEMS_PER_BEAT; i++)
            ybuf[i] = POISON_LANE;
        if ((bad = pim_memcpy_ctx(c, ygpr, ybuf, (size_t)nw * EMU_WORD_BYTES,
                              PIM_TO_DEV, 0))) goto out;

        pim_prog_init(&prog, storage, pim_gemv_nisr(w, gpl, mode));
        if ((bad = pim_gemv_program(g, w, g0, ng, xword, yword, rows, mode, &prog)))
            goto out;
        if ((bad = pim_exec_run(e, &prog, &info))) goto out;
        if (st) {
            st->launch_us += info.us;
            st->polls     += info.polls;
            st->nisr      += prog.n;
            st->nwrvec    += pim_gemv_nwrvec(w, ng, mode);
            st->nlaunch   += 1;
            st->saw_done   = info.saw_done;
        }

        // ---- collect ----------------------------------------------------
        // done decides nothing (it is a held level and can be stale); the poison
        // does.  Every word must have lost it — a channel that never executed leaves
        // its own word untouched, which is exactly the failure a fan-out program has
        // to show, and a latch that was never selected leaves its group's too.
        if ((bad = pim_memcpy_ctx(c, ybuf, ygpr, (size_t)nw * EMU_WORD_BYTES,
                              PIM_FROM_DEV, 0))) goto out;
        for (uint32_t i = 0; i < nw; i++) {
            bool landed = false;
            for (uint32_t l = 0; l < ELEMS_PER_BEAT; l++)
                if (ybuf[i * ELEMS_PER_BEAT + l] != POISON_LANE) { landed = true; break; }
            if (!landed) {
                snprintf(gv_err, sizeof gv_err,
                         "result word %u (supergroup %u, channel %u) is still the "
                         "poison after %llu us (%u polls, done %s).  That RD_MAC "
                         "never wrote — check CH_MASK fan-out, or ISR[35] if this "
                         "was a dual-latch run.", i, g0 + i / w->nch, i % w->nch,
                         (unsigned long long)info.us, info.polls,
                         info.saw_done ? "rose" : "never rose");
                bad = gv_err;
                goto out;
            }
        }

        // Word (sg*nch + ch), lane b  ->  output sg*16*nch + ch*16 + b.  That is
        // bank(j) = j % 16, channel(j) = (j/16) % nch, supergroup(j) = j / (16*nch)
        // read backwards, and it is the mapping pim_gemv_offset used to place them.
        // Outputs past w->n are the padding and are dropped here.
        for (uint32_t sg = 0; sg < ng; sg++)
            for (uint32_t ch = 0; ch < w->nch; ch++)
                for (uint32_t b = 0; b < w->nbank; b++) {
                    uint32_t j = ((g0 + sg) * w->nch + ch) * w->nbank + b;
                    if (j < w->n)
                        y[j] = ybuf[(sg * w->nch + ch) * ELEMS_PER_BEAT + b];
                }
    }

out:
    free(ybuf); free(storage); free(rows);
    return bad;
}

// ------------------------------------------------------------- reference ---
// Not an fp32 loop.  See pim_gemv.h and the top of pim_mac_exact.c: a beat is block
// floating point, so the loss happens in the alignment shift rather than in any
// addition, and the ORDER of the adds is irrelevant while the max-exponent of the
// beat is everything.  pim_mac_exact() is the RTL transcribed; this only walks it
// over the outputs.
void pim_gemv_golden(const uint16_t *W, uint32_t n, uint32_t k,
                     const uint16_t *x, uint16_t *y)
{
    for (uint32_t j = 0; j < n; j++)
        y[j] = pim_mac_exact(W + (size_t)j * k, x, k);
}

//////////////////////////////////////////////////////////////////////////////////
// THE TWO-PASS FORM.  Same schedule, same order, same ISRs — but the address fields
// are left blank and a note says what fills them.  Compare with pim_gemv_program()
// above: the loops are identical and every emu_isr_build() call is the same.  What
// is gone is the resolving (there is no unit_row[] to be handed) and the
// RD_MAC-per-channel expansion, which pim_prog_lower() now does from the split hint.
//
// KEPT SEPARATE FROM pim_gemv_program RATHER THAN REPLACING IT.  That one is what
// the board-verified path runs, and it is checked field-by-field on the host by
// gemv_test's phase 0b.  Having both lets lower_test prove they agree ISR for ISR,
// which is a stronger statement than either alone.
//////////////////////////////////////////////////////////////////////////////////
uint32_t pim_gemv_logical_nisr(const pim_gemv_w *w, uint32_t count, pim_acc_mode mode)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t n = 1 + (hoist_wrvec(w) ? 1u : 0u);        // EOS, and the hoisted WRVEC

    for (uint32_t g0 = 0; g0 < count; g0 += nlatch) {
        uint32_t ng = count - g0 < nlatch ? count - g0 : nlatch;
        n += w->nchunks * ng            // one MAC per group per chunk
           + ng;                        // ONE logical drain per group (not nch)
        if (!hoist_wrvec(w)) n += w->nchunks;
    }
    return n;
}

const char *pim_gemv_logical(const pim_geometry *g, const pim_gemv_w *w,
                             uint32_t group_first, uint32_t group_count,
                             const void *xgpr, size_t xbytes,
                             const void *ygpr, size_t ybytes,
                             pim_acc_mode mode, pim_logical *out)
{
    uint32_t all_ch = (1u << g->nch) - 1u;
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    bool     hoist  = hoist_wrvec(w);
    const char *bad;

    if (!g || !w || !xgpr || !ygpr || !out)
        return "pim_gemv_logical: null argument";
    if (group_first + group_count > w->ngroups)
        return "pim_gemv_logical: the group range runs past the allocation";

    if (hoist) {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
        struct emu_isr      isr;
        s.opsize = w->last_beats;
        s.row    = 0;                    // blank; the reference below fills it
        if ((bad = emu_isr_build(&isr, &s))) return bad;
        if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                        PIM_REF_GPR_WORD, PIM_SPLIT_NONE,
                                        xgpr, xbytes, 0, pim_gemv_xtag(w))))
            return bad;
    }

    for (uint32_t g0 = 0; g0 < group_count; g0 += nlatch) {
        uint32_t ng    = group_count - g0 < nlatch ? group_count - g0 : nlatch;
        uint32_t first = out->nisr;      // this pass owns its accumulators from here

        for (uint32_t ck = 0; ck < w->nchunks; ck++) {
            uint32_t beats = (ck == w->nchunks - 1) ? w->last_beats : BEATS_PER_ROW;
            struct emu_isr_spec s;
            struct emu_isr      isr;

            if (!hoist) {
                s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
                s.opsize = beats;
                s.row    = 0;
                if ((bad = emu_isr_build(&isr, &s))) return bad;
                if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                                PIM_REF_GPR_WORD, PIM_SPLIT_NONE,
                                                xgpr, xbytes, ck * BEATS_PER_ROW,
                                                pim_gemv_xtag(w))))
                    return bad;
            }

            for (uint32_t t = 0; t < ng; t++) {
                uint32_t unit = pim_gemv_unit(w, group_first + g0 + t, ck);

                s = emu_isr_default_ch(ISR_OP_MAC, all_ch);
                s.opsize     = beats;
                s.row        = 0;
                s.col        = 0;
                s.pu_mask    = (1u << g->nbank) - 1u;
                s.gb_mc_mask = s.pu_mask;
                if ((bad = emu_isr_build(&isr, &s))) return bad;
                if (t) emu_isr_set(&isr, ISR_F_T, 1u);
                if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                                PIM_REF_DRAM_UNIT, PIM_SPLIT_NONE,
                                                w->w, pim_gemv_bytes(g, w), unit,
                                                pim_gemv_tag(g, w))))
                    return bad;
            }
        }

        // ONE DRAIN PER GROUP, not one per (group, channel).  The word index is the
        // FIRST of this group's nch words and PIM_SPLIT_PER_CHANNEL says the rest
        // follow; pim_prog_lower() emits the other nch-1 and one-hots each CH_MASK.
        //
        // CH_MASK IS BUILT AS CHANNEL 0 RATHER THAN ALL-CHANNELS, and that is not a
        // detail: emu_isr_build() REFUSES a multicast RD_MAC outright, so a logical
        // program cannot hold the illegal word even briefly.  The intermediate form
        // stays legal by construction and the split hint carries the "this stands
        // for nch of them" meaning instead.  The lowerer overwrites CH_MASK anyway.
        for (uint32_t t = 0; t < ng; t++) {
            struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u);
            struct emu_isr      isr;

            s.opsize = 0;
            s.row    = 0;
            if ((bad = emu_isr_build(&isr, &s))) return bad;
            if (t) emu_isr_set(&isr, ISR_F_T, 1u);
            if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                            PIM_REF_GPR_WORD, PIM_SPLIT_PER_CHANNEL,
                                            ygpr, ybytes, (g0 + t) * g->nch, 0)))
                return bad;
        }

        // The pass is one accumulation region: first MAC to last RD_MAC.  Cutting
        // inside it across a doorbell strands a running sum in a latch.
        if ((bad = pim_logical_atom(out, first, out->nisr - 1))) return bad;
    }

    {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_EOS, all_ch);
        struct emu_isr      isr;
        if ((bad = emu_isr_build(&isr, &s))) return bad;
        if ((bad = pim_logical_push(out, (const pim_isr *)&isr))) return bad;
    }
    return NULL;
}
