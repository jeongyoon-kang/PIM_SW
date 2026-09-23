// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_matvec.c — pass 1.  The header carries the design; this is the schedule.
//
// THE SHAPE OF THE PROGRAM, and why it is this and not something else:
//
//     [WRVEC]                       once, if the whole reduction fits one chunk
//     for each pass of nlatch output groups:
//         for each chunk of the reduction range:
//             [WRVEC]               unless hoisted
//             MAC x nlatch          same row set, different T
//         RD_MAC x nlatch           one logical drain per group; lowering fans it
//                                   out to one per channel
//     EOS
//
// GROUP-OUTER, CHUNK-INNER.  The accumulator is fp32 and survives across ISRs
// [measured 2026-08-10], so a dot product longer than one row is chained and
// rounded ONCE at the RD_MAC.  Chunk-outer would need an adder somewhere else.
//
// THE HOIST.  When the reduction range lies inside a single chunk, the vector goes
// up once and every output group MACs off that one load — a GB-sourced MAC rewinds
// the buffer and RD_MAC does not touch it [measured, emu_chain --test rewind].  For
// Q.K^T that is the whole schedule: red_len is D, always one chunk, so ONE WRVEC
// serves every token group in the layer.  It is also why DUAL buys nothing there
// and tlatch_test says so out loud.
//
// ONE ATOM PER PASS.  Everything from a pass's first MAC to its last RD_MAC owns
// the latches, because the latch clears only on RD_MAC and on reset.  Cutting
// inside that range across a doorbell strands a running sum and the next program
// accumulates on top of it — a wrong number, no error.
//////////////////////////////////////////////////////////////////////////////////
#include "pimrt/pim_matvec.h"

#include <stdio.h>
#include <string.h>

#include "emu_regs.h"

#define ELEMS_PER_BEAT  PIM_TENSOR_ELEMS_PER_BEAT        /* 16   */
#define ELEMS_PER_ROW   PIM_TENSOR_ELEMS_PER_ROW         /* 1024 */
#define BEATS_PER_ROW   (ELEMS_PER_ROW / ELEMS_PER_BEAT) /* 64   */

static char mv_err[300];

// The reduction slice as whole beats.  Rounding the START DOWN and the END UP is
// what makes COL and OPSIZE expressible at all: both count beats, so a range that
// began mid-beat could not be named.  In practice red_off is a head boundary and
// already aligned; it is red_len that is ragged.
static uint32_t slice_lo(uint32_t red_off)
{ return red_off / ELEMS_PER_BEAT * ELEMS_PER_BEAT; }

static uint32_t slice_hi(uint32_t red_off, uint32_t red_len)
{
    uint32_t end = red_off + red_len;
    return (end + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT * ELEMS_PER_BEAT;
}

uint32_t pim_matvec_beats(uint32_t red_off, uint32_t red_len)
{
    if (!red_len) return 0;
    return (slice_hi(red_off, red_len) - slice_lo(red_off)) / ELEMS_PER_BEAT;
}

// How many chunks the slice spans, and therefore how many (WRVEC, MAC) the inner
// loop runs.  A chunk is one DRAM row; crossing one is crossing into another unit.
static uint32_t slice_chunks(uint32_t red_off, uint32_t red_len)
{
    uint32_t lo, hi;
    if (!red_len) return 0;
    lo = slice_lo(red_off) / ELEMS_PER_ROW;
    hi = (slice_hi(red_off, red_len) - 1) / ELEMS_PER_ROW;
    return hi - lo + 1;
}

// The hoist condition, and it is exactly "one chunk" — see the file header.
static bool hoist_wrvec(uint32_t red_off, uint32_t red_len)
{ return slice_chunks(red_off, red_len) == 1; }

// Chunk `i` of the slice, as (absolute chunk index, beats into that row, beats).
static void slice_part(uint32_t red_off, uint32_t red_len, uint32_t i,
                       uint32_t *chunk, uint32_t *col, uint32_t *beats)
{
    uint32_t lo  = slice_lo(red_off);
    uint32_t hi  = slice_hi(red_off, red_len);
    uint32_t ck  = lo / ELEMS_PER_ROW + i;
    uint32_t beg = ck * ELEMS_PER_ROW;
    uint32_t end = beg + ELEMS_PER_ROW;

    if (beg < lo) beg = lo;
    if (end > hi) end = hi;
    *chunk = ck;
    *col   = (beg % ELEMS_PER_ROW) / ELEMS_PER_BEAT;
    *beats = (end - beg) / ELEMS_PER_BEAT;
}

// Elements of the vector consumed before chunk `i` — the GPR word the WRVEC starts
// at.  THE VECTOR IS INDEXED FROM red_off, NOT FROM ZERO: the GB is filled from
// beat 0 however far into the row COL reaches, so a Q vector for head 7 sits at
// word 0 of its buffer and not at word 7*D/16.  Getting this backwards is a wrong
// number with no error, which is why it is a function and not an expression.
static uint32_t slice_vec_word(uint32_t red_off, uint32_t red_len, uint32_t i)
{
    uint32_t lo = slice_lo(red_off), used = 0;
    for (uint32_t j = 0; j < i; j++) {
        uint32_t ck, col, beats;
        slice_part(red_off, red_len, j, &ck, &col, &beats);
        used += beats;
    }
    (void)lo;
    return used;      /* one GPR word is one beat: 32 B, 16 BF16 */
}

uint32_t pim_matvec_nwrvec(const pim_tensor *m, uint32_t out_count,
                           uint32_t red_off, uint32_t red_len, pim_acc_mode mode)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t nck    = slice_chunks(red_off, red_len);

    (void)m;
    if (!out_count || !nck) return 0;
    if (hoist_wrvec(red_off, red_len)) return 1;
    return (out_count + nlatch - 1) / nlatch * nck;
}

uint32_t pim_matvec_nisr(const pim_tensor *m, uint32_t out_count,
                         uint32_t red_off, uint32_t red_len, pim_acc_mode mode)
{
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t nck    = slice_chunks(red_off, red_len);
    uint32_t n      = 1;                              /* EOS */

    (void)m;
    if (!out_count || !nck) return n;
    n += pim_matvec_nwrvec(m, out_count, red_off, red_len, mode);
    for (uint32_t g0 = 0; g0 < out_count; g0 += nlatch) {
        uint32_t ng = out_count - g0 < nlatch ? out_count - g0 : nlatch;
        n += nck * ng          /* one MAC per group per chunk           */
           + ng;               /* ONE logical drain per group, not nch  */
    }
    return n;
}

const char *pim_matvec_logical(const pim_geometry *g, const pim_tensor *m,
                               uint32_t out_first, uint32_t out_count,
                               uint32_t red_off, uint32_t red_len,
                               const void *vgpr, size_t vbytes, uint64_t vtag,
                               const void *ygpr, size_t ybytes,
                               pim_acc_mode mode, pim_logical *out)
{
    const char *bad = pim_matvec_logical_part(g, m, out_first, out_count, red_off,
                                              red_len, vgpr, vbytes, vtag,
                                              ygpr, ybytes, mode, out);
    if (bad) return bad;
    return pim_matvec_eos(g, out);
}

const char *pim_matvec_eos(const pim_geometry *g, pim_logical *out)
{
    struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_EOS, (1u << g->nch) - 1u);
    struct emu_isr      isr;
    const char *bad;

    if ((bad = emu_isr_build(&isr, &s))) return bad;
    return pim_logical_push(out, (const pim_isr *)&isr);
}

const char *pim_matvec_logical_part(const pim_geometry *g, const pim_tensor *m,
                               uint32_t out_first, uint32_t out_count,
                               uint32_t red_off, uint32_t red_len,
                               const void *vgpr, size_t vbytes, uint64_t vtag,
                               const void *ygpr, size_t ybytes,
                               pim_acc_mode mode, pim_logical *out)
{
    uint32_t all_ch = (1u << g->nch) - 1u;
    uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t nck, hi;
    bool     hoist;
    const char *bad;

    if (!g || !m || !vgpr || !ygpr || !out)
        return "pim_matvec_logical: null argument";
    if (!red_len)  return "pim_matvec_logical: red_len is zero";
    if (!out_count) return NULL;                 /* nothing to do, legally */

    if (out_first + out_count > m->ngroups) {
        snprintf(mv_err, sizeof mv_err,
                 "pim_matvec_logical: output groups [%u, %u) run past the %u this "
                 "tensor has", out_first, out_first + out_count, m->ngroups);
        return mv_err;
    }
    hi = slice_hi(red_off, red_len);
    if (hi > m->nredpad) {
        snprintf(mv_err, sizeof mv_err,
                 "pim_matvec_logical: the reduction slice [%u, %u) rounds to %u "
                 "beats ending at %u, past the %u this tensor reserved",
                 red_off, red_off + red_len,
                 pim_matvec_beats(red_off, red_len), hi, m->nredpad);
        return mv_err;
    }
    nck   = slice_chunks(red_off, red_len);
    hoist = hoist_wrvec(red_off, red_len);

    // A hoisted WRVEC loads the WHOLE slice, because there is only one chunk.
    if (hoist) {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
        struct emu_isr      isr;
        uint32_t ck, col, beats;

        slice_part(red_off, red_len, 0, &ck, &col, &beats);
        s.opsize = beats;
        s.row    = 0;                    /* blank; the reference below fills it */
        if ((bad = emu_isr_build(&isr, &s))) return bad;
        if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                        PIM_REF_GPR_WORD, PIM_SPLIT_NONE,
                                        vgpr, vbytes, 0, vtag)))
            return bad;
    }

    for (uint32_t g0 = 0; g0 < out_count; g0 += nlatch) {
        uint32_t ng    = out_count - g0 < nlatch ? out_count - g0 : nlatch;
        uint32_t first = out->nisr;      /* this pass owns its latches from here */

        for (uint32_t i = 0; i < nck; i++) {
            uint32_t ck, col, beats;
            struct emu_isr_spec s;
            struct emu_isr      isr;

            slice_part(red_off, red_len, i, &ck, &col, &beats);

            if (!hoist) {
                s = emu_isr_default_ch(ISR_OP_WRVEC, all_ch);
                s.opsize = beats;
                s.row    = 0;
                if ((bad = emu_isr_build(&isr, &s))) return bad;
                if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                                PIM_REF_GPR_WORD, PIM_SPLIT_NONE,
                                                vgpr, vbytes,
                                                slice_vec_word(red_off, red_len, i),
                                                vtag)))
                    return bad;
            }

            for (uint32_t t = 0; t < ng; t++) {
                uint32_t unit = pim_tensor_unit(m, out_first + g0 + t, ck);

                s = emu_isr_default_ch(ISR_OP_MAC, all_ch);
                s.opsize     = beats;
                s.row        = 0;
                s.col        = col;       /* which operand inside the row */
                s.pu_mask    = (1u << g->nbank) - 1u;
                s.gb_mc_mask = s.pu_mask;
                if ((bad = emu_isr_build(&isr, &s))) return bad;
                if (t) emu_isr_set(&isr, ISR_F_T, 1u);
                if ((bad = pim_logical_push_ref(out, (const pim_isr *)&isr,
                                                PIM_REF_DRAM_UNIT, PIM_SPLIT_NONE,
                                                m->base, pim_tensor_bytes(g, m),
                                                unit, m->tag)))
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
        // for nch of them" meaning instead.
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

        if ((bad = pim_logical_atom(out, first, out->nisr - 1))) return bad;
    }

    return NULL;
}
