// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_gemv.c — the schedules.  This file decides the ORDER of the two loops and
// how much of a GEMV goes behind one doorbell; it never touches a register.
//
// THE TWO LOOPS, AND WHY THE CHOICE IS NOT OBVIOUS
//   A GEMV is (output group g) x (K-chunk c).  There is exactly one accumulator
//   per bank, so only one output group can be ACCUMULATING at a time — but that
//   does not fix which loop is outer, because the accumulation can also be cut at
//   chunk boundaries and finished on the host.
//
//     group-outer   for g { for c { WRVEC(c); MAC(g,c) } RD_MAC(g) }
//                   G*C vector loads, G reads, ONE rounding per output
//     chunk-outer   for c { WRVEC(c); for g { MAC(g,c); RD_MAC(g,c) } }
//                   C vector loads, G*C reads, C roundings per output
//
//   A 64-beat WRVEC costs ~1.37 us and a 64-beat MAC ~0.82 us [measured], so the
//   G*C vector loads dominate group-outer.  Both were run on the board over
//   K = 2048..8192:
//     chunk-outer is 1.4x faster at G=64 and 1.63x at G=128, and its answer is at
//     most ONE BF16 ulp from group-outer's — never more, at any C tried.
//     group-outer is BIT-IDENTICAL to pim_gemv_golden().
//
//   So: PIM_FAST is chunk-outer and is the default; PIM_EXACT is group-outer and
//   exists because being able to reproduce the device exactly is what separates
//   "the hardware is wrong" from "the schedule rounded".  On a machine with no
//   validity gate, giving that up would be a bad trade.
//
// SPLITTING
//   A GEMV larger than pim_config.max_isrs is issued as several launches, cut on
//   OUTPUT GROUP boundaries.  A group boundary is the right cut: every launch then
//   ends with its accumulators drained, so no state crosses the seam — which
//   matters because the accumulator is cleared only by RD_MAC and by reset.
//
//   The split is nearly free.  The IMEM bytes are the same either way; only the
//   per-doorbell fixed cost (~14 us) is duplicated, so cutting an 8003-ISR program
//   into five costs about 56 us — under a tenth of a percent of a decode step.
//////////////////////////////////////////////////////////////////////////////////
#include "pim_internal.h"

#include <stdlib.h>

// Where chunk c's slice of the vector sits in the GPR.  Chunks are laid out
// 64 words apart whatever their OPSIZE, so this is just an index — and the last
// chunk simply uses fewer of its 64 words.
static uint32_t vec_word_of(const pim_dev *d, uint32_t chunk)
{
    return d->gpr_vec_base + chunk * PIM_BEATS_PER_ROW;
}

// x -> GPR, padded to whole beats with zeros.  A zero element contributes exactly
// 0.0f to the sum in any order, so the padding cannot move the result.
static const char *upload_vector(pim_dev *d, const pim_weight *w, const uint16_t *x)
{
    const uint32_t nwords = (w->nchunks - 1) * PIM_BEATS_PER_ROW + w->last_opsize;
    if (nwords > d->gpr_vec_words)
        return pim_err(d, "k=%u needs %u GPR words of vector staging, but only %u are "
                          "reserved (pim_config.gpr_vec_words)", w->k, nwords,
                       d->gpr_vec_words);
    uint16_t *buf = malloc((size_t)nwords * PIM_LANES * sizeof *buf);
    if (!buf) return pim_err(d, "out of memory staging the vector");
    memset(buf, 0, (size_t)nwords * PIM_LANES * sizeof *buf);
    memcpy(buf, x, (size_t)w->k * sizeof *x);
    // Element k is the 1.0 that turns the weight's bias column into a bias.
    if (w->has_bias) buf[w->k] = pim_f32_to_bf16(1.0f);
    const char *e = pim_gpr_write(d, d->gpr_vec_base, buf, nwords);
    free(buf);
    return e;
}

// ---- a job: several weights sharing one x, viewed as one flat group list ------
struct job {
    const pim_weight *const *ws;
    unsigned                 nw;
    uint16_t *const         *ys;
    uint32_t                 total_groups;
};

static void decompose(const struct job *j, uint32_t gi, unsigned *wi, uint32_t *g)
{
    for (unsigned i = 0; i < j->nw; i++) {
        if (gi < j->ws[i]->ngroups) { *wi = i; *g = gi; return; }
        gi -= j->ws[i]->ngroups;
    }
    *wi = 0; *g = 0;               // unreachable: callers bound gi by total_groups
}

// Scatter one RD_MAC word (16 lanes = 16 consecutive outputs) straight into y.
// Used by group-outer, where the device already did the whole sum.
static void scatter(const struct job *j, uint32_t gi, const uint16_t *lane)
{
    unsigned wi; uint32_t g;
    decompose(j, gi, &wi, &g);
    const pim_weight *w = j->ws[wi];
    uint16_t *y = j->ys[wi];
    for (unsigned b = 0; b < PIM_BANKS; b++) {
        uint32_t o = g * PIM_BANKS + b;
        if (o >= w->n) continue;                       // padded output row
        y[o] = lane[b];
    }
}

// The host half of chunk-outer.  Each chunk's partial arrives ALREADY rounded to
// BF16 — that rounding is the price of the schedule and cannot be avoided.  What
// CAN be avoided is doing it again on the host, and that is not a small detail:
// accumulating the partials in BF16 instead of fp32 doubles the number of
// roundings and takes the error from at most 1 ulp to 2 [caught by pim_test at
// C = 4].  So the partials go into an fp32 accumulator and are rounded ONCE.
static void accumulate(float *acc, uint32_t slot, const uint16_t *lane)
{
    for (unsigned b = 0; b < PIM_BANKS; b++)
        acc[(size_t)slot * PIM_BANKS + b] += pim_bf16_to_f32(lane[b]);
}

static void finish(const struct job *j, uint32_t gi, uint32_t slot, const float *acc)
{
    unsigned wi; uint32_t g;
    decompose(j, gi, &wi, &g);
    const pim_weight *w = j->ws[wi];
    uint16_t *y = j->ys[wi];
    for (unsigned b = 0; b < PIM_BANKS; b++) {
        uint32_t o = g * PIM_BANKS + b;
        if (o >= w->n) continue;
        y[o] = pim_f32_to_bf16(acc[(size_t)slot * PIM_BANKS + b]);
    }
}

// ---- chunk-outer -------------------------------------------------------------
static const char *run_fast(pim_dev *d, const struct job *j, uint32_t C)
{
    // C + Gs*C*2 + 1 <= cap
    if (d->max_isrs < C + 2 * C + 1)
        return pim_err(d, "max_isrs %u cannot hold even one output group at %u chunks",
                       d->max_isrs, C);
    const uint32_t per = (d->max_isrs - C - 1) / (2 * C);

    for (uint32_t lo = 0; lo < j->total_groups; lo += per) {
        uint32_t hi = lo + per;
        if (hi > j->total_groups) hi = j->total_groups;
        const uint32_t gs = hi - lo, nres = gs * C;

        struct pim_prog p;
        pim_prog_init(&p, d);
        const char *e;
        for (uint32_t c = 0; c < C; c++) {
            const uint32_t op = pim_chunk_opsize(j->ws[0], c);
            if ((e = pim_emit_wrvec(&p, vec_word_of(d, c), op))) return e;
            for (uint32_t gi = lo; gi < hi; gi++) {
                unsigned wi; uint32_t g, row, col;
                decompose(j, gi, &wi, &g);
                if ((e = pim_place_of(d, j->ws[wi], g, c, &row, &col))) return e;
                if ((e = pim_emit_mac(&p, row, col, op))) return e;
                if ((e = pim_emit_rd_mac(&p, d->gpr_res_base + c * gs + (gi - lo)))) return e;
            }
        }
        if ((e = pim_emit_eos(&p))) return e;
        if ((e = pim_launch(d, &p, d->gpr_res_base, nres, d->res))) return e;

        memset(d->acc, 0, (size_t)gs * PIM_BANKS * sizeof *d->acc);
        for (uint32_t c = 0; c < C; c++)
            for (uint32_t gi = lo; gi < hi; gi++)
                accumulate(d->acc, gi - lo,
                           d->res + (size_t)(c * gs + (gi - lo)) * PIM_LANES);
        for (uint32_t gi = lo; gi < hi; gi++) finish(j, gi, gi - lo, d->acc);
    }
    return NULL;
}

// ---- group-outer -------------------------------------------------------------
static const char *run_exact(pim_dev *d, const struct job *j, uint32_t C)
{
    if (d->max_isrs < 2 * C + 2)
        return pim_err(d, "max_isrs %u cannot hold even one output group at %u chunks",
                       d->max_isrs, C);
    // Worst case per group is still C WRVEC + C MAC + 1 RD_MAC; hoisting only ever
    // emits fewer, so this bound stays safe.
    const uint32_t per = (d->max_isrs - 1) / (2 * C + 1);

    for (uint32_t lo = 0; lo < j->total_groups; lo += per) {
        uint32_t hi = lo + per;
        if (hi > j->total_groups) hi = j->total_groups;
        const uint32_t gs = hi - lo;

        struct pim_prog p;
        pim_prog_init(&p, d);
        const char *e;
        // Which chunk the GB currently holds.  A WRVEC is only worth issuing when
        // the vector actually has to change: the GB rewinds at the start of every
        // GB-sourced MAC, so it keeps serving until something overwrites it.
        //
        // With C > 1 the chunk alternates every MAC and this saves nothing — that
        // is the real cost of group-outer.  With C == 1 the vector never changes
        // at all, and without this the schedule reloads it once PER OUTPUT GROUP:
        // 9496 redundant 64-beat WRVECs for a [151936][1024] lm_head, about 13 ms
        // of pure vector traffic per token.  At C == 1 the two schedules are the
        // same program and must cost the same; they now do.
        //
        // -1 at the top of every launch, not once per gemv: a program starts with
        // the GB holding whatever the PREVIOUS program left, which may be another
        // caller's.
        int32_t gb_chunk = -1;
        for (uint32_t gi = lo; gi < hi; gi++) {
            unsigned wi; uint32_t g, row, col;
            decompose(j, gi, &wi, &g);
            for (uint32_t c = 0; c < C; c++) {
                const uint32_t op = pim_chunk_opsize(j->ws[0], c);
                if ((int32_t)c != gb_chunk) {
                    if ((e = pim_emit_wrvec(&p, vec_word_of(d, c), op))) return e;
                    gb_chunk = (int32_t)c;
                }
                if ((e = pim_place_of(d, j->ws[wi], g, c, &row, &col))) return e;
                if ((e = pim_emit_mac(&p, row, col, op))) return e;
            }
            if ((e = pim_emit_rd_mac(&p, d->gpr_res_base + (gi - lo)))) return e;
        }
        if ((e = pim_emit_eos(&p))) return e;
        if ((e = pim_launch(d, &p, d->gpr_res_base, gs, d->res))) return e;

        for (uint32_t gi = lo; gi < hi; gi++)
            scatter(j, gi, d->res + (size_t)(gi - lo) * PIM_LANES);
    }
    return NULL;
}

// ---- public ------------------------------------------------------------------
const char *pim_gemv_multi(pim_dev *d, const pim_weight *const *ws, unsigned nw,
                           const uint16_t *x, uint16_t *const *ys, pim_mode mode)
{
    if (nw == 0) return pim_err(d, "pim_gemv_multi with no weights");
    for (unsigned i = 1; i < nw; i++)
        if (ws[i]->k != ws[0]->k || ws[i]->has_bias != ws[0]->has_bias)
            return pim_err(d, "weight %u is k=%u bias=%d and weight 0 is k=%u bias=%d; "
                              "they cannot share a vector", i, ws[i]->k, ws[i]->has_bias,
                           ws[0]->k, ws[0]->has_bias);

    struct job j = { .ws = ws, .nw = nw, .ys = ys, .total_groups = 0 };
    for (unsigned i = 0; i < nw; i++) {
        if (ws[i]->buf.size == 0)
            return pim_err(d, "weight %u has no allocation", i);
        j.total_groups += ws[i]->ngroups;
    }

    const char *e = upload_vector(d, ws[0], x);
    if (e) return e;

    const uint32_t C = ws[0]->nchunks;
    e = (mode == PIM_EXACT) ? run_exact(d, &j, C) : run_fast(d, &j, C);
    if (e) {
        // A failed launch leaves the accumulators in an unknown state.  Clearing
        // them here means the NEXT call starts clean, so one bad kernel does not
        // quietly corrupt everything after it.
        //
        // The message has to be saved first: pim_scrub() runs a program of its own
        // and will overwrite d->err, which is very often exactly where `e` points.
        char keep[PIM_ERRLEN];
        snprintf(keep, sizeof keep, "%s", e);
        pim_scrub(d);
        snprintf(d->err, PIM_ERRLEN, "%s", keep);
        return d->err;
    }
    return NULL;
}

// Emit exactly what run_fast / run_exact would emit for the FIRST launch, and
// print it instead of running it.  Kept next to them on purpose: if the schedules
// change and this does not, the two drift and the dump starts lying.
const char *pim_gemv_dump(pim_dev *d, const pim_weight *w, pim_mode mode,
                          FILE *out, unsigned max_words)
{
    const uint32_t C = w->nchunks;
    const uint32_t G = w->ngroups;
    const bool exact = (mode == PIM_EXACT);
    const uint32_t per = exact ? (d->max_isrs - 1) / (2 * C + 1)
                              : (d->max_isrs - C - 1) / (2 * C);
    const uint32_t gs = G < per ? G : per;

    fprintf(out, "\n%s  [%u][%u] -> %u groups x %u chunks, %u group%s per page, "
                 "pages %u..%u, OPSIZE %u..%u\n",
            exact ? "PIM_EXACT (group-outer)" : "PIM_FAST (chunk-outer)",
            w->n, w->k, G, C, w->groups_per_row, w->groups_per_row == 1 ? "" : "s",
            (unsigned)(w->buf.off / PIM_ROBACO_ROW),
            (unsigned)((w->buf.off + w->buf.size - 1) / PIM_ROBACO_ROW),
            w->full_opsize, w->last_opsize);
    fprintf(out, "  this launch covers output groups 0..%u of %u  ->  y[0..%u]\n",
            gs - 1, G, gs * PIM_BANKS - 1);

    struct pim_prog p;
    pim_prog_init(&p, d);
    const char *e;
    if (exact) {
        for (uint32_t g = 0; g < gs; g++) {
            for (uint32_t c = 0; c < C; c++) {
                const uint32_t op = pim_chunk_opsize(w, c);
                uint32_t row, col;
                if ((e = pim_emit_wrvec(&p, vec_word_of(d, c), op))) return e;
                if ((e = pim_place_of(d, w, g, c, &row, &col))) return e;
                if ((e = pim_emit_mac(&p, row, col, op))) return e;
            }
            if ((e = pim_emit_rd_mac(&p, d->gpr_res_base + g))) return e;
        }
    } else {
        for (uint32_t c = 0; c < C; c++) {
            const uint32_t op = pim_chunk_opsize(w, c);
            if ((e = pim_emit_wrvec(&p, vec_word_of(d, c), op))) return e;
            for (uint32_t g = 0; g < gs; g++) {
                uint32_t row, col;
                if ((e = pim_place_of(d, w, g, c, &row, &col))) return e;
                if ((e = pim_emit_mac(&p, row, col, op))) return e;
                if ((e = pim_emit_rd_mac(&p, d->gpr_res_base + c * gs + g))) return e;
            }
        }
    }
    if ((e = pim_emit_eos(&p))) return e;
    if ((e = pim_prog_verify(&p))) return e;
    fprintf(out, "  %u ISRs, one doorbell\n", p.n);
    pim_prog_print(&p, out, max_words);
    return NULL;
}

const char *pim_gemv(pim_dev *d, const pim_weight *w, const uint16_t *x,
                     uint16_t *y, pim_mode mode)
{
    const pim_weight *ws[1] = { w };
    uint16_t *ys[1] = { y };
    return pim_gemv_multi(d, ws, 1, x, ys, mode);
}
