// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_op.c — host pointer in, host pointer out.
//
// The header says what this is for.  What is worth saying beside the code is the
// order of the five things one op does, because three of them are only there to
// catch a failure that is otherwise silent:
//
//   1  PAD AND UPLOAD THE VECTOR.  red_len rounds up to a whole beat and the tail
//      is zeroed here.  Not an optimisation: a WRVEC reads whole beats and there is
//      no lane mask, so a stale lane that decodes as Inf turns every output of its
//      bank into NaN [measured, runtime/test/tensor_board].
//   2  POISON THE RESULT WORDS.  Before the launch, not after.
//   3  BUILD, LOWER, RING.  pass 1 has no addresses in it; pass 2 puts them in.
//   4  CHECK THE POISON IS GONE.  `done` is a held level and can be stale from a
//      previous run, so it decides nothing.  A word that still holds the poison had
//      no RD_MAC land on it, which is what a mis-fanned CH_MASK or an unselected
//      latch looks like — and the only other symptom would be a plausible number.
//   5  UNPACK.  Word (group, channel), lane b -> output group*nch*nbank + ch*16 + b.
//
// SPLITTING IS BY SUPERGROUP AND NOWHERE ELSE.  A supergroup owns its accumulators
// from its first MAC to its RD_MAC; cutting anywhere inside that strands a running
// sum in a latch across a doorbell, and the next program accumulates on top of it.
//////////////////////////////////////////////////////////////////////////////////
#include "pimrt/pim_op.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim/pim_addr.h"
#include "emu_regs.h"

#define ELEMS_PER_BEAT  PIM_TENSOR_ELEMS_PER_BEAT
#define POISON_LANE     0xDEAD

struct pim_rt {
    pim_ctx  *ctx;
    pim_exec *exec;
    const pim_geometry *g;
    pim_rt_config cfg;

    void     *vgpr;      /* the vector, one allocation reused by every op    */
    void     *ygpr;      /* result words for one launch                      */
    size_t    vbytes, ybytes;
    uint32_t  max_launch_groups;

    uint16_t *vbuf;      /* host staging: padded vector, then result words   */
    uint16_t *ybuf;

    pim_isr  *lisr;      /* pass 1 storage                                   */
    pim_ref  *lref;
    pim_atom *latom;
    uint32_t  lisr_cap, lref_cap, latom_cap;
    pim_isr  *pisr;      /* the lowered program                              */
    uint32_t  pisr_cap;

    pim_rt_stat stat;
    char      err[320];
};

static const char *fail(pim_rt *r, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r->err, sizeof r->err, fmt, ap);
    va_end(ap);
    return r->err;
}

// How many supergroups fit behind one doorbell.  Counted rather than solved: the
// ISR count is a small closed form but it has a hoist in it, and a formula that
// disagreed with pim_matvec_nisr by one would overrun IMEM only on certain shapes.
static uint32_t groups_per_launch(const pim_rt *r, uint32_t red_off, uint32_t red_len,
                                  pim_acc_mode mode, uint32_t want)
{
    uint32_t cap = pim_exec_max_isrs(r->exec), n = 0;

    while (n < want) {
        uint32_t next = n + 1;
        if (pim_matvec_nisr(NULL, next, red_off, red_len, mode) > cap) break;
        if (next > r->max_launch_groups) break;
        n = next;
    }
    return n;
}

const char *pim_rt_open(pim_ctx *c, const pim_rt_config *cfg, pim_rt **out)
{
    static char open_err[320];
    pim_rt_config def = { .max_red = 4096, .max_out_groups = 64 };
    pim_rt *r;
    const pim_geometry *g;
    uint32_t redpad, isr_bound;

    if (!c || !out) return "pim_rt_open: null argument";
    *out = NULL;
    g = pim_geom_ctx(c);
    if (!g) return "pim_rt_open: no geometry";
    if (!cfg) cfg = &def;
    if (!cfg->max_red || !cfg->max_out_groups)
        return "pim_rt_open: max_red and max_out_groups must both be nonzero";

    r = calloc(1, sizeof *r);
    if (!r) return "out of host memory for a pim_rt";
    r->ctx = c;
    r->g   = g;
    r->cfg = *cfg;

    {
        pim_exec_config ec = { 0 };
        const char *bad = pim_exec_open(c, &ec, &r->exec);
        if (bad) { free(r); snprintf(open_err, sizeof open_err, "%s", bad); return open_err; }
    }

    // A launch is capped by IMEM as well as by what the caller declared, so the
    // scratch only ever needs the smaller of the two.  Sized with the LONGEST
    // reduction, which is the case that fits fewest groups.
    redpad = (cfg->max_red + ELEMS_PER_BEAT - 1) / ELEMS_PER_BEAT * ELEMS_PER_BEAT;
    r->max_launch_groups = cfg->max_out_groups;
    {
        uint32_t cap = pim_exec_max_isrs(r->exec), n = 0;
        while (n < cfg->max_out_groups &&
               pim_matvec_nisr(NULL, n + 1, 0, redpad, PIM_ACC_SINGLE) <= cap)
            n++;
        if (!n) {
            pim_exec_close(r->exec); free(r);
            snprintf(open_err, sizeof open_err,
                     "not even one supergroup fits in %u ISRs at max_red = %u; the "
                     "engine's program limit is too low for this session", cap,
                     cfg->max_red);
            return open_err;
        }
        r->max_launch_groups = n;
    }

    r->vbytes = (size_t)redpad * 2;
    r->ybytes = (size_t)r->max_launch_groups * g->nch * EMU_WORD_BYTES;
    r->vgpr = pim_alloc_ctx(c, r->vbytes, PIM_MEM_GPR);
    r->ygpr = pim_alloc_ctx(c, r->ybytes, PIM_MEM_GPR);

    isr_bound = pim_matvec_nisr(NULL, r->max_launch_groups, 0, redpad, PIM_ACC_SINGLE);
    r->lisr_cap  = isr_bound;
    r->lref_cap  = isr_bound;
    r->latom_cap = r->max_launch_groups + 1;
    // Every PIM_SPLIT_PER_CHANNEL reference can become nch ISRs, and only the drains
    // are split — one per supergroup per pass.
    r->pisr_cap  = isr_bound + r->max_launch_groups * (g->nch ? g->nch - 1 : 0);

    r->vbuf  = malloc(r->vbytes);
    r->ybuf  = malloc(r->ybytes);
    r->lisr  = malloc((size_t)r->lisr_cap  * sizeof *r->lisr);
    r->lref  = malloc((size_t)r->lref_cap  * sizeof *r->lref);
    r->latom = malloc((size_t)r->latom_cap * sizeof *r->latom);
    r->pisr  = malloc((size_t)r->pisr_cap  * sizeof *r->pisr);

    if (!r->vgpr || !r->ygpr || !r->vbuf || !r->ybuf || !r->lisr || !r->lref ||
        !r->latom || !r->pisr) {
        snprintf(open_err, sizeof open_err,
                 "pim_rt_open: %s (vector %zu B, results %zu B, %u ISRs)",
                 (!r->vgpr || !r->ygpr) ? pim_last_error_ctx(c) : "out of host memory",
                 r->vbytes, r->ybytes, isr_bound);
        pim_rt_close(r);
        return open_err;
    }
    *out = r;
    return NULL;
}

void pim_rt_close(pim_rt *r)
{
    if (!r) return;
    if (r->vgpr) pim_free_ctx(r->ctx, r->vgpr);
    if (r->ygpr) pim_free_ctx(r->ctx, r->ygpr);
    free(r->vbuf); free(r->ybuf);
    free(r->lisr); free(r->lref); free(r->latom); free(r->pisr);
    if (r->exec) pim_exec_close(r->exec);
    free(r);
}

pim_exec *pim_rt_exec(pim_rt *r) { return r ? r->exec : NULL; }
pim_ctx  *pim_rt_ctx (pim_rt *r) { return r ? r->ctx  : NULL; }

void pim_rt_stat_get(const pim_rt *r, pim_rt_stat *out)
{ if (r && out) *out = r->stat; }
void pim_rt_stat_reset(pim_rt *r)
{ if (r) memset(&r->stat, 0, sizeof r->stat); }

uint32_t pim_op_outputs(const pim_rt *r, uint32_t out_count)
{ return r ? out_count * r->g->nch * r->g->nbank : 0; }

// The vector's own layout contract.  The rt zeroes the tail every time, so this is
// not catching the rt — it is what lowering compares against, and it keeps the
// vector reference on the same footing as the tensor's.  A caller who one day passes
// their own GPR buffer will need to stamp the same thing.
static uint64_t vtag(uint32_t beats)
{
    uint64_t h = 0xcbf29ce484222325ull;
    h ^= 0x6f707665u; h *= 0x100000001b3ull;      /* 'opve' */
    h ^= beats;       h *= 0x100000001b3ull;
    h ^= ELEMS_PER_BEAT; h *= 0x100000001b3ull;
    return h ? h : 1;
}

const char *pim_op_matvec(pim_rt *r, const pim_tensor *m,
                          uint32_t out_first, uint32_t out_count,
                          uint32_t red_off, uint32_t red_len,
                          const uint16_t *v, uint16_t *y,
                          pim_acc_mode mode)
{
    const pim_geometry *g;
    uint32_t beats, redpad, gpl, vword, vn, yword, yn;
    const char *bad;

    if (!r || !m || !v || !y) return "pim_op_matvec: null argument";
    if (!out_count) return NULL;
    if (mode != PIM_ACC_SINGLE && mode != PIM_ACC_DUAL)
        return fail(r, "pim_op_matvec: mode is neither SINGLE nor DUAL");
    if (mode == PIM_ACC_DUAL && !r->cfg.allow_t_latch)
        return fail(r, "PIM_ACC_DUAL needs pim_rt_config.allow_t_latch, which says "
                       "this image was shown to decode ISR[35].  Without that a "
                       "second-latch schedule reads back one group's partial sum "
                       "as both groups' answers, and nothing reports it");
    g = r->g;

    beats  = pim_matvec_beats(red_off, red_len);
    redpad = beats * ELEMS_PER_BEAT;
    if ((size_t)redpad * 2 > r->vbytes)
        return fail(r, "red_len %u rounds to %u elements, past the %zu this rt was "
                       "opened for (max_red)", red_len, redpad, r->vbytes / 2);
    if (out_count > r->cfg.max_out_groups)
        return fail(r, "%u output supergroups, past the %u this rt was opened for "
                       "(max_out_groups)", out_count, r->cfg.max_out_groups);

    // ---- 1. pad and upload the vector --------------------------------------
    // The tail from red_len to the beat boundary is zeroed HERE and this is the
    // only place it happens.  See the file header for why it is not optional.
    memset(r->vbuf, 0, (size_t)redpad * 2);
    memcpy(r->vbuf, v, (size_t)red_len * 2);
    if ((bad = pim_memcpy_ctx(r->ctx, r->vgpr, r->vbuf, (size_t)redpad * 2,
                              PIM_TO_DEV, 0)))
        return bad;
    if ((bad = pim_tag_set_ctx(r->ctx, r->vgpr, vtag(beats)))) return bad;

    if ((bad = pim_addr_gpr_words_ctx(r->ctx, r->vgpr, (size_t)redpad * 2,
                                      &vword, &vn))) return bad;
    (void)vword; (void)vn;

    gpl = groups_per_launch(r, red_off, red_len, mode, out_count);
    if (!gpl)
        return fail(r, "not even one supergroup fits in %u ISRs at red_len %u",
                    pim_exec_max_isrs(r->exec), red_len);

    for (uint32_t g0 = 0; g0 < out_count; g0 += gpl) {
        uint32_t ng = out_count - g0 < gpl ? out_count - g0 : gpl;
        uint32_t nw = ng * g->nch;
        pim_logical lp;
        pim_prog    prog;
        pim_launch  info;

        if ((bad = pim_addr_gpr_words_ctx(r->ctx, r->ygpr,
                                          (size_t)nw * EMU_WORD_BYTES,
                                          &yword, &yn))) return bad;
        (void)yword; (void)yn;

        // ---- 2. poison, before the launch ----------------------------------
        for (size_t i = 0; i < (size_t)nw * ELEMS_PER_BEAT; i++)
            r->ybuf[i] = POISON_LANE;
        if ((bad = pim_memcpy_ctx(r->ctx, r->ygpr, r->ybuf,
                                  (size_t)nw * EMU_WORD_BYTES, PIM_TO_DEV, 0)))
            return bad;

        // ---- 3. build, lower, ring -----------------------------------------
        pim_logical_init(&lp, r->lisr, r->lisr_cap, r->lref, r->lref_cap,
                         r->latom, r->latom_cap, g->nch,
                         mode == PIM_ACC_DUAL ? 2u : 1u);
        if ((bad = pim_matvec_logical(g, m, out_first + g0, ng, red_off, red_len,
                                      r->vgpr, (size_t)redpad * 2, vtag(beats),
                                      r->ygpr, (size_t)nw * EMU_WORD_BYTES,
                                      mode, &lp)))
            return bad;
        pim_prog_init(&prog, r->pisr, r->pisr_cap);
        if ((bad = pim_prog_lower_ctx(r->ctx, &lp, &prog,
                                      r->cfg.allow_t_latch ? PIM_LOWER_ALLOW_T : 0)))
            return bad;
        if ((bad = pim_exec_run(r->exec, &prog, &info))) return bad;

        r->stat.launch_us += info.us;
        r->stat.polls     += info.polls;
        r->stat.nisr      += prog.n;
        r->stat.nwrvec    += pim_matvec_nwrvec(m, ng, red_off, red_len, mode);
        r->stat.nlaunch   += 1;

        // ---- 4. the poison decides, not `done` -----------------------------
        if ((bad = pim_memcpy_ctx(r->ctx, r->ybuf, r->ygpr,
                                  (size_t)nw * EMU_WORD_BYTES, PIM_FROM_DEV, 0)))
            return bad;
        for (uint32_t i = 0; i < nw; i++) {
            bool landed = false;
            for (uint32_t l = 0; l < ELEMS_PER_BEAT; l++)
                if (r->ybuf[i * ELEMS_PER_BEAT + l] != POISON_LANE) { landed = true; break; }
            if (!landed)
                return fail(r, "result word %u (supergroup %u, channel %u) still "
                               "holds the poison after %llu us (%u polls, done %s). "
                               "That RD_MAC never wrote — check CH_MASK fan-out, or "
                               "ISR[35] if this was a dual-latch run.",
                            i, out_first + g0 + i / g->nch, i % g->nch,
                            (unsigned long long)info.us, info.polls,
                            info.saw_done ? "rose" : "never rose");
        }

        // ---- 5. unpack ------------------------------------------------------
        // The inverse of pim_tensor_offset's output mapping: bank = out % nbank,
        // ch = (out/nbank) % nch, group = out / (nbank*nch).  Written from the same
        // three lines so the two cannot drift.
        for (uint32_t sg = 0; sg < ng; sg++)
            for (uint32_t ch = 0; ch < g->nch; ch++)
                for (uint32_t b = 0; b < g->nbank; b++)
                    y[((g0 + sg) * g->nch + ch) * g->nbank + b] =
                        r->ybuf[(sg * g->nch + ch) * ELEMS_PER_BEAT + b];
    }
    r->stat.nop += 1;
    return NULL;
}
