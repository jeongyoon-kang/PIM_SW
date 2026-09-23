// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_op.c — host pointer in, host pointer out.
//
// The header says what this is for.  What is worth saying beside the code is the
// order of the three things one op does:
//
//   1  PAD AND UPLOAD THE VECTOR.  red_len rounds up to a whole beat and the tail
//      is zeroed here.  Not an optimisation: a WRVEC reads whole beats and there is
//      no lane mask, so a stale lane that decodes as Inf turns every output of its
//      bank into NaN [measured, runtime/test/tensor_board].
//   2  BUILD, LOWER, RING.  pass 1 has no addresses in it; pass 2 puts them in.
//   3  UNPACK.  Word (group, channel), lane b -> output group*nch*nbank + ch*16 + b.
//
// WHAT DECIDES THAT THE LAUNCH FINISHED is EOS.  Every program ends with one and the
// dispatcher runs in order, so `done` rising means every RD_MAC before it has
// executed.  pim_exec_run treats a timeout as the error it is; there is nothing for
// this layer to re-check.
//
// THIS USED TO POISON THE RESULT WORDS and refuse a word that still held the
// sentinel.  It was removed because the case it claimed to catch, it does not: with
// ISR[35] undecoded both RD_MACs still execute and still write, so no word keeps the
// poison and only the ANSWER is wrong.  The other case — a CH_MASK fan-out that
// lowering failed to expand — is a software bug, and lower_test compares logical
// against direct ISR for ISR on the host with no board.
//
// SPLITTING IS BY SUPERGROUP AND NOWHERE ELSE.  A supergroup owns its accumulators
// from its first MAC to its RD_MAC; cutting anywhere inside that strands a running
// sum in a latch across a doorbell, and the next program accumulates on top of it.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pimrt/pim_op.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pim/pim_addr.h"
#include "emu_regs.h"

#define ELEMS_PER_BEAT  PIM_TENSOR_ELEMS_PER_BEAT

// CLOCK_MONOTONIC, the same clock pim_exec_run uses, so run_us and launch_us are
// comparable and their difference is a real interval rather than two clocks' skew.
static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

// Charge [since, now) to a counter and return now, so phases chain without a second
// variable per phase and without a gap between one ending and the next starting.
static uint64_t mark(uint64_t *acc, uint64_t since)
{
    uint64_t t = now_us();
    *acc += t - since;
    return t;
}

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

    /* ---- the batch being built, if any -------------------------------------
     * A stacked program keeps every piece's vector AND result words live at once,
     * so the two GPR buffers are carved by a bump pointer rather than reused. */
    uint32_t  max_batch;
    bool      batching;
    pim_acc_mode bmode;
    pim_logical lp;
    uint32_t  vword_used;      /* GPR words handed out of vgpr this program */
    uint32_t  yword_used;      /* ...and out of ygpr                        */
    struct pim_piece {
        uint16_t *y;           /* where the caller wants the answer         */
        uint32_t  out_count;
        uint32_t  yword;       /* words into ygpr this piece's results land  */
    } *piece;
    uint32_t  npiece;

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
    uint32_t cap = pim_exec_max_isrs(r->exec), lo = 0, hi;

    // BISECTED, because pim_matvec_nisr does not decrease as groups are added and
    // lm_head asks for four thousand of them — a linear walk did four thousand
    // calls of it on every op to answer a question whose answer has twelve bits.
    hi = want < r->max_launch_groups ? want : r->max_launch_groups;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo + 1) / 2;
        if (pim_matvec_nisr(NULL, mid, red_off, red_len, mode) > cap) hi = mid - 1;
        else lo = mid;
    }
    return lo;
}

const char *pim_rt_open(pim_ctx *c, const pim_rt_config *cfg, pim_rt **out)
{
    static char open_err[320];
    pim_rt_config def = { .max_red = 4096, .max_out_groups = 64, .max_batch = 1 };
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
    r->max_batch = cfg->max_batch ? cfg->max_batch : 1u;

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
        // COUNTED AS LOWERED, not as built.  pim_matvec_nisr is the logical count
        // and every drain becomes nch ISRs, so bounding the logical one leaves a
        // program that passes here and is refused at the doorbell.
        while (n < cfg->max_out_groups &&
               pim_matvec_nisr(NULL, n + 1, 0, redpad, PIM_ACC_SINGLE)
                 + (n + 1) * (g->nch ? g->nch - 1u : 0u) + 1u <= cap)
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

    // EVERY STACKED PIECE NEEDS ITS OWN WORDS, live for the whole launch.  What
    // the derived size assumes is that every piece is the largest one; a caller
    // stacking many small pieces says so instead and the bump allocator packs
    // them at their true size.  Raised, either way, to fit one largest piece —
    // below that an op would be impossible rather than merely unstacked.
    {
        size_t one_v = (size_t)redpad * 2;
        size_t one_y = (size_t)r->max_launch_groups * g->nch * EMU_WORD_BYTES;
        r->vbytes = cfg->vec_bytes ? cfg->vec_bytes : one_v * r->max_batch;
        r->ybytes = cfg->res_bytes ? cfg->res_bytes : one_y * r->max_batch;
        if (r->vbytes < one_v) r->vbytes = one_v;
        if (r->ybytes < one_y) r->ybytes = one_y;
    }
    r->vgpr = pim_alloc_ctx(c, r->vbytes, PIM_MEM_GPR);
    r->ygpr = pim_alloc_ctx(c, r->ybytes, PIM_MEM_GPR);

    // THE PROGRAM ARRAYS ARE BOUNDED BY IMEM, NOT BY max_batch.  A program that
    // does not fit the engine is split before it is built — add_one checks the
    // lowered count against pim_exec_max_isrs and submits what it has — so sizing
    // these as "every piece is the largest, times max_batch" reserves for a
    // program that can never be constructed.  At max_batch 8192 that was tens of
    // millions of ISRs of host memory for a 16383-instruction ceiling.
    isr_bound = pim_exec_max_isrs(r->exec);
    r->pisr_cap  = isr_bound;
    r->lisr_cap  = isr_bound;   /* lowering only ever grows a program */
    r->lref_cap  = isr_bound;
    // An atom is a first MAC through its RD_MAC, so it costs at least two ISRs.
    r->latom_cap = isr_bound / 2u + 2u;

    // vbuf stages ONE piece at a time (the upload happens inside add_one, before
    // the next piece is touched), so it does not scale with max_batch.  ybuf holds
    // the whole program's results, read back in one pread at submit, so it does.
    r->vbuf  = malloc((size_t)redpad * 2);
    r->ybuf  = malloc(r->ybytes);
    r->lisr  = malloc((size_t)r->lisr_cap  * sizeof *r->lisr);
    r->lref  = malloc((size_t)r->lref_cap  * sizeof *r->lref);
    r->latom = malloc((size_t)r->latom_cap * sizeof *r->latom);
    r->pisr  = malloc((size_t)r->pisr_cap  * sizeof *r->pisr);
    r->piece = malloc((size_t)r->max_batch * sizeof *r->piece);

    if (!r->vgpr || !r->ygpr || !r->vbuf || !r->ybuf || !r->lisr || !r->lref ||
        !r->latom || !r->pisr || !r->piece) {
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
    free(r->lisr); free(r->lref); free(r->latom); free(r->pisr); free(r->piece);
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

//////////////////////////////////////////////////////////////////////////////////
// §5  STACKING
//
// A doorbell costs an IMEM transfer, two MMIO writes and a poll loop, and that is
// the same whether eleven instructions are behind it or eleven thousand.  Attention
// pays it per HEAD — 32 launches a layer for programs of eleven instructions — so
// stacking the heads into one program changes not a single MAC, only how often the
// fixed cost is paid.
//
// WHAT MAKES IT MORE THAN A LOOP.  Every stacked piece keeps its vector AND its
// result words live for the whole launch: a single-piece program uploads its vector,
// rings, and reads the answer back before anything else touches the buffer, while
// eight stacked pieces have eight vectors resident at once.  Both GPR buffers are
// therefore carved with a bump pointer, and the host side of each piece's answer has
// to be remembered until submit.
//////////////////////////////////////////////////////////////////////////////////
const char *pim_op_begin(pim_rt *r, pim_acc_mode mode)
{
    if (!r) return "pim_op_begin: null argument";
    if (mode != PIM_ACC_SINGLE && mode != PIM_ACC_DUAL)
        return fail(r, "pim_op_begin: mode is neither SINGLE nor DUAL");
    if (mode == PIM_ACC_DUAL && !r->cfg.allow_t_latch)
        return fail(r, "PIM_ACC_DUAL needs pim_rt_config.allow_t_latch, which says "
                       "this image was shown to decode ISR[35].  Without that a "
                       "second-latch schedule reads back one group's partial sum "
                       "as both groups' answers, and nothing reports it");

    pim_logical_init(&r->lp, r->lisr, r->lisr_cap, r->lref, r->lref_cap,
                     r->latom, r->latom_cap, r->g->nch,
                     mode == PIM_ACC_DUAL ? 2u : 1u);
    r->bmode      = mode;
    r->batching   = true;
    r->vword_used = 0;
    r->yword_used = 0;
    r->npiece     = 0;
    return NULL;
}

const char *pim_op_submit(pim_rt *r)
{
    const pim_geometry *g;
    pim_prog   prog;
    pim_launch info;
    const char *bad;
    uint64_t   t;

    if (!r) return "pim_op_submit: null argument";
    if (!r->batching) return fail(r, "pim_op_submit: no program is open");
    g = r->g;
    r->batching = false;
    if (!r->npiece) return NULL;                  /* nothing was added; legal */

    t = now_us();
    if ((bad = pim_matvec_eos(g, &r->lp))) return bad;

    pim_prog_init(&prog, r->pisr, r->pisr_cap);
    if ((bad = pim_prog_lower_ctx(r->ctx, &r->lp, &prog,
                                  r->cfg.allow_t_latch ? PIM_LOWER_ALLOW_T : 0)))
        return bad;
    t = mark(&r->stat.low_us, t);
    if ((bad = pim_exec_run(r->exec, &prog, &info))) return bad;
    t = mark(&r->stat.run_us, t);

    r->stat.launch_us += info.us;
    r->stat.polls     += info.polls;
    r->stat.nisr      += prog.n;
    r->stat.nlaunch   += 1;

    // ONE READBACK FOR THE WHOLE PROGRAM.  The pieces' words are consecutive, so
    // what was one pread per launch stays one pread per launch however many pieces
    // share it — which is the other half of what stacking buys.
    if ((bad = pim_memcpy_ctx(r->ctx, r->ybuf, r->ygpr,
                              (size_t)r->yword_used * EMU_WORD_BYTES,
                              PIM_FROM_DEV, 0)))
        return bad;
    t = mark(&r->stat.back_us, t);

    for (uint32_t k = 0; k < r->npiece; k++) {
        const struct pim_piece *pc = &r->piece[k];
        // Word (group, channel), lane b -> output group*nch*nbank + ch*nbank + b.
        // The inverse of pim_tensor_offset's output mapping, written from the same
        // three lines so the two cannot drift.
        for (uint32_t sg = 0; sg < pc->out_count; sg++)
            for (uint32_t ch = 0; ch < g->nch; ch++)
                for (uint32_t b = 0; b < g->nbank; b++)
                    pc->y[(sg * g->nch + ch) * g->nbank + b] =
                        r->ybuf[(pc->yword + sg * g->nch + ch) * ELEMS_PER_BEAT + b];
    }
    mark(&r->stat.unpack_us, t);
    r->npiece = 0;
    return NULL;
}

// One piece that is KNOWN to fit behind a single doorbell.  pim_op_add below does
// the splitting; this only decides whether the CURRENT program has room.
static const char *add_one(pim_rt *r, const pim_tensor *m,
                           uint32_t out_first, uint32_t out_count,
                           uint32_t red_off, uint32_t red_len,
                           const uint16_t *v, uint16_t *y)
{
    const pim_geometry *g = r->g;
    uint32_t beats, redpad, vwords, ywords, want_isr;
    const char *bad;

    beats  = pim_matvec_beats(red_off, red_len);
    redpad = beats * ELEMS_PER_BEAT;
    vwords = beats;                               /* one GPR word is one beat */
    ywords = out_count * g->nch;

    if ((size_t)redpad * 2 > r->vbytes)
        return fail(r, "red_len %u rounds to %u elements (%zu B), past the %zu B "
                       "this rt reserved in GPR for a vector",
                    red_len, redpad, (size_t)redpad * 2, r->vbytes);
    // WOULD IT FIT?  Five ways it might not, and the answer to all five is to send
    // what we have and start again — refusing would make every caller write the
    // same retry, and the results are identical either way.
    //
    // IMEM IS THE ONE THAT IS NOT A HOST ARRAY, and checking only the arrays is how
    // the first version of this broke: lm_head splits into pieces that each fit a
    // launch, and with a large max_batch the host arrays were big enough to swallow
    // all of them, so they accumulated past the engine's 16383 and pim_exec_run
    // refused.  The count that matters is the LOWERED one, because a
    // PIM_SPLIT_PER_CHANNEL drain becomes nch instructions.
    want_isr = pim_matvec_nisr(m, out_count, red_off, red_len, r->bmode);
    if (r->npiece &&
        (r->npiece >= r->max_batch ||
         (size_t)(r->vword_used + vwords) * EMU_WORD_BYTES > r->vbytes ||
         (size_t)(r->yword_used + ywords) * EMU_WORD_BYTES > r->ybytes ||
         r->lp.nisr + want_isr + 1u > r->lisr_cap ||
         pim_lower_isr_count(&r->lp) + want_isr + out_count * (g->nch - 1u) + 1u
             > pim_exec_max_isrs(r->exec))) {
        pim_acc_mode mode = r->bmode;
        if ((bad = pim_op_submit(r))) return bad;
        if ((bad = pim_op_begin(r, mode))) return bad;
    }

    // ---- this piece's slice of the two GPR buffers -------------------------
    {
        char *vp = (char *)r->vgpr + (size_t)r->vword_used * EMU_WORD_BYTES;
        char *yp = (char *)r->ygpr + (size_t)r->yword_used * EMU_WORD_BYTES;

        // PAD AND UPLOAD.  The tail from red_len to the beat boundary is zeroed
        // here and this is the only place it happens: a WRVEC reads whole beats and
        // there is no lane mask, so a stale lane that decodes as Inf turns every
        // output of its bank into NaN [measured, runtime/test/tensor_board].
        uint64_t t = now_us();
        memset(r->vbuf, 0, (size_t)redpad * 2);
        memcpy(r->vbuf, v, (size_t)red_len * 2);
        t = mark(&r->stat.pad_us, t);
        if ((bad = pim_memcpy_ctx(r->ctx, vp, r->vbuf, (size_t)redpad * 2,
                                  PIM_TO_DEV, 0)))
            return bad;
        if ((bad = pim_tag_set_ctx(r->ctx, vp, vtag(beats)))) return bad;
        t = mark(&r->stat.vup_us, t);

        if ((bad = pim_matvec_logical_part(g, m, out_first, out_count,
                                           red_off, red_len,
                                           vp, (size_t)redpad * 2, vtag(beats),
                                           yp, (size_t)ywords * EMU_WORD_BYTES,
                                           r->bmode, &r->lp)))
            return bad;
        mark(&r->stat.gen_us, t);

        r->piece[r->npiece].y         = y;
        r->piece[r->npiece].out_count = out_count;
        r->piece[r->npiece].yword     = r->yword_used;
        r->npiece++;
        r->vword_used += vwords;
        r->yword_used += ywords;
    }
    r->stat.nwrvec += pim_matvec_nwrvec(m, out_count, red_off, red_len, r->bmode);
    return NULL;
}

const char *pim_op_add(pim_rt *r, const pim_tensor *m,
                       uint32_t out_first, uint32_t out_count,
                       uint32_t red_off, uint32_t red_len,
                       const uint16_t *v, uint16_t *y)
{
    uint32_t gpl;
    const char *bad;

    if (!r || !m || !v || !y) return "pim_op_add: null argument";
    if (!r->batching) return fail(r, "pim_op_add: call pim_op_begin first");
    if (!out_count) return NULL;
    if (!red_len)   return fail(r, "pim_op_add: red_len is zero");

    // SPLIT BY SUPERGROUP AND NOWHERE ELSE.  lm_head is 4008 supergroups and IMEM
    // holds 16383 instructions, so a piece this wide cannot be one program — but a
    // supergroup owns its accumulators from its first MAC to its RD_MAC, and cutting
    // inside that strands a running sum in a latch across a doorbell.
    gpl = groups_per_launch(r, red_off, red_len, r->bmode, out_count);
    if (!gpl)
        return fail(r, "not even one supergroup fits in %u ISRs at red_len %u",
                    pim_exec_max_isrs(r->exec), red_len);

    for (uint32_t g0 = 0; g0 < out_count; g0 += gpl) {
        uint32_t ng = out_count - g0 < gpl ? out_count - g0 : gpl;
        // Each sub-range writes its own slice of the caller's buffer.
        if ((bad = add_one(r, m, out_first + g0, ng, red_off, red_len, v,
                           y + (size_t)g0 * r->g->nch * r->g->nbank)))
            return bad;
    }
    return NULL;
}

// The single-piece call, which is begin + add + submit.  Still the right shape for
// an op that already fills IMEM; stacking buys nothing there.
const char *pim_op_matvec(pim_rt *r, const pim_tensor *m,
                          uint32_t out_first, uint32_t out_count,
                          uint32_t red_off, uint32_t red_len,
                          const uint16_t *v, uint16_t *y,
                          pim_acc_mode mode)
{
    const char *bad;

    if (!r) return "pim_op_matvec: null argument";
    if ((bad = pim_op_begin(r, mode))) return bad;
    if ((bad = pim_op_add(r, m, out_first, out_count, red_off, red_len, v, y))) {
        r->batching = false;
        return bad;
    }
    if ((bad = pim_op_submit(r))) return bad;
    r->stat.nop += 1;
    return NULL;
}
