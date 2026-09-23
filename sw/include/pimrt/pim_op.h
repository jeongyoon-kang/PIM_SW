/* SPDX-License-Identifier: MIT */
/*
 * pim_op.h — the surface a host language binds to.
 *
 * Everything below this header speaks in allocations, ISRs and GPR words.  A caller
 * coming from PyTorch has a pointer to some BF16 and wants a pointer to some BF16
 * back, and does not want to learn any of that.  So:
 *
 *     y = pim_op_matvec(rt, M, out range, red range, v)
 *
 * with v and y ORDINARY HOST MEMORY — torch's data_ptr() goes straight in.  The GPR
 * staging, the vector's zero padding, pass 1, lowering, the doorbell, the polling,
 * splitting across doorbells when IMEM is the limit, and the readback all happen
 * inside.
 *
 * ONE OP, NOT THREE.  pim_matvec_logical showed that a linear layer, a Q.K^T and an
 * S.V are the same operation with different arguments; adding pim_op_linear,
 * pim_op_attn_qk and pim_op_attn_sv here would put the model's vocabulary into C
 * for no gain, and three names that differ only in argument order are three places
 * for the argument order to be wrong.  The names belong in the Python layer, where
 * the model does.
 *
 * WHAT pim_rt IS FOR.  A launch must not allocate: the GPR buffers, the ISR arrays
 * and the reference tables are all sized by the shapes a session will use, and
 * asking malloc for them per token would put the allocator on the critical path for
 * no reason.  pim_rt owns them, sized once at open from the largest shape you
 * declare.  It is the compute-side sibling of pim_ctx, which owns the memory.
 *
 * SINGLE THREADED, like pim_ctx.  One board, one command stream, one doorbell.
 */
#ifndef PIM_OP_H
#define PIM_OP_H

#include <stdbool.h>
#include <stdint.h>

#include "pim/pim.h"
#include "pimrt/pim_exec.h"
#include "pimrt/pim_matvec.h"
#include "pimrt/pim_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pim_rt pim_rt;

typedef struct {
	/* The largest shape this session will ask for.  Both are used to size the
	 * scratch at open, and both are CHECKED at every call — an op that exceeds
	 * what was declared is refused rather than quietly reallocating, because a
	 * realloc on the launch path is the thing this exists to avoid.
	 *
	 * For a transformer: max_red is the largest of (hidden, intermediate, S_max),
	 * and max_out_groups is the largest out_features divided by nch*nbank. */
	uint32_t max_red;          /* elements on the reduction axis          */
	uint32_t max_out_groups;   /* supergroups of outputs in ONE op        */

	/* HOW MANY PIECES MAY BE STACKED INTO ONE PROGRAM.  1 keeps the old
	 * behaviour: every op is its own doorbell.
	 *
	 * A DOORBELL COSTS THE SAME WHATEVER IS BEHIND IT — an IMEM transfer, two MMIO
	 * writes and a poll loop — so a kernel whose real work is a handful of MACs
	 * pays the same fixed cost as one with eleven thousand.  Attention pays it per
	 * HEAD: 32 launches a layer for programs of eleven instructions each.  Stacking
	 * them changes not one MAC, only how often that fixed cost is paid.
	 *
	 * What it costs is GPR: every stacked piece needs its own vector words and its
	 * own result words, live at the same time, where a single-piece program reuses
	 * one buffer.  Sized at open from this number. */
	uint32_t max_batch;

	/* GPR RESERVED FOR THE STACKED PIECES, vectors and results.  0 means "derive
	 * them from max_red and max_batch", which is right for a caller stacking a few
	 * large pieces.
	 *
	 * A CALLER STACKING THOUSANDS OF SMALL ONES MUST SET THESE.  Attention over a
	 * whole prompt is S_q * H_q pieces whose vectors are a HEAD wide — 128 B — and
	 * deriving the buffer from max_red would reserve the FFN's 16 KiB for each of
	 * them, 128 times what they use.  The allocator underneath is a bump pointer
	 * and already packs pieces at their true size; this is only how much room it
	 * has.  Both are raised to fit one largest piece, so neither can be set so
	 * small that an op is impossible. */
	size_t vec_bytes;
	size_t res_bytes;

	/* Evidence that this image decodes ISR[35], which PIM_ACC_DUAL needs.  Passed
	 * to lowering as PIM_LOWER_ALLOW_T; without it a DUAL op is refused early
	 * rather than producing a silently wrong number at the doorbell. */
	bool allow_t_latch;

} pim_rt_config;

/* `cfg` may be NULL for defaults, which are deliberately small — a default big
 * enough for a model would hide a caller that forgot to declare its shapes. */
const char *pim_rt_open (pim_ctx *c, const pim_rt_config *cfg, pim_rt **out);
void        pim_rt_close(pim_rt *r);

/* The engine underneath, for a caller that wants violation counters or a scrub. */
pim_exec   *pim_rt_exec (pim_rt *r);
pim_ctx    *pim_rt_ctx  (pim_rt *r);

/* Statistics for the last op, or cumulative since pim_rt_stat_reset().
 *
 * WHY THE PHASE BREAKDOWN IS HERE AND NOT IN A PROFILER.  A host profiler sees one
 * call — pim_op_matvec — and can say only that it took four milliseconds.  Whether
 * those were spent generating instructions, moving a vector over PCIe, or waiting
 * on a bank is invisible from outside, and those three have nothing in common: one
 * is fixed by caching, one by batching transfers, and one not at all.  Measuring
 * launches was how we learnt that cutting them eightfold changed nothing.
 *
 * THE PHASES ARE DISJOINT and, summed, are the time inside pim_op_*.  launch_us is
 * NOT one of them — it is the poll-only subset of run_us, so run_us - launch_us is
 * what the IMEM transfer and the two MMIO writes cost. */
typedef struct {
	uint64_t launch_us;   /* doorbell -> done, summed over every launch */
	uint32_t polls;
	uint32_t nlaunch;     /* doorbells; more than one when IMEM is the limit */
	uint32_t nisr;
	uint32_t nwrvec;      /* vector loads, the number DUAL halves  */
	uint32_t nop;

	/* --- disjoint phases, microseconds ------------------------------------ */
	uint64_t pad_us;      /* zero the vector's tail beat, copy to staging     */
	uint64_t vup_us;      /* staging -> GPR, one pwrite per piece             */
	uint64_t gen_us;      /* pass 1: build the logical program                */
	uint64_t low_us;      /* pass 2: resolve, check, legalise, relocate       */
	uint64_t run_us;      /* IMEM transfer + doorbell + poll                  */
	uint64_t back_us;     /* GPR -> staging, one pread per launch             */
	uint64_t unpack_us;   /* staging -> the caller's buffer                   */
} pim_rt_stat;

void pim_rt_stat_get  (const pim_rt *r, pim_rt_stat *out);
void pim_rt_stat_reset(pim_rt *r);

/* THE OP.
 *
 *     y[0 .. out_count * nch * nbank)  =  M[.][red_off .. red_off+red_len) . v
 *
 *   m              a pim_tensor, filled and tagged.  Lowering checks the tag, so a
 *                  tensor that was allocated and never uploaded is refused here
 *                  rather than read as whatever was in those granules.
 *   out_first/
 *   out_count      SUPERGROUPS.  One is nch*nbank outputs and is what a single
 *                  all-bank MAC covers.
 *   red_off/
 *   red_len        ELEMENTS.  red_len is rounded UP to a whole beat and the tail is
 *                  ZEROED ON BOTH SIDES: this function zeroes the vector's, and the
 *                  tensor's is pim_tensor's zero invariant.  Measured: a tail lane
 *                  that is merely large changes nothing, but an Inf or NaN turns
 *                  every output of its bank into NaN (runtime/test/tensor_board).
 *   v              red_len BF16 of host memory.  Host order, unpadded.
 *   y              out_count * nch * nbank BF16 of host memory, written in output
 *                  order — the padding outputs of the last supergroup are written
 *                  too, so size it to the padded count and slice afterwards.
 *
 * Returns NULL, or a reason. */
const char *pim_op_matvec(pim_rt *r, const pim_tensor *m,
                          uint32_t out_first, uint32_t out_count,
                          uint32_t red_off, uint32_t red_len,
                          const uint16_t *v, uint16_t *y,
                          pim_acc_mode mode);

/* ============================== stacking ===================================
 * The same op, split so several can share one doorbell.
 *
 *     pim_op_begin(rt, mode)
 *     pim_op_add(rt, ...)      x N        <- 32 attention heads, say
 *     pim_op_submit(rt)                   <- one launch
 *
 * pim_op_matvec() is exactly begin + add + submit and stays the right call for a
 * single large op; there is nothing to gain from stacking a program that already
 * fills IMEM.
 *
 * pim_op_add MAY LAUNCH.  When the next piece would not fit — IMEM, the GPR
 * scratch, or max_batch — it submits what it has and starts a fresh program with
 * that piece.  The results are identical either way; only the launch count moves,
 * and pim_rt_stat reports it.  Refusing instead would make every caller implement
 * the same retry.
 *
 * `y` MUST STAY VALID UNTIL submit, because that is when it is written. */
const char *pim_op_begin (pim_rt *r, pim_acc_mode mode);
const char *pim_op_add   (pim_rt *r, const pim_tensor *m,
                          uint32_t out_first, uint32_t out_count,
                          uint32_t red_off, uint32_t red_len,
                          const uint16_t *v, uint16_t *y);
const char *pim_op_submit(pim_rt *r);

/* Outputs pim_op_matvec writes for a given out_count — the size `y` must have.
 * Worth a function because it is nch*nbank*out_count and getting it from the tensor
 * instead would be right only when out_count is all of them. */
uint32_t pim_op_outputs(const pim_rt *r, uint32_t out_count);

#ifdef __cplusplus
}
#endif
#endif /* PIM_OP_H */
