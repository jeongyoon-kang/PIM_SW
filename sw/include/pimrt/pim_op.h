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

/* Statistics for the last op, or cumulative since pim_rt_stat_reset(). */
typedef struct {
	uint64_t launch_us;   /* summed over every doorbell            */
	uint32_t polls;
	uint32_t nlaunch;     /* doorbells; more than one when IMEM is the limit */
	uint32_t nisr;
	uint32_t nwrvec;      /* vector loads, the number DUAL halves  */
	uint32_t nop;
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

/* Outputs pim_op_matvec writes for a given out_count — the size `y` must have.
 * Worth a function because it is nch*nbank*out_count and getting it from the tensor
 * instead would be right only when out_count is all of them. */
uint32_t pim_op_outputs(const pim_rt *r, uint32_t out_count);

#ifdef __cplusplus
}
#endif
#endif /* PIM_OP_H */
