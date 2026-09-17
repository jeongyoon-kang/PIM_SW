/* SPDX-License-Identifier: MIT */
/*
 * pim_matvec.h — pass 1, for every kernel this machine has.
 *
 *     y[out_first .. out_first+out_count)  =  M[.][red_off .. red_off+red_len)
 *                                             .  v[0 .. red_len)
 *
 * ONE BUILDER, THREE KERNELS.  A linear layer, a Q.K^T and an S.V differ in nothing
 * the ISR encoder can see — same WRVEC, same MAC, same RD_MAC — so they differ here
 * only in what the caller passes:
 *
 *                 M              out range          red_off      red_len
 *   linear        W  (OUT_MAJOR) every group        0            k, fixed
 *   Q . K^T       K  (OUT_MAJOR) the token groups   head * D     D
 *   S . V         V  (RED_MAJOR) the dim groups     0            S_kv, GROWS
 *
 * WHY red_off AND red_len ARE ARGUMENTS AND NOT FIELDS OF THE TENSOR.  Two reasons,
 * one per attention kernel, and neither applies to a linear layer — which is why
 * pim_gemv got this far without them.
 *
 *   red_off  SEVERAL OPERANDS SHARE A ROW.  A DRAM row holds 1024 BF16 and a KV
 *            head is D of them, so for D = 128 eight heads fit in one row and COL
 *            picks which.  That is only legal because COL offsets the DRAM side
 *            alone — the GB is read from beat 0 whatever COL says [measured,
 *            runtime/test/col_probe].  Without it a K cache would waste 1024/D of
 *            itself, 8x at D = 128.
 *
 *   red_len  THE REDUCTION AXIS GROWS.  S.V reduces over the sequence, so its
 *            length changes every token while the ALLOCATION stays the size it was
 *            reserved at.  A tensor cannot carry it; only the call knows.
 *
 * BEAT ALIGNMENT.  OPSIZE counts beats, so both are rounded UP to a multiple of 16
 * elements.  For red_off that is free — a head boundary is already aligned for any
 * sane D.  For red_len it means the final beat reads PAST what the caller wrote,
 * which is exactly the range pim_tensor.h's zero invariant covers.  A RED_MAJOR
 * tensor that has not kept it will produce NaN here and no error.
 *
 * WHAT THIS PRODUCES is a pim_logical: ISRs with blank address fields plus the
 * notes saying what fills them.  It needs the geometry and nothing else — no
 * context, no allocator, no board.  pim_prog_lower() turns it into something
 * runnable.  See pim_logical.h for why that is two passes.
 */
#ifndef PIM_MATVEC_H
#define PIM_MATVEC_H

#include <stdint.h>

#include "pim/pim.h"
#include "pimrt/pim_exec.h"
#include "pimrt/pim_logical.h"
#include "pimrt/pim_tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* HOW MANY ACCUMULATOR LATCHES THE SCHEDULE USES.
 *
 * SINGLE is one output group at a time.  DUAL pairs two groups across ISR[35] so
 * they share one vector load, which halves the WRVECs when the reduction takes more
 * than one chunk and buys exactly nothing when it takes one — there is no
 * accumulator to protect between groups if the vector only goes up once.  Measured
 * on ch2: 8 -> 4 loads, 25.6 -> 19.7 us (runtime/test/tlatch_test).
 *
 * DUAL declares nlatch = 2, which pim_prog_lower() REFUSES unless the caller passes
 * PIM_LOWER_ALLOW_T — dropping to one latch is a different schedule, not a rewrite
 * of an ISR, so lowering may only say no. */
typedef enum { PIM_ACC_SINGLE = 0, PIM_ACC_DUAL = 1 } pim_acc_mode;

/* The reduction slice, rounded out to whole beats.  Exposed because a caller sizing
 * a vector buffer needs the same number the builder will use, and computing it
 * twice is how the two come to disagree. */
uint32_t pim_matvec_beats(uint32_t red_off, uint32_t red_len);

/* ISRs this will emit, as an exact count rather than a bound — pass 1 knows the
 * schedule.  (pim_lower_isr_count() gives the bound AFTER splitting.) */
uint32_t pim_matvec_nisr  (const pim_tensor *m, uint32_t out_count,
                           uint32_t red_off, uint32_t red_len, pim_acc_mode mode);
/* Vector loads, which is the number DUAL halves. */
uint32_t pim_matvec_nwrvec(const pim_tensor *m, uint32_t out_count,
                           uint32_t red_off, uint32_t red_len, pim_acc_mode mode);

/* Build it.
 *
 *   out_first/out_count   in SUPERGROUPS, not outputs.  One supergroup is
 *                         nch * nbank outputs and is what a single all-bank MAC
 *                         covers; a partial one is not expressible.
 *   red_off/red_len       in ELEMENTS, rounded up to beats as above.
 *   vgpr/vbytes           the vector, red_len elements at word 0 of `vgpr`.  Its
 *                         index is red - red_off, NOT red: the GB is filled from
 *                         beat 0 however far into the row COL reaches.
 *   vtag                   the layout contract the vector carries, or 0 for none.
 *                         A WRVEC reads whole beats and the tail past red_len is
 *                         multiplied like any other lane, so a buffer whose tail
 *                         nobody zeroed is a wrong answer with no error.  The tag
 *                         is how "somebody zeroed it on purpose" is stated, since
 *                         the bytes cannot say.  pim_prog_lower() compares it.
 *   ygpr/ybytes           nch words per output group, in group order.
 *
 * Returns NULL, or a reason. */
const char *pim_matvec_logical(const pim_geometry *g, const pim_tensor *m,
                               uint32_t out_first, uint32_t out_count,
                               uint32_t red_off, uint32_t red_len,
                               const void *vgpr, size_t vbytes, uint64_t vtag,
                               const void *ygpr, size_t ybytes,
                               pim_acc_mode mode, pim_logical *out);

#ifdef __cplusplus
}
#endif
#endif /* PIM_MATVEC_H */
