/* SPDX-License-Identifier: MIT */
/*
 * pim_logical.h — a program before it knows where anything is.
 *
 * CODE GENERATION IS TWO PASSES (design note, 2026-08-25).
 *
 *   pass 1   BUILD FROM VIRTUAL ADDRESSES.  A kernel emits ISRs whose address
 *            fields are left blank, plus a note saying what each blank means —
 *            "the DRAM row of unit 7 of allocation W", "the GPR word 128 words into
 *            allocation X".  It considers NO hardware constraint, so the result
 *            cannot run.  It needs the geometry and nothing else: no allocator, no
 *            driver, no board.
 *
 *   pass 2   LOWER IT.  pim_prog_lower() asks libpim where those allocations
 *            actually landed, rewrites whatever the hardware cannot express in one
 *            command, fills the blanks in, and hands back a pim_prog the engine can
 *            run.  This is where every constraint lives.
 *
 * WHY THE SPLIT IS WORTH A HEADER.  Two reasons that are not style:
 *
 *   the constraints stop being copied.  RD_MAC cannot be a multicast, so a logical
 *   "drain this group" becomes nch ISRs with a one-hot CH_MASK.  pim_gemv.c does
 *   that expansion by hand today; every future kernel would have to do it again,
 *   and each copy is a chance to get it silently wrong.
 *
 *   one shape, many tensors.  A transformer has the same [n x k] GEMV in every
 *   layer with different weights.  Pass 1 depends on the shape only, so it runs
 *   ONCE and each layer re-runs pass 2 against its own allocation.
 *
 * WHAT PASS 2 MAY AND MAY NOT DO.  It may split an ISR into several and it may
 * refuse.  It may NOT reorder, and it may not cut inside an ATOM (below), because
 * both would move an accumulation across a boundary that owns it.
 */
#ifndef PIM_LOGICAL_H
#define PIM_LOGICAL_H

#include <stdint.h>

#include "pim/pim.h"
#include "pimrt/pim_exec.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================== address references =========================
 * WHAT A BLANK MEANS.  Both kinds land in the ISR's ROW field, which carries two
 * unrelated things depending on the opcode — a DRAM row for MAC, a GPR word for
 * WRVEC and RD_MAC.  Naming the KIND rather than the field is what stops a kernel
 * from putting one where the other belongs; getting that backwards is not an error
 * anywhere in the hardware, it is a wrong number.
 */
typedef enum {
	/* ROW = the DRAM row of broadcast unit `index` of `base`.  DRAM pointers
	 * only.  The §4.1 invariant makes this a single row and never two. */
	PIM_REF_DRAM_UNIT = 0,
	/* ROW = the GPR word `index` words past the start of `base`.  GPR pointers
	 * only. */
	PIM_REF_GPR_WORD  = 1,
} pim_ref_kind;

/* HOW PASS 2 MAY REWRITE THIS ISR IF ONE COMMAND CANNOT EXPRESS IT.
 *
 * This is the kernel telling the lowerer what is SAFE, not what to do.  The lowerer
 * knows the hardware rules; it does not know whether splitting an operation changes
 * the answer, and only the kernel does.  So the default is the strict one. */
typedef enum {
	/* One ISR in, one ISR out.  If the hardware cannot express it, lowering
	 * fails rather than inventing a rewrite. */
	PIM_SPLIT_NONE = 0,
	/* Expand into one ISR per channel: CH_MASK becomes one-hot and `index`
	 * advances by one per channel.  This is what a RD_MAC needs — a multicast
	 * would have two channels writing one GPR word and one result would be lost
	 * with nothing to say so. */
	PIM_SPLIT_PER_CHANNEL = 1,
} pim_split;

typedef struct {
	uint32_t     isr;    /* which ISR of the logical program this blank is in */
	pim_ref_kind kind;
	pim_split    split;
	const void  *base;   /* a pim_alloc() pointer */
	size_t       len;    /* bytes of `base` this program uses — bounds the ref */
	uint32_t     index;  /* unit number (DRAM) or word offset (GPR)            */

	/* THE LAYOUT THIS REFERENCE ASSUMES `base` WAS FILLED IN.  0 = no claim.
	 *
	 * "Unit 7 of this allocation" only names a row; it says nothing about what is
	 * IN that row, and the arrangement is a contract between whoever wrote the
	 * bytes and whoever reads them.  Nothing in the hardware checks it — a matrix
	 * written linearly into an allocation whose program expects one output per
	 * bank row produces a wrong number in silence, and for k > 1024 a single
	 * output would even straddle two banks and be summed into two accumulators.
	 *
	 * So the filler stamps the allocation (pim_tag_set) and this states what the
	 * kernel assumed; pim_prog_lower() refuses when they differ.  Leave it 0 for
	 * an allocation whose contents have no layout — a GPR vector, say, which is
	 * just consecutive words. */
	uint64_t     tag;
} pim_ref;

/* ============================== atoms ======================================
 * A RANGE OF ISRs THAT MUST NOT BE CUT APART.
 *
 * The accumulator latch is cleared only by RD_MAC and by reset, so everything from
 * a group's first MAC to its RD_MAC owns that latch.  Splitting a program across
 * doorbells anywhere inside that range strands a running sum in a latch, and the
 * next program accumulates its own work on top of it — a wrong number, no error.
 *
 * Pass 1 knows where those ranges are, because it chose the schedule.  Pass 2 only
 * has to respect them, so it does not need to understand the schedule at all.
 *
 * Indices are into the LOGICAL program.  Lowering maps them forward itself, since
 * only it knows how many ISRs each one became.
 */
typedef struct { uint32_t first, last; } pim_atom;   /* inclusive */

/* ============================== the logical program ======================== */
typedef struct {
	pim_isr  *isr;   uint32_t nisr,  isr_cap;
	pim_ref  *ref;   uint32_t nref,  ref_cap;
	pim_atom *atom;  uint32_t natom, atom_cap;
	uint32_t  nch;   /* channels the kernel built for; lowering checks it */

	/* HOW MANY ACCUMULATOR LATCHES THIS SCHEDULE USES.  1, or 2 for a schedule
	 * that pairs output groups across ISR[35].
	 *
	 * DECLARED RATHER THAN HINTED, because lowering cannot act on it: dropping
	 * from two latches to one is not a rewrite of an ISR, it is a different
	 * schedule with a different number of WRVECs.  So pass 1 commits and pass 2
	 * may only REFUSE — which it does when the caller has not said it has
	 * evidence that this image decodes ISR[35] (pim_exec_config.allow_t_latch).
	 *
	 * Refusing here rather than at the doorbell is the point: the failure is a
	 * silent wrong number, so it should be caught as early as something can. */
	uint32_t  nlatch;
} pim_logical;

/* Storage is the caller's, the same as pim_prog — a kernel can build into stack
 * arrays with no allocation on the launch path. */
void pim_logical_init(pim_logical *lp, pim_isr *isr, uint32_t isr_cap,
                      pim_ref *ref, uint32_t ref_cap,
                      pim_atom *atom, uint32_t atom_cap,
                      uint32_t nch, uint32_t nlatch);

/* Append an ISR whose address field is already final (EOS, or a WRVEC whose word is
 * known without an allocation).  Returns NULL, or a reason. */
const char *pim_logical_push(pim_logical *lp, const pim_isr *isr);

/* Append an ISR with a blank ROW, and the note that says what fills it.  The ISR's
 * ROW field is ignored and overwritten. */
const char *pim_logical_push_ref(pim_logical *lp, const pim_isr *isr,
                                 pim_ref_kind kind, pim_split split,
                                 const void *base, size_t len, uint32_t index,
                                 uint64_t tag);

/* Mark [first, last] as one accumulation region.  Call it after pushing the range. */
const char *pim_logical_atom(pim_logical *lp, uint32_t first, uint32_t last);

/* ============================== pass 2 =====================================
 * Resolve, check, legalise, relocate.  In that order, and the order is forced: it
 * takes an address to know whether a constraint is violated at all.
 *
 * `out` receives the runnable program; its storage is the caller's and must hold at
 * least pim_lower_isr_count(lp) ISRs, which is an upper bound rather than the exact
 * count — the exact one is not known until the addresses are.
 *
 * Uses the default context.  A kernel that was handed one explicitly should call
 * pim_prog_lower_ctx(). */
#define PIM_LOWER_ALLOW_T  0x1u   /* the caller has evidence ISR[35] is decoded */
const char *pim_prog_lower(const pim_logical *lp, pim_prog *out, unsigned flags);
const char *pim_prog_lower_ctx(pim_ctx *c, const pim_logical *lp, pim_prog *out,
                               unsigned flags);

/* An upper bound on how many ISRs lowering can produce, for sizing `out`.  Every
 * PIM_SPLIT_PER_CHANNEL reference can become nch. */
uint32_t pim_lower_isr_count(const pim_logical *lp);

/* Where the atoms ended up in the lowered program, so a caller that has to split
 * across doorbells knows where it may cut.  `out` receives up to `cap` entries and
 * *n gets how many there were; pass NULL to ask for the count only.
 *
 * SEPARATE FROM pim_prog_lower BECAUSE SPLITTING IS THE ENGINE'S PROBLEM, not the
 * address side's: what fits behind one doorbell depends on IMEM, and IMEM is a
 * property of the board rather than of the allocation. */
const char *pim_lower_atoms(const pim_logical *lp, pim_atom *out, uint32_t cap,
                            uint32_t *n);

#ifdef __cplusplus
}
#endif
#endif /* PIM_LOGICAL_H */
