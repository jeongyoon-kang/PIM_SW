/* SPDX-License-Identifier: MIT */
/*
 * pim_addr.h — an allocation's coordinates, as the ISA names them.
 *
 * THIS IS PART OF libpim, NOT OF THE COMPUTE RUNTIME.  What it does is address
 * translation: it takes pim_resolve()'s answer and expresses it the way a command
 * field wants it.  It knows nothing about opcodes, schedules or the dispatcher.
 *
 * THAT SPLIT IS THE POINT.  Code generation happens in two passes (design note,
 * 2026-08-25):
 *
 *   pass 1  build the program from VIRTUAL addresses.  No hardware constraint is
 *           considered and the result cannot run.  Needs the geometry only.
 *   pass 2  ask THIS FILE where everything actually landed, split whatever the
 *           hardware cannot express in one command, and fill the address fields in.
 *
 * So the runtime asks libpim for coordinates rather than reaching for them itself,
 * and pass 1 stays testable with no allocator, no driver and no board.
 *
 * WHY THE ANSWERS ARE SHAPED LIKE THIS.  Design §4.1 fixes the DRAM side completely:
 * a broadcast unit divides a hugepage and hugepages are hugepage-aligned, so no
 * operation can straddle one, so an operation's operands are always ONE contiguous
 * extent per unit.  The GPR side is simpler still — it is linear, so a coordinate is
 * a word index and a count.
 */
#ifndef PIM_ADDR_H
#define PIM_ADDR_H

#include <stddef.h>
#include <stdint.h>

#include "pim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================== operands ===================================
 * One broadcast unit of an allocation, resolved to where the hardware will find it.
 *
 * A unit is the footprint of a single all-bank, all-channel operation (design §3.2).
 * Because of the §4.1 invariant it is always one contiguous run of card addresses,
 * so one unit is one (row, channel-set) pair and never two.
 *
 * `row` is what goes in a command's ROW field.  Under RoChBaCo every channel of the
 * unit shares it — which is exactly why the ISA can carry one ROW field and a
 * channel mask rather than a row per channel. */
typedef struct {
	uint64_t card_addr;	/* AXI address of the unit's first byte */
	uint64_t row;		/* the ROW field a command would carry   */
	uint32_t unit_index;	/* which unit of the allocation this is  */
} pim_unit;

/* How many units back [p, p+len).  0 if p is not a live allocation, or if the range
 * runs past its end. */
size_t pim_addr_unit_count(const void *p, size_t len);
size_t pim_addr_unit_count_ctx(pim_ctx *c, const void *p, size_t len);

/* Resolve unit `i` of [p, p+len).  NULL on success, else a reason.
 *
 * Walking these is the whole address half of issuing an operation: for each unit,
 * decode card_addr with pim_geom_decode() to get (row, ch, bank, col) and emit
 * whatever the command encoding wants.  It needs no syscall — that is the point of
 * design §4's hugepage sizing.
 *
 * DRAM ONLY.  A GPR pointer is refused rather than decoded: the GPR is linear and
 * has no (row, ch, bank), so answering at all would be inventing coordinates. */
const char *pim_addr_unit(const void *p, size_t len, size_t i, pim_unit *out);
const char *pim_addr_unit_ctx(pim_ctx *c, const void *p, size_t len,
                              size_t i, pim_unit *out);

/* ============================== GPR operands ================================
 * The GPR word a pointer names.  WRVEC multicasts from a word into every channel's
 * global buffer and RD_MAC unicasts results back into one, so a word index is the
 * only GPR coordinate the ISA has — and this is the whole conversion.
 *
 * *word receives the first word, *nwords how many of them [p, p+len) covers.  Both a
 * vector operand and a result buffer go through here; that they are allocated out of
 * the same pool (design §9.4) is exactly why one function serves both.
 *
 * IT REQUIRES THE RANGE TO BE CONTIGUOUS IN THE GPR, and says so if it is not.  It
 * normally is — a GPR allocation of n pages from one extent is one span — but a
 * fragmented pool can split one, and a WRVEC that walked off the end of a run would
 * read someone else's vector rather than fail. */
const char *pim_addr_gpr_words(const void *p, size_t len,
                               uint32_t *word, uint32_t *nwords);
const char *pim_addr_gpr_words_ctx(pim_ctx *c, const void *p, size_t len,
                                   uint32_t *word, uint32_t *nwords);

/* ============================== not yet =====================================
 * pim_op(), pim_rt_open(), completion, and the tiled-layout question of §9.2 all
 * wait on the submission decision above.  When it is made:
 *
 *   plan A  adds PIM_IOC_SUBMIT / PIM_IOC_WAIT at uapi/pim_ioctl.h 0x04.., and the
 *           driver grows an ioremap and a batch writer.  The command words stay
 *           OPAQUE to the driver so that "the kernel does not know geometry"
 *           (design §5) survives.
 *   plan B  adds nothing to the driver at all.  The ring lives in a card-memory
 *           region carved out at insmod (outside the hugepage pool, so the allocator
 *           can never hand it out) and everything is pwrite/pread through libpim.
 *
 * Either way the code above this comment does not change, because the address side
 * does not depend on how the words get there.
 */

#ifdef __cplusplus
}
#endif
#endif /* PIM_ADDR_H */
