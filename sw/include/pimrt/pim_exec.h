/* SPDX-License-Identifier: MIT */
/*
 * pim_exec.h — the engine.  Loading a program and ringing the doorbell.
 *
 * WHAT MAKES THIS DIFFERENT FROM THE MEMORY SIDE.  Memory was a PARTITIONING problem
 * and the ledger solved it.  The engine is not partitionable: there is one
 * dispatcher, one IMEM, one doorbell, and — the sharp one — a per-bank accumulator
 * latch that is cleared only by RD_MAC and by reset, so it survives across programs
 * and across processes.  A program that dies half way leaves it dirty and the next
 * program inherits someone else's partial sum with nothing to show for it.
 *
 * SO THIS LAYER ASSUMES ONE PROCESS OWNS THE BOARD (design §11, 2026-08-21).  There
 * is no lease and no arbitration here.  Two processes running programs at once will
 * corrupt each other and nothing in this file will notice.
 *
 * WHAT IS TRANSIENT AND THEREFORE NOT ALLOCATED.  IMEM is a single buffer the
 * dispatcher fetches from word 0 up to PROG_LEN, so it belongs to whoever is running
 * a program and there is nothing for an allocator to hand out.  The GPR is the
 * opposite — it is memory, it persists, and it comes from pim_alloc(PIM_MEM_GPR).
 * That split is design §9.4 and it is why this header has no allocation calls.
 *
 * ------------------------------------------------------------------------------
 * THE ONE DESIGN DECISION MADE HERE, AND IT IS REVERSIBLE
 *
 * Design §6.1 says the control BAR is not mapped into userspace.  This file maps
 * 32 KiB of it — the CFR page and the violation CSRs — because the doorbell,
 * PROG_LEN and STATUS are AXI-Lite registers and there is no other way to reach them
 * today.  Everything bulk still goes over QDMA: the program to IMEM, the vector and
 * the results to and from the GPR.
 *
 * Why it is acceptable NOW: the two reasons §6.1 gave were arbitrating between
 * processes and keeping registers away from arbitrary ones.  The first is moot while
 * one process owns the board, and the second is explicitly deferred (§11 leaves
 * isolation out of scope).
 *
 * What it costs: a second process CAN ring the doorbell, and nothing stops it.
 *
 * How to undo it: everything MMIO in this file is behind cfr_rd/cfr_wr and
 * exec_open's mapping.  Design §9.1 plan A replaces those with a PIM_IOC_LAUNCH that
 * hands the driver PROG_LEN and waits — reserved at uapi/pim_ioctl.h 0x04.  Nothing
 * above this file changes, because nothing above it knows how the words get out.
 * ------------------------------------------------------------------------------
 */
#ifndef PIM_EXEC_H
#define PIM_EXEC_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "pim/pim.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One 256-bit IMEM cell.  Same layout as hwdef's struct emu_isr, deliberately — the
 * encoder lives there and this is the wire form.  Kept structurally identical rather
 * than including emu_regs.h here, so an application need not carry hwdef's include
 * path to hold a program. */
typedef struct { uint64_t w[4]; } pim_isr;

typedef struct pim_exec pim_exec;

/* The DRAM timing model's registers.
 *
 * THIS LAYER DOES NOT WRITE THEM.  hwdef/test/emu_timing owns them, and one owner is
 * the whole point: that tool exists to set a value and ask what it does, and an
 * engine that wrote its own defaults at every open would silently revert the
 * experiment between setting it and measuring it.  `emu_timing --scale 6 --keep`
 * followed by any program that opened an engine used to end up back at the SIM set
 * without saying so.
 *
 * SO THERE IS NO SETTER HERE.  pim_exec_open READS them, refuses on the one value
 * that is wrong without saying so (T_CCD < 2), and hands them back through
 * pim_exec_timing_at_open() so a run can report the conditions it ran under.  When
 * something eventually owns the control plane the write goes there, with the
 * doorbell — not back into a compute library.
 *
 * T_CCD MUST NOT DROP BELOW 2.  acc_top is a two-stage feedback pipe with no
 * forwarding, so back-to-back beats into one latch accumulate every OTHER beat —
 * with no violation counter and no diagnostic anywhere.  The answer is simply
 * halved.
 *
 * T_RCD RAISES RCD_RD VIOLATIONS AND THAT IS NOT A DEFECT.  The counter is the
 * emulator reporting that its memory was SLOWER than the allowance the model was
 * configured with; the results are unaffected.  Measured 2026-08-21 on ch2, one
 * 2048-element GEMV tile, MC-resident weights:
 *
 *     T_RCD      4     8    16    32    64
 *     RCD_RD    32    32    32     4     4     (per channel)
 *     overrun   81    70    61    37    15     (worst, cycles)
 *     lanes   32/32 32/32 32/32 32/32 32/32    (bit-exact at every setting)
 *
 * So PIM_TIMING_QUIET's T_RCD = 8 does NOT quiet this workload — it takes about 32.
 * hwdef/emu_regs.h calls 8 quiet, which held for the single-beat probe it was
 * measured with and does not hold here.  Raise it only to make the CSR readable;
 * it buys no correctness. */
typedef struct { uint8_t faw, rrd, rcd, ccd, rtp, rp, wr, ras; } pim_timing;
#define PIM_TIMING_SIM    ((pim_timing){ 30, 6, 4, 2, 3, 3, 4, 6 })
#define PIM_TIMING_QUIET  ((pim_timing){ 30, 6, 8, 2, 3, 3, 4, 6 })

/* Read the eight registers back off the board.  They are 32-bit AXI-Lite and this
 * is eight MMIO reads; there is no caching, so this is what the device HAS, not
 * what anyone last asked for. */
const char *pim_exec_get_timing(pim_exec *e, pim_timing *out);

/* What the registers held when this engine opened.  Cheaper than re-reading and it
 * is what the run actually used, so a report can name its own conditions. */
const pim_timing *pim_exec_timing_at_open(const pim_exec *e);


/* The register offsets in the order the struct lists them, for a tool that wants to
 * name them.  CFR_T_FAW .. CFR_T_RAS are contiguous 4-byte registers. */
extern const char *const pim_timing_names[8];

typedef struct {
    const char *bdf;         /* NULL -> "0000:01:00.0" */
    uint32_t    timeout_ms;  /* 0 -> 2000.  Covers both idle-wait and done-wait. */
    /* Longest program this engine will accept.  0 -> the hardware maximum, which
     * is what PROG_LEN can express.  A caller lowers it to force the kernels above
     * to split across launches on a shape that would otherwise fit — which is the
     * only way to test that path without a model-sized GEMV. */
    uint32_t    max_isrs;
    /* THE TIMING REGISTERS ARE NOT HERE, on purpose — see the pim_timing block
     * above.  emu_timing sets them; this reads them. */

    /* ISR[35] SELECTS ONE OF TWO PER-BANK ACCUMULATOR LATCHES, and on this image
     * it may not be wired.  the platform conf says:
     *
     *     FEATURE_T_LATCH=0
     *     emulator_controller.v:1854 ties latch_sel to 1'b0, so T=1 is silently
     *     wrong today: both MACs pile into latch 0 and both RD_MACs read the same
     *     value.  Flip to 1 only after the HW revision that decodes ISR[35].
     *
     * "Silently wrong" is the whole problem: a program using T=1 on an image that
     * ties it produces a WRONG NUMBER and no error.  So T=1 is refused by default,
     * and a caller that sets this is stating it has evidence.
     *
     * THE EVIDENCE NOW EXISTS, for one image.  Measured 2026-08-21 on ch2 by
     * runtime/test/tlatch_test.c, k = 2048 split across the two latches:
     *
     *     latch 0's partial == dot(first 1024)   32 of 32 outputs
     *     latch 0's partial == dot(all 2048)      0 of 32 outputs
     *     latch 1's partial == dot(last 1024)    32 of 32 outputs
     *
     * So ISR[35] IS decoded on this bitstream and the conf's claim is stale.  The
     * flag still defaults to off, because that was measured on ONE image and the
     * failure it guards against is silent — ch1 and ch4 have not been asked. */
    bool        allow_t_latch;
} pim_exec_config;

/* Opens the control plane and checks two things the memory layer cannot:
 *   - the board's MODE_CTRL agrees with the address map libpim was told about.
 *     A mismatch makes every weight land somewhere other than where the ROW field
 *     will look for it, with no error anywhere.
 *   - the dispatcher is idle.
 * Also installs IMEM as libpim's raw-AXI window (pim_axi_allow). */
const char *pim_exec_open (pim_ctx *c, const pim_exec_config *cfg, pim_exec **out);
void        pim_exec_close(pim_exec *e);
const char *pim_exec_error(const pim_exec *e);

/* The longest program pim_exec_run() will accept.  Kernels ask this before they
 * decide how much of themselves fits in one doorbell. */
uint32_t    pim_exec_max_isrs(const pim_exec *e);

/* ------------------------------------------------------------- program ---
 * A program is an array of ISR words the caller owns.  This is a cursor over it,
 * not a container: the storage is the caller's so a kernel can build into a stack
 * array without an allocation on the launch path.
 */
typedef struct {
    pim_isr *w;
    uint32_t n, cap;
} pim_prog;

void        pim_prog_init(pim_prog *p, pim_isr *storage, uint32_t cap);
const char *pim_prog_push(pim_prog *p, const pim_isr *isr);

/* ------------------------------------------------------- looking inside ---
 * A program is opaque to the engine, so these exist for the two callers that must
 * see into one anyway: a dump, and a test.
 *
 * THE OPCODES ARE RESTATED HERE rather than pulled from hwdef/emu_regs.h, so that
 * holding and inspecting a program needs only this header.  pim_exec.c static-
 * asserts every one of them against the ISA's own definition, so they cannot drift
 * without a build failure. */
#define PIM_OP_MAC      0x0Cu
#define PIM_OP_EWMUL    0x0Du
#define PIM_OP_COPY     0x0Eu
#define PIM_OP_WRVEC    0x0Fu
#define PIM_OP_RD_MAC   0x10u
#define PIM_OP_EOS      0x11u

typedef struct {
    uint32_t opcode;
    uint32_t opsize;     /* BEATS, not elements                              */
    uint32_t row;        /* a DRAM row for MAC; a GPR WORD for WRVEC/RD_MAC   */
    uint32_t col;
    uint32_t ch_mask;
    uint32_t pu_mask;
    uint32_t gb_mc_mask;
    uint32_t t;          /* ISR[35], the accumulator latch select — see below */
} pim_isr_info;

void        pim_isr_decode(const pim_isr *w, pim_isr_info *out);
const char *pim_isr_opname(uint32_t opcode);
void        pim_prog_dump (const pim_prog *p, FILE *out);

/* THE RULES A SINGLE ISR CANNOT SHOW, and the hardware checks none of them — the
 * validity gate was pulled out of the fetch path for timing on 2026-07-29 and has
 * not come back, so breaking one of these produces a wrong NUMBER, not an error.
 *
 *   1. a MAC's OPSIZE must equal the OPSIZE of the WRVEC that filled the GB.  The
 *      per-bank skid below the GB has no flush port, so a mismatch leaves beats
 *      behind and the NEXT MAC starts its vector at the wrong element.
 *   2. every MAC must be drained by a RD_MAC before the program ends, or the latch
 *      is inherited by whatever runs next.
 *   3. two RD_MACs must not land on the same GPR word.
 *   4. the last ISR must not produce a result: STATUS[31] rises when the fetcher
 *      ACCEPTS the last ISR, so its result would still be in flight.  Hence EOS.
 *
 *   5. no ISR may set T unless the caller has evidence that this image decodes
 *      ISR[35].  See pim_exec_config.allow_t_latch.
 *
 * pim_exec_run() calls this and refuses rather than launching. */
#define PIM_VERIFY_ALLOW_T  0x1u
const char *pim_prog_verify(const pim_prog *p, unsigned flags);

/* ------------------------------------------------------------- launching --- */
typedef struct {
    uint32_t status_before, status_after;
    uint32_t polls;
    uint64_t us;
    bool     saw_done;
} pim_launch;

/* Verify, wait for idle, load IMEM, set PROG_LEN, ring, wait for done.
 *
 * A TIMEOUT IS AN ERROR.  Every program ends with EOS and the dispatcher runs in
 * order, so done rising means every RD_MAC before it executed — there is nothing
 * left for a caller to re-check, and callers no longer try.
 *
 * ONE GAP STAYS OPEN, and it is here rather than hidden: done is a HELD LEVEL, and
 * require_idle accepts it as idle, so a 1 left over from the previous run can
 * satisfy the very first poll of this one.  Closing that needs to know whether the
 * doorbell clears the bit, which has not been measured.  What this does catch is the
 * launch that hangs, which is the failure that strands a caller. */
const char *pim_exec_run(pim_exec *e, const pim_prog *p, pim_launch *out);

/* Drain every bank's accumulator and throw the results away.
 *
 * THE LATCH IS LIVE STATE THAT CROSSES DOORBELLS AND PROCESSES.  It is cleared only
 * by RD_MAC and by reset, and a MAC ACCUMULATES INTO whatever is there — so a
 * program that died half way, or one from a previous session, leaves a partial sum
 * that the next kernel silently adds its own work to.  The next RD_MAC then returns
 * a number that is wrong by someone else's arithmetic, with nothing anywhere to say
 * so.  A kernel's own trailing RD_MAC does not save it: by then the stale value has
 * already been accumulated into.
 *
 * So: run this once at the start of a session and after any failed launch.  It is a
 * program of nch RD_MACs and an EOS — one doorbell, microseconds.
 *
 * gpr_word names nch consecutive GPR words the caller owns, where the discarded
 * results land.  They are overwritten and not read. */
const char *pim_exec_scrub(pim_exec *e, uint32_t gpr_word);

/* The violation CSR for one channel.  Read it after a launch: it is the only thing
 * the hardware says about a program it disliked.
 *
 * READ THE COUNTS, NOT JUST `any`.  Two of these are routine and one is not:
 *   RCD_RD       routine.  The timing model saying the memory was slower than its
 *                allowance — see pim_timing above.  Scales with T_RCD, never with
 *                correctness.
 *   RECOVERY_WR  routine on this stack.  Raised by the MC WRITE path, so a weight
 *                upload produces them and a launch produces none [measured
 *                2026-08-21: upload 384/channel, launch 0].
 *   CCD_WR, ewmul_drop  NOT routine.  Nothing here should raise them. */
const char *pim_exec_violations(pim_exec *e, unsigned ch, uint32_t *any);

/* Per-bank detail for one channel: the summed counters across all 16 banks, and the
 * worst overrun any of them saw.  Counters saturate at 255 per bank and never wrap. */
typedef struct {
    uint32_t sticky;        /* OR of every bank's sticky bits    */
    uint32_t rcd_rd, ccd_rd, rcd_wr, ccd_wr, recovery_wr;
    uint32_t worst_rcd_rd;  /* cycles                            */
} pim_viol;
const char *pim_exec_violation_detail(pim_exec *e, unsigned ch, pim_viol *out);
const char *pim_exec_clear_violations(pim_exec *e);

#ifdef __cplusplus
}
#endif
#endif /* PIM_EXEC_H */
