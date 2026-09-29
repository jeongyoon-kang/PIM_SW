// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_regs.h — address map and register offsets for emulator_top.
//
// Layer 0: pure header, no code, no dependencies.  Everything above it (the
// AXI-Lite sanity check, the DMA loopback, and later the ISR builder) shares
// these numbers so there is exactly one place they can be wrong.
//
// SOURCE: the block design's assign_bd_address.  ISR encoding — field placement,
// opcode values and the per-opcode shapes this header checks — comes from
// HW/r1p0/docs/pim_isr_opcode_사용가이드.md, which defers to pim_isr_defs.vh.
//
// SCOPE: this header covers the address map only.  ISR encoding is deliberately
// NOT here yet; it belongs with the layer that can actually run a program.
//////////////////////////////////////////////////////////////////////////////////
#ifndef EMU_REGS_H
#define EMU_REGS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>       // snprintf, used to print an ISR word as hex
#include <string.h>      // memset/memcpy, used by the ISR builder and the BF16 pair

// ============================== BAR2 -> AXI ===================================
// The CPM QDMA BAR-to-AXI translation is a plain base add:
//     AXI = EMU_AXI_BASE + (offset within BAR2)
// The base is 8 MB aligned, so no carry is possible and the low six hex digits of
// an AXI address ARE the BAR2 offset.  That is why one set of offsets serves both
// the MMIO path (mmap of resource2) and the DMA path (AXI address as file offset).
#define EMU_AXI_BASE        0x020200000000ULL
#define EMU_BAR2_SIZE       0x800000u        // 8 MB

// ============================== transport policy ==============================
// ONE TRANSPORT OWNS EACH WINDOW.
//   AXI-Lite  (CFR, violation CSR)              -> MMIO only
//   256 b     (IMEM, GPR, HBM, MC s_axi)        -> QDMA only, 32 B aligned
//
// THIS SPLIT IS NO LONGER FORCED BY THE HARDWARE, and the reason it used to be is
// worth keeping because it is what the policy was built around:
//
//   [measured 2026-07-31, an earlier image]  a 32 B MMIO store was split into <=16 B
//   TLPs and those slaves had no WSTRB, so each fragment was expanded to a full 32 B
//   word write and the second one won:  wrote b0..bf c0..cf, read back 00..00 c0..cf
//
//   [measured 2026-08-19, the ch4 image]  it lands.  MMIO stores of 1, 4, 8, 16 and
//   32 B all write exactly their own bytes and nothing else, MMIO reads of the GPR
//   return what DMA put there, and sub-beat DMA is exact on GPR, the direct aperture
//   and MC alike.  WSTRB is honoured now.
//
// The policy stays anyway, for reasons that never depended on WSTRB: DMA is an order
// of magnitude faster for anything bulk, and emu_sanity has to judge a freshly
// programmed board over MMIO before any queue exists.  What changes is that a
// violation is now a performance mistake rather than silent corruption — do not
// quote the 07-31 behaviour as current.
//
// The AXI-Lite blocks keep MMIO for the opposite reason: they are 32 bit registers,
// one TLP each, with no split to suffer — and emu_sanity must be able to judge a
// freshly programmed board before any DMA queue exists.

// [measured 2026-08-10] the layout the IP's assign_bd_address produces, and every
// window below has now answered on the board: emu_sanity write/readback for the two
// AXI-Lite blocks, emu_gpr_loop for the GPR, and a GEMV run for IMEM.
//
//   block                        AXI                  size  -> BAR2 offset
//   gpr_wrap_0/s_axi             0x202_0000_0000       4M      +0x000000  GPR
//   dispatcher_top_0/s_lite      0x202_0040_0000       4K      +0x400000  CFR
//   viol_csr_0/s_axi             0x202_0040_1000       4K      +0x401000  violation CSR
//   dispatcher_top_0/s_axi       0x202_0060_0000     512K      +0x600000  IMEM
//   emulator_controller_0/s_axi  0x201_0000_0000       4G                 MC s_axi
//
// The gaps (+0x402000..0x5fffff and +0x680000..0x7fffff) are undecoded and read
// 0xffffffff.  emu_sanity's all-ones check keys on that: a window that has moved
// shows up there before anything else notices.
#define EMU_OFF_GPR         0x000000u        //   4 MiB, R/W            gpr_wrap_0/s_axi
#define EMU_OFF_CFR         0x400000u        //   4 KB, AXI4-Lite, R/W  dispatcher_top_0/s_lite
#define EMU_OFF_VIOL        0x401000u        //   4 KB, AXI4-Lite, R (+CTRL W)  viol_csr_0/s_axi
#define EMU_OFF_MODE        0x405000u        //   4 KB, AXI4-Lite, R/W  channel address map
#define EMU_OFF_IMEM        0x600000u        // 512 KB, write-only (AR/R is a stub)  dispatcher_top_0/s_axi

#define EMU_SIZE_IMEM       0x080000u
#define EMU_SIZE_CFR        0x001000u
#define EMU_SIZE_VIOL       0x001000u
#define EMU_SIZE_GPR        0x400000u
#define EMU_SIZE_MODE       0x001000u

#define EMU_AXI_IMEM        (EMU_AXI_BASE + EMU_OFF_IMEM)
#define EMU_AXI_CFR         (EMU_AXI_BASE + EMU_OFF_CFR)
#define EMU_AXI_VIOL        (EMU_AXI_BASE + EMU_OFF_VIOL)
#define EMU_AXI_GPR         (EMU_AXI_BASE + EMU_OFF_GPR)
#define EMU_AXI_MODE        (EMU_AXI_BASE + EMU_OFF_MODE)

// ============================== geometry ======================================
// 256 b everywhere: an ISR word, a GPR word and a DRAM beat are all 32 B.
#define EMU_NBANKS          16u              // banks per channel, fixed by the BD
#define EMU_WORD_BYTES      32u              // 256 b: one IMEM cell, one GPR word
#define EMU_LANES_PER_WORD  16u              // BF16 lanes per 256 b beat
#define EMU_IMEM_WORDS      16384u           // 512 KB / 32 B, index = addr[18:5]
#define EMU_GPR_WORDS       131072u          // 4 MiB / 32 B, index = addr[21:5]
#define EMU_ROW_BYTES       2048u            // one DRAM row = 64 beats
#define EMU_BEATS_PER_ROW   64u

// ============================== what is NOT here ==============================
// The channel count, the HBM aperture, the bank stride, the bank window and the
// MC base all VARY BETWEEN IMAGES and live in platform/*.conf, read at run time by
// pim_platform.{h,c}.  They were briefly -D-overridable defaults here, which put
// the same number in two places with nothing comparing them: build without
// -DEMU_NCH=2, run against the ch2 image, and a probe covers half the hardware and
// prints PASS.
//
//     #include "pim_platform.h"
//     struct pim_platform P;
//     const char *e = pim_platform_load(opt_platform, &P);   // NULL name -> active
//     if (e) { fprintf(stderr, "%s\n", e); return 2; }
//     pim_platform_banner(&P);
//
// Then pim_bank(&P, ch, b), pim_mc_bank(&P, ch, b), pim_viol_axi(&P, ch),
// pim_dma_check(&P, axi, len), pim_window_name(&P, axi).

// ============================== CFR ===========================================
// ONE 4 KiB WINDOW, THREE PAGES.  v1 decoded only addr[7:0], so the register block
// aliased every 256 B and that aliasing was how a probe identified it.  v2.0 ended
// that: the statistics live at 0x400 and the clear/summary registers at 0x800, so
// the decode is at least addr[11:0] and there is no aliasing to lean on.
//
//   0x000 - 0x03F   registers, RW
//   0x040 - 0x3FF   reserved, reads 0
//   0x400 - 0x7FF   per-bank violation statistics, RO
//   0x800 - 0x808   clear and summary
//
// Nothing may be written at an offset whose low byte is 0x00 unless a doorbell is
// intended: 0x00 is CTRL and CTRL[0] is the doorbell.
#define CFR_CTRL            0x00u   // W  [0] doorbell (self-clearing)  [1] mode
#define CFR_STATUS          0x04u   // R  [3:0] fetch FSM state  [31] done
#define CFR_T_FAW           0x08u
#define CFR_T_RRD           0x0Cu
#define CFR_T_RCD           0x10u
#define CFR_T_CCD           0x14u
#define CFR_T_RTP           0x18u
#define CFR_T_RP            0x1Cu   // SINGLE-bank PRE -> ACT
#define CFR_T_WR            0x20u
#define CFR_T_RAS           0x24u
#define CFR_PROG_LEN        0x28u   // 14 bits wide (dispatcher_top.v IMEM_AW)
#define CFR_RUN_CYC_LO      0x2Cu   // RO  v2.0
#define CFR_RUN_CYC_HI      0x30u   // RO  v2.0
#define CFR_T_MOD           0x34u   // v2.0  REGISTER <-> BANK mode switch stall
#define CFR_T_RP_AB         0x38u   // v2.0  ALL-bank PRE -> ACT
#define CFR_T_GB            0x3Cu   // v2.0  WRVEC beat -> next beat into the Global Buffer

#define CFR_CTRL_DOORBELL   (1u << 0)
#define CFR_CTRL_MODE_ALLBK (1u << 1)
#define CFR_STATUS_DONE     (1u << 31)
#define CFR_STATUS_STATE(s) ((s) & 0xFu)
#define CFR_PROG_LEN_MAX    ((1u << 14) - 1u)
#define CFR_TIMING_MAX      ((1u << 10) - 1u)   // TW = 10 bits, so 0..1023

// RUN_CYC is a 64-bit count of the cycles the fetcher ran, cleared at the doorbell
// and frozen when STATUS[31] rises.  READ IT AFTER done, never during: LO and HI
// are separate reads and LO can roll between them.
//
// IT DOES NOT INCLUDE WHAT EOS LEAVES BEHIND.  v2.0 is open-page, so the write-back
// of the rows EOS precharges happens after this value stops moving.
#define CFR_RUN_CYC(lo, hi) (((uint64_t)(hi) << 32) | (uint32_t)(lo))

// The eleven timing budgets, in PL clock cycles, 10 bits each (0..1023).  v2.0 has
// three beyond v1's eight:
//
//   T_MOD    the stall when a channel switches between REGISTER mode (WRVEC, RD_MAC)
//            and BANK mode (MAC, COPY, EWMUL).  WRVEC -> MAC -> RD_MAC pays it twice.
//   T_RP_AB  PRECHARGE ALL -> ACT.  An all-bank span and EOS's flush are charged
//            with it; a single-bank PRE with T_RP.
//   T_GB     the spacing of WRVEC's beats into the Global Buffer.  Those beats ride
//            no bank command, so T_CCD does not space them; at 0 they move at the
//            GPR read rate, 2 cycles a beat.  The actual spacing is the larger of the
//            two.
//
// The fields are in the order of the registers' names, not their offsets:
// T_MOD, T_RP_AB and T_GB sit above PROG_LEN and RUN_CYC.
struct emu_timing {
    uint16_t faw, rrd, rcd, ccd, rtp, rp, wr, ras;
    uint16_t mod;      // v2.0  T_MOD
    uint16_t rp_ab;    // v2.0  T_RP_AB
    uint16_t gb;       // v2.0  T_GB
};

// HANDOFF §2's starting values.  Two warnings travel with them:
//   T_CCD MUST NOT drop below 2 — mac_top.sv:26-33, two beats into the same latch
//   back to back accumulate every OTHER beat, with no diagnostic.
//   T_RCD=4 DOES raise ACT_FILL violations and that is correct: HANDOFF §2 measured
//   overrun 3 against an ideal zero-delay memory model.  The emulator is reporting
//   that memory was slower than the model.  T_RCD=8 is quiet if that is wanted.
// T_MOD and T_RP_AB are placeholders: nothing has measured what a mode switch or an
// all-bank precharge costs on this image.  T_GB takes T_CCD's value, as the AiM
// timing does (nCCDS = nCCDL).
#define EMU_TIMING_SIM    ((struct emu_timing){ 30, 6, 4, 2, 3, 3, 4, 6, 0, 3, 2 })
#define EMU_TIMING_QUIET  ((struct emu_timing){ 30, 6, 8, 2, 3, 3, 4, 6, 0, 3, 2 })

// ============================== channel address map ===========================
// The PL channel switch in front of the MC s_axi (0x0204_0000_0000 + ch*4 GiB) splits
// host addresses into channels; its MODE register (BAR2 +0x405000) picks how.
// WHICH FIELD OF AN ADDRESS NAMES THE CHANNEL.  One bit, and it changes what every
// host address means — so it is the one piece of device state that can make a
// correctly written program read someone else's bytes without any error at all.
//
//   0  ChRoBaCo (bypass)      | CH | ROW | BA | CO | byte |
//      Channel is the TOP field, so it changes once every 4 GiB.  A linear transfer
//      stays in one channel: this is what the address map was before the mode
//      existed, when each channel simply had its own 4 GiB of MC window.
//
//   1  RoChBaCo (interleave)  | ROW | CH | BA | CO | byte |
//      Channel drops to just under ROW, so it changes every 32 KiB.  What is left
//      below it — BA + CO + byte — is exactly one RoBaCo row, which is what one
//      all-bank MAC consumes: an operand never straddles a channel.  A linear
//      transfer now spreads across every channel's controller.
//
// THE DIRECT APERTURE IS NOT AFFECTED.  It names (channel, bank, offset) itself and
// never goes through the MC.  What changes is the CONVERSION — which channel and
// bank a given MC-space offset lives in — so a tool that writes through one face and
// reads through the other has to use the conversion for the mode that is set.
//
// CHANGING THIS INVALIDATES EVERYTHING RESIDENT.  The bytes do not move; the meaning
// of the address that reaches them does.  Set it before placing data, not after.
#define MODE_CTRL            0x00u   // RW [0]
#define MODE_STATUS          0x04u   // RO [0] busy: a transaction is in flight on
                                     //     some slave port.  Read it as clear before
                                     //     writing MODE_CTRL.
#define MODE_CTRL_INTERLEAVE (1u << 0)
#define MODE_STATUS_BUSY     (1u << 0)

// ============================== violation statistics ==========================
// INSIDE THE CFR NOW.  v1 gave each channel its own viol_csr slave at BAR2
// +0x401000 + ch*0x1000; v2.0 returned that window and put the statistics in the
// dispatcher's own 4 KiB at 0x400, with the channel as an address field.
//
//     addr = CFR + 0x400 + ch*0x100 + bank*0x10 + reg
//
// THE OLD WINDOW STILL READS.  0x401000-0x404FFF is decoded and empty, so a v1
// host reads zeros there and concludes the run was clean.  That is why this is a
// rename and not an addition -- the v1 names must not survive.
//
// TWO KINDS, NOT SIX.  v1 split tRCD and tCCD into read and write and added a
// recovery and an EWMUL drop.  v2.0 moved every budget into the controller, which
// cannot violate a budget it enforces on itself: the only things that can overrun
// are the two PHYSICAL waits, where the bank takes longer than the budget says.
//
//     ACT_FILL    ACT issued -> row buffer filled     budget T_RCD
//     PRE_DRAIN   PRE issued -> write-back finished   budget T_RP, or T_RP_AB
//                                                     for an all-bank span and for
//                                                     the flush EOS performs
//
// Read-only by construction -- the host cannot forge a clean run.  Counts saturate
// at 0xFF and never wrap, because a wrapped 0 is indistinguishable from a run that
// had none.
#define VIOL_BASE           0x400u
#define VIOL_BANK_BASE(ch, b)  (VIOL_BASE + (uint32_t)(ch) * 0x100u \
                                          + (uint32_t)(b)  * 0x10u)
#define VIOL_STICKY         0x00u   // [0]ACT_FILL [1]PRE_DRAIN [31]any
#define VIOL_CNT            0x04u   // [7:0]ACT_FILL [15:8]PRE_DRAIN, saturating
#define VIOL_MAX            0x08u   // [9:0]ACT_FILL [25:16]PRE_DRAIN, cycles over
#define VIOL_CTRL           0x800u  // W [0] clrstats, self-clearing; reads 0
#define VIOL_ANY0           0x804u  // R [15:0] ch0 banks  [31:16] ch1 banks
#define VIOL_ANY1           0x808u  // R [15:0] ch2 banks  [31:16] ch3 banks

// Two reads name the guilty bank instead of sixty-four.
#define VIOL_ANY_REG(ch)    ((ch) < 2 ? VIOL_ANY0 : VIOL_ANY1)
#define VIOL_ANY_SHIFT(ch)  (((ch) & 1u) ? 16u : 0u)

#define VIOL_STICKY_ACT     (1u << 0)
#define VIOL_STICKY_PRE     (1u << 1)
#define VIOL_STICKY_ANY     (1u << 31)
#define VIOL_CNT_ACT(v)     ((v) & 0xFFu)
#define VIOL_CNT_PRE(v)     (((v) >> 8) & 0xFFu)
#define VIOL_MAX_ACT(v)     ((v) & 0x3FFu)
#define VIOL_MAX_PRE(v)     (((v) >> 16) & 0x3FFu)
#define VIOL_STICKY_DROP    (1u << 8)

// ============================== what DMA may touch ============================
// MOVED to pim_platform.{h,c}.  The allowed set depends on the channel count and
// on the bank window/stride pair, none of which is knowable at compile time —
// use pim_dma_check(&P, axi, len).
//
// What is OBSERVED about getting it wrong, kept here because it is a property of
// this host and driver rather than of any image, and kept separate from what is
// inferred because the inference has been overstated before:
//   observed  a MM transfer to 0x2020_0010_0000 (an address that decodes nowhere)
//             returned EIO, and dmesg showed H2C_MM_ERR_CODE at that moment.
//             Every later transfer also EIO'd, so the engine latched.  The read
//             engine was unaffected.
//   observed  kern.log and syslog reached 19.8 GB each; their content is
//             error_intr_handler + hw_error_process + a full engine register dump,
//             ~9 KB per occurrence, 571 occurrences in the last 5 MB alone.
//   observed  driver source: error_intr_handler (qdma_intr.c:137) prints, calls
//             qdma_hw_error_process, then RE-ARMS without necessarily clearing the
//             condition.
//   inferred  that those three compose into an interrupt storm, and that such a
//             storm is what made the host unresponsive on 2026-07-31.  NOT proven:
//             the mis-addressed transfers happened before an earlier reboot, and
//             every transfer after the last queue setup was in range and succeeded.
//             Do not repeat this as established fact.
//
// This is why the gap above a bank window is checked in SOFTWARE and never probed
// with a real transfer.

// ============================== ISR ==========================================
// One ISR = 256 b = 32 B = one IMEM cell.  Little endian: bit[0] is the LSB of
// byte[0].  A word with the fields in the WRONG PLACES still decodes to a
// plausible instruction — opcode and COL sit low and would survive almost any
// misplacement above them — so a mis-encoded program is wrong SILENTLY rather
// than rejected.  EMU_GOLDEN_* below are the v2.0 document's §4.1 words, and
// emu_isr_golden_check() compares the encoder against them.
#define ISR_OP_MAC      0x0Cu   // verified: all-bank, GB-sourced
#define ISR_OP_EWMUL    0x0Du   // NOT verified on FPGA
#define ISR_OP_COPY     0x0Eu   // read half verified (bank->GB), and it MULTICASTS on
                                // CH_MASK [measured 2026-08-23, emu_gemv --chs 0,1:
                                // CH_MASK=0x3, 32/32 lanes, identical to WRVEC].
                                // Write half is still unverified.
                                //
                                // A MULTICAST COPY MAKES EVERY CHANNEL READ ITS OWN
                                // BANK.  The ISR carries one ROW and each channel
                                // fetches from it locally, so a vector that is SHARED
                                // across channels has to be REPLICATED into each of
                                // them first.  WRVEC does not: it fills the GB from
                                // the GPR, which is one host write for every channel.
                                // So for a shared operand WRVEC is strictly less
                                // work, and COPY earns its place only when the vector
                                // is ALREADY in DRAM — which on this datapath means
                                // an EWMUL result, since RD_MAC lands in the GPR.
#define ISR_OP_WRVEC    0x0Fu   // verified: GPR -> GB vector load
#define ISR_OP_RD_MAC   0x10u   // verified: 16 BF16 lanes land in a GPR word
#define ISR_OP_EOS      0x11u   // verified: end of program
#define ISR_OP_WR_SBK   0x12u   // v2.0: GPR -> one bank.  On hold, the encoder refuses it
#define ISR_OP_RD_SBK   0x13u   // v2.0: one bank -> GPR.  On hold, the encoder refuses it

// route[i] port numbering: 0..15 name a bank, 16 is the GB, 31 is "unused".
// The default MUST be 31 — 0 means "send to bank 0", which is a real destination.
#define GB_PORT         16u
#define GB_NULL_PORT    31u
#define EMU_MAX_OPSIZE  64u     // GB depth, DRAM row beats, and COL+OPSIZE<=64
#define EMU_ISR_ROW_MAX ((1u << 17) - 1u)

struct emu_isr { uint64_t w[4]; };   // w[0] = bits[63:0] ... w[3] = bits[255:192]

static inline void emu_isr_set(struct emu_isr *p, unsigned lo, unsigned width, uint64_t v)
{
    for (unsigned i = 0; i < width; i++) {
        unsigned bit = lo + i;
        if ((v >> i) & 1ULL) p->w[bit >> 6] |= 1ULL << (bit & 63);
    }
}
static inline uint64_t emu_isr_get(const struct emu_isr *p, unsigned lo, unsigned width)
{
    uint64_t v = 0;
    for (unsigned i = 0; i < width; i++) {
        unsigned bit = lo + i;
        if ((p->w[bit >> 6] >> (bit & 63)) & 1ULL) v |= 1ULL << i;
    }
    return v;
}

// THE v2.0 PACKING.  v1 left 93 bits of holes between the fields; v2.0 closed
// them and pulled everything down, so the word is contiguous from 0 to 179 and
// [255:180] is reserved.  What did NOT move is the bottom 36 bits -- COL, ROW,
// BK, CH_MASK and T are where they were -- which is exactly why crossing an
// image with the wrong build does not fail: a v1 program loads, runs, and
// answers with the wrong numbers.
//
//   field        v1          v2.0
//   COL           0, 6        0, 6     unchanged
//   ROW           6,17        6,17     unchanged
//   BK           23, 4       23, 4     unchanged
//   CH_MASK      27, 8       27, 8     unchanged
//   T            35, 1       35, 1     unchanged
//   OPSIZE       49,10       36,10
//   OPCODE       59, 5       46, 5
//   PU_MASK      78,16       51,16
//   ROUTE(i)   96+5i, 5    67+5i, 5
//   GB_MC       176,16      147,16     v1's bits 180..191 are v2's reserved zone
//   GPR_ADDR2      --       163,17     new: the GPR side of WR_SBK / RD_SBK
//
// Positions are the RTL's, via hw/ch2/version2.0/pim_v2.0_ISR_주소맵.md §1, which
// names src/front_end/emulator_controller/rtl/pim_isr_defs.vh as the authority.
#define ISR_F_OPCODE    46, 5
#define ISR_F_OPSIZE    36, 10
#define ISR_F_T         35, 1
#define ISR_F_CHMASK    27, 8
#define ISR_F_BK        23, 4
#define ISR_F_ROW        6, 17
#define ISR_F_COL        0, 6
#define ISR_F_PUMASK    51, 16
#define ISR_F_GBMC     147, 16
#define ISR_F_GPRADDR2 163, 17
#define ISR_F_ROUTE(i) (67 + 5 * (i)), 5

// Everything above 179 must be zero.  The encoder asserts it rather than trusting
// itself: a field written at a v1 offset lands here, and that is the one mistake
// this repack makes easy.
#define ISR_RESERVED_LO 180u

struct emu_isr_spec {
    uint32_t opcode;
    uint32_t opsize;      // BEATS, not elements.  K/16 for a dot product.
    uint32_t row;         // DRAM row for MAC/COPY/EWMUL; GPR word for WRVEC/RD_MAC
    uint32_t col;
    uint32_t bk;
    uint32_t ch_mask;     // CH_MASK[i] = 1 runs this ISR on channel i.  ISR[34:27]
    uint32_t pu_mask;     // banks whose PU computes.  MAC/EWMUL only.
    // MAC: the banks the GB sends the vector to, equal to pu_mask; 0 means the
    // vector comes from a peer bank instead.  COPY: the banks the write half fills.
    uint32_t gb_mc_mask;
    uint8_t  route[16];   // 31 = unused.  0 would mean "send to bank 0".
    // Accumulator latch, 0 or 1.  A RD_MAC reads and clears the latch it names, so it
    // has to name the one its MACs accumulated into; the other reads back 0.
    bool     t;
};

// A GB-sourced MAC's gb_mc_mask: one bank, one bank of every bank group
// (0x1111, 0x2222, 0x4444, 0x8888), or all sixteen (v2.0 document §2.1).
static inline bool emu_gb_mc_valid(uint32_t m)
{
    return __builtin_popcount(m) == 1 || m == 0x1111u || m == 0x2222u ||
           m == 0x4444u || m == 0x8888u || m == 0xFFFFu;
}

// The per-ISR preconditions of the v2.0 document §2 and §5 (V1-V4).  NULL if
// legal, else the reason.  The hardware checks none of them, so an ISR that breaks
// one gives a wrong number or a hang rather than an error.
static inline const char *emu_isr_check(const struct emu_isr_spec *s)
{
    if (s->opcode > 0x1Fu)           return "opcode is wider than 5 bits";
    if (s->opcode == ISR_OP_WR_SBK || s->opcode == ISR_OP_RD_SBK)
        return "WR_SBK and RD_SBK are on hold (v2.0 document §2.7): their data path is "
               "being redesigned, and they overwrite the vector WRVEC left in the GB";
    if (s->ch_mask == 0)             return "CH_MASK must not be 0 (use 0x01)";
    if (s->ch_mask > 0xFFu)          return "CH_MASK is wider than 8 bits";
    if (s->row > EMU_ISR_ROW_MAX)    return "ROW/GPR_ADDR exceeds 17 bits";
    if (s->col >= EMU_BEATS_PER_ROW) return "COL must be < 64";
    if (s->bk >= EMU_NBANKS)         return "BK must be < 16";

    bool sized = (s->opcode == ISR_OP_MAC || s->opcode == ISR_OP_WRVEC ||
                  s->opcode == ISR_OP_COPY || s->opcode == ISR_OP_EWMUL);
    if (sized) {
        if (s->opsize == 0)                         return "OPSIZE must be >= 1";
        if (s->opsize > EMU_MAX_OPSIZE)             return "OPSIZE > 64 wedges WRVEC forever with no message";
        if (s->col + s->opsize > EMU_BEATS_PER_ROW) return "COL + OPSIZE > 64 splits the madi";
    }
    if (s->pu_mask > 0xFFFFu)    return "pu_mask is wider than 16 bits";
    if (s->gb_mc_mask > 0xFFFFu) return "gb_mc_mask is wider than 16 bits";

    // gb_mc_mask is NOT simply "who receives the GB broadcast".  On MAC it is also
    // the SOURCING SELECTOR: nonzero means the vector comes from the GB (and route
    // is ignored), zero means it comes from a peer bank over the crossbar (and
    // route carries the pairing).  A check that demanded gb_mc_mask == pu_mask
    // unconditionally would make peer sourcing and EWMUL unencodable.
    if (s->opcode == ISR_OP_MAC) {
        if (s->pu_mask == 0)     return "pu_mask is 0: no bank would compute";
        if (s->gb_mc_mask && s->pu_mask != s->gb_mc_mask)
            return "GB-sourced MAC (gb_mc_mask != 0): gb_mc_mask must equal pu_mask, "
                   "or a bank receives the vector without computing (or the reverse)";
        if (s->gb_mc_mask && !emu_gb_mc_valid(s->gb_mc_mask))
            return "GB-sourced MAC: gb_mc_mask must be one bank, 0x1111/0x2222/0x4444/"
                   "0x8888, or 0xFFFF; the hardware does not split other masks";
    }
    if (s->opcode == ISR_OP_EWMUL) {
        if (__builtin_popcount(s->pu_mask) != 1)
            return "EWMUL runs exactly one group at a time: popcount(pu_mask) must be 1";
        if (s->gb_mc_mask)
            return "EWMUL sources from a peer bank, so gb_mc_mask must be 0";
    }
    if (s->opcode == ISR_OP_RD_MAC) {
        if (s->opsize != 0)                    return "RD_MAC OPSIZE must be 0: the result is always one aggregate beat";
        if (s->pu_mask || s->gb_mc_mask)       return "RD_MAC broadcasts to every bank; pu_mask and gb_mc_mask must be 0";
        if (__builtin_popcount(s->ch_mask) != 1)
            return "RD_MAC CH_MASK must be 1-hot, or two channels write the same GPR word";
    }
    // WRVEC, RD_MAC and EOS route nothing.  Their route and gb_mc_mask still reach
    // the crossbar while the channel waits (EOS: the whole flush), so route stays
    // all 31 and gb_mc_mask 0.
    if (s->opcode == ISR_OP_WRVEC || s->opcode == ISR_OP_RD_MAC || s->opcode == ISR_OP_EOS) {
        if (s->pu_mask || s->gb_mc_mask)
            return "WRVEC, RD_MAC and EOS take pu_mask = gb_mc_mask = 0";
        for (int i = 0; i < 16; i++)
            if (s->route[i] != GB_NULL_PORT)
                return "WRVEC, RD_MAC and EOS take route[] all 31 (NULL); 0 is bank 0";
    }

    // route[i]: 0..15 = that bank, 16 = the GB, 31 = unused.  17..30 name nothing.
    for (int i = 0; i < 16; i++) {
        if (s->route[i] > 31u)                       return "route[i] must be <= 31";
        if (s->route[i] > GB_PORT && s->route[i] != GB_NULL_PORT)
            return "route[i] must be a bank (0..15), the GB (16), or unused (31)";
        if (s->route[i] == (uint8_t)i)               return "route[i] = i is forbidden: a bank cannot feed itself";
    }
    // V1: each bank is the destination of at most one stream.
    for (int i = 0; i < 16; i++) {
        if (s->route[i] >= EMU_NBANKS) continue;      // GB or unused
        for (int j = i + 1; j < 16; j++)
            if (s->route[j] == s->route[i])
                return "two route entries name the same destination bank";
        if ((s->gb_mc_mask >> s->route[i]) & 1u)
            return "a route destination is also in gb_mc_mask: that bank would get two streams";
    }
    return NULL;
}

static inline const char *emu_isr_build(struct emu_isr *out, const struct emu_isr_spec *s)
{
    const char *bad = emu_isr_check(s);
    if (bad) return bad;
    memset(out, 0, sizeof *out);
    emu_isr_set(out, ISR_F_OPCODE, s->opcode);
    emu_isr_set(out, ISR_F_OPSIZE, s->opsize);
    emu_isr_set(out, ISR_F_T,      s->t ? 1u : 0u);
    emu_isr_set(out, ISR_F_CHMASK, s->ch_mask);
    emu_isr_set(out, ISR_F_BK,     s->bk);
    emu_isr_set(out, ISR_F_ROW,    s->row);
    emu_isr_set(out, ISR_F_COL,    s->col);
    emu_isr_set(out, ISR_F_PUMASK, s->pu_mask);
    emu_isr_set(out, ISR_F_GBMC,   s->gb_mc_mask);
    for (int i = 0; i < 16; i++) emu_isr_set(out, ISR_F_ROUTE(i), s->route[i]);
    return NULL;
}

// Defaults that are right for nearly every ISR.  route[] all-31 matters: 0 would
// mean "send to bank 0".
static inline struct emu_isr_spec emu_isr_default(uint32_t opcode)
{
    struct emu_isr_spec s;
    memset(&s, 0, sizeof s);
    s.opcode  = opcode;
    s.ch_mask = 0x01u;      // channel 0.  emu_isr_default_ch() for anything else.
    for (int i = 0; i < 16; i++) s.route[i] = GB_NULL_PORT;
    return s;
}

// Same, on a chosen channel.  ch_mask is a BITMASK, so 1<<c is one channel and
// 0b11 is a multicast — which is legal for WRVEC/COPY/MAC but NOT for RD_MAC,
// where two channels would write the same GPR word (emu_isr_check enforces that).
static inline struct emu_isr_spec emu_isr_default_ch(uint32_t opcode, uint32_t ch_mask)
{
    struct emu_isr_spec s = emu_isr_default(opcode);
    s.ch_mask = ch_mask;
    return s;
}

// STATUS[31] fires when the FETCHER accepts the last ISR, not when the kernel
// finishes.  The ISR stream is single-outstanding, so everything BEFORE the last
// ISR has retired by then — but the last one has not.  A program ending in RD_MAC
// therefore reports done while its own result is still in flight.
static inline bool emu_isr_produces_result(uint32_t opcode) { return opcode == ISR_OP_RD_MAC; }

// The v2.0 document's §4.1 GEMV tile, worked out by the RTL side: WRVEC 64 beats
// from GPR word 0, MAC 64 beats of row 100 from beat 0 on all 16 banks, RD_MAC into
// GPR word 1041, EOS; channel 0, T 0.  MSB on the left.
#define EMU_GOLDEN_WRVEC  "0000000000000000000000000007fffffffffffffffffff80003c40008000000"
#define EMU_GOLDEN_MAC    "000000000000000000000007fffffffffffffffffffffffffffb040008001900"
#define EMU_GOLDEN_RD_MAC "0000000000000000000000000007fffffffffffffffffff80004000008010440"
#define EMU_GOLDEN_EOS    "0000000000000000000000000007fffffffffffffffffff80004400008000000"

// An ISR word as 64 hex digits, MSB on the left, the form EMU_GOLDEN_* are written in.
static inline void emu_isr_hex(const struct emu_isr *p, char out[65])
{
    snprintf(out, 65, "%016llx%016llx%016llx%016llx",
             (unsigned long long)p->w[3], (unsigned long long)p->w[2],
             (unsigned long long)p->w[1], (unsigned long long)p->w[0]);
}

// Builds the §4.1 tile with the encoder and compares it with EMU_GOLDEN_*.  NULL if
// every word matches, else which one differs.  Needs no board.
static inline const char *emu_isr_golden_check(void)
{
    static const char *const want[4] = {
        EMU_GOLDEN_WRVEC, EMU_GOLDEN_MAC, EMU_GOLDEN_RD_MAC, EMU_GOLDEN_EOS };
    static const char *const name[4] = {
        "WRVEC differs from the v2.0 document §4.1 word",
        "MAC differs from the v2.0 document §4.1 word",
        "RD_MAC differs from the v2.0 document §4.1 word",
        "EOS differs from the v2.0 document §4.1 word" };
    struct emu_isr_spec s[4];
    struct emu_isr w;
    char hex[65];

    s[0] = emu_isr_default(ISR_OP_WRVEC);  s[0].opsize = 64; s[0].row = 0;
    s[1] = emu_isr_default(ISR_OP_MAC);    s[1].opsize = 64; s[1].row = 100; s[1].col = 0;
    s[1].pu_mask = 0xFFFFu; s[1].gb_mc_mask = 0xFFFFu;
    s[2] = emu_isr_default(ISR_OP_RD_MAC); s[2].opsize = 0;  s[2].row = 1041;
    s[3] = emu_isr_default(ISR_OP_EOS);
    for (int i = 0; i < 4; i++) {
        if (emu_isr_build(&w, &s[i])) return name[i];
        emu_isr_hex(&w, hex);
        if (strcmp(hex, want[i])) return name[i];
    }
    return NULL;
}

// ============================== BF16 =========================================
static inline uint16_t emu_f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    if (((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu))  // NaN stays a NaN
        return (uint16_t)((u >> 16) | 0x40u);
    uint32_t lsb = (u >> 16) & 1u;                        // round to nearest even
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}
static inline float emu_bf16_to_f32(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}
// Compare as VALUES: +0 and -0 are the same number, and a bitwise test would
// report a spurious mismatch on a zero row.
static inline bool emu_bf16_same(uint16_t a, uint16_t b)
{
    if (a == b) return true;
    return ((a | b) & 0x7FFFu) == 0;
}

#endif // EMU_REGS_H
