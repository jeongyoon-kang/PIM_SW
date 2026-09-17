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
// dispatcher_top.v decodes only addr[7:0], so the block REPEATS every 256 B
// inside its 4 KB segment.  Two consequences for host code:
//   - that aliasing identifies the CFR behaviourally, and
//   - nothing may be written at an offset whose low byte is 0x00 unless a
//     doorbell is intended, because 0x00 is CTRL and CTRL[0] is the doorbell.
#define CFR_CTRL            0x00u   // W  [0] doorbell (self-clearing)  [1] mode
#define CFR_STATUS          0x04u   // R  [3:0] fetch FSM state  [31] done
#define CFR_T_FAW           0x08u
#define CFR_T_RRD           0x0Cu
#define CFR_T_RCD           0x10u
#define CFR_T_CCD           0x14u
#define CFR_T_RTP           0x18u
#define CFR_T_RP            0x1Cu
#define CFR_T_WR            0x20u
#define CFR_T_RAS           0x24u
#define CFR_PROG_LEN        0x28u   // 14 bits wide (dispatcher_top.v IMEM_AW)

#define CFR_CTRL_DOORBELL   (1u << 0)
#define CFR_CTRL_MODE_ALLBK (1u << 1)
#define CFR_STATUS_DONE     (1u << 31)
#define CFR_STATUS_STATE(s) ((s) & 0xFu)
#define CFR_PROG_LEN_MAX    ((1u << 14) - 1u)

struct emu_timing { uint8_t faw, rrd, rcd, ccd, rtp, rp, wr, ras; };

// HANDOFF §2's starting values.  Two warnings travel with them:
//   T_CCD MUST NOT drop below 2 — mac_top.sv:26-33, two beats into the same latch
//   back to back accumulate every OTHER beat, with no diagnostic.
//   T_RCD=4 DOES raise RCD_RD violations and that is correct: HANDOFF §2 measured
//   overrun 3 against an ideal zero-delay memory model.  The emulator is reporting
//   that memory was slower than the model.  T_RCD=8 is quiet if that is wanted.
#define EMU_TIMING_SIM    ((struct emu_timing){ 30, 6, 4, 2, 3, 3, 4, 6 })
#define EMU_TIMING_QUIET  ((struct emu_timing){ 30, 6, 8, 2, 3, 3, 4, 6 })

// ============================== channel address map ===========================
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

// ============================== violation CSR =================================
// Read-only by construction — the host cannot forge a clean run.  Counts saturate
// at 0xFF and never wrap.  Everything resets to zero (emu_viol_csr.v:283-292), and
// an offset the block does NOT decode also reads zero (emu_viol_csr.v:47) — which
// is what tells a live block apart from undecoded BAR space, since the latter
// reads 0xffffffff.
#define VIOL_BANK_BASE(b)   ((uint32_t)(b) * 0x40u)
#define VIOL_STICKY         0x00u   // [0]RCD_RD [1]CCD_RD [2]RCD_WR [3]CCD_WR
                                    // [4]RECOVERY_WR [8]ewmul_drop [31]any
#define VIOL_CNT_A          0x04u   // [7:0]RCD_RD [15:8]CCD_RD [23:16]RCD_WR [31:24]CCD_WR
#define VIOL_CNT_B          0x08u   // [7:0]RECOVERY_WR [15:8]ewmul_drop
#define VIOL_MAX_A          0x0Cu   // same packing as CNT_A, worst overrun in cycles
#define VIOL_MAX_B          0x10u   // [7:0]RECOVERY_WR
#define VIOL_CTRL           0x400u  // W [0] clrstats, self-clearing; reads 0
#define VIOL_ANY            0x404u  // R [15:0] one bit per bank — read this first

#define VIOL_STICKY_ANY     (1u << 31)
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
// than rejected.  EMU_GOLDEN_* below are HANDOFF §3.5's words, to check the
// encoder against something the hardware actually executed.
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

#define ISR_F_OPCODE    59, 5
#define ISR_F_OPSIZE    49, 10
#define ISR_F_T         35, 1
#define ISR_F_CHMASK    27, 8
#define ISR_F_BK        23, 4
#define ISR_F_ROW        6, 17
#define ISR_F_COL        0, 6
#define ISR_F_PUMASK    78, 16
#define ISR_F_GBMC     176, 16
#define ISR_F_ROUTE(i) (96 + 5 * (i)), 5

struct emu_isr_spec {
    uint32_t opcode;
    uint32_t opsize;      // BEATS, not elements.  K/16 for a dot product.
    uint32_t row;         // DRAM row for MAC/COPY/EWMUL; GPR word for WRVEC/RD_MAC
    uint32_t col;
    uint32_t bk;
    // CH_MASK[i]=1 means "execute this ISR on channel i".  ISR[34:27].
    // Whether the ch2 build actually fans out on it is UNVERIFIED: fetch_decode.v
    // carries `// TODO(CH_MASK)` and the only RTL that reads the field is the
    // parked validity_gate.  A tool that sets it to 1<<ch and gets the wrong
    // channel's answer has found that out; that is the point of setting it.
    uint32_t ch_mask;
    uint32_t pu_mask;     // banks whose PU computes.  MAC/EWMUL only.
    uint32_t gb_mc_mask;  // GB broadcast destinations.  MUST equal pu_mask.
    uint8_t  route[16];   // 31 = unused.  0 would mean "send to bank 0".
    bool     t;           // must be 0: the MC hard-ties latch_sel
};

// HANDOFF §3.7's per-ISR preconditions.  NULL if legal, else the reason.
// These are PRECONDITIONS, not advice: the validity gate was removed from the
// fetch path for timing, so the hardware neither checks an ISR nor reports a bad
// one.  Breaking one of these produces a wrong NUMBER, not an error.
static inline const char *emu_isr_check(const struct emu_isr_spec *s)
{
    if (s->opcode > 0x1Fu)           return "opcode is wider than 5 bits";
    if (s->t)                        return "T must be 0 (the MC hard-ties latch_sel; a 1 is silently ignored)";
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

// HANDOFF §3.5's golden words — what simulation actually pushed through IMEM,
// MSB on the left.  If the encoder reproduces these the field placement is right.
#define EMU_GOLDEN_WRVEC  "00000000000000000000ffffffffffffffffffff000000007880000008000000"
#define EMU_GOLDEN_MAC    "0000000000000000ffffffffffffffffffffffff3fffc0006080000008001900"
#define EMU_GOLDEN_RD_MAC "00000000000000000000ffffffffffffffffffff000000008000000008010440"
#define EMU_GOLDEN_EOS    "00000000000000000000ffffffffffffffffffff000000008800000008000000"

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
