// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_platform — the addresses, as compile-time constants.
//
// The values come from platform/<name>.conf via `hwdef/gen_config.sh --defs`, which
// scripts/setup.sh turns into -D arguments for the whole build.  pim_config.h holds
// a default for each one (ch4) so a bare `make` still compiles; PIM_CONFIG_FROM_CONF
// records which of the two happened, and pim_platform_check() refuses the defaults.  They are fixed for as long as one bitstream is on the board, so
// there is nothing to load, nothing to pass around, and nothing to get wrong at
// run time — the same status PIM_NBANK and the 256 b geometry always had.  There
// used to be a struct threaded through every call and a conf parser to fill it;
// see gen_config.sh for why that went.
//
// THE ONE THING THAT IS STILL CHECKED AT RUN TIME
// The hardware cannot report its own channel count, so nothing can compare this
// build against the board.  What CAN be compared is this build against the
// platform the user has selected, and that is the mistake that actually happens:
// setup.sh --platform ch1, forget to make, run a ch2 binary.  pim_platform_check()
// is a readlink and a strcmp.
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_PLATFORM_H
#define PIM_PLATFORM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "emu_regs.h"       // EMU_ROW_BYTES / EMU_NBANKS: the MC address math needs
                            // the DRAM page size, and it has exactly one home.
#include "pim_config.h"     // static defaults; -D overrides them.  See gen_config.sh

// ---- what the conf must satisfy, checked where a violation is a build error ----
// CH4 halves the bank stride while the window stays 256 MiB, and getting that pair
// wrong by hand is the likeliest editing mistake in those files.
//
// THESE DO NOT CATCH A PARTIAL -D.  They relate the geometry values to each other,
// and PIM_NCH appears in none of them — a build that overrode the channel count and
// nothing else passes all five while addressing the wrong stride.  That is why
// gen_config.sh emits every key every time instead of only the ones that differ.
_Static_assert(PIM_NCH >= 1 && PIM_NCH <= 8, "PIM_NCH out of range");
_Static_assert(PIM_NBANK == 16, "the BD instantiates 16 banks per channel");
_Static_assert(PIM_BANK_WINDOW <= PIM_BANK_STRIDE, "bank windows would overlap");
_Static_assert((uint64_t)PIM_NBANK * PIM_BANK_STRIDE <= PIM_HBM_CH_SPAN,
               "the banks of one channel overrun HBM_CH_SPAN");
_Static_assert(PIM_BANK_WINDOW % 2048 == 0, "BANK_WINDOW is not a whole 2048 B row");

// Derived, never -D'd: overriding it would let it disagree with PIM_NCH, and
// pim_hbm_decode() uses it to decide what is inside the aperture at all.
#define PIM_HBM_APERTURE  ((uint64_t)PIM_NCH * PIM_HBM_CH_SPAN)

// ---- addresses ---------------------------------------------------------------
// Path A, the direct aperture.  WINDOW IS NOT STRIDE: a bank decodes
// PIM_BANK_WINDOW bytes and the next one starts PIM_BANK_STRIDE later, so there is
// an undecoded gap between them.  One constant doing both jobs is what let a probe
// push 1 GiB into a 256 MiB window.
static inline uint64_t pim_bank(unsigned ch, unsigned b)
{ return PIM_HBM_BASE + (uint64_t)ch * PIM_HBM_CH_SPAN + (uint64_t)b * PIM_BANK_STRIDE; }

// Path B, the MC.  THE MC DECODES RoBaCo INSIDE A CHANNEL, which is the runtime's
// own model — so bank is a MIDDLE field, not the top one, and "the base address of
// bank b" does not exist.  There is no pim_mc_bank(); there cannot be.
//
//     robaco(row, bank, col) = row*32768 + bank*2048 + col*32
//
// The consequence that bites every caller: a run of bytes that is CONTIGUOUS inside
// one bank is NOT contiguous here.  It is 2048 B every 32768 B.  Conversely one
// contiguous 32 KiB transfer through this window covers all 16 banks' page — which
// is exactly what an all-bank MAC reads, and why a linear upload can now go through
// the MC at all.  Anything that transfers per-bank must walk page by page.
#define PIM_ROBACO_ROW_BYTES ((uint64_t)EMU_NBANKS * EMU_ROW_BYTES)   // 32 KiB

// AND THAT IS THE WHOLE INTERFACE.  There is no pim_mc_addr(ch, bank, off): a
// caller that has a bank in hand is holding the direct window's coordinates, and
// converting them here would let that thinking cross into a path where it does not
// belong.  An MC address is a channel offset; the controller splits it up.
//
// The conversion, for when a message has to name both faces: a byte at channel
// offset A is at pim_mc_ch_base(ch) + A on this window, and at
// pim_bank(ch, (A % 32768) / 2048) + (A / 32768) * 2048 + A % 2048 on the direct
// one.  pim_window_name() prints the MC side of it.
static inline uint64_t pim_mc_ch_base(unsigned ch)
{ return PIM_MC_BASE + (uint64_t)ch * PIM_MC_CH_SPAN; }

// An offset in the MC space reaches the same byte at PIM_MC_BASE + a IN BOTH MODES —
// the mode changes how the controller splits `a`, not where the window is.
static inline uint64_t pim_mc_at(uint64_t a) { return PIM_MC_BASE + a; }

// ---- the channel address map --------------------------------------------------
// COMPILED IN, like everything else here.  It is a register on the board
// (MODE_CTRL, emu_regs.h) and could have been read at run time — but then every
// caller would carry a map argument, and two callers could disagree.  It is chosen
// once by scripts/setup.sh --map, exactly the way the channel count is, and a build
// checks the board against itself at open rather than following it.
//
// So: nothing above this file passes a map around.  pim_decode() answers with the
// one this build was made for.  pim_decode_as() exists only for the probes, whose
// job is to exercise BOTH maps against one board.
typedef enum { PIM_MAP_CHROBACO = 0, PIM_MAP_ROCHBACO = 1 } pim_addr_map;

#define PIM_ADDR_MAP_NAME  (PIM_ADDR_MAP ? "RoChBaCo" : "ChRoBaCo")
#define PIM_ADDR_MAP_WHAT  (PIM_ADDR_MAP ? "interleave, channel every 32 KiB" \
                                         : "bypass, channel every 4 GiB")

// Where MC-space offset `a` physically lives, so the DIRECT aperture can reach it:
//   direct address = pim_bank(ch, bank) + bkoff
// The direct aperture itself does not change with the map; only this does.
static inline void pim_decode_as(pim_addr_map m, uint64_t a,
                                 unsigned *ch, unsigned *bank, uint64_t *bkoff)
{
    uint64_t rowf, within;
    unsigned c;
    if (m == PIM_MAP_ROCHBACO) {
        rowf   = a / ((uint64_t)PIM_NCH * PIM_ROBACO_ROW_BYTES);
        c      = (unsigned)((a / PIM_ROBACO_ROW_BYTES) % PIM_NCH);
        within = a % PIM_ROBACO_ROW_BYTES;
    } else {
        c      = (unsigned)(a / PIM_MC_CH_SPAN);
        uint64_t inch = a % PIM_MC_CH_SPAN;
        rowf   = inch / PIM_ROBACO_ROW_BYTES;
        within = inch % PIM_ROBACO_ROW_BYTES;
    }
    if (ch)    *ch    = c;
    if (bank)  *bank  = (unsigned)(within / EMU_ROW_BYTES);
    if (bkoff) *bkoff = rowf * EMU_ROW_BYTES + (within % EMU_ROW_BYTES);
}

// The first MC-space offset that belongs to channel `ch`.  4 GiB apart when the
// channel is the top field, one RoBaCo row apart when it is interleaved.
static inline uint64_t pim_ch_first_as(pim_addr_map m, unsigned ch)
{ return (uint64_t)ch * (m == PIM_MAP_ROCHBACO ? PIM_ROBACO_ROW_BYTES : PIM_MC_CH_SPAN); }

// ---- what everything above hwdef actually calls -------------------------------
static inline void pim_decode(uint64_t a, unsigned *ch, unsigned *bank, uint64_t *bkoff)
{ pim_decode_as((pim_addr_map)PIM_ADDR_MAP, a, ch, bank, bkoff); }

static inline uint64_t pim_ch_first(unsigned ch)
{ return pim_ch_first_as((pim_addr_map)PIM_ADDR_MAP, ch); }

// The direct-aperture address of MC-space offset `a`, in one call.
static inline uint64_t pim_direct_at(uint64_t a)
{ unsigned c, b; uint64_t o; pim_decode(a, &c, &b, &o); return pim_bank(c, b) + o; }

// NULL if the board's MODE_CTRL agrees with what this build was compiled for, else a
// reason.  `mode_ctrl` is the register's value, which the caller reads over MMIO —
// this file does not touch the device.  Same shape as pim_platform_check(): the
// build states what it needs and refuses to run against anything else, rather than
// quietly following whatever it finds.
const char *pim_addr_map_check(uint32_t mode_ctrl);

// One violation CSR per channel.
static inline uint64_t pim_viol_axi(unsigned ch)
{ return PIM_BAR2_AXI_BASE + PIM_OFF_VIOL + (uint64_t)ch * 0x1000ull; }

// Decompose a direct-aperture address.  false if `axi` is outside it entirely.
// *in_window separates "inside a decoded bank" from "in the gap above one".
bool pim_hbm_decode(uint64_t axi, unsigned *ch, unsigned *bank,
                    uint64_t *off, bool *in_window);

// NULL if [axi, axi+len) lies wholly inside one DMA-legal window; otherwise the
// reason, which the caller should print and then NOT attempt the transfer.
//
// THERE IS NO ALIGNMENT RULE HERE ANY MORE.  It used to refuse anything that was not
// whole 32 B beats, on the grounds that the slaves had no WSTRB and a partial beat
// would overwrite its neighbours.  The hardware does narrow transfers now: a 1 B
// access lands as exactly 1 B on GPR, on the direct aperture and on MC, over DMA and
// over MMIO alike, including across a 32 B boundary [measured 2026-08-19].  Refusing
// what the bus performs correctly only makes callers work around this function.
//
// What is still checked is what software cannot recover from: an address that
// decodes nowhere, the undecoded gap above a bank window (a DMA there latches the
// H2C engine), a transfer that runs off the end of its window, and the AXI-Lite
// blocks — where the GPR ends one beat before the CFR, whose offset 0 is the
// doorbell.
#define PIM_DMA_ALLOW_MC   0x1u
const char *pim_dma_check_ex(uint64_t axi, size_t len, unsigned flags);
static inline const char *pim_dma_check(uint64_t axi, size_t len)
{ return pim_dma_check_ex(axi, len, 0u); }

// Which window an address is in, for messages.  NULL if none.  Not reentrant.
const char *pim_window_name(uint64_t axi);

// One line every tool prints at startup, so a run made against the wrong topology
// says so in its own output rather than in the reader's memory.
void pim_platform_banner(void);

// NULL if this build matches platform/active, else a reason to print and exit.
// A tree that has been moved (PIM_CONF_PATH unreadable) returns NULL with a warning
// on stderr: it cannot tell agreement from disagreement, so it claims neither.
const char *pim_platform_check(void);

#endif // PIM_PLATFORM_H
