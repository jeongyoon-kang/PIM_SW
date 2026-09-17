// SPDX-License-Identifier: MIT
// pim_platform — see pim_platform.h.
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "pim_platform.h"
#include "emu_regs.h"

bool pim_hbm_decode(uint64_t axi, unsigned *ch, unsigned *bank,
                    uint64_t *off, bool *in_window)
{
    if (axi < PIM_HBM_BASE || axi >= PIM_HBM_BASE + PIM_HBM_APERTURE) return false;
    uint64_t rel  = axi - PIM_HBM_BASE;
    unsigned c    = (unsigned)(rel / PIM_HBM_CH_SPAN);
    uint64_t inch = rel % PIM_HBM_CH_SPAN;
    unsigned b    = (unsigned)(inch / PIM_BANK_STRIDE);
    uint64_t o    = inch % PIM_BANK_STRIDE;
    if (ch)        *ch        = c;
    if (bank)      *bank      = b;
    if (off)       *off       = o;
    if (in_window) *in_window = (b < PIM_NBANK) && (o < PIM_BANK_WINDOW);
    return true;
}

const char *pim_dma_check_ex(uint64_t axi, size_t len, unsigned flags)
{
    if (len == 0) return NULL;

    // HBM first: it is the only aperture whose decoded size differs from its stride,
    // so it needs a two-part answer a {base, size} table cannot give.
    {
        bool inw;
        uint64_t off;
        if (pim_hbm_decode(axi, NULL, NULL, &off, &inw)) {
            if (!inw)
                return "address is in the undecoded gap ABOVE a bank window.  A bank "
                       "decodes BANK_WINDOW bytes but the banks are spaced "
                       "BANK_STRIDE apart; anything past the window reaches nothing, "
                       "and a DMA there latches the H2C engine.";
            if (off + len > PIM_BANK_WINDOW)
                return "transfer starts inside a bank window but runs past its end "
                       "into the undecoded gap above it";
            return NULL;
        }
    }

    struct win { uint64_t base, size; bool dma_ok; };
    static const struct win W[] = {
        { PIM_BAR2_AXI_BASE + PIM_OFF_IMEM, EMU_SIZE_IMEM, true  },
        { PIM_BAR2_AXI_BASE + PIM_OFF_GPR,  EMU_SIZE_GPR,  true  },
        // AXI-Lite: MMIO-only by the transport policy, so DMA is refused.  Not only
        // tidiness — the GPR ENDS where the CFR BEGINS, so a GPR word index that is
        // too large lands on the CFR, whose offset 0x00 is CTRL and whose bit 0 is
        // the DOORBELL.  A mis-aimed operand write would start a kernel.
        { PIM_BAR2_AXI_BASE + PIM_OFF_CFR,  EMU_SIZE_CFR,  false },
        { PIM_BAR2_AXI_BASE + PIM_OFF_VIOL, (uint64_t)PIM_NCH * EMU_SIZE_VIOL, false },
        { EMU_AXI_MODE,                     EMU_SIZE_MODE,                     false },
        { PIM_MC_BASE, (uint64_t)PIM_NCH * PIM_MC_CH_SPAN,                     false },
    };
    for (unsigned i = 0; i < sizeof W / sizeof W[0]; i++) {
        if (axi < W[i].base || axi >= W[i].base + W[i].size) continue;
        if (!W[i].dma_ok) {
            if (W[i].base == PIM_MC_BASE) {
                if (!(flags & PIM_DMA_ALLOW_MC))
                    return "address is in MC s_axi.  It decodes, and it now maps "
                           "RoBaCo like the runtime does — but UPLOAD_PATH is direct, "
                           "so nothing in the library should be addressing it.  A tool "
                           "that means to use this window passes allow_mc.";
            } else {
                return "address is an AXI-Lite block (CFR / violation CSR), which is "
                       "MMIO-only — and the CFR begins one beat past the end of the "
                       "GPR, so this is what a GPR word index that is too large looks "
                       "like.  CFR offset 0x00 is CTRL and CTRL[0] is the doorbell";
            }
        }
        if (axi + len > W[i].base + W[i].size)
            return "transfer starts inside a window but runs past its end";
        return NULL;
    }
    return "address decodes nowhere: it is neither a control-plane window "
           "(IMEM/CFR/violation CSR/GPR under BAR2) nor an HBM bank window "
           "(HBM_BASE + ch*CH_SPAN + bank*BANK_STRIDE) nor MC s_axi.";
}

const char *pim_window_name(uint64_t axi)
{
    static char buf[64];
    unsigned c, b;
    bool inw;
    if (pim_hbm_decode(axi, &c, &b, NULL, &inw)) {
        snprintf(buf, sizeof buf, "HBM ch%u bank%u%s", c, b, inw ? "" : " (GAP)");
        return buf;
    }
    if (axi >= PIM_MC_BASE && axi < PIM_MC_BASE + (uint64_t)PIM_NCH * PIM_MC_CH_SPAN) {
        // RoBaCo inside the channel, so the bank comes out of the MIDDLE of the
        // offset and the row is what the top of it names.
        uint64_t rel  = axi - PIM_MC_BASE;
        uint64_t inch = rel % PIM_MC_CH_SPAN;
        snprintf(buf, sizeof buf, "MC ch%u row%" PRIu64 " bank%u col%" PRIu64,
                 (unsigned)(rel / PIM_MC_CH_SPAN),
                 inch / PIM_ROBACO_ROW_BYTES,
                 (unsigned)((inch % PIM_ROBACO_ROW_BYTES) / EMU_ROW_BYTES),
                 (inch % EMU_ROW_BYTES) / EMU_WORD_BYTES);
        return buf;
    }
    const uint64_t base = PIM_BAR2_AXI_BASE;
    if (axi >= base + PIM_OFF_GPR  && axi < base + PIM_OFF_GPR  + EMU_SIZE_GPR)  return "GPR";
    if (axi >= base + PIM_OFF_CFR  && axi < base + PIM_OFF_CFR  + EMU_SIZE_CFR)  return "CFR";
    if (axi >= base + PIM_OFF_IMEM && axi < base + PIM_OFF_IMEM + EMU_SIZE_IMEM) return "IMEM";
    if (axi >= EMU_AXI_MODE && axi < EMU_AXI_MODE + EMU_SIZE_MODE) return "channel map CSR";
    if (axi >= base + PIM_OFF_VIOL &&
        axi < base + PIM_OFF_VIOL + (uint64_t)PIM_NCH * EMU_SIZE_VIOL) {
        snprintf(buf, sizeof buf, "violation CSR ch%u",
                 (unsigned)((axi - base - PIM_OFF_VIOL) / EMU_SIZE_VIOL));
        return buf;
    }
    return NULL;
}

const char *pim_addr_map_check(uint32_t mode_ctrl)
{
    static char err[512];
    unsigned on;

    // ONE CHANNEL MAKES THE TWO MAPS THE SAME ADDRESS.  ChRoBaCo and RoChBaCo differ
    // only in where the channel field sits, and a field zero bits wide does not sit
    // anywhere — every other field keeps its place, so pim_decode() cannot tell them
    // apart and there is nothing for the board to disagree with.
    //
    // It also has to be the answer for an image that has NO MODE BLOCK.  ch1 r1p0
    // predates the register; +0x405000 is undecoded there and reads all-ones, which
    // this function would otherwise take for "interleave on" and refuse a build that
    // is not wrong.  Checking a register that cannot matter is how a correct setup
    // gets rejected.
    if (PIM_NCH == 1u) return NULL;

    on = (mode_ctrl & MODE_CTRL_INTERLEAVE) ? 1u : 0u;
    if (on == (unsigned)PIM_ADDR_MAP) return NULL;
    snprintf(err, sizeof err,
        "this binary was built for channel map %s (%s) but the board is set to %s.\n"
        "       The map decides which channel and bank every address reaches, so a\n"
        "       mismatch reads the wrong bytes without any error at all.\n"
        "       Set the board to match:  hwdef/test/emu_sanity\n"
        "       Or rebuild for what the board has:  scripts/setup.sh --platform %s --map %u",
        PIM_ADDR_MAP_NAME, PIM_ADDR_MAP_WHAT, on ? "RoChBaCo" : "ChRoBaCo",
        PIM_PLATFORM_NAME, on + 1u);
    return err;
}

void pim_platform_banner(void)
{
    printf("platform %s (%s)  —  %u ch x %u bank, window %llu MiB, "
           "stride %llu MiB\n",
           PIM_PLATFORM_NAME,
           PIM_CONFIG_FROM_CONF ? "compiled in"
                                : "COMPILED-IN DEFAULTS - no platform selected",
           PIM_NCH, PIM_NBANK,
           (unsigned long long)(PIM_BANK_WINDOW >> 20),
           (unsigned long long)(PIM_BANK_STRIDE >> 20));
    printf("         chanmap=%s (%s)%s\n", PIM_ADDR_MAP_NAME, PIM_ADDR_MAP_WHAT,
           PIM_NCH == 1 ? "  — one channel, so both maps are the same" : "");
    // FOUR CONF VALUES ARE DELIBERATELY NOT PRINTED HERE: UPLOAD_PATH, SCHEDULE,
    // NTAIL_POLICY and FEATURE_T_LATCH.  Nothing reads them — not this file, not a
    // probe — so a banner that listed them next to the values that DO decide
    // something (channel count, bank stride, address map) invited the reading that
    // editing the conf would change behaviour.  It never did.
    //
    // They are still validated by gen_config.sh and compiled in, so putting the line
    // back is one printf.  Do not, unless something has started reading them.
}

const char *pim_platform_check(void)
{
    static char err[512];

    // FIRST: was a platform chosen at all?  The constants have compiled-in defaults
    // so that a bare `make` produces a working binary — but a binary nobody chose a
    // platform for is exactly the one that must not run.  It carries ch4's channel
    // count and bank stride because something had to be there, not because anyone
    // said ch4, and there is no reading of the board that would tell the difference.
    // So this is a refusal, not the warning below.
    if (!PIM_CONFIG_FROM_CONF) {
        snprintf(err, sizeof err,
            "this binary was built from pim_config.h's COMPILED-IN DEFAULTS (%s) "
            "because no platform was selected.\n"
            "       Those values are a placeholder, not a choice — a wrong channel "
            "count returns plausible numbers.\n"
            "       Build through:  scripts/setup.sh --platform chN",
            PIM_PLATFORM_NAME);
        return err;
    }

    // A readlink and a strcmp.  The board cannot report its channel count, so this
    // is not "does the build match the hardware" — it is "does the build match what
    // the user selected", which is the mistake that actually happens: choose a
    // platform, forget to rebuild, run yesterday's binary.
    char target[512];
    ssize_t k = readlink(PIM_CONF_PATH, target, sizeof target - 1);
    if (k <= 0) {
        // Only reachable with PIM_CONFIG_FROM_CONF, i.e. a path that WAS read at
        // build time and is not readable now — a moved tree.  It cannot tell
        // agreement from disagreement, so it claims neither.
        fprintf(stderr,
            "WARNING: cannot read %s, so this build's platform (%s) was not checked "
            "against the selected one.  Moved tree?\n", PIM_CONF_PATH, PIM_PLATFORM_NAME);
        return NULL;
    }
    target[k] = '\0';

    const char *base = strrchr(target, '/');
    base = base ? base + 1 : target;
    char name[sizeof target];
    snprintf(name, sizeof name, "%s", base);
    char *dot = strstr(name, ".conf");
    if (dot) *dot = '\0';

    if (strcmp(name, PIM_PLATFORM_NAME) == 0) return NULL;

    // %.31s: `name` came from a symlink target and could in principle be long; the
    // message must not be truncated at the point where it says what to do about it.
    snprintf(err, sizeof err,
        "this binary was built for platform '%s' but '%.31s' is selected.\n"
        "       The topology is compiled in, so it does not follow the symlink.\n"
        "       Rebuild:  scripts/setup.sh --platform %.31s\n"
        "       (selected via %s)",
        PIM_PLATFORM_NAME, name, name, PIM_CONF_PATH);
    return err;
}
