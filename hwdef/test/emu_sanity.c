// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_sanity — "did the board come back healthy?", for reprogram.sh.
//
//     ./emu_sanity                 # write/readback the CFR, leave the sim timings
//     ./emu_sanity --bdf X --bar 2
//     ./emu_sanity --no-set        # verify only; do not leave the sim values
//     ./emu_sanity --quiet         # one line of output
//
// LAYER 1.  This is the bottom of the stack and depends on nothing above it: one
// pure header (emu_regs.h) for the offsets, and libc.  reprogram.sh must not need
// libemu, emu_probe, or a working DMA queue to decide whether programming worked.
//
// WHY WRITE/READBACK AND NOT A RESET SIGNATURE
// The two earlier platforms each had something to test that this one does not:
//   read_channel/write_channel   keyed on rresp != 0
//   bank_controller_top          keyed on a nonzero reset signature
//                                (TIMING0=0x08040408, TIMING1=0x00001004, ...)
// Here BOTH are unavailable.  Every register in this design resets to zero
// (dispatcher_top.v:272 clears all eight timing registers, PROG_LEN and mode;
// emu_viol_csr.v:283-292 clears every counter), and BAR2 offset 0 is the IMEM
// slave whose read path is a hardwired zero-return stub.  So a healthy, freshly
// programmed board reads ALL ZEROS everywhere it can be read, and no read-only
// check can tell that apart from a dead one.
//
// Writing and reading back settles it, and settles more than liveness: it proves
// the BAR is mapped, that the BAR-to-AXI translation lands on the CFR, that the
// NoC carries it, that the AXI-Lite slave decodes each register separately, and
// that the return path works.  HANDOFF §5 step 1 says exactly this.
//
// THE PROBE VALUES ARE DISTINCT ON PURPOSE
// The eight simulation values (30 6 4 2 3 3 4 6) contain three duplicate pairs,
// so writing only those cannot detect a decode that maps two registers onto one
// address — the duplicate would read back "correct".  So the probe pass writes
// eight distinct values first, and the simulation values are written afterwards,
// once, as the parting state.
//
// NOTHING IS WRITTEN AT AN OFFSET ENDING IN 0x00
// CFR decode uses only addr[7:0] (dispatcher_top.v:281), so the block repeats
// every 256 B and offset 0x00 is CTRL — whose bit 0 is the doorbell.  The eight
// timing registers at 0x08..0x24 are safe; nothing here touches CTRL.
//
// Exit: 0 healthy   1 unhealthy   2 usage
//////////////////////////////////////////////////////////////////////////////////
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "emu_regs.h"
#include "pim_platform.h"

// ---- platform ------------------------------------------------------------------
// The topology (channel count, bank stride, bank window, MC base) is NOT a compile
// time constant: it is read at startup from platform/*.conf, the same file the
// shell scripts read.  g_ch is the channel this tool is addressing right now —
// tools that walk every channel set it in their loop, tools that take --ch set it
// once.  Every tool prints pim_platform_banner(), so a run made against the wrong
// topology says so in its own output.
// No g_ch here: this tool reads the control plane, which is one gpr_wrap, one
// dispatcher and one doorbell no matter how many channels the image has.  The one


#define DEFAULT_BDF "0000:01:00.0"

static const struct { uint32_t off; const char *name; uint8_t sim; uint8_t probe; } REG[8] = {
    { CFR_T_FAW, "T_FAW", 30, 0x11 }, { CFR_T_RRD, "T_RRD",  6, 0x22 },
    { CFR_T_RCD, "T_RCD",  4, 0x33 }, { CFR_T_CCD, "T_CCD",  2, 0x44 },
    { CFR_T_RTP, "T_RTP",  3, 0x55 }, { CFR_T_RP,  "T_RP",   3, 0x66 },
    { CFR_T_WR,  "T_WR",   4, 0x77 }, { CFR_T_RAS, "T_RAS",  6, 0x88 },
};

static volatile uint8_t *g_bar;

static uint32_t rd(uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off);
    __sync_synchronize();
    return v;
}

static void wr(uint32_t off, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off) = v;
    __sync_synchronize();
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--bdf <b:d.f>] [--bar <n>] [--no-set] [--quiet]\n"
"\n"
"Proves the control plane came back healthy after programming, by writing the\n"
"eight CFR timing registers and reading them back (HANDOFF §5 step 1).  A\n"
"read-only check is impossible on this platform: every register resets to zero.\n"
"\n"
"  --bdf <b:d.f>  PCIe address (default %s)\n"
"  --bar <n>      which BAR carries the control plane (default 2)\n"
"  --no-set       verify only; leave the registers at the probe values\n"
"  --quiet        one line\n"
"\n"
"Exit: 0 healthy, 1 unhealthy, 2 usage\n", p, DEFAULT_BDF);
}

int main(int argc, char **argv)
{
    const char *bdf = DEFAULT_BDF;
    unsigned bar = 2;
    bool quiet = false, set_sim = true;

    static const struct option lo[] = {
        { "bdf",    required_argument, NULL, 1 },
        { "bar",    required_argument, NULL, 2 },
        { "quiet",  no_argument,       NULL, 3 },
        { "no-set", no_argument,       NULL, 4 },
        { "help",   no_argument,       NULL, 'h' },
        { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: bdf = optarg; break;
        case 2: bar = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: quiet = true; break;
        case 4: set_sim = false; break;

        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 2;
        }
    }

    // The topology is compiled in.  What is checked is that this build matches the
    // platform currently SELECTED — choosing one and forgetting to rebuild is the
    // mistake that survives, since the board cannot report its own channel count.
    {
        const char *e = pim_platform_check();
        if (e) { fprintf(stderr, "ERROR: %s\n", e); return 2; }
        pim_platform_banner();
    }

    char path[256];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource%u", bdf, bar);
    int fd = open(path, O_RDWR | O_SYNC);
    if (fd < 0) {
        if (errno == ENOENT)
            fprintf(stderr, "UNHEALTHY: %s does not exist — the device is not "
                            "enumerated, or this bitstream exposes no BAR%u.\n", path, bar);
        else if (errno == EACCES)
            fprintf(stderr, "ERROR: %s: permission denied.  sudo ./setup_permissions.sh\n", path);
        else
            fprintf(stderr, "ERROR: open(%s): %s\n", path, strerror(errno));
        return 1;
    }

    struct stat st;
    size_t span = EMU_BAR2_SIZE;
    if (fstat(fd, &st) == 0 && st.st_size > 0) span = (size_t)st.st_size;
    if (span < EMU_OFF_CFR + 0x1000u) {
        fprintf(stderr, "UNHEALTHY: BAR%u is %zu B; the CFR alone sits at +0x%x.  "
                        "This is not the emulator_top image.\n", bar, span, EMU_OFF_CFR);
        close(fd);
        return 1;
    }

    void *m = mmap(NULL, span, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        fprintf(stderr, "ERROR: mmap(%s): %s\n", path, strerror(errno));
        close(fd);
        return 1;
    }
    g_bar = m;

    // ---- pass 1: eight DISTINCT values, so a decode that merges two registers
    //      cannot hide behind a duplicate.
    for (int i = 0; i < 8; i++) wr(REG[i].off, REG[i].probe);
    uint32_t got[8];
    int bad = 0, ones = 0;
    for (int i = 0; i < 8; i++) {
        got[i] = rd(REG[i].off);
        if (got[i] == 0xffffffffu) ones++;
        if (got[i] != REG[i].probe) bad++;
    }

    if (bad) {
        printf("UNHEALTHY: %d/8 CFR timing registers did not read back.\n", bad);
        for (int i = 0; i < 8; i++)
            printf("  %-6s +0x%02x  wrote 0x%02x  read 0x%08x%s\n", REG[i].name,
                   REG[i].off, REG[i].probe, got[i],
                   got[i] == REG[i].probe ? "" : "   <--");
        if (ones == 8)
            printf("\n  All ones: the PCIe mapping is STALE.  Programming a PDI over JTAG\n"
                   "  invalidates enumeration; the host must remove and rescan.\n");
        else if (ones == 0 && !got[0] && !got[7])
            printf("\n  All zero after a write: the window is mapped but nothing is\n"
                   "  decoding it.  Either BAR%u is not the control plane on this image,\n"
                   "  or the CFR is not at +0x%x.\n", bar, EMU_OFF_CFR);
        else
            printf("\n  Mixed: some registers answered.  Suspect the BAR-to-AXI base or a\n"
                   "  partially programmed device.\n");
        munmap(m, span); close(fd);
        return 1;
    }

    // ---- pass 2: leave the board configured, so the check is not something that
    //      has to be undone before real work.  T_CCD < 2 is refused upstream in
    //      emu_regs.h's contract; these values satisfy it.
    uint32_t got_sim[8];
    for (int i = 0; i < 8; i++) got_sim[i] = 0;
    if (set_sim) {
        for (int i = 0; i < 8; i++) wr(REG[i].off, REG[i].sim);
        for (int i = 0; i < 8; i++) got_sim[i] = rd(REG[i].off);
        for (int i = 0; i < 8; i++) {
            if (got_sim[i] != REG[i].sim) {
                printf("UNHEALTHY: %s took the probe value but not the final value "
                       "(wrote %u, read 0x%08x).\n", REG[i].name, REG[i].sim, got_sim[i]);
                for (int k = 0; k < 8; k++)
                    printf("  %-6s +0x%02x  probe 0x%02x -> 0x%08x   final %2u -> 0x%08x%s\n",
                           REG[k].name, REG[k].off, REG[k].probe, got[k],
                           REG[k].sim, got_sim[k],
                           got_sim[k] == REG[k].sim ? "" : "   <--");
                munmap(m, span); close(fd);
                return 1;
            }
        }
    }

    uint32_t status = rd(CFR_STATUS), prog_len = rd(CFR_PROG_LEN);

    // ---- pass 3: the OTHER AXI-Lite slave.
    //
    // The violation CSR is read-only by construction — the host cannot forge a
    // clean run — so there is no write/readback to do.  And at reset every
    // register in it is zero (emu_viol_csr.v:283-292 clears all five kinds'
    // sticky/count/max plus the drop counters).  So "it reads 0" cannot by itself
    // prove the block is alive.
    //
    // What it CAN prove is that the window is MAPPED, and the discriminator is
    // sharp on this board: the undecoded parts of BAR2 read 0xffffffff (measured
    // 2026-07-31 — roughly 3.5 MB of the 8 MB does), while this block returns 0
    // even for an offset it does not decode (emu_viol_csr.v:47).  Zero is
    // therefore the CORRECT reading here and all-ones is the fault — the exact
    // opposite of what a generic BAR liveness heuristic would conclude.
    //
    // What it does NOT prove: that the counters work.  emu_viol_csr has no
    // testbench upstream and nobody has exercised a register read on it.  That
    // gets settled on the first real program run, which cannot come out clean:
    // HANDOFF §2 measured RCD_RD overrun 3 at T_RCD=4 against an ideal memory, so
    // a run with the values this tool leaves behind MUST make ANY nonzero.
    struct { uint32_t off; const char *name; } V[] = {
        { VIOL_ANY,                        "ANY   (0x404)" },
        { VIOL_CTRL,                       "CTRL  (0x400, reads 0 by design)" },
        { VIOL_BANK_BASE(0)  + VIOL_STICKY, "bank0 STICKY" },
        { VIOL_BANK_BASE(0)  + VIOL_CNT_A,  "bank0 CNT_A" },
        { VIOL_BANK_BASE(0)  + VIOL_MAX_A,  "bank0 MAX_A" },
        { VIOL_BANK_BASE(15) + VIOL_STICKY, "bank15 STICKY" },
        { VIOL_BANK_BASE(0)  + 0x14u,       "bank0 +0x14 (unmapped -> must read 0)" },
    };
    const unsigned NV = sizeof V / sizeof V[0];
    // ONE violation CSR PER CHANNEL, at OFF_VIOL + ch*0x1000.  How many there are
    // comes from the conf and nothing else: this tool reads exactly nch blocks and
    // never one more.  Reading past the last one would be a channel-count PROBE,
    // and probing an address the image does not decode is not something a sanity
    // check should do — the hardware gets a register for that in a later revision.
    uint32_t vval[PIM_NCH][sizeof V / sizeof V[0]];
    int v_ones = 0;
    unsigned ones_ch = 0;
    for (unsigned ch = 0; ch < PIM_NCH; ch++) {
        for (unsigned i = 0; i < NV; i++) {
            __sync_synchronize();
            vval[ch][i] = *(volatile uint32_t *)
                (g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + V[i].off);
            __sync_synchronize();
            if (vval[ch][i] == 0xffffffffu) { v_ones++; ones_ch = ch; }
        }
    }
    if (v_ones) {
        printf("UNHEALTHY: violation CSR ch%u (+0x%06" PRIx64 ") reads all-ones on "
               "%d/%u probes.\n", ones_ch,
               (uint64_t)(PIM_OFF_VIOL + (uint64_t)ones_ch * 0x1000u), v_ones, NV * PIM_NCH);
        for (unsigned ch = 0; ch < PIM_NCH; ch++)
            for (unsigned i = 0; i < NV; i++)
                printf("  ch%u %-34s 0x%08x%s\n", ch, V[i].name, vval[ch][i],
                       vval[ch][i] == 0xffffffffu ? "   <--" : "");
        printf("\n  All-ones is an undecoded address on this board, not a full register.\n"
               "  Either BAR%u is not the control plane, or this image does not have\n"
               "  %u channels — check that platform/active matches the PDI on the board.\n",
               bar, PIM_NCH);
        munmap(m, span); close(fd);
        return 1;
    }

    // ---- the channel address map ------------------------------------------
    // MODE_CTRL resets to 0 with the bitstream, so after a programming run the board
    // is ChRoBaCo whatever the build wants.  Setting it here is what closes the loop:
    // reprogram.sh runs this probe last, so the board ends up matching what
    // setup.sh --map compiled, and nothing above hwdef has to think about it again.
    //
    // --no-set skips it, for the same reason it skips the timing registers: a probe
    // that only wants to look should not leave the board different.
    //
    // ON ONE CHANNEL IT IS NOT CHECKED AND NOT WRITTEN.  Both maps name the same
    // address when there is no channel field to move, so the register decides
    // nothing — and the ch1 r1p0 image does not carry it at all, reading all-ones at
    // +0x405000.  Refusing on that would fail a board that is set up correctly.
    if (PIM_NCH == 1u) {
        printf("  channel map not checked — one channel, so ChRoBaCo and RoChBaCo\n"
               "              are the same address and the image need not carry the CSR\n");
    } else {
        volatile uint32_t *mode = (volatile uint32_t *)(g_bar + EMU_OFF_MODE);
        __sync_synchronize();
        uint32_t before = mode[MODE_CTRL / 4], st = mode[MODE_STATUS / 4];
        __sync_synchronize();
        if (before == 0xffffffffu) {
            printf("UNHEALTHY: the channel map CSR (+0x%06x) reads all-ones — this "
                   "image has no such block.\n", EMU_OFF_MODE);
            munmap(m, span); close(fd);
            return 1;
        }
        if (set_sim) {
            if (st & MODE_STATUS_BUSY) {
                printf("UNHEALTHY: MODE_STATUS says busy; something is mid-transaction.\n");
                munmap(m, span); close(fd);
                return 1;
            }
            __sync_synchronize();
            mode[MODE_CTRL / 4] = PIM_ADDR_MAP ? MODE_CTRL_INTERLEAVE : 0u;
            __sync_synchronize();
        }
        uint32_t now = mode[MODE_CTRL / 4];
        const char *bad = pim_addr_map_check(now);
        if (bad) { printf("UNHEALTHY: %s\n", bad); munmap(m, span); close(fd); return 1; }
        printf("  channel map %s (%s)%s\n", PIM_ADDR_MAP_NAME, PIM_ADDR_MAP_WHAT,
               (now != before) ? "  <-- changed; anything already resident is now"
                                 " addressed differently" : "");
    }

    if (quiet) {
        printf("HEALTHY: CFR at BAR%u+0x%x answered 8/8\n", bar, EMU_OFF_CFR);
    } else {
        printf("HEALTHY: the control plane is alive.\n");
        printf("  BAR%u = %zu B, windows onto AXI 0x%011" PRIx64 "\n",
               bar, span, (uint64_t)EMU_AXI_BASE);
        // Show the transactions, not just the verdict.  Pass 1 uses eight DISTINCT
        // values so a decode that merges two registers cannot hide; pass 2 leaves
        // the simulation values, which contain three duplicate pairs and therefore
        // could NOT have proven separate decode on their own.
        // The check is one sentence: write a value, read it back, compare.  Show
        // every transaction so the verdict can be checked rather than believed.
        printf("\n  THE CHECK: write each CFR timing register, read it back, compare.\n");
        printf("  Done twice.  Pass 1 is the test; pass 2 leaves the board configured.\n\n");
        printf("    register  addr      PASS 1 (test)        PASS 2 (final state)\n");
        printf("    --------  ------    ------------------   --------------------\n");
        for (int i = 0; i < 8; i++) {
            printf("    %-8s +0x%02x     wrote 0x%02x read 0x%02x %-3s",
                   REG[i].name, REG[i].off, REG[i].probe, got[i] & 0xFFu,
                   got[i] == REG[i].probe ? "ok" : "BAD");
            if (set_sim)
                printf("  wrote %-2u   read %-2u %s\n", REG[i].sim, got_sim[i] & 0xFFu,
                       got_sim[i] == REG[i].sim ? "ok" : "BAD");
            else
                printf("  (--no-set: skipped)\n");
        }
        printf("\n    PASS 1 is the real test.  The eight values are all DIFFERENT, so if\n"
               "    two registers shared one address, one of them would read back the\n"
               "    other's value and this would say BAD.  All eight matching means:\n"
               "      - the BAR is mapped and reachable\n"
               "      - the BAR-to-AXI translation lands on the CFR, not somewhere else\n"
               "      - each register decodes to its own address\n"
               "      - the read return path works\n");
        if (set_sim)
            printf("\n    PASS 2 is NOT a test — it is the state left behind.  The run values\n"
                   "    30/6/4/2/3/3/4/6 contain three duplicate pairs (6 twice, 4 twice,\n"
                   "    3 twice), so writing only these could not have caught a shared\n"
                   "    address.  That is why pass 1 exists and why it goes first.\n"
                   "    (T_RCD=4 WILL raise RCD_RD violations later — HANDOFF §2 measured\n"
                   "     overrun 3 against an ideal memory model.  Expected, not a fault.)\n");
        printf("  STATUS = 0x%08x (done=%u fetch_state=%u), PROG_LEN = %u\n",
               status, (status & CFR_STATUS_DONE) ? 1u : 0u,
               CFR_STATUS_STATE(status), prog_len);

        for (unsigned ch = 0; ch < PIM_NCH; ch++) {
            printf("\n  violation CSR ch%u at +0x%06" PRIx64 " (AXI 0x%011" PRIx64
                   "), read-only:\n",
                   ch, (uint64_t)(PIM_OFF_VIOL + (uint64_t)ch * 0x1000u),
                   pim_viol_axi(ch));
            for (unsigned i = 0; i < NV; i++)
                printf("    %-38s 0x%08x\n", V[i].name, vval[ch][i]);
        }
        printf("    ZERO IS THE CORRECT READING HERE.  Everything resets to 0\n"
               "    (emu_viol_csr.v:283-292) and an offset the block does not decode\n"
               "    also reads 0 (emu_viol_csr.v:47).  What would be wrong is\n"
               "    0xffffffff, which is what undecoded BAR2 space returns — so these\n"
               "    reads prove the window is MAPPED, and nothing more.\n"
               "    The counters themselves are still unproven: emu_viol_csr has no\n"
               "    testbench upstream.  They get proven on the first program run,\n"
               "    which cannot come out clean at T_RCD=4 (HANDOFF 2) — ANY must go\n"
               "    nonzero then, and if it does not, this block is broken.\n");

        printf("\n  This proves the CONTROL PLANE only.  The datapath needs a program run.\n");
    }

    munmap(m, span);
    close(fd);
    return 0;
}
