// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_hbm_direct — is every HBM bank window individually reachable, and are the 16
//               of them actually distinct?
//
//     ./emu_hbm_direct                # 3 probe points per bank, all 16 banks
//     ./emu_hbm_direct --kib 256      # bigger probe blocks
//     ./emu_hbm_direct --mib 8        # also a contiguous 8 MiB per bank (chunking)
//     ./emu_hbm_direct --banks 0,3,15 # only these
//
// LAYER 2b.  Depends on emu_regs.h and libc.  Run after emu_sanity (layer 1) and
// emu_gprloop (layer 2): if the DMA path itself were broken this would only tell
// you that again, more slowly.
//
// WHY WRITE ALL SIXTEEN BEFORE READING ANY
// The obvious shape — for each bank: write, read, compare — CANNOT detect the
// failure that matters.  If all sixteen windows aliased onto one piece of memory,
// every bank would still read back exactly what it had just written, and the test
// would pass sixteen times while the hardware was completely wrong.  So this
// writes every bank first and reads afterwards: with a distinct signature per
// bank, an alias means a later write clobbered an earlier one and the readback
// says so, naming which bank's data turned up.
//
// WHERE THE PROBE POINTS ARE, AND WHY
//   +0                     the base the ISR's ROW field addresses from
//   +128 MiB               the middle.  Testing only the two ends would pass on a
//                          window that decodes its edges and nothing between
//   +256 MiB - block       the LAST decoded byte.  The bank window is 256 MiB,
//                          which is also the real per-bank capacity and exactly
//                          what ISR ROW reaches (17 b x 2048 B = 2^28)
// Testing only offset 0 would pass on a window one beat wide.
//
// WHAT IS NOT TRANSFERRED, AND WHY
// The banks are spaced 512 MiB or 1 GiB apart depending on the image, so there is
// an UNDECODED GAP above every 256 MiB window.  This tool does NOT transfer into
// it.  A DMA to an address that decodes nowhere returned EIO and latched the H2C
// engine for every later transfer [observed], and the driver's error handler
// re-arms without clearing — so probing the gap costs a queue teardown at best.
// Instead the gap is checked in SOFTWARE: pim_dma_check() must REFUSE it.  That
// answers the same question ("is the window really only 256 MiB?") without
// touching the bus.
//
// An earlier version of this tool probed +1 GiB - block and passed on all 16
// banks, which is how the 1 GiB window in the old header survived.  It should not
// have passed; whatever answered there was outside the assigned window.  The check
// below is deliberately a refusal test, not a re-run of that transfer.
//
// Exit: 0 pass   1 fail   2 usage
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
// No g_ch: this tool walks EVERY (channel, bank) slot rather than addressing one,
// because the failure it exists to catch — two slots reaching the same memory —


#define H2C_DEFAULT "/dev/qdma01000-MM-0"
#define C2H_DEFAULT "/dev/qdma01000-MM-1"
#define CHUNK       (4u << 20)
#define POISON      0xA5

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

// One 32 B word carrying WHERE it belongs: which bank, which byte offset.
// A mismatch can then say "bank 5 offset 0 holds bank 4's data" instead of
// "bytes differ", which separates an aliased window from a shifted address from
// real corruption.
// The signature carries the CHANNEL as well as the bank.  Without it, two
// channels' bank 3 look identical and a channel that folded onto another would
// read back its own expected content and pass.
#define MAGIC 0xB47Cu
static void fill_word(uint64_t *w, unsigned ch, unsigned bank, uint64_t off)
{
    uint64_t id = ((uint64_t)ch << 4) | (bank & 0xFu);
    w[0] = ((uint64_t)MAGIC << 48) | (id << 40) | (off & 0xFFFFFFFFFFull);
    w[1] = ~w[0];
    w[2] = (off ^ (id * 0x9E3779B97F4A7C15ull));
    w[3] = (id << 56) | (off >> 8);
}
static bool word_origin(const uint64_t *w, unsigned *ch, unsigned *bank, uint64_t *off)
{
    if ((w[0] >> 48) != MAGIC) return false;
    unsigned id = (unsigned)((w[0] >> 40) & 0xFFu);
    *ch   = id >> 4;
    *bank = id & 0xFu;
    *off  = w[0] & 0xFFFFFFFFFFull;
    return true;
}

static int xfer(int fd, const char *dir, uint64_t axi, void *buf, size_t len, bool writing)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) {
        fprintf(stderr, "  REFUSED %s 0x%011" PRIx64 " (%zu B): %s\n", dir, axi, len, bad);
        return -1;
    }
    uint8_t *p = buf;
    size_t done = 0;
    while (done < len) {
        size_t want = len - done;
        if (want > CHUNK) want = CHUNK;
        ssize_t n = writing ? pwrite(fd, p + done, want, (off_t)(axi + done))
                            : pread (fd, p + done, want, (off_t)(axi + done));
        if (n < 0) {
            fprintf(stderr, "  %s 0x%011" PRIx64 " +%zu (%zu B): %s\n",
                    dir, axi, done, want, strerror(errno));
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "  %s 0x%011" PRIx64 " +%zu made no progress\n", dir, axi, done);
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

struct point { uint64_t off; const char *what; };

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--ch LIST] [--kib N] [--mib N] [--banks LIST]\n"
"       [--h2c PATH] [--c2h PATH]\n"
"\n"
"Writes a (channel, bank, offset) signature into every bank window, THEN reads\n"
"them all back — so that two windows aliased onto one memory cannot pass.\n"
"\n"
"  --ch LIST     comma-separated channels, e.g. 0,1 (default: all of them)\n"
"\n"
"  This tool reaches the banks through the DIRECT aperture and never through the MC\n"
"  port, so --ch names a PHYSICAL channel: it verifies one emulator channel on its\n"
"  own.  That makes it independent of the channel address map — ChRoBaCo or\n"
"  RoChBaCo only changes how an MC-space address is split up, and nothing here is an\n"
"  MC-space address.  (emu_mc is the one that is: its --ch is a place in MC space,\n"
"  which is why that one is ChRoBaCo-only.)\n"
"  --kib N       probe block size, default 64\n"
"  --mib N       additionally test a contiguous N MiB from each base address\n"
"                (N > 4 exercises the multi-chunk path; max 256 = the DECODED\n"
"                window, which is SMALLER than the bank spacing; needs\n"
"                2 x N MiB of host RAM; default 0 = skip)\n"
"  --banks LIST  comma-separated, e.g. 0,3,15 (default all 16 per channel)\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p);
}

int main(int argc, char **argv)
{
    const char *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    unsigned kib = 64, mib = 0;
    bool want[EMU_NBANKS];
    bool wch[PIM_NCH];
    for (unsigned i = 0; i < EMU_NBANKS; i++) want[i] = true;
    for (unsigned i = 0; i < PIM_NCH; i++) wch[i] = true;
    const char *ch_list = NULL;

    static const struct option lo[] = {
        { "ch",       required_argument, NULL, 'C' },
        { "kib",   required_argument, NULL, 1 }, { "mib",  required_argument, NULL, 2 },
        { "banks", required_argument, NULL, 3 }, { "h2c",  required_argument, NULL, 4 },
        { "c2h",   required_argument, NULL, 5 }, { "help", no_argument,       NULL, 'h' },
        { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: kib = (unsigned)strtoul(optarg, NULL, 0); break;
        case 2: mib = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: {
            for (unsigned i = 0; i < EMU_NBANKS; i++) want[i] = false;
            for (char *s = strtok(optarg, ","); s; s = strtok(NULL, ",")) {
                unsigned b = (unsigned)strtoul(s, NULL, 0);
                if (b < EMU_NBANKS) want[b] = true;
            }
            break; }
        case 4: h2c_path = optarg; break;
        case 5: c2h_path = optarg; break;
        case 'C': ch_list = optarg; break;

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
    if (ch_list) {
        for (unsigned i = 0; i < PIM_NCH; i++) wch[i] = false;
        char buf[64];
        snprintf(buf, sizeof buf, "%s", ch_list);
        for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
            unsigned c = (unsigned)strtoul(t, NULL, 0);
            if (c >= PIM_NCH) {
                fprintf(stderr, "ERROR: --ch %u but %s has %u channel(s)\n",
                        c, PIM_PLATFORM_NAME, PIM_NCH);
                return 2;
            }
            wch[c] = true;
        }
    }
    if (kib < 1 || kib > 4096) { fprintf(stderr, "--kib must be 1..4096\n"); return 2; }
    // The only real ceiling is the DECODED window, which is smaller than the
    // spacing between banks: a contiguous run may not cross out of the 256 MiB a
    // bank actually answers for.  pim_dma_check() refuses it anyway; catching it
    // here gives a better message.  Host RAM is the other limit and is checked at
    // the allocation below, which needs 2 x this.
    const unsigned MIB_MAX = (unsigned)(PIM_BANK_WINDOW >> 20);
    if (mib > MIB_MAX) {
        fprintf(stderr, "--mib must be <= %u: a bank decodes %u MiB and a contiguous\n"
                        "run may not cross into the undecoded gap above it.\n",
                MIB_MAX, MIB_MAX);
        return 2;
    }

    const size_t blk = (size_t)kib << 10;
    if (blk % EMU_WORD_BYTES) { fprintf(stderr, "--kib must give a multiple of 32 B\n"); return 2; }

    const struct point PT[3] = {
        { 0,                                  "base (ISR ROW 0)" },
        { PIM_BANK_WINDOW / 2,                "128 MiB (middle)" },
        { PIM_BANK_WINDOW - blk,              "256 MiB - blk (last decoded byte)" },
    };
    const unsigned NPT = 3;

    unsigned nb = 0, ncsel = 0;
    for (unsigned c = 0; c < PIM_NCH; c++) if (wch[c]) ncsel++;
    for (unsigned b = 0; b < EMU_NBANKS; b++) if (want[b]) nb++;
    nb *= ncsel;                       // total (channel, bank) slots under test

    printf("host DMA reach over the HBM address range\n");
    printf("  range    0x%011" PRIx64 " .. 0x%011" PRIx64 "  (%" PRIu64 " GiB)\n",
           (uint64_t)PIM_HBM_BASE,
           (uint64_t)(PIM_HBM_BASE + PIM_HBM_APERTURE - 1),
           (uint64_t)(PIM_HBM_APERTURE >> 30));
    printf("  slots    %u  (%u channel(s) x %u bank),  window %" PRIu64 " MiB every "
           "%" PRIu64 " MiB\n",
           nb, ncsel, nb / (ncsel ? ncsel : 1u),
           (uint64_t)(PIM_BANK_WINDOW >> 20), (uint64_t)(PIM_BANK_STRIDE >> 20));
    printf("  probes   %u offsets x %u KiB at each of those:\n", NPT, kib);
    for (unsigned i = 0; i < NPT; i++)
        printf("             +0x%09" PRIx64 "  %s\n", PT[i].off, PT[i].what);
    if (mib) printf("  bulk     %u MiB contiguous from each base (%s multi-chunk)\n",
                    mib, mib > 4 ? "exercises" : "does not exercise");
    printf("  order    EVERY point written first, THEN all read back — so an address\n"
           "           range that folds onto itself cannot pass.\n\n");

    // ---------------- phase 0 : the gap, checked WITHOUT touching the bus -----
    // The first byte past a bank window decodes nowhere.  Transferring there
    // latches the H2C engine, so this asks pim_dma_check() instead — which is the
    // guard every tool actually relies on, and therefore the thing worth testing.
    {
        unsigned refused = 0, checked = 0;
        for (unsigned c = 0; c < PIM_NCH; c++) {
            if (!wch[c]) continue;
            for (unsigned b = 0; b < EMU_NBANKS; b++) {
                if (!want[b]) continue;
                checked++;
                if (pim_dma_check(pim_bank(c, b) + PIM_BANK_WINDOW,
                                  blk)) refused++;
            }
        }
        printf("gap check (software only, no transfer)\n");
        printf("  +256 MiB in each selected bank must be REFUSED: %u/%u refused  %s\n\n",
               refused, checked, refused == checked ? "ok" : "FAIL");
        if (refused != checked) {
            printf("FAIL: emu_dma_check accepts an address in the undecoded gap above a\n"
                   "      bank window.  Every tool trusts that guard; fix it before\n"
                   "      running anything that transfers.\n");
            return 1;
        }
    }

    int h2c = open(h2c_path, O_WRONLY), c2h = open(c2h_path, O_RDONLY);
    if (h2c < 0 || c2h < 0) {
        fprintf(stderr, "ERROR: cannot open the MM queues (%s / %s): %s\n"
                        "       sudo ./qdma_queues.sh setup\n",
                h2c_path, c2h_path, strerror(errno));
        return 1;
    }

    uint8_t *tx = aligned_alloc(4096, blk), *rx = aligned_alloc(4096, blk);
    if (!tx || !rx) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    // ---------------- phase A : write every bank, every probe point ----------
    uint64_t t0 = now_us();
    size_t written = 0;
    for (unsigned c = 0; c < PIM_NCH; c++) {
        if (!wch[c]) continue;
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!want[b]) continue;
            for (unsigned i = 0; i < NPT; i++) {
                for (size_t o = 0; o < blk; o += EMU_WORD_BYTES)
                    fill_word((uint64_t *)(tx + o), c, b, PT[i].off + o);
                if (xfer(h2c, "H2C", pim_bank(c, b) + PT[i].off, tx, blk, true) < 0) {
                    printf("FAIL: 0x%011" PRIx64 " (ch%u bank%u, %s) could not be written\n",
                           (uint64_t)(pim_bank(c, b) + PT[i].off), c, b, PT[i].what);
                    return 1;
                }
                written += blk;
            }
        }
    }
    uint64_t t1 = now_us();

    // ---------------- phase B : read every bank back and verify --------------
    unsigned bad_banks = 0, shown = 0;
    for (unsigned c = 0; c < PIM_NCH; c++) {
      if (!wch[c]) continue;
      for (unsigned b = 0; b < EMU_NBANKS; b++) {
        if (!want[b]) continue;
        bool bank_ok = true;
        for (unsigned i = 0; i < NPT; i++) {
            uint64_t axi = pim_bank(c, b) + PT[i].off;
            memset(rx, POISON, blk);
            if (xfer(c2h, "C2H", axi, rx, blk, false) < 0) { bank_ok = false; continue; }

            bool untouched = true;
            for (size_t o = 0; o < blk && untouched; o++) if (rx[o] != POISON) untouched = false;
            if (untouched) {
                printf("  0x%011" PRIx64 " %-30s READ RETURNED NOTHING\n",
                       (uint64_t)axi, PT[i].what);
                bank_ok = false;
                continue;
            }
            for (size_t o = 0; o < blk; o += EMU_WORD_BYTES)
                fill_word((uint64_t *)(tx + o), c, b, PT[i].off + o);
            if (!memcmp(tx, rx, blk)) continue;

            bank_ok = false;
            if (shown < 12) {
                // Name where the wrong data came from — that is the whole point of
                // the signature.
                size_t o = 0;
                while (o < blk && !memcmp(tx + o, rx + o, EMU_WORD_BYTES)) o += EMU_WORD_BYTES;
                unsigned oc, ob; uint64_t oo;
                printf("  ch%u bank%-2u 0x%011" PRIx64 " %-26s MISMATCH at +0x%zx: ",
                       c, b, (uint64_t)axi, PT[i].what, o);
                if (!word_origin((const uint64_t *)(rx + o), &oc, &ob, &oo))
                    printf("content is not a signature word\n");
                else if (oc != c || ob != b)
                    printf("holds what was written at ch%u bank%u 0x%011" PRIx64
                           " — two addresses reach the same memory\n",
                           oc, ob, (uint64_t)(pim_bank(oc, ob) + oo));
                else
                    printf("right bank, wrong offset 0x%" PRIx64 " (expected 0x%" PRIx64
                           ")\n", oo, PT[i].off + o);
                shown++;
            }
        }
        if (!bank_ok) bad_banks++;
        else printf("  ch%u bank%-2u 0x%011" PRIx64 "  %u/%u offsets OK\n",
                    c, b, (uint64_t)pim_bank(c, b), NPT, NPT);
      }
    }
    uint64_t t2 = now_us();

    // ---------------- optional bulk pass, to exercise multi-chunk ------------
    if (!bad_banks && mib) {
        size_t blen = (size_t)mib << 20;
        uint8_t *btx = aligned_alloc(4096, blen), *brx = aligned_alloc(4096, blen);
        if (!btx || !brx) {
            fprintf(stderr, "ERROR: could not allocate 2 x %u MiB for the bulk pass\n",
                    mib);
            return 1;
        }
        printf("\n  bulk: %u MiB per bank from offset 0 (%zu chunks of %u MiB each)\n",
               mib, (blen + CHUNK - 1) / CHUNK, CHUNK >> 20);
        for (unsigned c = 0; c < PIM_NCH && !bad_banks; c++) {
          if (!wch[c]) continue;
          for (unsigned b = 0; b < EMU_NBANKS && !bad_banks; b++) {
            if (!want[b]) continue;
            for (size_t o = 0; o < blen; o += EMU_WORD_BYTES)
                fill_word((uint64_t *)(btx + o), c, b, o);
            if (xfer(h2c, "H2C", pim_bank(c, b), btx, blen, true) < 0) { bad_banks++; break; }
            memset(brx, POISON, blen);
            if (xfer(c2h, "C2H", pim_bank(c, b), brx, blen, false) < 0) { bad_banks++; break; }
            if (memcmp(btx, brx, blen)) {
                size_t o = 0;
                while (o < blen && !memcmp(btx + o, brx + o, EMU_WORD_BYTES)) o += EMU_WORD_BYTES;
                unsigned oc, ob; uint64_t oo;
                printf("  ch%u bank%-2u 0x%011" PRIx64 " BULK MISMATCH at +0x%zx: ",
                       c, b, (uint64_t)pim_bank(c, b), o);
                if (word_origin((const uint64_t *)(brx + o), &oc, &ob, &oo))
                    printf("holds ch%u bank%u 0x%011" PRIx64 "\n",
                           oc, ob, (uint64_t)(pim_bank(oc, ob) + oo));
                else printf("not a signature word\n");
                bad_banks++;
                break;
            }
            printf("  ch%u bank%-2u  bulk %u MiB OK\n", c, b, mib);
          }
        }
        free(btx); free(brx);
    }

    printf("\n");
    if (bad_banks) {
        printf("FAIL: %u of %u base addresses did not verify.\n", bad_banks, nb);
        return 1;
    }

    double wmb = (double)written / (double)(t1 - t0);
    double rmb = (double)written / (double)(t2 - t1);
    printf("PASS: all %u base addresses reachable at all %u offsets, and every one\n"
           "      holds its OWN signature after all of them had been written.\n",
           nb, NPT);
    printf("      %zu KiB each way, write %.0f MB/s, read %.0f MB/s\n",
           written >> 10, wmb, rmb);
    
    free(tx); free(rx); close(h2c); close(c2h);
    return 0;
}
