// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_gpr_loop — write the whole 4 MiB GPR over QDMA, read it back, verify.
//
//     ./emu_gpr_loop                  # 4 MiB loopback
//     ./emu_gpr_loop --mib 1          # a smaller slice
//
// LAYER 2.  Depends on emu_regs.h and libc, nothing else.  Layer 1 (emu_sanity)
// must pass first: if the CFR is not answering, a GPR mismatch says nothing.
//
// WHY THE GPR IS THE RIGHT THING TO LOOP BACK
// It is the only large region on this board that is BOTH host-writable and
// host-readable.  IMEM is write-only (dispatcher AR/R is a zero-return stub), the
// CFR is eight bytes of scalars, and the violation CSR is read-only.  So the GPR
// is where the DMA path can be proven end to end.
//
// ONE TRANSPORT.  The GPR is a 256 b slave and is therefore DMA-only — see
// emu_regs.h's transport policy.  MMIO reaches the AXI-Lite blocks (CFR,
// violation CSR) and nothing else, so there is only one address view of the GPR
// and nothing here has to establish that two of them agree.
//
// THE PATTERN CARRIES ITS OWN ADDRESS
// Every 32 B word holds the word index it belongs at.  A plain random pattern
// tells you "byte 91234 differs"; this tells you "word 2851 contains word 2850's
// data", which separates a corrupted transfer from an off-by-one address, a
// wrapped window, and a stride error — all of which look identical otherwise.
// bank_controller_top's pim_pattern_fill(PIM_PAT_ADDR) does the same thing for
// the same reason.
//
// TRANSFER RULES CARRIED FROM bank_controller_top's pim_dma
//   - the AXI address goes in the pwrite/pread CALL, never via lseek: this
//     driver's cdev never advances *pos, so an omitted seek silently reuses the
//     previous transfer's address.
//   - no-progress (a 0 return) is an error, not something to loop on.
//   - 4 MB per call.  NOT the 0x7ffff000 the vendor apps use: this driver build
//     allocates a physically contiguous SGL proportional to the length, so a huge
//     request fails allocation; and it printk()s once per call, so tiny chunks
//     flood the kernel log.
//   - page-aligned buffers, to keep the pinned-page count minimal.
//   - the read buffer is POISONED first, because a pread can return a positive
//     count without having transferred, which otherwise reads as valid data.
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
// No g_ch here on purpose: the GPR is ONE 4 MiB block shared by every channel
// (there is one gpr_wrap in CH1, CH2 and CH4 alike), so this tool has nothing to


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

// One 32 B GPR word, carrying the index it belongs at.
//   [0] magic | word          [1] ~word        [2] word * golden      [3] magic2 | word
// Any single u64 identifies the word; four of them make a corrupted word obvious
// rather than merely different.
#define MAGIC1 0xE30Full
#define MAGIC2 0x69C5ull
static void fill_word(uint64_t *w, uint32_t idx)
{
    w[0] = (MAGIC1 << 48) | idx;
    w[1] = ~(uint64_t)idx;
    w[2] = (uint64_t)idx * 0x9E3779B97F4A7C15ull;
    w[3] = (MAGIC2 << 48) | (uint64_t)idx;
}

// If a word is wrong, which word does its content claim to be?  -1 if unreadable.
static long claimed_index(const uint64_t *w)
{
    if ((w[0] >> 48) == MAGIC1) return (long)(w[0] & 0xFFFFFFFFull);
    if ((w[3] >> 48) == MAGIC2) return (long)(w[3] & 0xFFFFFFFFull);
    return -1;
}

static int xfer(int fd, const char *dir, uint64_t axi, void *buf, size_t len, bool writing)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) {
        fprintf(stderr, "REFUSED %s 0x%011" PRIx64 " (%zu B): %s\n", dir, axi, len, bad);
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
            fprintf(stderr, "%s 0x%011" PRIx64 " +%zu (%zu B): %s\n",
                    dir, axi, done, want, strerror(errno));
            return -1;
        }
        if (n == 0) {
            fprintf(stderr, "%s 0x%011" PRIx64 " +%zu made no progress\n", dir, axi, done);
            return -1;
        }
        done += (size_t)n;
    }
    return 0;
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--mib N] [--h2c PATH] [--c2h PATH]\n"
"\n"
"Writes the GPR over a QDMA MM queue, reads it back, and verifies.  Every 32 B\n"
"word carries its own index, so a mismatch names the word its content came from.\n"
"The GPR is a 256 b slave and is DMA-only; MMIO reaches AXI-Lite alone.\n"
"\n"
"  --mib N     how much of the 4 MiB GPR to use (default 4)\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p);
}

int main(int argc, char **argv)
{
    const char *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    unsigned mib = 4;

    static const struct option lo[] = {
        { "mib",     required_argument, NULL, 1 },
        { "h2c",     required_argument, NULL, 3 },
        { "c2h",     required_argument, NULL, 4 },
        { "help",    no_argument,       NULL, 'h' },
        { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: mib = (unsigned)strtoul(optarg, NULL, 0); break;
        case 3: h2c_path = optarg; break;
        case 4: c2h_path = optarg; break;

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
    if (mib < 1 || mib > 4) { fprintf(stderr, "--mib must be 1..4\n"); return 2; }

    const size_t len   = (size_t)mib << 20;
    const uint32_t nw  = (uint32_t)(len / EMU_WORD_BYTES);
    const uint64_t axi = EMU_AXI_GPR;

    printf("GPR loopback over QDMA MM\n");
    printf("  target  AXI 0x%011" PRIx64 "  (BAR2 +0x%06x)\n", (uint64_t)axi, EMU_OFF_GPR);
    printf("  size    %u MiB = %u words of %u B, 32 B aligned\n", mib, nw, EMU_WORD_BYTES);

    int h2c = open(h2c_path, O_WRONLY);
    int c2h = open(c2h_path, O_RDONLY);
    if (h2c < 0 || c2h < 0) {
        fprintf(stderr,
            "\nERROR: cannot open the MM queues (%s / %s): %s\n"
            "       They do not survive a reboot or a PCIe re-enumeration.\n"
            "         sudo ./qdma_queues.sh setup\n", h2c_path, c2h_path, strerror(errno));
        if (h2c >= 0) close(h2c);
        if (c2h >= 0) close(c2h);
        return 1;
    }

    uint8_t *tx = aligned_alloc(4096, len);
    uint8_t *rx = aligned_alloc(4096, len);
    if (!tx || !rx) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }

    for (uint32_t i = 0; i < nw; i++)
        fill_word((uint64_t *)(tx + (size_t)i * EMU_WORD_BYTES), i);

    uint64_t t0 = now_us();
    if (xfer(h2c, "H2C write", axi, tx, len, true) < 0) return 1;
    uint64_t t1 = now_us();

    memset(rx, POISON, len);          // a pread can report success without transferring
    if (xfer(c2h, "C2H read", axi, rx, len, false) < 0) return 1;
    uint64_t t2 = now_us();

    bool untouched = true;
    for (size_t i = 0; i < len && untouched; i++) if (rx[i] != POISON) untouched = false;
    if (untouched) {
        printf("\nFAIL: the read returned but the buffer is still poison — the driver\n"
               "      reported success without transferring anything.\n");
        return 1;
    }

    // ---- compare, word by word, and say WHERE a wrong word came from ----
    uint32_t bad = 0, shown = 0;
    for (uint32_t i = 0; i < nw; i++) {
        const uint64_t *a = (const uint64_t *)(tx + (size_t)i * EMU_WORD_BYTES);
        const uint64_t *b = (const uint64_t *)(rx + (size_t)i * EMU_WORD_BYTES);
        if (!memcmp(a, b, EMU_WORD_BYTES)) continue;
        bad++;
        if (shown < 8) {
            long c = claimed_index(b);
            printf("  word %-7u (AXI 0x%011" PRIx64 "): ", i,
                   (uint64_t)(axi + (uint64_t)i * EMU_WORD_BYTES));
            if (c < 0)          printf("content is not a valid pattern word\n");
            else if (c == i)    printf("right index, corrupted payload\n");
            else                printf("holds word %ld's data (off by %+ld)\n", c, c - (long)i);
            shown++;
        }
    }

    double wmb = (double)len / (double)(t1 - t0);      // B/us == MB/s
    double rmb = (double)len / (double)(t2 - t1);

    if (bad) {
        printf("\nFAIL: %u/%u words differ after a %u MiB round trip.\n", bad, nw, mib);
        if (shown < bad) printf("      (%u more not shown)\n", bad - shown);
        printf("      An 'off by N' above is an address error, not data corruption:\n"
               "      the bytes are intact but landed at the wrong word index.\n");
        return 1;
    }
    printf("\nPASS: %u/%u words identical.  write %.0f MB/s (%" PRIu64 " us), "
           "read %.0f MB/s (%" PRIu64 " us)\n", nw, nw, wmb, t1 - t0, rmb, t2 - t1);

    free(tx); free(rx); close(h2c); close(c2h);
    return 0;
}
