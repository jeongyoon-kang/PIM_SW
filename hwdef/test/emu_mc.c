// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_mc — does the memory controller decode the way the host thinks it does, and
//          how fast does it move bytes.
//
//     ./emu_mc --read                     # MC reads back what direct put there
//     ./emu_mc --write                    # MC writes, direct reads it back
//     ./emu_mc --read --ref               # prove the direct face first
//     ./emu_mc --read --mib 64            # a bigger range, for a bandwidth number
//     ./emu_mc --read --block 4           # split the MC side, to measure the cost
//     ./emu_mc --write --beats 4          # the last 4 beats, one at a time
//
// LAYER 2.  Depends on emu_regs.h, pim_platform.h and libc.  emu_sanity first.
//
// THE TWO PATHS ARE NOT PEERS
//   DIRECT is a back door.  Each bank decodes its own window, so the host names the
//   bank itself and the MC never sees the transfer.  Nothing about DRAM timing or
//   address decoding is exercised.  It exists so this test can place known bytes in
//   known banks without trusting the thing under test.
//
//   MC is the real path.  One contiguous space per channel; the controller splits an
//   address into ROW / BA / CO itself:
//
//        31                    15 14      11 10       5 4      0
//       | ROW (17b)              | BA (4b)  | CO (6b)  | byte   |
//
//   which is exactly the RoBaCo order the runtime lays weights out in, so an MC
//   address IS a payload offset.  There is no bank arithmetic on this side, and any
//   appearing in a tool is a sign the direct window's thinking leaked across.
//
// WHAT IS BEING MEASURED
//   1. does the MC put the payload where the host expects — decoded correctly
//   2. how fast the MC moves it — the number that decides UPLOAD_PATH
//
// THE RANGE IS MC SPACE, AND MC SPACE HAS NO CHANNEL OR BANK IN IT
//   --at and the size name bytes in the whole per-image MC space, because that is
//   what an MC transfer addresses.  Splitting an address into a channel, a bank, a
//   row and a column is the controller's job; the caller states an offset and a
//   length and nothing else.
//
//   They used to be per-bank, with --bank picking which one to move — which only
//   made sense when the MC packed each bank into its own 256 MiB slot.  It does not,
//   so a transfer covering "one bank" is not expressible and that option is gone.
//   --ch survives only under ChRoBaCo, where the channel really is the top address
//   field and "channel N" therefore is a place in the range.
//
// A RANGE IS WHOLE RoBaCo ROWS.  32 KiB = 16 banks x one 2048 B page: the allocator's
// granule and what a single all-bank MAC consumes.  Both faces are then one
// contiguous transfer each — the MC over the range, direct over each bank's share —
// and neither has an edge case at the ends.
//
// POISON FIRST, AND EVERYTHING BEFORE ANY READ-BACK.  A run that writes and reads one
// piece at a time cannot tell a permutation from a pass: a transfer landing in the
// wrong place is read back from that same wrong place.
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
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "emu_regs.h"
#include "pim_platform.h"

#define H2C_DEFAULT "/dev/qdma01000-MM-0"
#define C2H_DEFAULT "/dev/qdma01000-MM-1"

// ONE TRANSFER, UP TO A LIMIT THAT IS THE DRIVER'S AND NOT THE HARDWARE'S.
//
// There used to be a 4 MiB cap here, applied inside xfer() UNDERNEATH --block — so
// --one, which says "the whole range in ONE transfer", quietly went out as 4 MiB
// pieces anyway and the option measured nothing.
//
// SPLITTING NEVER BOUNDED ANYTHING.  The payload is one host allocation of the whole
// range either way (see the aligned_alloc in main), so a block only ever cost
// syscalls.  [measured 2026-08-23] 64 MiB through the MC at 4 / 8 / 16 / 32 / 64 MiB
// blocks: write 880.9 / 883.0 / 884.1 / 884.7 / 886.0 MB/s, read 636.4 / 637.9 /
// 638.3 / 638.6 / 638.9.  One call is marginally the FASTEST — the syscall vanishes
// under a bottleneck that is the controller itself.
//
// BUT ONE pwrite CANNOT BE ARBITRARILY LARGE.  The QDMA driver pins the pages and
// builds a descriptor chain for the WHOLE request, and it runs out: [measured
// 2026-08-23] 400 MiB in one call is fine and 416 MiB returns ENOMEM.  That boundary
// is allocation pressure rather than a register, so it is not a constant — it moves
// with how much of this host is free.
//
// So the default is well under it, and equal to PIM_BANK_WINDOW on purpose: an 8 GiB
// range then goes out in 32 transfers on this face and 32 (channel, bank) shares on
// the direct one.  The same number on both sides is one less thing to explain when
// the two are compared, which is this tool's whole job.
#define MC_XFER_MAX  (256u << 20)

#define ROBACO_ROW  ((size_t)PIM_ROBACO_ROW_BYTES)   // 32 KiB
#define WIN         256u            // bytes poisoned/read around each beat probe

// --ch is sugar for --at, and only under ChRoBaCo — see where it is applied.
static unsigned g_ch = 0;
static bool     g_ch_given = false;
static volatile uint8_t *g_bar = NULL;

// The payload: every 8 B word says which channel offset it belongs to.  Keying it to
// the offset rather than to a (bank, index) pair is deliberate — the MC decodes an
// address into ROW and BA together, so a word that turns up in the wrong place has
// to be able to name where it came from without assuming which field went wrong.
#define PAT_MAGIC     0xC5A17ull
#define POISON_MAGIC  0xDEAD0ull
static uint64_t pat(uint64_t a) { return (PAT_MAGIC    << 44) | (a & 0xFFFFFFFFFFull); }
static uint64_t poi(uint64_t a) { return (POISON_MAGIC << 44) | (a & 0xFFFFFFFFFFull); }

// THE PAYLOAD IS A BYTE STREAM, not a sequence of words.  Byte at channel offset `a`
// is byte a%8 of the word for the 8 B group it falls in — so any range at all is a
// well-defined slice of it, down to a single byte, and a range does not have to start
// or end on any boundary.  Nothing on the bus requires one: sub-beat access lands
// exactly on every window, over DMA and MMIO alike [measured 2026-08-19].
static uint8_t pat_b(uint64_t a) { return (uint8_t)(pat(a & ~7ull) >> (8 * (a & 7))); }
static uint8_t poi_b(uint64_t a) { return (uint8_t)(poi(a & ~7ull) >> (8 * (a & 7))); }

// The same stream, a word at a time where it can be.  Byte at a time over 8 GiB is
// 8.6 billion calls and it dominated the numbers this tool exists to report.
static void fill(uint8_t *p, uint64_t base, size_t n, bool poison)
{
    size_t i = 0;
    while (i < n && ((base + i) & 7)) { p[i] = poison ? poi_b(base + i) : pat_b(base + i); i++; }
    for (; i + 8 <= n; i += 8) {
        uint64_t w = poison ? poi(base + i) : pat(base + i);
        memcpy(p + i, &w, 8);
    }
    while (i < n) { p[i] = poison ? poi_b(base + i) : pat_b(base + i); i++; }
}
static bool     is_pat(uint64_t v) { return (v >> 44) == PAT_MAGIC; }
static bool     is_poi(uint64_t v) { return (v >> 44) == POISON_MAGIC; }
static uint64_t p_off(uint64_t v)  { return v & 0xFFFFFFFFFFull; }

// An MC-space offset, in the coordinates the DIRECT face uses.  Everything here goes
// through pim_decode() so the mode is honoured in exactly one place.
static unsigned a_ch(uint64_t a)   { unsigned c; pim_decode(a,&c,NULL,NULL); return c; }
static unsigned a_bank(uint64_t a) { unsigned b; pim_decode(a,NULL,&b,NULL); return b; }
static uint64_t a_bkoff(uint64_t a){ uint64_t o; pim_decode(a,NULL,NULL,&o); return o; }
static uint64_t a_row(uint64_t a)  { return a_bkoff(a) / EMU_ROW_BYTES; }
static uint64_t a_col(uint64_t a)  { return (a % EMU_ROW_BYTES) / EMU_WORD_BYTES; }
static uint64_t a_direct(uint64_t a)
{ unsigned c,b; uint64_t o; pim_decode(a,&c,&b,&o); return pim_bank(c,b) + o; }

// MODE_CTRL is AXI-Lite, so MMIO — the transport policy for it is the CFR's, not the
// GPR's.  Reading it rather than taking it on the command line is the point: a tool
// that assumed the map would convert with the wrong one and report a wall of
// mismatches that look like a decode fault.
static uint32_t mode_rd(unsigned off)
{ __sync_synchronize(); uint32_t v = *(volatile uint32_t *)(g_bar + EMU_OFF_MODE + off);
  __sync_synchronize(); return v; }


// Sizes the way a person reads them, into one of four rotating buffers so several
// can sit in one printf.
static const char *hsize(uint64_t n)
{
    static char b[4][32]; static unsigned i;
    char *p = b[i++ & 3];
    if      (n >= (1ull << 30)) snprintf(p, 32, "%.4g GiB", (double)n / (double)(1ull << 30));
    else if (n >= (1ull << 20)) snprintf(p, 32, "%.4g MiB", (double)n / (double)(1ull << 20));
    else if (n >= (1ull << 10)) snprintf(p, 32, "%.4g KiB", (double)n / (double)(1ull << 10));
    else                        snprintf(p, 32, "%" PRIu64 " B", n);
    return p;
}

// ---- violation CSR ----------------------------------------------------------
// emu_mc 는 PIM 커널이 아니라 MC 경로 DMA 검증 도구지만, 그 전송도 뱅크
// 컨트롤러를 지나므로 타이밍 위반 카운터가 움직인다.  실행 전에 지우고 뒤에
// 읽어야 이번 전송이 만든 값만 남는다.  카운터는 채널마다 따로 있다.
static uint32_t cfr_rd(uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off);
    __sync_synchronize();
    return v;
}

static uint32_t viol_rd(unsigned ch, uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + off);
    __sync_synchronize();
    return v;
}
static void viol_clear_all(void)
{
    for (unsigned ch = 0; ch < PIM_NCH; ch++) {
        __sync_synchronize();
        *(volatile uint32_t *)(g_bar + PIM_OFF_VIOL + (size_t)ch * 0x1000u + VIOL_CTRL) = 1u;
        __sync_synchronize();
    }
}
// CNT_A/MAX_A 는 [7:0]RCD_RD [15:8]CCD_RD [23:16]RCD_WR [31:24]CCD_WR 로 묶여 있다.
// ANY 비트가 0이어도 값을 찍는다.  "0 건" 과 "안 읽었음" 은 눈으로 구분돼야 한다.
// 위반 표를 볼 때 "무슨 예산에서 나온 수치인가" 가 같이 있어야 해석이 된다.
// 이 값들은 ./emu_timing --scale N --keep 이 걸어둔 것이고, 커널은 건드리지 않는다.
static void timing_print_now(void)
{
    printf("    timing : faw=%u rrd=%u rcd=%u ccd=%u rtp=%u rp=%u wr=%u ras=%u\n",
           cfr_rd(CFR_T_FAW), cfr_rd(CFR_T_RRD), cfr_rd(CFR_T_RCD), cfr_rd(CFR_T_CCD),
           cfr_rd(CFR_T_RTP), cfr_rd(CFR_T_RP),  cfr_rd(CFR_T_WR),  cfr_rd(CFR_T_RAS));
}

// REC_WR (write recovery) and ewmul_drop live in CNT_B/MAX_B, not CNT_A/MAX_A, so
// they have to be read separately — and ANY (+0x404) ORs all of them, which is why a
// read-only run can show ANY=0xffff with every CNT_A column at zero.  Only a write
// run can move REC_WR, so its two columns are printed only then; on a read they would
// be a pair of guaranteed zeros widening the table for nothing.
static void viol_report(bool writing)
{
    for (unsigned ch = 0; ch < PIM_NCH; ch++) {
        uint32_t any = viol_rd(ch, VIOL_ANY);
        printf("\n  violation CSR ch%u ANY: 0x%04x%s\n", ch, any,
               any ? "" : "   (no violations — every row below should read 0)");
        timing_print_now();
        printf("    bank |   RCD_RD    |   CCD_RD    |   RCD_WR    |   CCD_WR    |%s\n",
               writing ? "   REC_WR    | drop |" : "");
        printf("         | cnt    max  | cnt    max  | cnt    max  | cnt    max  |%s\n",
               writing ? " cnt    max  |  cnt |" : "");
        printf("    -----+-------------+-------------+-------------+-------------+%s\n",
               writing ? "-------------+------+" : "");
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            uint32_t ca = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_CNT_A);
            uint32_t ma = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_MAX_A);
            char wr[32] = "";

            if (writing) {
                uint32_t cb = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_CNT_B);
                uint32_t mb = viol_rd(ch, VIOL_BANK_BASE(b) + VIOL_MAX_B);
                snprintf(wr, sizeof wr, " %3u    %3u  | %4u |",
                         cb & 0xffu, mb & 0xffu, (cb >> 8) & 0xffu);
            }
            printf("    %4u | %3u    %3u  | %3u    %3u  | %3u    %3u  | %3u    %3u  |%s%s\n",
                   b,
                   ca         & 0xffu, ma         & 0xffu,
                   (ca >>  8) & 0xffu, (ma >>  8) & 0xffu,
                   (ca >> 16) & 0xffu, (ma >> 16) & 0xffu,
                   (ca >> 24) & 0xffu, (ma >> 24) & 0xffu,
                   wr,
                   ((any >> b) & 1u) ? "  <-- ANY" : "");
        }
    }
}

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

// Sum of the driver's qdma "-error" MSI-X counters.  A nonzero delta across a
// transfer is the leading edge of the interrupt storm that can take the host down;
// a plain EIO with no delta is a contained failure.
static uint64_t err_irq(void)
{
    FILE *f = fopen("/proc/interrupts", "r");
    if (!f) return (uint64_t)-1;
    char line[8192];
    uint64_t total = 0; bool found = false;
    while (fgets(line, sizeof line, f)) {
        if (!strstr(line, "qdma") || !strstr(line, "-error")) continue;
        found = true;
        char *p = strchr(line, ':');
        if (!p) continue;
        for (p++; *p; ) {
            while (*p == ' ' || *p == '\t') p++;
            if (*p < '0' || *p > '9') break;
            total += strtoull(p, &p, 10);
        }
    }
    fclose(f);
    return found ? total : (uint64_t)-1;
}

// Chunk by chunk.  On failure returns the byte offset the failing chunk started at
// and sets *err; on success returns len.  Never gives up silently.
static size_t xfer(int fd, const char *what, uint64_t axi, void *buf, size_t len,
                   bool writing, bool mc, int *err)
{
    *err = 0;
    const char *bad = pim_dma_check_ex(axi, len, mc ? PIM_DMA_ALLOW_MC : 0u);
    if (bad) {
        fprintf(stderr, "  REFUSED %s 0x%011" PRIx64 " (%zu B): %s\n", what, axi, len, bad);
        *err = EINVAL;
        return 0;
    }
    uint8_t *p = buf;
    size_t done = 0;
    // ONE CALL, and the loop is only for a SHORT transfer — pread and pwrite are
    // allowed to move less than asked, and treating that as an error would turn a
    // legal partial into a false failure.  It is not a deliberate split.
    while (done < len) {
        ssize_t n = writing ? pwrite(fd, p + done, len - done, (off_t)(axi + done))
                            : pread (fd, p + done, len - done, (off_t)(axi + done));
        if (n <= 0) { *err = n < 0 ? errno : EIO; return done; }
        done += (size_t)n;
    }
    return done;
}

// ---------------------------------------------------------------- the two faces --
//
// MC: one address, one transfer.  `blk` bounds the host buffer, nothing else — the
// controller does not care where a transfer starts or stops.
static size_t mc_move(int fd, const char *what, uint64_t at, uint8_t *buf, size_t len,
                      size_t blk, bool writing, unsigned *nxfer, int *err)
{
    size_t done = 0;
    *err = 0;
    while (done < len) {
        size_t want = len - done;
        if (blk && want > blk) want = blk;
        size_t got = xfer(fd, what, pim_mc_at(at + done),
                          buf + done, want, writing, true, err);
        (*nxfer)++;
        if (*err) return done + got;
        done += want;
    }
    return done;
}

// DIRECT: where bank b's share of [at, at+len) sits in that bank's own space.
//
// IT IS ALWAYS ONE CONTIGUOUS RUN, whatever the range — so the direct face is still
// one transfer per bank even when the range is not a whole number of rows.  The only
// partial pages are at the two ends: a partial HEAD still reaches its page end, and
// the next row's page for the same bank begins exactly where it left off in bank
// space, so nothing in between is missing.  Only the last row can stop short.
// [checked by construction over 58k ranges, including sub-page and row-straddling]
//
// The run does NOT start at a page boundary in general — at=0x7e0 gives bank 0 a
// 32 B piece at bank offset 2016 and bank 1 a 32 B piece at 0.
static bool bank_share(uint64_t at, size_t len, unsigned c, unsigned b,
                       uint64_t *bkoff, size_t *n)
{
    const uint64_t A0 = at, A1 = at + len;
    bool any = false; uint64_t lo = 0, hi = 0;
    for (uint64_t p = A0 / EMU_ROW_BYTES; p <= (A1 - 1) / EMU_ROW_BYTES; p++) {
        uint64_t ps = p * EMU_ROW_BYTES, pe = ps + EMU_ROW_BYTES;
        if (a_ch(ps) != c || a_bank(ps) != b) continue;      // not this (ch, bank)'s page
        uint64_t s = ps > A0 ? ps : A0, e = pe < A1 ? pe : A1;
        if (s >= e) continue;
        if (!any) { lo = s; any = true; }
        hi = e;
    }
    if (!any) return false;                      // a short range misses most banks
    *bkoff = a_bkoff(lo);
    *n     = (size_t)(a_bkoff(hi - 1) + 1 - a_bkoff(lo));
    return true;
}

// Move bank b's share between the payload (channel order) and a contiguous buffer
// holding what that bank's window sees.  Row by row, because the payload side is
// strided even where the bank side is not.
static void bank_copy(uint8_t *bankbuf, uint8_t *payload, uint64_t at, size_t len,
                      unsigned c, unsigned b, uint64_t bkoff, bool to_bank)
{
    const uint64_t A0 = at, A1 = at + len;
    for (uint64_t p = A0 / EMU_ROW_BYTES; p <= (A1 - 1) / EMU_ROW_BYTES; p++) {
        uint64_t ps = p * EMU_ROW_BYTES, pe = ps + EMU_ROW_BYTES;
        if (a_ch(ps) != c || a_bank(ps) != b) continue;
        uint64_t s = ps > A0 ? ps : A0, e = pe < A1 ? pe : A1;
        if (s >= e) continue;
        uint8_t *pb = bankbuf + (a_bkoff(s) - bkoff);
        uint8_t *pp = payload + (s - A0);
        if (to_bank) memcpy(pb, pp, (size_t)(e - s));
        else         memcpy(pp, pb, (size_t)(e - s));
    }
}

// Name the first place a read-back diverges, in every coordinate needed to go look
// at it — including where the byte that DID arrive belongs, which is the difference
// between "the bytes are damaged" and "the decode sent them somewhere else".
static void locate(const uint8_t *got, uint64_t at, size_t len, size_t i)
{
    uint64_t a = at + i;
    printf("     FIRST DIVERGENCE at channel offset 0x%09" PRIx64 "\n", a);
    printf("       expected  0x%02x   got 0x%02x\n", pat_b(a), got[i]);

    // If the whole 8 B group this byte belongs to is inside the range, the word can
    // be put back together and asked where it came from — which is the difference
    // between damaged bytes and bytes that were decoded to the wrong place.
    uint64_t w0 = (a & ~7ull);
    uint64_t v = 0; bool whole = (w0 >= at) && (w0 + 8 <= at + len);
    if (whole) memcpy(&v, got + (size_t)(w0 - at), 8);
    if (!whole)          printf("       (its 8 B group is cut by the range, so the origin cannot be read)\n");
    else if (is_poi(v))  printf("       word 0x%016" PRIx64 " — still POISON, nothing was written here\n", v);
    else if (!is_pat(v)) printf("       word 0x%016" PRIx64 " — neither the pattern nor the poison\n", v);
    else                 printf("       word 0x%016" PRIx64 " — the payload from offset 0x%09" PRIx64 "\n",
                                v, p_off(v));
    printf("       expected at  ch%u bank %2u  row %6" PRIu64 "  col %2" PRIu64
           "   direct 0x%011" PRIx64 "   MC 0x%011" PRIx64 "\n",
           a_ch(a), a_bank(a), a_row(a), a_col(a), a_direct(a), pim_mc_at(a));
    if (whole && is_pat(v)) {
        uint64_t o = p_off(v);
        printf("       arrived from ch%u bank %2u  row %6" PRIu64 "  col %2" PRIu64 "%s\n",
               a_ch(o), a_bank(o), a_row(o), a_col(o),
               a_ch(o)   != a_ch(a)   ? "   <-- WRONG CHANNEL"
             : a_bank(o) != a_bank(a) ? "   <-- WRONG BANK"
             : a_row(o)  != a_row(a)  ? "   <-- WRONG ROW" : "");
    }
}

static void usage(const char *p)
{
    printf(
"Usage: %s [--read | --write] [--ch N] [--at OFF] [--mib N | --kib N]\n"
"\n"
"Checks that the memory controller decodes ROW/BA/CO the way the host does, and\n"
"measures what it moves.  The direct aperture is the reference face: it reaches the\n"
"banks without going through the MC, so it can place known bytes without trusting\n"
"the thing under test.\n"
"\n"
"  --read          MC is the READ side: direct writes the payload, MC reads it back\n"
"  --write         MC is the WRITE side: MC writes, direct reads it back (default)\n"
"\n"
"  --ch N          ChRoBaCo ONLY: shorthand for --at N*4GiB, because there the\n"
"                  channel IS the top address field.  Rejected under RoChBaCo,\n"
"                  where a range of any size covers every channel\n"
"  --at OFF        where in MC space to start.  An MC address is exactly this\n"
"                  number, and MC space has no channel or bank in it — splitting\n"
"                  an address up is the controller's job, not the caller's\n"
"  --mib N         how much, in MiB (default 4)\n"
"  --kib N         how much, in KiB\n"
"  --bytes N       how much, in bytes.  ANY size from 1 up — the range does not\n"
"                  have to be rows, beats or even 8 B: sub-byte is the only thing\n"
"                  a bus cannot do.  The direct face handles ends mid-page\n"
"\n"
"  --ref           read the payload back through the DIRECT aperture first, so a\n"
"                  failure on the MC side cannot be blamed on the reference face\n"
"  --block N       split the MC side into N MiB transfers.  OFF by default: the\n"
"                  payload is one host allocation either way, so this bounds\n"
"                  nothing and costs syscalls.  It is here to MEASURE the split,\n"
"                  which is the only reason to want it\n"
"                  controller itself does not care\n"
"  --one           the default, kept so a script can say it: one transfer.  The\n"
"                  host buffer is the whole range regardless — see --block\n"
"                  in host memory\n"
"  --beats N       probe the last N 32 B beats one at a time, checking that the\n"
"                  neighbouring beats are NOT disturbed.  There is no WSTRB on this\n"
"                  bus, so a partial beat rewrites its whole word\n"
"  --settle-ms N   wait N ms between the transfer and the read-back\n"
"  --dump PATH     write the read-back buffer to PATH before judging it.  Every 8 B\n"
"                  word names the channel offset it was written for, so the file\n"
"                  says where each byte CAME FROM, not just that it is wrong.\n"
"  --h2c PATH      default %s\n"
"  --c2h PATH      default %s\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p, H2C_DEFAULT, C2H_DEFAULT);
}

int main(int argc, char **argv)
{
    int dir = -1;                       // -1 = MC writes, +1 = MC reads
    uint64_t at = 0;
    unsigned long bytes = 4ul << 20;
    unsigned beats = 0, settle_ms = 0;
    // ONE TRANSFER BY DEFAULT.  The payload is already ONE host allocation of the
    // whole range (see the aligned_alloc below), so splitting it bought nothing but
    // syscalls: [measured 2026-08-23] 64 MiB through the MC at 4/8/16/32/64 MiB
    // blocks ran at 880.9/883.0/884.1/884.7/886.0 MB/s writing and 636.4/637.9/
    // 638.3/638.6/638.9 reading — one call is marginally the fastest, because the
    // bottleneck is the controller and the syscall vanishes under it.
    size_t blk = MC_XFER_MAX;       // see MC_XFER_MAX; --block / --one override
    bool ref = false;
    const char *h2c_path = H2C_DEFAULT, *c2h_path = C2H_DEFAULT;
    // --dump: 판정 직전의 read-back 버퍼를 그대로 파일에 쓴다.  판정은 맞다/틀리다만
    // 말하고 첫 불일치 하나만 보여주는데, 어긋난 데이터가 어디서 온 것인지는 전 구간을
    // 봐야 알 수 있다.  payload 워드가 자기 채널 오프셋을 값에 담고 있으므로(pat()),
    // 이 파일 하나면 바이트마다 출처를 되물을 수 있다 — 모델을 세우지 않고.
    const char *dump_path = NULL;

    static const struct option lo[] = {
        { "ch",        required_argument, NULL, 'C' },
        { "read",      no_argument,       NULL,  1 }, { "write",     no_argument,       NULL,  2 },
        { "at",        required_argument, NULL,  3 }, { "mib",       required_argument, NULL,  4 },
        { "kib",       required_argument, NULL,  5 }, { "ref",       no_argument,       NULL,  6 },
        { "bytes",     required_argument, NULL, 13 },
        { "block",     required_argument, NULL,  7 }, { "one",       no_argument,       NULL,  8 },
        { "beats",     required_argument, NULL,  9 }, { "settle-ms", required_argument, NULL, 10 },
        { "h2c",       required_argument, NULL, 11 }, { "c2h",       required_argument, NULL, 12 },
        { "dump",      required_argument, NULL, 14 },
        { "help",      no_argument,       NULL, 'h' },
        { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1:  dir = +1; break;
        case 2:  dir = -1; break;
        case 3:  at = strtoull(optarg, NULL, 0); break;
        case 4:  bytes = strtoul(optarg, NULL, 0) << 20; break;
        case 5:  bytes = strtoul(optarg, NULL, 0) << 10; break;
        case 6:  ref = true; break;
        case 7:  blk = (size_t)strtoul(optarg, NULL, 0) << 20; break;
        case 8:  blk = 0; break;
        case 9:  beats = (unsigned)strtoul(optarg, NULL, 0); break;
        case 10: settle_ms = (unsigned)strtoul(optarg, NULL, 0); break;
        case 11: h2c_path = optarg; break;
        case 12: c2h_path = optarg; break;
        case 13: bytes = strtoul(optarg, NULL, 0); break;
        case 14: dump_path = optarg; break;
        case 'C': g_ch = (unsigned)strtoul(optarg, NULL, 0); g_ch_given = true; break;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }

    const char *pe = pim_platform_check();
    if (pe) { fprintf(stderr, "ERROR: %s\n", pe); return 2; }
    pim_platform_banner();

    // The map is COMPILED IN, so this reads the register only to refuse a board that
    // disagrees — the same shape as pim_platform_check().  It does not write it:
    // exactly one thing does, and that is emu_sanity, which reprogram.sh already
    // runs.  A probe that could flip the map at run time would be the one build in
    // the tree whose conversion disagreed with everything else's, which is the
    // confusion this whole arrangement exists to remove.
    {
        char rp[256]; snprintf(rp, sizeof rp, "/sys/bus/pci/devices/%s/resource2", "0000:01:00.0");
        int bf = open(rp, O_RDWR | O_SYNC);
        if (bf < 0) { fprintf(stderr, "ERROR: open %s: %s\n       sudo ./setup_permissions.sh\n",
                              rp, strerror(errno)); return 1; }
        void *mm = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, bf, 0);
        if (mm == MAP_FAILED) { perror("mmap BAR2"); return 1; }
        g_bar = mm;
        const char *bad = pim_addr_map_check(mode_rd(MODE_CTRL));
        if (bad) { fprintf(stderr, "ERROR: %s\n", bad); return 2; }
    }

    // --at AND THE SIZE NAME MC SPACE, AND MC SPACE HAS NO CHANNEL IN IT.  Splitting
    // an address into a channel is the controller's job; a range is just a range.
    // --ch is the one concession, and only where it means something: under ChRoBaCo
    // the channel is the top address field, so "channel N" IS a place in the range —
    // 4 GiB in.  Under RoChBaCo it is not a place at all: the channel changes every
    // 32 KiB, so every range of any size covers all of them.
    if (g_ch_given) {
        if (PIM_ADDR_MAP == PIM_MAP_ROCHBACO) {
            fprintf(stderr,
                "ERROR: --ch means nothing under RoChBaCo.  The channel changes every "
                "%zu KiB,\n"
                "       so any range covers every channel and none can be singled out.\n"
                "       Drop --ch, or build for ChRoBaCo:  scripts/setup.sh --platform %s --map 1\n",
                ROBACO_ROW >> 10, PIM_PLATFORM_NAME);
            return 2;
        }
        if (g_ch >= PIM_NCH) {
            fprintf(stderr, "ERROR: --ch %u but %s has %u channel\n",
                    g_ch, PIM_PLATFORM_NAME, PIM_NCH);
            return 2;
        }
        at += pim_ch_first(g_ch);       // ChRoBaCo: exactly --at ch*4GiB
    }
    const size_t len = (size_t)bytes;
    // ALL of MC space, not one channel's worth: NCH x (16 banks x 256 MiB).  It was
    // the per-channel figure, which made --at past the first channel — the only way
    // to reach channel 1 under ChRoBaCo — look like it ran off the end.
    const uint64_t MCSPACE = (uint64_t)PIM_NCH * PIM_BANK_WINDOW * PIM_NBANK;
    if (!len)                       { fprintf(stderr, "ERROR: zero length\n"); return 2; }
    if (at + len > MCSPACE) {
        fprintf(stderr, "ERROR: the range runs past MC space (%" PRIu64 " GiB = %u ch x "
                        "%u banks x %" PRIu64 " MiB)\n",
                MCSPACE >> 30, PIM_NCH, PIM_NBANK, (uint64_t)(PIM_BANK_WINDOW >> 20));
        return 2;
    }
    if (beats > 64)  { fprintf(stderr, "ERROR: --beats must be <= 64\n"); return 2; }
    if (blk && blk % ROBACO_ROW) blk = (blk / ROBACO_ROW) * ROBACO_ROW;
    if (blk && !blk) blk = ROBACO_ROW;

    viol_clear_all();   // 이번 전송분만 세도록

    const bool reading = (dir > 0);
    // A bank's share is one contiguous run, but the ends of the range make the runs
    // differ in length and start — so each is asked for rather than assumed.
    // A range is (channel, bank) shares now, not bank shares: under RoChBaCo it
    // crosses channels every 32 KiB, so the direct face has up to NCH x 16 of them.
    static uint64_t bkoff[PIM_NCH][EMU_NBANKS];
    static size_t   bklen[PIM_NCH][EMU_NBANKS];
    unsigned nbk = 0; size_t maxbk = 0;
    for (unsigned c = 0; c < PIM_NCH; c++)
        for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!bank_share(at, len, c, b, &bkoff[c][b], &bklen[c][b])) { bklen[c][b] = 0; continue; }
            nbk++;
            if (bklen[c][b] > maxbk) maxbk = bklen[c][b];
        }

    if (PIM_NCH > 1 && PIM_ADDR_MAP == PIM_MAP_CHROBACO)
        printf("  channels start %" PRIu64 " GiB apart; --ch N is --at N*that\n",
               (uint64_t)(PIM_MC_CH_SPAN >> 30));
    printf("\nMC is the %s side\n", reading ? "READ" : "WRITE");
    printf("  range   MC +0x%09" PRIx64 " .. +0x%09" PRIx64 "   %s\n",
           at, at + len - 1, hsize(len));
    printf("  MC      one address 0x%011" PRIx64 ", %s\n", pim_mc_at(at),
           blk ? "blocked" : "ONE transfer");
    printf("  direct  %u of %u (channel, bank) shares, %s each at most\n",
           nbk, PIM_NCH * EMU_NBANKS, hsize(maxbk));
    if (blk) printf("  blocks  %s per MC transfer\n", hsize(blk));

    uint8_t *payload = aligned_alloc(4096, len);
    uint8_t *back    = aligned_alloc(4096, len);
    uint8_t *bank    = aligned_alloc(4096, maxbk ? maxbk : 8);
    if (!payload || !back || !bank) { fprintf(stderr, "ERROR: out of memory\n"); return 1; }
    for (size_t i = 0; i < len; i++) payload[i] = pat_b(at + i);

    int h2c = open(h2c_path, O_WRONLY), c2h = open(c2h_path, O_RDONLY);
    if (h2c < 0 || c2h < 0) {
        fprintf(stderr, "ERROR: cannot open the MM queues (%s / %s): %s\n"
                        "       sudo ./qdma_queues.sh setup\n",
                h2c_path, c2h_path, strerror(errno));
        return 1;
    }
    uint64_t irq0 = err_irq();      // reported at the end, as a delta
    int rc = 0, e;

    // ---- 0. poison, through the direct face, every bank before anything reads ----
    // THE CLOCK STARTS AFTER THE BUFFER IS BUILT.  Generating the stream is this
    // tool's own cost, not the path's — and counting it in one step and not the other
    // is what made poison look 7x slower than an identical write.
    fill(payload, at, len, true);
    uint64_t t0 = now_us();
    for (unsigned c = 0; c < PIM_NCH; c++)
      for (unsigned b = 0; b < EMU_NBANKS; b++) {
        if (!bklen[c][b]) continue;
        bank_copy(bank, payload, at, len, c, b, bkoff[c][b], true);
        xfer(h2c, "poison", pim_bank(c, b) + bkoff[c][b], bank, bklen[c][b], true, false, &e);
        if (e) {
            printf("0. POISON FAILED on ch%u bank %u: %s\n"
                   "   the direct write path is not working — the H2C engine is probably\n"
                   "   latched from an earlier failure.  Recycle the queues:\n"
                   "     sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n",
                   c, b, strerror(e));
            rc = 1; goto done;
        }
      }
    uint64_t t0e = now_us();
    fill(payload, at, len, false);

    // ---- 1. the payload ---------------------------------------------------------
    printf("\n  step           via       transfers        MB/s\n");
    printf("  -------------- --------- --------- -----------\n");
    printf("  poison         direct    %9u %11.0f\n", nbk,
           (double)len / (double)(t0e - t0));

    unsigned nxfer = 0;
    uint64_t t1 = now_us(), t2;
    if (reading) {
        for (unsigned c = 0; c < PIM_NCH; c++)
          for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!bklen[c][b]) continue;
            bank_copy(bank, payload, at, len, c, b, bkoff[c][b], true);
            xfer(h2c, "write", pim_bank(c, b) + bkoff[c][b], bank, bklen[c][b], true, false, &e);
            nxfer++;
            if (e) { printf("1. DIRECT WRITE FAILED on ch%u bank %u: %s\n", c, b, strerror(e));
                     rc = 1; goto done; }
          }
        t2 = now_us();
        printf("  write payload  direct    %9u %11.0f\n", nxfer,
               (double)len / (double)(t2 - t1));
    } else {
        mc_move(h2c, "write", at, payload, len, blk, true, &nxfer, &e);
        t2 = now_us();
        if (e) {
            printf("1. MC WRITE FAILED: %s\n"
                   "   the H2C engine latches on an error; every later write fails too.\n"
                   "     sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n",
                   strerror(e));
            rc = 1; goto done;
        }
        printf("  write payload  MC        %9u %11.0f   <-- MC WRITE\n", nxfer,
               (double)len / (double)(t2 - t1));
    }
    if (settle_ms) usleep(settle_ms * 1000u);

    // ---- 2. optional reference read, direct -> direct ----------------------------
    if (ref && reading) {
        memset(back, 0, len);
        for (unsigned c = 0; c < PIM_NCH; c++)
          for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!bklen[c][b]) continue;
            xfer(c2h, "ref read", pim_bank(c, b) + bkoff[c][b], bank, bklen[c][b], false, false, &e);
            if (e) { printf("  reference      direct    FAILED ch%u bank %u: %s\n", c, b, strerror(e)); rc = 1; goto done; }
            bank_copy(bank, back, at, len, c, b, bkoff[c][b], false);
          }
        size_t i = 0;
        while (i < len && back[i] == pat_b(at + i)) i++;
        if (i < len) {
            printf("  reference      direct    ALREADY WRONG\n");
            locate(back, at, len, i);
            printf("\nFAIL: the reference face itself is wrong.  Nothing can be concluded\n"
                   "      about the MC until the direct aperture round-trips.\n");
            rc = 1; goto done;
        }
        printf("  reference      direct    %9u %11s\n", nbk, "clean");
    }

    // ---- 3. read back -----------------------------------------------------------
    memset(back, 0, len);
    nxfer = 0;
    uint64_t t3 = now_us(), t4;
    if (reading) {
        mc_move(c2h, "read", at, back, len, blk, false, &nxfer, &e);
        t4 = now_us();
        if (e) { printf("  read back      MC        FAILED: %s\n", strerror(e)); rc = 1; goto done; }
        printf("  read back      MC        %9u %11.0f   <-- MC READ\n", nxfer,
               (double)len / (double)(t4 - t3));
    } else {
        for (unsigned c = 0; c < PIM_NCH; c++)
          for (unsigned b = 0; b < EMU_NBANKS; b++) {
            if (!bklen[c][b]) continue;
            xfer(c2h, "read", pim_bank(c, b) + bkoff[c][b], bank, bklen[c][b], false, false, &e);
            nxfer++;
            if (e) { printf("  read back      direct    FAILED ch%u bank %u: %s\n", c, b, strerror(e));
                     rc = 1; goto done; }
            bank_copy(bank, back, at, len, c, b, bkoff[c][b], false);
          }
        t4 = now_us();
        printf("  read back      direct    %9u %11.0f\n", nxfer,
               (double)len / (double)(t4 - t3));
    }

    // ---- dump ---------------------------------------------------------------------
    // 판정 전에 쓴다.  판정이 실패해도 (goto done) 파일은 남아야 진단이 된다.
    if (dump_path) {
        FILE *df = fopen(dump_path, "wb");
        if (!df) {
            fprintf(stderr, "WARN: --dump %s: %s\n", dump_path, strerror(errno));
        } else {
            size_t w = fwrite(back, 1, len, df);
            if (fclose(df) || w != len)
                fprintf(stderr, "WARN: --dump %s: wrote %zu of %zu B\n", dump_path, w, len);
            else
                printf("  read-back buffer -> %s (%zu B)\n", dump_path, len);
        }
    }

    // ---- verdict ----------------------------------------------------------------
    size_t bad = 0, first = len, good = 0;
    for (size_t i = 0; i < len; i++) {
        if (back[i] == pat_b(at + i)) { if (i == good) good++; continue; }
        if (first == len) first = i;
        bad++;
    }
    printf("\n");
    if (!bad) {
        printf("PASS  %s, every byte where the host put it\n"
               "      %u (channel, bank) shares, all poisoned and all transferred\n"
               "      before any read-back — a permuted address could not have passed\n",
               hsize(len), nbk);
    } else {
        printf("FAIL: %zu of %zu B wrong (%.1f%%)\n", bad, len, 100.0 * (double)bad / (double)len);
        locate(back, at, len, first);
        if (bad == len - good)
            printf("     everything from +0x%09" PRIx64 " to the end is wrong — a tail,"
                   " not a hole\n", at + good);
        else
            printf("     NOT a clean tail: %zu B after +0x%09" PRIx64 " are still correct,"
                   " so this is damage in places\n", (len - good) - bad, at + good);
        rc = 1;
    }

    // ---- beat probe -------------------------------------------------------------
    // One 32 B beat at a time from the end, stopping at the first failure — a failed
    // H2C latches the engine and everything after it would be meaningless.  The beats
    // on either side must stay poison: a write that spills is as wrong as one that
    // fails.
    if (beats && !rc) {
        printf("\nthe last %u beats, one at a time\n", beats);
        uint8_t *w = aligned_alloc(4096, WIN);
        if (!w) { fprintf(stderr, "ERROR: out of memory\n"); rc = 1; goto done; }
        for (unsigned i = 0; i < beats; i++) {
            uint64_t a   = at + len - (uint64_t)(beats - i) * EMU_WORD_BYTES;
            uint64_t win = (a + EMU_WORD_BYTES >= WIN) ? (a + EMU_WORD_BYTES - WIN) : 0;
            win &= ~(uint64_t)(EMU_WORD_BYTES - 1);
            size_t in = (size_t)(a - win);

            printf("  probe %u: 32 B at channel offset 0x%09" PRIx64 " (end-%" PRIu64 ")\n",
                   i + 1, a, at + len - a);
            for (size_t o = 0; o < WIN; o++) w[o] = poi_b(win + o);
            unsigned n = 0;
            mc_move(h2c, "poison", win, w, WIN, 0, true, &n, &e);
            if (e) { printf("    poison failed: %s\n", strerror(e)); rc = 1; break; }

            for (size_t o = 0; o < EMU_WORD_BYTES; o++) w[o] = pat_b(a + o);
            n = 0;
            mc_move(h2c, "write", a, w, EMU_WORD_BYTES, 0, true, &n, &e);
            printf("    MC write: %s\n", e ? strerror(e) : "reported success");
            if (settle_ms) usleep(settle_ms * 1000u);

            memset(w, 0, WIN);
            n = 0;
            mc_move(c2h, "read", win, w, WIN, 0, false, &n, &e);
            if (e) { printf("    read back failed: %s\n", strerror(e)); rc = 1; break; }

            unsigned nbad = 0, spill = 0;
            for (size_t o = 0; o < EMU_WORD_BYTES; o++)
                if (w[in + o] != pat_b(a + o)) nbad++;
            for (size_t o = 0; o < WIN; o++) {
                if (o >= in && o < in + EMU_WORD_BYTES) continue;
                if (w[o] != poi_b(win + o)) spill++;
            }
            printf("    beat %s, neighbours %s\n",
                   nbad ? "WRONG" : "ok", spill ? "DISTURBED" : "untouched");
            if (nbad || spill) rc = 1;
            if (err_irq() != irq0) {
                printf("    qdma error IRQ moved — stopping before the engine latches\n");
                rc = 1; break;
            }
        }
        free(w);
    }

done:
    {
        uint64_t irq1 = err_irq();
        if (irq1 != irq0)
            printf("\nqdma error IRQ moved %" PRIu64 " -> %" PRIu64 " — the leading edge of\n"
                   "the interrupt storm that can take the host down.\n", irq0, irq1);
    }
    if (rc)
        printf("\nIf a write failed, the H2C engine is latched and every later write will\n"
               "fail too until the queues are recycled:\n"
               "  sudo ./qdma_queues.sh teardown && sudo ./qdma_queues.sh setup\n");
    free(payload); free(back); free(bank);
    close(h2c); close(c2h);
    viol_report(!reading);
    printf("%s\n", rc ? "FAIL" : "PASS");
    return rc;
}
