// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// geom_test — the address arithmetic, checked against something that already agreed
// with the board.
//
// NEEDS NO HARDWARE.  Everything here is pure arithmetic on a pim_geometry built by
// hand, which is the point: the numbers can be wrong long before a board is
// available to notice, and this is where that gets caught.
//
// THIS IS THE ONE PROGRAM THAT INCLUDES BOTH TOPOLOGY SOURCES ON PURPOSE.
//   sw/include/pim/pim_geometry.h   parameterised, from the driver at run time
//   hwdef/pim_platform.h            compiled in, from platform/*.conf via -D
// Nothing else may do that (see sw/README.md §6), because a binary with two answers
// for the channel count is worse than one with none.  But a CHECK has to hold both
// up against each other, and hwdef's is the version emu_mc drove over 8 GiB of real
// card, so it is the reference and this is the thing being checked.
//
// Build hwdef for the topology you want compared:  scripts/setup.sh --map 2
//////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "pim/pim_geometry.h"
#include "pim_platform.h"

static pim_geometry mk(unsigned nch, uint64_t hugepage, uint64_t gpr_page)
{
    pim_geometry g = {0};
    g.nch = nch; g.nbank = 16; g.row_bytes = 2048;
    g.map = PIM_MAP_RO_CH_BA_CO;
    g.unit_bytes = (uint64_t)nch * 16 * 2048;

    g.mem[PIM_MEM_DRAM].base        = 0x020400000000ull;      /* MC s_axi */
    g.mem[PIM_MEM_DRAM].bytes       = (uint64_t)nch * 0x100000000ull;
    g.mem[PIM_MEM_DRAM].hugepage_bytes = hugepage;
    g.mem[PIM_MEM_DRAM].nr_hugepages   = g.mem[PIM_MEM_DRAM].bytes / hugepage;

    g.mem[PIM_MEM_GPR].base         = 0x020200000000ull;      /* BAR2 + OFF_GPR */
    g.mem[PIM_MEM_GPR].bytes        = 4ull << 20;
    g.mem[PIM_MEM_GPR].hugepage_bytes  = gpr_page;
    g.mem[PIM_MEM_GPR].nr_hugepages    = (4ull << 20) / gpr_page;

    pim_geom_derive(&g);
    return g;
}

int main(void)
{
    int fail = 0;

    // ---- 1. geom_check accepts the three real topologies -------------------
    for (unsigned nch = 1; nch <= 4; nch <<= 1) {
        pim_geometry g = mk(nch, 2u<<20, 4096);
        const char *bad = pim_geom_check(&g);
        printf("ch%u  dram: unit %6llu B  %2u/hugepage  shift %2u   |   gpr: gran %llu B  %u/hugepage  -> %s\n",
               nch, (unsigned long long)g.unit_bytes,
               g.mem[PIM_MEM_DRAM].grans_per_hugepage, g.mem[PIM_MEM_DRAM].gran_shift,
               (unsigned long long)g.mem[PIM_MEM_GPR].gran,
               g.mem[PIM_MEM_GPR].grans_per_hugepage, bad ? bad : "OK");
        if (bad) fail++;
    }

    // ---- 2. the invariant must be REFUSED when it fails --------------------
    { pim_geometry g = mk(4, 64u<<10, 4096);      // hugepage 64 KiB < unit 128 KiB
      const char *bad = pim_geom_check(&g);
      printf("\nchunk 64 KiB with unit 128 KiB -> %s\n", bad ? "refused" : "ACCEPTED (BUG)");
      if (!bad) fail++; else printf("   \"%s\"\n", bad); }
    { pim_geometry g = mk(1, 16u<<20, 4096);      // 512 units per hugepage > 64
      const char *bad = pim_geom_check(&g);
      printf("hugepage 16 MiB with unit 32 KiB  -> %s\n", bad ? "refused" : "ACCEPTED (BUG)");
      if (!bad) fail++; else printf("   \"%s\"\n", bad); }
    { pim_geometry g = mk(3, 2u<<20, 4096);       // nch not a power of two
      const char *bad = pim_geom_check(&g);
      printf("nch = 3                        -> %s\n", bad ? "refused" : "ACCEPTED (BUG)");
      if (!bad) fail++; }

    // ---- 2b. the GPR's own rules -------------------------------------------
    { pim_geometry g = mk(4, 2u<<20, 16);     // page smaller than one 32 B word
      const char *bad = pim_geom_check(&g);
      printf("gpr page 16 B (< one 32 B word) -> %s\n", bad ? "refused" : "ACCEPTED (BUG)");
      if (!bad) fail++; else printf("   \"%s\"\n", bad); }
    { pim_geometry g = mk(4, 2u<<20, 4096);
      uint64_t base = g.mem[PIM_MEM_GPR].base, bad_w = 0;
      for (uint64_t w = 0; w < g.mem[PIM_MEM_GPR].bytes / 32; w += 997)
          if (pim_gpr_word_of(&g, base + w * 32) != w) bad_w++;
      printf("gpr word index round trip      -> %s\n", bad_w ? "MISMATCH" : "ok");
      if (bad_w) fail++; }

    // ---- 3. decode/encode round trip ---------------------------------------
    {
        pim_geometry g = mk(4, 2u<<20, 4096);
        uint64_t bad_rt = 0, n = 0;
        for (uint64_t off = 0; off < g.mem[PIM_MEM_DRAM].bytes; off += 977 * 32 + 1) {
            pim_coord c; pim_geom_decode(&g, g.mem[PIM_MEM_DRAM].base + off, &c);
            if (pim_geom_encode(&g, &c) != g.mem[PIM_MEM_DRAM].base + off) bad_rt++;
            n++;
        }
        printf("\nround trip RoChBaCo ch4 : %"PRIu64" of %"PRIu64" mismatched\n", bad_rt, n);
        if (bad_rt) fail++;
    }
    {
        pim_geometry g = mk(4, 2u<<20, 4096); g.map = PIM_MAP_CH_RO_BA_CO;
        uint64_t bad_rt = 0, n = 0;
        for (uint64_t off = 0; off < g.mem[PIM_MEM_DRAM].bytes; off += 977 * 32 + 1) {
            pim_coord c; pim_geom_decode(&g, g.mem[PIM_MEM_DRAM].base + off, &c);
            if (pim_geom_encode(&g, &c) != g.mem[PIM_MEM_DRAM].base + off) bad_rt++;
            n++;
        }
        printf("round trip ChRoBaCo ch4 : %"PRIu64" of %"PRIu64" mismatched\n", bad_rt, n);
        if (bad_rt) fail++;
    }

    // ---- 4. agree with hwdef, which agreed with the board -------------------
    {
        pim_geometry g = mk(PIM_NCH, 2u<<20, 4096);
        uint64_t n = 0, bad_ch = 0, bad_bk = 0, bad_off = 0;
        printf("\ncross-check vs hwdef (built for %s, map %s):\n",
               PIM_PLATFORM_NAME, PIM_ADDR_MAP_NAME);
        for (uint64_t off = 0; off < g.mem[PIM_MEM_DRAM].bytes; off += 4093 * 8 + 3) {
            unsigned hch, hbk; uint64_t hoff;
            pim_coord c;
            pim_decode_as(PIM_MAP_ROCHBACO, off, &hch, &hbk, &hoff);
            pim_geom_decode(&g, g.mem[PIM_MEM_DRAM].base + off, &c);
            if (c.ch != hch)   bad_ch++;
            if (c.bank != hbk) bad_bk++;
            if (c.row * g.row_bytes + c.col_byte != hoff) bad_off++;
            n++;
        }
        printf("  %"PRIu64" points: ch %"PRIu64" bad, bank %"PRIu64" bad, bank-offset %"PRIu64" bad\n",
               n, bad_ch, bad_bk, bad_off);
        if (bad_ch || bad_bk || bad_off) fail++;
    }

    printf("\n%s\n", fail ? "FAIL" : "all checks passed");
    return fail != 0;
}
