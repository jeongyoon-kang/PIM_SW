// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pool_test — both allocators, driven against a SIMULATED driver.
//
// NEEDS NO HARDWARE and no loaded module.  It links the real pim_pool.c,
// pim_xlate.c, pim_vspace.c, pim_geom.c and the runtime's pim_rt.c, and replaces
// only the two ioctl wrappers — so what runs is the shipping code with a fake card
// underneath it.
//
// LINKS OBJECTS, NOT libpim.so — deliberately, and it is the exception.  The rule
// elsewhere in this tree is that a test may only call what an application could
// call, so that it tests what ships rather than the implementation.  That rule
// cannot hold here: substituting the driver means substituting a symbol that is
// internal by design.  The integration test that goes through libpim.so needs a
// loaded pim.ko and a board, and is a different program.
//
// WHAT IT IS ACTUALLY LOOKING FOR is aliasing.  Every other property here would
// survive a subtly wrong granule index; two live allocations resolving to the same
// card address would not, and it is the failure that would be hardest to diagnose
// from a wrong GEMV result later.  With two pools (design §9.4) there is a second
// version of the same worry — a DRAM index used against the GPR's granule — so the
// alias check runs across BOTH pools at once rather than once per pool.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fake_drv.h"
#include "pim/pim_addr.h"

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); fail++; } }while(0)

/* Every live granule, in either pool, must map to a distinct AXI address. */
static void check_no_alias(pim_ctx *c, void **p, size_t *sz, int n)
{
    static uint64_t seen[8192]; size_t ns = 0;
    for (int i = 0; i < n; i++) {
        pim_mem w;
        if (!p[i] || !pim_where_ctx(c, p[i], &w)) continue;
        for (size_t off = 0; off < sz[i]; off += c->geom.mem[w].gran) {
            pim_loc loc; size_t run;
            const char *bad = pim_resolve_ctx(c, (char*)p[i]+off, 1, &loc, &run);
            CHECK(!bad, "resolve alloc %d off %zu: %s", i, off, bad?bad:"");
            if (bad) continue;
            CHECK(loc.mem == w, "alloc %d resolved into the wrong pool", i);
            for (size_t k = 0; k < ns; k++)
                CHECK(seen[k] != loc.axi, "ALIAS: two live granules both at %#"PRIx64, loc.axi);
            if (ns < sizeof seen / sizeof *seen) seen[ns++] = loc.axi;
        }
    }
}

int main(void)
{
    pim_ctx *c = mkctx(4);
    printf("dram: unit %llu B, %u/hugepage, %u hugepages   gpr: page %llu B, %u pages\n\n",
           (unsigned long long)c->geom.unit_bytes,
           c->geom.mem[PIM_MEM_DRAM].grans_per_hugepage, NR_DRAM,
           (unsigned long long)c->geom.mem[PIM_MEM_GPR].gran, NR_GPR);

    // ---- 1. each pool rounds to its OWN granule ---------------------------
    { void *w = pim_alloc_ctx(c, 1, PIM_MEM_DRAM);
      void *v = pim_alloc_ctx(c, 1, PIM_MEM_GPR);
      pim_mem where;
      CHECK(w && v, "1-byte allocations failed");
      CHECK(pim_usable_ctx(c,w) == c->geom.unit_bytes, "dram 1 B -> %zu, want a broadcast unit", pim_usable_ctx(c,w));
      CHECK(pim_usable_ctx(c,v) == 4096, "gpr 1 B -> %zu, want a 4 KiB page", pim_usable_ctx(c,v));
      CHECK(pim_where_ctx(c,w,&where) && where == PIM_MEM_DRAM, "dram alloc reports the wrong pool");
      CHECK(pim_where_ctx(c,v,&where) && where == PIM_MEM_GPR,  "gpr alloc reports the wrong pool");
      printf("1. dram 1 B -> %zu B, gpr 1 B -> %zu B; pools reported correctly\n",
             pim_usable_ctx(c,w), pim_usable_ctx(c,v));
      pim_free_ctx(c,w); pim_free_ctx(c,v); }

    // ---- 2. a GPR pointer is a WORD to the ISA ----------------------------
    { void *v = pim_alloc_ctx(c, 2048 * 2, PIM_MEM_GPR);      /* a k=2048 bf16 vector */
      uint32_t word = 0, nw = 0;
      const char *bad = pim_addr_gpr_words_ctx(c, v, 2048*2, &word, &nw);
      CHECK(!bad, "pim_addr_gpr_words: %s", bad?bad:"");
      CHECK(nw == 2048*2/32, "%u words for 4096 B, want %u", nw, 2048*2/32);
      printf("2. a 4096 B vector is GPR word %u..%u (%u words of %u B)\n",
             word, word + nw - 1, nw, PIM_GPR_WORD_BYTES);

      void *w = pim_alloc_ctx(c, 1<<20, PIM_MEM_DRAM);
      bad = pim_addr_gpr_words_ctx(c, w, 4096, &word, &nw);
      CHECK(bad != NULL, "a DRAM pointer was accepted as a GPR operand");
      printf("   a DRAM pointer is refused: \"%.60s...\"\n", bad ? bad : "");
      bad = pim_addr_unit_ctx(c, v, 4096, 0, &(pim_unit){0});
      CHECK(bad != NULL, "a GPR pointer was accepted as a broadcast operand");
      printf("   a GPR pointer is refused:  \"%.60s...\"\n", bad ? bad : "");
      pim_free_ctx(c,v); pim_free_ctx(c,w); }

    // ---- 3. both pools loaded at once, nothing aliases --------------------
    { enum { N = 80 };
      void *p[N]; size_t sz[N];
      for (int i = 0; i < N; i++) {
          pim_mem m = (i % 3 == 0) ? PIM_MEM_GPR : PIM_MEM_DRAM;
          sz[i] = (m == PIM_MEM_GPR) ? (size_t)((i%7)+1) * 3000
                                     : (size_t)((i*7 % 11) + 1) * 100000;
          p[i] = pim_alloc_ctx(c, sz[i], m);
          CHECK(p[i] != NULL, "alloc %d of %zu B in %s failed: %s", i, sz[i],
                m ? "gpr":"dram", pim_last_error_ctx(c));
          if (p[i]) sz[i] = pim_usable_ctx(c, p[i]);
      }
      check_no_alias(c, p, sz, N);
      uint64_t a0,p0,b0,a1,p1,b1;
      pim_meminfo_ctx(c, PIM_MEM_DRAM, &a0,&p0,&b0);
      pim_meminfo_ctx(c, PIM_MEM_GPR,  &a1,&p1,&b1);
      printf("3. %d mixed allocations, 0 aliases\n"
             "   dram %"PRIu64" MiB in %"PRIu64" hugepages (%u ioctls)   "
             "gpr %"PRIu64" KiB in %"PRIu64" pages (%u ioctls)\n",
             N, a0>>20, b0, ioctl_allocs[0], a1>>10, b1, ioctl_allocs[1]);

      for (int i = 0; i < N; i += 2) { pim_free_ctx(c, p[i]); p[i] = NULL; }
      void *big = pim_alloc_ctx(c, 8u<<20, PIM_MEM_DRAM);
      CHECK(big != NULL, "8 MiB after fragmenting failed: %s", pim_last_error_ctx(c));
      sz[0] = big ? pim_usable_ctx(c,big) : 0; p[0] = big;
      check_no_alias(c, p, sz, N);
      printf("4. after freeing every other, an 8 MiB dram alloc still succeeds\n");
      for (int i = 0; i < N; i++) if (p[i]) pim_free_ctx(c, p[i]); }

    // ---- 5. contiguity: one transfer each ---------------------------------
    { void *w = pim_alloc_ctx(c, 2u<<20, PIM_MEM_DRAM);
      void *v = pim_alloc_ctx(c, 64u<<10, PIM_MEM_GPR);
      pim_loc lw, lv; size_t rw = 0, rv = 0;
      CHECK(!pim_resolve_ctx(c, w, 2u<<20,  &lw, &rw), "resolve dram");
      CHECK(!pim_resolve_ctx(c, v, 64u<<10, &lv, &rv), "resolve gpr");
      printf("5. 2 MiB dram -> one run of %zu B at %#"PRIx64"\n"
             "   64 KiB gpr -> one run of %zu B at %#"PRIx64" (word %"PRIu64")\n",
             rw, lw.axi, rv, lv.axi, lv.off / PIM_GPR_WORD_BYTES);
      CHECK(rw == (2u<<20),  "a whole hugepage should be one extent, got %zu", rw);
      CHECK(rv == (64u<<10), "16 gpr pages should be one extent, got %zu", rv);
      pim_free_ctx(c,w); pim_free_ctx(c,v); }

    // ---- 6. everything comes back, in both pools --------------------------
    pim_pool_trim(c, PIM_MEM_DRAM, true);
    pim_pool_trim(c, PIM_MEM_GPR,  true);
    printf("6. after trim: dram %u hugepages out, gpr %u hugepages out\n", out[0], out[1]);
    CHECK(out[0] == 0, "%u dram hugepages leaked", out[0]);
    CHECK(out[1] == 0, "%u gpr hugepages leaked",  out[1]);

    // ---- 7. exhaustion names the pool it happened in -----------------------
    { void *v = pim_alloc_ctx(c, 8u<<20, PIM_MEM_GPR);   /* the GPR is only 4 MiB */
      CHECK(v == NULL, "an 8 MiB GPR allocation succeeded; the region is 4 MiB");
      printf("7. over-sized gpr request refused: \"%s\"\n", pim_last_error_ctx(c)); }

    printf("\n%s\n", fail ? "FAIL" : "all checks passed");
    return fail != 0;
}
