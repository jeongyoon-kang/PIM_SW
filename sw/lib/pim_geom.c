// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_geom.c — validating and deriving from what the driver reported.
//
// Every check here is a precondition of code somewhere else, and the message says
// which.  A geometry that fails one of these does not produce a wrong answer later;
// it fails at open, once, with a sentence.
//
// THE TWO REGIONS ARE NOT CHECKED THE SAME WAY, and the asymmetry is the design:
// DRAM carries an interleaved address map and must satisfy §4.1, while the GPR is
// linear and only has to be a whole number of 32 B words.  Applying the DRAM rules
// to the GPR would refuse a perfectly good 4 KiB page for failing an invariant that
// does not exist there.
//////////////////////////////////////////////////////////////////////////////////
#include "pim/pim_geometry.h"
#include "uapi/pim_ioctl.h"

#include <stdio.h>

// The public names and the ABI's numbers are two vocabularies for one thing.
_Static_assert((int)PIM_MEM_DRAM == (int)PIM_REGION_DRAM, "region numbering drifted");
_Static_assert((int)PIM_MEM_GPR  == (int)PIM_REGION_GPR,  "region numbering drifted");
_Static_assert(PIM_NMEM == PIM_NREGION, "region count drifted");

static bool pow2(uint64_t v) { return v && !(v & (v - 1)); }

static uint32_t log2u(uint64_t v)
{
    uint32_t s = 0;
    while ((1ull << s) < v) s++;
    return s;
}

// Runs BEFORE pim_geom_check(), on numbers a module handed us — so it must survive
// a zero rather than dividing by one.  The check is what rejects them; this only has
// to get there.
void pim_geom_derive(pim_geometry *g)
{
    // DRAM's granule is the broadcast unit: what one all-bank, all-channel operation
    // consumes, and therefore the smallest thing that can be an operand.
    g->mem[PIM_MEM_DRAM].gran = g->unit_bytes;
    // The GPR's granule is the driver's page.  Linear memory imposes no operand
    // shape, so there is no second level to cut (design §9.4) and v1 does not guess
    // at a size-class layer before any size distribution has been measured.
    g->mem[PIM_MEM_GPR].gran  = g->mem[PIM_MEM_GPR].hugepage_bytes;

    for (int m = 0; m < PIM_NMEM; m++) {
        pim_region *r = &g->mem[m];
        r->gran_shift      = r->gran ? log2u(r->gran) : 0;
        r->grans_per_hugepage = r->gran ? (uint32_t)(r->hugepage_bytes / r->gran) : 0;
    }
}

// Not reentrant.  Called once, at open, on a path that has already serialised.
static char buf[240];

static const char *check_region(const pim_region *r, const char *name)
{
    if (!pow2(r->hugepage_bytes) || !pow2(r->gran)) {
        snprintf(buf, sizeof buf,
                 "%s: hugepage %llu B and granule %llu B must both be powers of two — "
                 "the pointer-to-address translation is a shift",
                 name, (unsigned long long)r->hugepage_bytes,
                 (unsigned long long)r->gran);
        return buf;
    }
    if (r->hugepage_bytes % r->gran) {
        snprintf(buf, sizeof buf,
                 "%s: granule %llu B does not divide hugepage %llu B",
                 name, (unsigned long long)r->gran,
                 (unsigned long long)r->hugepage_bytes);
        return buf;
    }
    if (r->hugepage_bytes / r->gran > 64) {
        snprintf(buf, sizeof buf,
                 "%s: %llu granules per hugepage; libpim's per-hugepage bitmap is a "
                 "uint64_t and holds 64.  Lower the hugepage size, or give this a real "
                 "bitmap.", name,
                 (unsigned long long)(r->hugepage_bytes / r->gran));
        return buf;
    }
    if (r->base % r->hugepage_bytes) {
        snprintf(buf, sizeof buf,
                 "%s: base %#llx is not hugepage-aligned — hugepage index arithmetic would "
                 "be off by a partial hugepage", name, (unsigned long long)r->base);
        return buf;
    }
    if (!r->bytes || r->bytes % r->hugepage_bytes) {
        snprintf(buf, sizeof buf,
                 "%s: the region is empty or not a whole number of hugepages", name);
        return buf;
    }
    if (r->nr_hugepages != r->bytes / r->hugepage_bytes) {
        snprintf(buf, sizeof buf,
                 "%s: nr_hugepages disagrees with bytes / hugepage_bytes", name);
        return buf;
    }
    return NULL;
}

const char *pim_geom_check(const pim_geometry *g)
{
    const pim_region *dram = &g->mem[PIM_MEM_DRAM];
    const pim_region *gpr  = &g->mem[PIM_MEM_GPR];
    const char *bad;

    // ---- topology, which only the DRAM side has --------------------------
    if (!pow2(g->nch) || g->nch > 8)
        return "channel count is not a power of two in 1..8 — RoChBaCo puts the "
               "channel in a bit field, so nothing else is a geometry this "
               "hardware can have";
    if (!pow2(g->nbank) || !pow2(g->row_bytes) || g->row_bytes < 32)
        return "banks-per-channel and row size must be powers of two, row >= 32";

    if (g->unit_bytes != (uint64_t)g->nch * g->nbank * g->row_bytes) {
        snprintf(buf, sizeof buf,
                 "driver reported unit %llu B but nch*nbank*row is %llu B — the two "
                 "sides computed the broadcast unit differently",
                 (unsigned long long)g->unit_bytes,
                 (unsigned long long)g->nch * g->nbank * g->row_bytes);
        return buf;
    }

    // ---- both regions, same rules ---------------------------------------
    if ((bad = check_region(dram, "dram"))) return bad;
    if ((bad = check_region(gpr,  "gpr")))  return bad;

    // ---- what is specific to each ---------------------------------------
    // DRAM: THE INVARIANT (design §4.1).  Already enforced by check_region's
    // "granule divides hugepage", because DRAM's granule IS the broadcast unit — this
    // only restates it where a reader will look for it.
    if (dram->hugepage_bytes % g->unit_bytes) {
        snprintf(buf, sizeof buf,
                 "broadcast unit %llu B does not divide hugepage %llu B — an all-bank "
                 "operation could straddle two hugepages.  Raise pim_hugepage_size on the "
                 "driver.",
                 (unsigned long long)g->unit_bytes,
                 (unsigned long long)dram->hugepage_bytes);
        return buf;
    }
    if (g->map == PIM_MAP_CH_RO_BA_CO && dram->bytes % g->nch)
        return "ChRoBaCo needs the DRAM region to divide evenly by the channel count";

    // GPR: linear, so the only shape constraint is the ISA's word.  WRVEC and
    // RD_MAC address 32 B words, and a page that is not a whole number of them
    // would put an allocation's first word off the grid.
    if (gpr->gran % PIM_GPR_WORD_BYTES) {
        snprintf(buf, sizeof buf,
                 "gpr: granule %llu B is not a whole number of %u B words — WRVEC "
                 "and RD_MAC address words, not bytes",
                 (unsigned long long)gpr->gran, PIM_GPR_WORD_BYTES);
        return buf;
    }

    return NULL;
}
