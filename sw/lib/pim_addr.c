// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_rt.c — the address half of issuing an operation.
//
// This file is small, and that is the honest state of this layer rather than a
// placeholder.  Design §9 splits into two halves and only one of them is settled:
//
//   WHERE the operands are    settled, and implemented here.  For DRAM by §4.1: a
//                             broadcast unit divides a hugepage and hugepages are
//                             hugepage-aligned, so an operand is one contiguous card
//                             extent per unit.  For the GPR by §9.4: it is linear,
//                             so an operand is a word index and a count.
//   HOW the command gets out   still open — §9.1 plan A (kernel-proxied ioctl) vs
//                              plan B (command ring in card memory), and §11 says
//                              the FPGA's command interface decides it.
//
// So what exists here is the part that does not change either way: turning
// allocations into the coordinates a command would name.  When the submission
// decision lands, the encoder and the doorbell go beside this and this does not move.
//////////////////////////////////////////////////////////////////////////////////
#include "pim/pim_addr.h"

#include <stdio.h>

// Not reentrant.  These are diagnostics on a path that has already failed.
static char rt_err[256];

/*
 * IS `p` UNIT-ALIGNED?  It has to be: a broadcast operation reads the whole unit in
 * every bank, so an operand that starts mid-unit is not addressable by one command.
 *
 * The check uses only the public API.  pim_usable() returns bytes-to-end-of-
 * allocation, and an allocation's size is always a whole number of granules, so the
 * remainder of that value modulo the granule IS the offset of p within its granule.
 * No need to expose vbase for this.
 */
size_t pim_addr_unit_count_ctx(pim_ctx *c, const void *p, size_t len)
{
    const pim_geometry *g = pim_geom_ctx(c);
    pim_mem where;
    size_t avail;

    if (!g || !p || !len)
        return 0;
    if (!pim_where_ctx(c, p, &where) || where != PIM_MEM_DRAM)
        return 0;
    avail = pim_usable_ctx(c, p);
    if (!avail || len > avail)
        return 0;
    if (avail % g->unit_bytes)          /* p is not at a unit boundary */
        return 0;

    return (len + g->unit_bytes - 1) / g->unit_bytes;
}

const char *pim_addr_unit_ctx(pim_ctx *c, const void *p, size_t len,
                        size_t i, pim_unit *out)
{
    const pim_geometry *g = pim_geom_ctx(c);
    size_t      n = pim_addr_unit_count_ctx(c, p, len);
    pim_loc     loc;
    size_t      run = 0;
    const char *bad;
    pim_coord   co;
    pim_mem     where;

    if (!out)
        return "pim_addr_unit: out is NULL";
    if (!n) {
        if (pim_where_ctx(c, p, &where) && where != PIM_MEM_DRAM)
            return "pim_addr_unit: that is a GPR pointer.  The GPR is linear and has "
                   "no (row, channel, bank) — use pim_addr_gpr_words().";
        snprintf(rt_err, sizeof rt_err,
                 "pim_addr_unit: %p+%zu is not a unit-aligned range inside one "
                 "allocation", p, len);
        return rt_err;
    }
    if (i >= n) {
        snprintf(rt_err, sizeof rt_err, "pim_addr_unit: unit %zu of %zu", i, n);
        return rt_err;
    }

    bad = pim_resolve_ctx(c, (const char *)p + i * g->unit_bytes,
                      (size_t)g->unit_bytes, &loc, &run);
    if (bad)
        return bad;

    // The §4.1 invariant says this cannot happen: a unit divides a hugepage and hugepages
    // are hugepage-aligned, so a unit is always one contiguous run.  Checked anyway,
    // because if it ever stops holding, every command built from here is silently
    // addressing the wrong half of an operand.
    if (run != g->unit_bytes) {
        snprintf(rt_err, sizeof rt_err,
                 "unit %zu is contiguous for only %zu of %llu B — the hugepage/unit "
                 "invariant (design 4.1) is broken", i, run,
                 (unsigned long long)g->unit_bytes);
        return rt_err;
    }

    pim_geom_decode(g, loc.axi, &co);
    out->card_addr  = loc.axi;
    out->row        = co.row;
    out->unit_index = (uint32_t)i;
    return NULL;
}

// ---------------------------------------------------------- GPR operands ----
const char *pim_addr_gpr_words_ctx(pim_ctx *c, const void *p, size_t len,
                             uint32_t *word, uint32_t *nwords)
{
    const pim_geometry *g = pim_geom_ctx(c);
    pim_loc     loc;
    size_t      run = 0;
    const char *bad;
    pim_mem     where;

    if (!g || !p || !len || !word || !nwords)
        return "pim_addr_gpr_words: null or empty argument";
    if (!pim_where_ctx(c, p, &where))
        return "pim_addr_gpr_words: not a live allocation";
    if (where != PIM_MEM_GPR)
        return "pim_addr_gpr_words: that is a DRAM pointer.  WRVEC and RD_MAC address "
               "GPR words; allocate the operand with PIM_MEM_GPR.";

    bad = pim_resolve_ctx(c, p, len, &loc, &run);
    if (bad)
        return bad;
    if (run < len) {
        snprintf(rt_err, sizeof rt_err,
                 "the GPR range is contiguous for only %zu of %zu B — one WRVEC "
                 "cannot span two runs.  Allocate it in one piece.", run, len);
        return rt_err;
    }
    if (loc.off % PIM_GPR_WORD_BYTES) {
        snprintf(rt_err, sizeof rt_err,
                 "GPR offset %llu is not a multiple of the %u B word",
                 (unsigned long long)loc.off, PIM_GPR_WORD_BYTES);
        return rt_err;
    }

    *word   = (uint32_t)(loc.off / PIM_GPR_WORD_BYTES);
    // Round up: a partial trailing word is still a whole word to the hardware, and
    // the caller's allocation covers it because requests round to a 4 KiB page.
    *nwords = (uint32_t)((len + PIM_GPR_WORD_BYTES - 1) / PIM_GPR_WORD_BYTES);
    return NULL;
}

// ------------------------------------------------- the default context ----
// Same three answers, for a caller that has not been handed a context.  The code
// generator's second pass runs on these: it asks "where did this allocation land"
// without needing to know that a context exists.
size_t pim_addr_unit_count(const void *p, size_t len)
{ return pim_addr_unit_count_ctx(pim_default(), p, len); }

const char *pim_addr_unit(const void *p, size_t len, size_t i, pim_unit *out)
{ return pim_addr_unit_ctx(pim_default(), p, len, i, out); }

const char *pim_addr_gpr_words(const void *p, size_t len,
                               uint32_t *word, uint32_t *nwords)
{ return pim_addr_gpr_words_ctx(pim_default(), p, len, word, nwords); }
