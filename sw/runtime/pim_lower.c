// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_lower.c — pass 2.  A logical program plus the addresses it turned out to have,
// into something the dispatcher can run.  See pim_logical.h for why this is a
// separate pass at all.
//
// FOUR STEPS, AND THE ORDER IS FORCED:
//
//   1. resolve    ask libpim where the allocation landed.  pim_addr_unit() for a
//                 DRAM unit, pim_addr_gpr_words() for a GPR word.
//   2. check      is what the kernel asked for expressible in one command?
//   3. legalise   if not, and the kernel said the rewrite is safe, split it.
//   4. relocate   write the resolved value into the ROW field.
//
// STEP 2 NEEDS STEP 1'S ANSWER.  "Is this operand contiguous", "does it cover every
// channel" are questions about where it landed, not about what was asked for — so
// the usual compiler order (legalise, then relocate) does not apply here.
//
// WHAT THIS FILE MAY NOT DO.  It may not reorder ISRs, and it may not decide that an
// accumulation can be cut — that is what the atoms are for, and they are carried
// through rather than interpreted.  It does not know what a GEMV is and must not
// learn: a rewrite that is only correct for one kernel belongs in that kernel, or in
// a pim_split value that says so.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim/pim_logical.h"
#include "pim/pim_addr.h"

#include <stdio.h>
#include <string.h>

#include "emu_regs.h"

static char lo_err[320];

// ------------------------------------------------------------- building -----
void pim_logical_init(pim_logical *lp, pim_isr *isr, uint32_t isr_cap,
                      pim_ref *ref, uint32_t ref_cap,
                      pim_atom *atom, uint32_t atom_cap,
                      uint32_t nch, uint32_t nlatch)
{
    if (!lp) return;
    lp->isr = isr;   lp->nisr = 0;   lp->isr_cap = isr_cap;
    lp->ref = ref;   lp->nref = 0;   lp->ref_cap = ref_cap;
    lp->atom = atom; lp->natom = 0;  lp->atom_cap = atom_cap;
    lp->nch = nch;
    lp->nlatch = nlatch ? nlatch : 1u;
}

const char *pim_logical_push(pim_logical *lp, const pim_isr *isr)
{
    if (!lp || !isr) return "pim_logical_push: null argument";
    if (lp->nisr == lp->isr_cap)
        return "pim_logical_push: the ISR buffer is full";
    lp->isr[lp->nisr++] = *isr;
    return NULL;
}

const char *pim_logical_push_ref(pim_logical *lp, const pim_isr *isr,
                                 pim_ref_kind kind, pim_split split,
                                 const void *base, size_t len, uint32_t index,
                                 uint64_t tag)
{
    const char *bad;

    if (!lp || !base) return "pim_logical_push_ref: null argument";
    if (lp->nref == lp->ref_cap)
        return "pim_logical_push_ref: the reference buffer is full";
    if ((bad = pim_logical_push(lp, isr))) return bad;

    lp->ref[lp->nref++] = (pim_ref){
        .isr = lp->nisr - 1, .kind = kind, .split = split,
        .base = base, .len = len, .index = index, .tag = tag,
    };
    return NULL;
}

const char *pim_logical_atom(pim_logical *lp, uint32_t first, uint32_t last)
{
    if (!lp) return "pim_logical_atom: null argument";
    if (lp->natom == lp->atom_cap)
        return "pim_logical_atom: the atom buffer is full";
    if (first > last || last >= lp->nisr)
        return "pim_logical_atom: the range is empty or runs past the program";
    lp->atom[lp->natom++] = (pim_atom){ first, last };
    return NULL;
}

// ------------------------------------------------------------- sizing -------
// How many ISRs each logical one can become.  Only a per-channel split grows.
static uint32_t fanout(const pim_logical *lp, uint32_t isr)
{
    for (uint32_t i = 0; i < lp->nref; i++)
        if (lp->ref[i].isr == isr && lp->ref[i].split == PIM_SPLIT_PER_CHANNEL)
            return lp->nch;
    return 1;
}

uint32_t pim_lower_isr_count(const pim_logical *lp)
{
    uint32_t n = 0;
    if (!lp) return 0;
    for (uint32_t i = 0; i < lp->nisr; i++)
        n += fanout(lp, i);
    return n;
}

// Where logical ISR `i` starts once everything before it has been expanded.
static uint32_t mapped(const pim_logical *lp, uint32_t i)
{
    uint32_t n = 0;
    for (uint32_t k = 0; k < i; k++)
        n += fanout(lp, k);
    return n;
}

const char *pim_lower_atoms(const pim_logical *lp, pim_atom *out, uint32_t cap,
                            uint32_t *n)
{
    if (!lp) return "pim_lower_atoms: null argument";
    if (n) *n = lp->natom;
    if (!out) return NULL;
    if (lp->natom > cap) return "pim_lower_atoms: the buffer is too small";

    for (uint32_t i = 0; i < lp->natom; i++) {
        uint32_t last = lp->atom[i].last;
        out[i].first = mapped(lp, lp->atom[i].first);
        // The atom ends at the LAST ISR the final logical one expanded into.
        out[i].last  = mapped(lp, last) + fanout(lp, last) - 1;
    }
    return NULL;
}

// -------------------------------------------------- resolve + relocate ------
// OVERWRITE A FIELD.  emu_isr_set() only RAISES bits — it never clears — because it
// was written for building a word from zero, which is what emu_isr_build does.  A
// lowering pass writes over fields an encoder already filled in, so it has to clear
// first.  Setting CH_MASK to 1<<1 over a word built with 1<<0 otherwise yields 0x3:
// a multicast RD_MAC, which is the exact thing the one-hot split exists to avoid,
// and it would have been a wrong number rather than an error.
static void isr_put(pim_isr *isr, unsigned lo, unsigned width, uint64_t v)
{
    struct emu_isr *e = (struct emu_isr *)isr;

    for (unsigned i = 0; i < width; i++) {
        unsigned bit = lo + i;
        e->w[bit >> 6] &= ~(1ULL << (bit & 63));
    }
    emu_isr_set(e, lo, width, v);
}

// The one place a resolved value meets an ISR word.  ROW is 17 bits and both kinds
// land in it, which is exactly why pim_ref names the KIND and not the field.
static const char *put_row(pim_isr *isr, uint64_t row, const char *what)
{
    if (row > EMU_ISR_ROW_MAX) {
        snprintf(lo_err, sizeof lo_err,
                 "%s resolved to ROW %llu and the ISR field holds %u",
                 what, (unsigned long long)row, (unsigned)EMU_ISR_ROW_MAX);
        return lo_err;
    }
    isr_put(isr, ISR_F_ROW, (uint32_t)row);
    return NULL;
}

// One reference, already known to need no splitting.
static const char *resolve_one(pim_ctx *c, const pim_ref *r, uint32_t ch_bump,
                               uint64_t *row)
{
    const char *bad;

    if (r->kind == PIM_REF_DRAM_UNIT) {
        pim_unit un;
        if ((bad = pim_addr_unit_ctx(c, r->base, r->len, r->index, &un)))
            return bad;
        *row = un.row;
        return NULL;
    } else {
        uint32_t word, nwords;
        if ((bad = pim_addr_gpr_words_ctx(c, r->base, r->len, &word, &nwords)))
            return bad;
        if (r->index + ch_bump >= nwords) {
            snprintf(lo_err, sizeof lo_err,
                     "GPR word %u is past the %u word(s) of that allocation",
                     r->index + ch_bump, nwords);
            return lo_err;
        }
        *row = word + r->index + ch_bump;
        return NULL;
    }
}

const char *pim_prog_lower_ctx(pim_ctx *c, const pim_logical *lp, pim_prog *out,
                               unsigned flags)
{
    const pim_geometry *g;

    if (!c)  return "pim_prog_lower: no context — is the module loaded?";
    if (!lp || !out) return "pim_prog_lower: null argument";
    g = pim_geom_ctx(c);
    if (!g) return "pim_prog_lower: the context has no geometry";
    if (lp->nch != g->nch) {
        snprintf(lo_err, sizeof lo_err,
                 "the program was built for %u channel(s) and this board has %u",
                 lp->nch, g->nch);
        return lo_err;
    }
    // THE ONE CAPABILITY THIS PASS CHECKS.  Everything else it verifies is about
    // where data landed; this is about what the bitstream decodes.  A program that
    // uses the second latch on an image that ties latch_sel does not fail — both
    // groups pile into latch 0 and both RD_MACs read the same value, so the answer
    // is wrong and nothing says so.  See pim_exec_config.allow_t_latch.
    if (lp->nlatch > 1 && !(flags & PIM_LOWER_ALLOW_T)) {
        snprintf(lo_err, sizeof lo_err,
                 "this program schedules %u accumulator latches; pass "
                 "PIM_LOWER_ALLOW_T only if you have evidence that this image "
                 "decodes ISR[35] (a tied latch_sel is a wrong number, not an error)",
                 lp->nlatch);
        return lo_err;
    }
    if (pim_lower_isr_count(lp) > out->cap) {
        snprintf(lo_err, sizeof lo_err,
                 "lowering needs up to %u ISRs and the buffer holds %u",
                 pim_lower_isr_count(lp), out->cap);
        return lo_err;
    }

    for (uint32_t i = 0; i < lp->nisr; i++) {
        const pim_ref *r = NULL;
        const char *bad;

        for (uint32_t k = 0; k < lp->nref; k++)
            if (lp->ref[k].isr == i) { r = &lp->ref[k]; break; }

        // No blank to fill: the kernel already knew the field.
        if (!r) {
            if ((bad = pim_prog_push(out, &lp->isr[i]))) return bad;
            continue;
        }

        // THE LAYOUT CHECK.  Cheap, and it is the only thing standing between a
        // program and an allocation that was filled some other way.
        if (r->tag) {
            uint64_t got = pim_tag_get_ctx(c, r->base);
            if (got != r->tag) {
                snprintf(lo_err, sizeof lo_err,
                         "ISR %u reads %p assuming layout %#llx, but that allocation "
                         "is %s.  A matrix written some other way is a wrong number, "
                         "not an error — see pim_tag_set().", i, r->base,
                         (unsigned long long)r->tag,
                         got ? "laid out differently" : "unstamped (a raw pim_memcpy?)");
                return lo_err;
            }
        }

        if (r->split == PIM_SPLIT_PER_CHANNEL) {
            // LEGALISE.  One logical "drain this" becomes one ISR per channel with
            // a one-hot CH_MASK, each landing on its own GPR word.  emu_isr_check
            // refuses a multicast RD_MAC for exactly this reason: two channels
            // would write one word and one result would vanish silently.
            for (uint32_t ch = 0; ch < g->nch; ch++) {
                pim_isr  isr = lp->isr[i];
                uint64_t row;

                if ((bad = resolve_one(c, r, ch, &row))) return bad;
                isr_put(&isr, ISR_F_CHMASK, 1u << ch);
                if ((bad = put_row(&isr, row, "a per-channel reference"))) return bad;
                if ((bad = pim_prog_push(out, &isr))) return bad;
            }
        } else {
            pim_isr  isr = lp->isr[i];
            uint64_t row;

            if ((bad = resolve_one(c, r, 0, &row))) return bad;
            if ((bad = put_row(&isr, row, "a reference"))) return bad;
            if ((bad = pim_prog_push(out, &isr))) return bad;
        }
    }
    return NULL;
}

const char *pim_prog_lower(const pim_logical *lp, pim_prog *out, unsigned flags)
{ return pim_prog_lower_ctx(pim_default(), lp, out, flags); }
