// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_prog.c — building a program, checking it, and running it.
//
// This is the only file in the runtime that produces a 256-bit ISR word or rings
// the doorbell.  Everything above it names rows and GPR words; it names opcodes.
//
// WHY A PROGRAM-LEVEL CHECKER EXISTS ON TOP OF emu_isr_check()
//   emu_isr_check() sees one ISR.  It cannot see the rules that only exist BETWEEN
//   ISRs, and those are the ones that fail silently:
//
//   1. A MAC's OPSIZE must equal the OPSIZE of the WRVEC that filled the GB.  The
//      per-bank skid below the GB has no flush port, so a mismatch leaves beats
//      behind and the NEXT MAC starts its vector at the wrong element.  Nothing
//      reports it; the numbers are just wrong.
//   2. Every MAC must be drained by a RD_MAC before the program ends.  The
//      accumulator is cleared only by RD_MAC and by reset, so an undrained latch is
//      inherited by whatever runs next — in this process or another one.
//   3. Two RD_MACs must not land on the same GPR word.
//   4. The last ISR must not produce a result.  STATUS[31] rises when the fetcher
//      ACCEPTS the last ISR, so a program ending in RD_MAC reports done while its
//      own result is still in flight.  Hence the trailing EOS.
//
//   The hardware checks none of these — the validity gate was pulled out of the
//   fetch path for timing on 2026-07-29 and has not come back.  This is the only
//   thing standing between a caller and a wrong number.
//
// WHY THE ONLY MAC SHAPE IS pu_mask = gb_mc_mask = 0xFFFF
//   The single-bank and 4-bank shapes exist, but RD_MAC broadcasts to all 16 banks
//   and clears them all, so with a partial mask the un-asked banks still have to be
//   drained on the same schedule.  For GEMV there is no reason to want that: every
//   bank holds a different output row and all of them are wanted.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim_internal.h"

#include <stdarg.h>
#include <stdlib.h>
#include <time.h>

#define POISON_LANE 0x7FC1u    // a quiet NaN payload nothing here computes

void pim_prog_init(struct pim_prog *p, pim_dev *d)
{
    memset(p, 0, sizeof *p);
    p->dev = d;
    p->w   = d->prog;
    p->cap = d->max_isrs;
}

static const char *fail(struct pim_prog *p, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static const char *fail(struct pim_prog *p, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(p->dev->err, PIM_ERRLEN, fmt, ap);
    va_end(ap);
    if (!p->bad) p->bad = p->dev->err;
    return p->dev->err;
}

static const char *push(struct pim_prog *p, const struct emu_isr_spec *s, const char *what)
{
    if (p->n >= p->cap)
        return fail(p, "program is full at %u ISRs while emitting %s.  The cap is a "
                       "runtime setting (pim_config.max_isrs), not the hardware's "
                       "16383 — a caller that hits it should split the work.",
                    p->cap, what);
    const char *e = emu_isr_build(&p->w[p->n], s);
    if (e) return fail(p, "ISR %u (%s) is illegal: %s", p->n, what, e);
    p->n++;
    return NULL;
}

const char *pim_emit_wrvec(struct pim_prog *p, uint32_t gpr_word, uint32_t opsize)
{
    if (opsize == 0 || opsize > EMU_MAX_OPSIZE)
        return fail(p, "WRVEC OPSIZE %u is outside 1..%u (the GB holds %u beats)",
                    opsize, EMU_MAX_OPSIZE, EMU_MAX_OPSIZE);
    if ((uint64_t)gpr_word + opsize > EMU_GPR_WORDS)
        return fail(p, "WRVEC would read GPR words [%u, %u), past the end", gpr_word,
                    gpr_word + opsize);
    struct emu_isr_spec s = emu_isr_default(ISR_OP_WRVEC);
    s.opsize = opsize;
    s.row    = gpr_word;                       // for WRVEC, ROW[22:6] is a GPR index
    const char *e = push(p, &s, "WRVEC");
    if (e) return e;
    p->gb_opsize = opsize;                     // the GB now holds exactly this many
    return NULL;
}

const char *pim_emit_mac(struct pim_prog *p, uint32_t row, uint32_t col, uint32_t opsize)
{
    if (p->gb_opsize == 0)
        return fail(p, "MAC at ISR %u sources from the GB, but no WRVEC has filled it "
                       "in this program.  It would multiply by whatever the previous "
                       "kernel left there.", p->n);
    if (opsize != p->gb_opsize)
        return fail(p, "MAC at ISR %u has OPSIZE %u but the live WRVEC loaded %u.  The "
                       "per-bank skid under the GB has no flush port, so the leftover "
                       "beats stay there and the NEXT MAC starts its vector at the "
                       "wrong element — silently.", p->n, opsize, p->gb_opsize);
    struct emu_isr_spec s = emu_isr_default(ISR_OP_MAC);
    s.opsize     = opsize;
    s.row        = row;
    s.col        = col;
    s.pu_mask    = 0xFFFF;
    s.gb_mc_mask = 0xFFFF;                     // nonzero = source from the GB
    const char *e = push(p, &s, "MAC");
    if (e) return e;
    p->undrained++;
    return NULL;
}

const char *pim_emit_rd_mac(struct pim_prog *p, uint32_t gpr_word)
{
    if (gpr_word >= EMU_GPR_WORDS)
        return fail(p, "RD_MAC destination word %u is past the GPR", gpr_word);
    // Results are required to be a contiguous ascending run.  That is not a
    // convenience: it is what lets a whole GEMV come back in ONE C2H transfer
    // instead of one per 16 outputs, and at 909 MB/s the difference is larger than
    // the compute.  It also makes a collision impossible by construction.
    if (!p->res_any) { p->res_lo = p->res_hi = gpr_word; p->res_any = true; }
    else if (gpr_word != p->res_hi + 1)
        return fail(p, "RD_MAC at ISR %u lands on GPR word %u, but the results so far "
                       "are [%u, %u].  A program's results must be one ascending run.",
                    p->n, gpr_word, p->res_lo, p->res_hi);
    else p->res_hi = gpr_word;

    struct emu_isr_spec s = emu_isr_default(ISR_OP_RD_MAC);
    s.opsize = 0;                              // always one aggregate beat
    s.row    = gpr_word;                       // ROW[22:6] is a GPR index here too
    const char *e = push(p, &s, "RD_MAC");
    if (e) return e;
    p->undrained = 0;                          // read-clear: every bank is empty now
    return NULL;
}

const char *pim_emit_eos(struct pim_prog *p)
{
    if (p->undrained)
        return fail(p, "%u MAC%s would be left undrained at EOS.  The accumulator is "
                       "cleared only by RD_MAC and by reset, so the leftover partial "
                       "sum is inherited by the next program — in this process or "
                       "another one — and nothing can detect it.",
                    p->undrained, p->undrained == 1 ? "" : "s");
    struct emu_isr_spec s = emu_isr_default(ISR_OP_EOS);
    return push(p, &s, "EOS");
}

const char *pim_prog_verify(const struct pim_prog *p)
{
    if (p->bad) return p->bad;
    if (p->n == 0) return pim_err(p->dev, "empty program");
    if (p->n > CFR_PROG_LEN_MAX)
        return pim_err(p->dev, "program is %u ISRs; PROG_LEN is 14 bits (max %u)",
                       p->n, CFR_PROG_LEN_MAX);
    uint32_t last = (uint32_t)emu_isr_get(&p->w[p->n - 1], ISR_F_OPCODE);
    if (last != ISR_OP_EOS)
        return pim_err(p->dev, "the last ISR is opcode 0x%02x, not EOS.  done rises "
                               "when the fetcher ACCEPTS the last ISR, so without a "
                               "trailing EOS control returns one ISR early and the "
                               "final result is still in flight.", last);
    if (p->undrained)
        return pim_err(p->dev, "%u MACs are undrained", p->undrained);
    return NULL;
}

// Decode a program back out of the words that will actually be written to IMEM.
// Printing the SPEC would prove nothing; printing what was ENCODED is what catches
// a field in the wrong place, which is the failure this hardware cannot report.
void pim_prog_print(const struct pim_prog *p, FILE *out, unsigned max_words)
{
    static const char *NM[] = {
        [ISR_OP_MAC]="MAC", [ISR_OP_EWMUL]="EWMUL", [ISR_OP_COPY]="COPY",
        [ISR_OP_WRVEC]="WRVEC", [ISR_OP_RD_MAC]="RD_MAC", [ISR_OP_EOS]="EOS",
    };
    fprintf(out, "  %5s  %-6s %-6s %-9s %-5s %-6s %-6s  %s\n",
            "IMEM", "opcode", "OPSIZE", "ROW/GPR", "COL", "pu", "gb_mc", "256-bit word (MSB left)");
    for (uint32_t i = 0; i < p->n; i++) {
        if (max_words && i >= max_words && i + 1 < p->n) {
            fprintf(out, "  ...    (%u more)\n", p->n - i);
            i = p->n - 1;
        }
        const struct emu_isr *w = &p->w[i];
        uint32_t op = (uint32_t)emu_isr_get(w, ISR_F_OPCODE);
        const char *nm = (op < sizeof NM / sizeof NM[0] && NM[op]) ? NM[op] : "?";
        fprintf(out, "  %5u  %-6s %-6llu %-9llu %-5llu 0x%04llx 0x%04llx  "
                     "%016llx%016llx%016llx%016llx\n",
                i, nm,
                (unsigned long long)emu_isr_get(w, ISR_F_OPSIZE),
                (unsigned long long)emu_isr_get(w, ISR_F_ROW),
                (unsigned long long)emu_isr_get(w, ISR_F_COL),
                (unsigned long long)emu_isr_get(w, ISR_F_PUMASK),
                (unsigned long long)emu_isr_get(w, ISR_F_GBMC),
                (unsigned long long)w->w[3], (unsigned long long)w->w[2],
                (unsigned long long)w->w[1], (unsigned long long)w->w[0]);
    }
}

// ---- launch ------------------------------------------------------------------
// The result is believed only when the POISON is gone.  STATUS[31] is a held level
// that rises when the fetcher accepts the last ISR, so it can be left over from the
// previous run and decides nothing.
const char *pim_launch(pim_dev *d, const struct pim_prog *p,
                       uint32_t res_word, uint32_t nres, uint16_t *lanes)
{
    const char *e = pim_prog_verify(p);
    if (e) return e;
    if (nres == 0 || !lanes) return pim_err(d, "launch with no result buffer");
    if (p->res_any && (res_word != p->res_lo || nres != p->res_hi - p->res_lo + 1))
        return pim_err(d, "launch asks for results [%u, %u) but the program writes "
                          "[%u, %u]", res_word, res_word + nres, p->res_lo, p->res_hi);

    if ((e = pim_require_idle(d, "load the device", 1000))) return e;
    if ((e = pim_axi_write(d, EMU_AXI_IMEM, p->w, (size_t)p->n * EMU_WORD_BYTES))) return e;

    for (uint32_t i = 0; i < nres * PIM_LANES; i++) lanes[i] = POISON_LANE;
    if ((e = pim_gpr_write(d, res_word, lanes, nres))) return e;

    if ((e = pim_require_idle(d, "ring the doorbell", 1000))) return e;
    pim_cfr_wr(d, CFR_PROG_LEN, p->n);
    if (pim_cfr_rd(d, CFR_PROG_LEN) != p->n)
        return pim_err(d, "PROG_LEN did not read back (wrote %u, read %u)", p->n,
                       pim_cfr_rd(d, CFR_PROG_LEN));
    (void)pim_cfr_rd(d, CFR_PROG_LEN);   // non-posted: orders the posted writes first

    uint64_t t0 = pim_now_us();
    pim_cfr_wr(d, CFR_CTRL, CFR_CTRL_DOORBELL);
    d->st.launches++;
    d->st.isrs += p->n;

    bool seen_done = false;
    while (pim_now_us() - t0 < 10000000ull) {
        d->st.polls++;
        uint32_t st = pim_cfr_rd(d, CFR_STATUS);
        if (st == 0xFFFFFFFFu)
            return pim_err(d, "the control plane went to all-ones while a kernel was "
                              "running (%u ISRs).  The device stopped answering.", p->n);
        if (st & CFR_STATUS_DONE) { seen_done = true; break; }
    }

    bool landed = false;
    for (unsigned tries = 0; tries < 64 && pim_now_us() - t0 < 10000000ull; tries++) {
        if ((e = pim_gpr_read(d, res_word, lanes, nres))) return e;
        landed = true;
        for (uint32_t i = 0; i < nres * PIM_LANES && landed; i++)
            if (lanes[i] == POISON_LANE) landed = false;
        if (landed) break;
        if (tries >= 4) { struct timespec ts = { 0, 1000000 }; nanosleep(&ts, NULL); }
    }
    d->st.device_us += pim_now_us() - t0;

    if (!landed)
        return pim_err(d, "the results never lost their poison (%u ISRs, %u result "
                          "words, done %s).  A GB fill that does not match its MACs, "
                          "an OPSIZE over 64, or a PROG_LEN past the program all hang "
                          "here with no diagnostic.  The accumulator is now in an "
                          "unknown state — call pim_scrub().",
                       p->n, nres, seen_done ? "rose" : "never rose");
    return NULL;
}

// ---- scrub -------------------------------------------------------------------
const char *pim_scrub(pim_dev *d)
{
    struct pim_prog p;
    pim_prog_init(&p, d);
    const char *e;
    // RD_MAC needs no MAC in front of it: a reset-state latch is readable and comes
    // back as 0x0000 [measured, report.md 1.3].  That is what makes this safe to
    // run on a device whose state is unknown.
    if ((e = pim_emit_rd_mac(&p, d->gpr_res_base))) return e;
    if ((e = pim_emit_eos(&p))) return e;
    uint16_t lanes[PIM_LANES];
    return pim_launch(d, &p, d->gpr_res_base, 1, lanes);
}
