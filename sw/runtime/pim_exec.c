// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_exec.c — the engine.  See pim_exec.h for the one design decision made here.
//
// FOUR WINDOWS, TWO TRANSPORTS, AND THE SPLIT IS NOT A PREFERENCE.
//   IMEM  256 b slave  -> QDMA.  A program is bulk and MMIO stores to a 256 b slave
//                         were observed splitting into <=16 B TLPs [2026-07-31].
//   GPR   256 b slave  -> QDMA, through libpim, because it is allocated memory.
//   CFR   AXI-Lite     -> MMIO.  32 bit registers, one TLP each, and a doorbell must
//                         not wait behind a DMA queue.
//   VIOL  AXI-Lite     -> MMIO, same reason.
//
// IDLE IS CHECKED BEFORE ANYTHING THE DISPATCHER OWNS, not just before the doorbell.
// The IMEM command port is shared between the host write path and the fetch read
// path with FETCH PRIORITY, so a host IMEM write while a kernel runs has WREADY held
// low; the DMA then stalls for the driver's 10 s timeout and returns EIO, and one
// EIO latches the H2C engine until the queue is torn down.  The GPR arbitrates the
// same way.  So: idle, then load, then idle again, then ring.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pimrt/pim_exec.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "emu_regs.h"      // the ISR encoding and the control plane.  Topology-free:
                           // GPR, CFR and IMEM are one each however many channels
                           // the image has, so including it here does NOT bring a
                           // compile-time channel count with it.  pim_platform.h
                           // would, and is deliberately not included.

#define BDF_DEFAULT   "0000:01:00.0"
#define TIMEOUT_MS    2000u

// The slice of BAR2 this maps: from the CFR page up through the MODE block, which
// covers the violation CSRs of up to four channels in between.  Not the whole 8 MiB
// — the GPR and IMEM live in the rest of it and they are DMA's, not MMIO's.
#define CTRL_OFF      EMU_OFF_CFR                 // 0x400000, page aligned
#define CTRL_LEN      0x8000u                     // CFR + VIOL[0..3] + MODE

struct pim_exec {
    pim_ctx           *ctx;
    int                bar_fd;
    volatile uint8_t  *ctrl;      // maps AXI [EMU_AXI_BASE+CTRL_OFF, +CTRL_LEN)
    uint32_t           timeout_ms;
    uint32_t           max_isrs;
    unsigned           verify_flags;
    pim_timing         timing_at_open;
    char               err[256];
};

static const char *ex_err(pim_exec *e, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));
static const char *ex_err(pim_exec *e, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->err, sizeof e->err, fmt, ap);
    va_end(ap);
    return e->err;
}

const char *pim_exec_error(const pim_exec *e)
{ return (e && e->err[0]) ? e->err : NULL; }

uint32_t pim_exec_max_isrs(const pim_exec *e)
{ return e ? e->max_isrs : CFR_PROG_LEN_MAX; }

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

// THE ONLY TWO FUNCTIONS THAT TOUCH A REGISTER.  Design §9.1 plan A replaces exactly
// these (and the mapping in pim_exec_open) with a PIM_IOC_LAUNCH; nothing else in
// this file knows the transport.
static uint32_t cfr_rd(pim_exec *e, uint32_t off)
{ return *(volatile uint32_t *)(e->ctrl + (EMU_OFF_CFR - CTRL_OFF) + off); }

static void cfr_wr(pim_exec *e, uint32_t off, uint32_t v)
{ *(volatile uint32_t *)(e->ctrl + (EMU_OFF_CFR - CTRL_OFF) + off) = v; }

// THE STATISTICS ARE CFR REGISTERS NOW.  v2.0 returned the per-channel viol_csr
// window; the channel is an address field inside the dispatcher's own 4 KiB, so
// these are cfr_rd/cfr_wr at a higher offset and nothing else.
//
// EMU_OFF_VIOL still names the old window in platform/ch2.conf, and that window
// still DECODES -- it reads zero.  So a reader left pointing there reports a clean
// run for every program, which is why nothing below uses it.

// Idle means done=1 (a program finished and the FSM is parked) or state=0 (fresh
// from reset).  Anything else is a kernel in progress.
static const char *require_idle(pim_exec *e, const char *what)
{
    uint64_t t = now_us();

    for (;;) {
        uint32_t st = cfr_rd(e, CFR_STATUS);

        if (st == 0xFFFFFFFFu)
            return ex_err(e, "cannot %s: the control plane reads all-ones.  Either "
                             "the BAR mapping is stale or the device stopped "
                             "answering (a JTAG reprogram does this).", what);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0)
            return NULL;
        if (now_us() - t > (uint64_t)e->timeout_ms * 1000ull)
            return ex_err(e, "cannot %s: the dispatcher is still busy after %u ms "
                             "(STATUS=%#08x, state=%u).  A previous kernel has not "
                             "finished.", what, e->timeout_ms, st,
                          CFR_STATUS_STATE(st));
    }
}

// The struct's field order, the register order, and this array are one list.  They
// are contiguous 4-byte registers from CFR_T_FAW, so a loop over the struct's bytes
// and a loop over the offsets stay in step by construction.
const char *const pim_timing_names[PIM_TIMING_NREG] =
    { "faw", "rrd", "rcd", "ccd", "rtp", "rp", "wr", "ras", "mod", "rpab" };

// NOT EIGHT IN A ROW ANY MORE.  v2.0 put T_MOD and T_RP_AB above PROG_LEN and
// RUN_CYC, so the offsets need a table; the index into it is the struct's field
// order, which is what keeps the loop below a loop.
static const uint32_t pim_timing_off[PIM_TIMING_NREG] = {
    CFR_T_FAW, CFR_T_RRD, CFR_T_RCD, CFR_T_CCD, CFR_T_RTP,
    CFR_T_RP,  CFR_T_WR,  CFR_T_RAS, CFR_T_MOD, CFR_T_RP_AB,
};

static uint16_t *timing_field(pim_timing *t, unsigned i)
{ return &((uint16_t *)t)[i]; }       // ten uint16_t, in declaration order

_Static_assert(sizeof(pim_timing) == 2 * PIM_TIMING_NREG,
               "pim_timing must stay ten packed uint16_t");

const char *pim_exec_get_timing(pim_exec *e, pim_timing *out)
{
    if (!e || !out) return "pim_exec_get_timing: null argument";
    for (unsigned i = 0; i < PIM_TIMING_NREG; i++) {
        uint32_t v = cfr_rd(e, pim_timing_off[i]);
        if (v == 0xFFFFFFFFu)
            return ex_err(e, "the CFR reads all-ones: the device is not answering");
        *timing_field(out, i) = (uint16_t)(v & CFR_TIMING_MAX);
    }
    return NULL;
}

const pim_timing *pim_exec_timing_at_open(const pim_exec *e)
{ return e ? &e->timing_at_open : NULL; }


// ------------------------------------------------------------------ open ----
const char *pim_exec_open(pim_ctx *c, const pim_exec_config *cfg, pim_exec **out)
{
    static char boot_err[512];
    const pim_geometry *g;
    pim_exec_config d;
    pim_exec *e;
    char path[256];
    const char *bad;
    uint32_t mode;
    void *m;

    if (!c || !out) { snprintf(boot_err, sizeof boot_err,
                               "pim_exec_open: null argument"); return boot_err; }
    *out = NULL;
    g = pim_geom_ctx(c);

    memset(&d, 0, sizeof d);
    if (cfg) d = *cfg;
    if (!d.bdf)        d.bdf        = BDF_DEFAULT;
    if (!d.timeout_ms) d.timeout_ms = TIMEOUT_MS;

    e = calloc(1, sizeof *e);
    if (!e) { snprintf(boot_err, sizeof boot_err, "out of memory"); return boot_err; }
    e->ctx = c;
    e->bar_fd = -1;
    e->timeout_ms = d.timeout_ms;
    e->max_isrs   = d.max_isrs ? d.max_isrs : CFR_PROG_LEN_MAX;
    if (e->max_isrs > CFR_PROG_LEN_MAX) e->max_isrs = CFR_PROG_LEN_MAX;
    e->verify_flags = d.allow_t_latch ? PIM_VERIFY_ALLOW_T : 0u;

    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource2", d.bdf);
    e->bar_fd = open(path, O_RDWR | O_SYNC);
    if (e->bar_fd < 0) {
        snprintf(boot_err, sizeof boot_err, "open(%s): %s — is the card enumerated?",
                 path, strerror(errno));
        free(e); return boot_err;
    }
    m = mmap(NULL, CTRL_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, e->bar_fd,
             (off_t)CTRL_OFF);
    if (m == MAP_FAILED) {
        snprintf(boot_err, sizeof boot_err, "mmap BAR2 +%#x: %s", CTRL_OFF,
                 strerror(errno));
        close(e->bar_fd); free(e); return boot_err;
    }
    e->ctrl = m;

    // THE CHECK THE MEMORY LAYER CANNOT MAKE.  libpim was told an address map by the
    // module's parameters; this is the board saying what it actually decodes.  A
    // mismatch puts every weight somewhere other than where the ROW field will look
    // for it, and nothing anywhere reports an error.
    mode = *(volatile uint32_t *)(e->ctrl + (EMU_OFF_MODE - CTRL_OFF) + MODE_CTRL);
    if (mode == 0xFFFFFFFFu) {
        snprintf(boot_err, sizeof boot_err,
                 "the MODE register reads all-ones: the device is not answering");
        pim_exec_close(e); return boot_err;
    }
    {
        unsigned board = (mode & MODE_CTRL_INTERLEAVE) ? 1u : 0u;
        unsigned built = (g->map == PIM_MAP_RO_CH_BA_CO) ? 1u : 0u;
        if (board != built) {
            snprintf(boot_err, sizeof boot_err,
                     "the board decodes %s but libpim was told %s.  Set MODE_CTRL "
                     "bit 0 to %u, or reload pim.ko with pim_addr_map=%u.",
                     board ? "RoChBaCo" : "ChRoBaCo",
                     built ? "RoChBaCo" : "ChRoBaCo", built, board);
            pim_exec_close(e); return boot_err;
        }
    }

    if ((bad = require_idle(e, "open the engine"))) {
        snprintf(boot_err, sizeof boot_err, "%s", bad);
        pim_exec_close(e); return boot_err;
    }

    // READ THE TIMING, DO NOT SET IT.  emu_timing owns these eight registers; an
    // engine that wrote its own defaults here would revert an experiment between the
    // moment it was set and the moment it was measured.
    //
    // T_CCD IS THE ONE THIS CANNOT MERELY OBSERVE.  Below 2, two beats into one
    // accumulator latch land only every OTHER beat: the answer is halved and no
    // counter, status bit or error reports it.  Every other register is a threshold
    // the emulator compares against and its counters are data; this one decides what
    // the arithmetic is.  So it is checked at open, where the run can still be
    // stopped, rather than trusted to whoever last set it.
    if ((bad = pim_exec_get_timing(e, &e->timing_at_open))) {
        snprintf(boot_err, sizeof boot_err, "%s", bad);
        pim_exec_close(e); return boot_err;
    }
    if (e->timing_at_open.ccd < 2) {
        snprintf(boot_err, sizeof boot_err,
                 "the board's T_CCD is %u.  Below 2, two beats into one accumulator "
                 "latch land only every OTHER beat — every MAC result is halved and "
                 "nothing reports it.  Set it with hwdef/test/emu_timing (--ccd 2, "
                 "or --reset for the whole SIM set) before opening an engine.",
                 e->timing_at_open.ccd);
        pim_exec_close(e); return boot_err;
    }

    // IMEM becomes the one raw window libpim will carry bytes to.  Everything else
    // it touches has to come from pim_alloc.
    if ((bad = pim_axi_allow_ctx(c, EMU_AXI_IMEM,
                             (uint64_t)EMU_IMEM_WORDS * EMU_WORD_BYTES))) {
        snprintf(boot_err, sizeof boot_err, "%s", bad);
        pim_exec_close(e); return boot_err;
    }

    *out = e;
    return NULL;
}

void pim_exec_close(pim_exec *e)
{
    if (!e) return;
    if (e->ctx) pim_axi_allow_ctx(e->ctx, 0, 0);       // shut the window again
    if (e->ctrl) munmap((void *)e->ctrl, CTRL_LEN);
    if (e->bar_fd >= 0) close(e->bar_fd);
    free(e);
}

// --------------------------------------------------------------- program ----
void pim_prog_init(pim_prog *p, pim_isr *storage, uint32_t cap)
{
    p->w = storage; p->n = 0; p->cap = cap;
}

const char *pim_prog_push(pim_prog *p, const pim_isr *isr)
{
    if (p->n >= p->cap)   return "the program is longer than its buffer";
    if (p->n >= EMU_IMEM_WORDS) return "the program is longer than IMEM";
    p->w[p->n++] = *isr;
    return NULL;
}

// The public opcode names must be the ISA's.  A drift here would build a program
// the verifier reads differently from the fetcher.
_Static_assert(PIM_OP_MAC    == ISR_OP_MAC,    "PIM_OP_MAC drifted from the ISA");
_Static_assert(PIM_OP_EWMUL  == ISR_OP_EWMUL,  "PIM_OP_EWMUL drifted from the ISA");
_Static_assert(PIM_OP_COPY   == ISR_OP_COPY,   "PIM_OP_COPY drifted from the ISA");
_Static_assert(PIM_OP_WRVEC  == ISR_OP_WRVEC,  "PIM_OP_WRVEC drifted from the ISA");
_Static_assert(PIM_OP_RD_MAC == ISR_OP_RD_MAC, "PIM_OP_RD_MAC drifted from the ISA");
_Static_assert(PIM_OP_EOS    == ISR_OP_EOS,    "PIM_OP_EOS drifted from the ISA");
_Static_assert(sizeof(pim_isr) == sizeof(struct emu_isr),
               "pim_isr and struct emu_isr must stay the same 32 B cell");

static uint32_t f(const pim_isr *w, unsigned lo, unsigned width)
{ return (uint32_t)emu_isr_get((const struct emu_isr *)w, lo, width); }

void pim_isr_decode(const pim_isr *w, pim_isr_info *out)
{
    out->opcode     = f(w, ISR_F_OPCODE);
    out->opsize     = f(w, ISR_F_OPSIZE);
    out->row        = f(w, ISR_F_ROW);
    out->col        = f(w, ISR_F_COL);
    out->ch_mask    = f(w, ISR_F_CHMASK);
    out->pu_mask    = f(w, ISR_F_PUMASK);
    out->gb_mc_mask = f(w, ISR_F_GBMC);
    out->t          = f(w, ISR_F_T);
}

const char *pim_isr_opname(uint32_t opcode)
{
    switch (opcode) {
    case ISR_OP_MAC:    return "MAC";
    case ISR_OP_EWMUL:  return "EWMUL";
    case ISR_OP_COPY:   return "COPY";
    case ISR_OP_WRVEC:  return "WRVEC";
    case ISR_OP_RD_MAC: return "RD_MAC";
    case ISR_OP_EOS:    return "EOS";
    default:            return "?";
    }
}

void pim_prog_dump(const pim_prog *p, FILE *out)
{
    if (!p || !out) return;
    fprintf(out, "  %-5s %-7s %-7s %-8s %-4s %-5s %-6s %-6s %-2s %s\n",
            "#", "op", "OPSIZE", "ROW", "COL", "CH", "PU", "GBMC", "T",
            "word (bit 255 first)");
    for (uint32_t i = 0; i < p->n; i++) {
        pim_isr_info d;
        pim_isr_decode(&p->w[i], &d);
        fprintf(out, "  %-5u %-7s %-7u %-8u %-4u %#-5x %#-6x %#-6x %-2u "
                "%016llx%016llx%016llx%016llx\n",
                i, pim_isr_opname(d.opcode), d.opsize, d.row, d.col,
                d.ch_mask, d.pu_mask, d.gb_mc_mask, d.t,
                (unsigned long long)p->w[i].w[3], (unsigned long long)p->w[i].w[2],
                (unsigned long long)p->w[i].w[1], (unsigned long long)p->w[i].w[0]);
    }
}

const char *pim_prog_verify(const pim_prog *p, unsigned flags)
{
    static char buf[260];
    uint32_t gb_len = 0;             // beats the last WRVEC wrote; the read position wraps here
    uint32_t gb_pos = 0;             // the Global Buffer's read position, in beats
    bool     gb_filled = false;
    uint32_t undrained[2] = { 0, 0 };  // MACs into each latch since its RD_MAC
    uint32_t seen[64];
    unsigned nseen = 0;

    if (!p->n) return "the program is empty";
    if (p->n > CFR_PROG_LEN_MAX)
        return "the program is longer than PROG_LEN can express";

    for (uint32_t i = 0; i < p->n; i++) {
        uint32_t op  = f(&p->w[i], ISR_F_OPCODE);
        uint32_t sz  = f(&p->w[i], ISR_F_OPSIZE);
        uint32_t row = f(&p->w[i], ISR_F_ROW);
        uint32_t t   = f(&p->w[i], ISR_F_T);

        // Rule 5.  On an image that ties latch_sel, T=1 is not an error anywhere in
        // the hardware — it is a wrong number.  Nothing downstream can tell the
        // difference, so the refusal has to be here.
        if (t && !(flags & PIM_VERIFY_ALLOW_T)) {
            snprintf(buf, sizeof buf,
                     "ISR %u sets T (accumulator latch 1), and this image may tie "
                     "latch_sel to 0 — in which case both latches are latch 0 and "
                     "the answer is silently wrong.  Set allow_t_latch only with "
                     "evidence; runtime/test/tlatch_test.c produces it.", i);
            return buf;
        }

        switch (op) {
        case ISR_OP_WRVEC:
            // Rule 1.  The vector before this one must have been read in whole passes.
            if (gb_pos) {
                snprintf(buf, sizeof buf,
                         "ISR %u: WRVEC while the GB read position is at beat %u of "
                         "%u; the MACs before it stopped part-way through the vector",
                         i, gb_pos, gb_len);
                return buf;
            }
            gb_len = sz; gb_pos = 0; gb_filled = true;
            break;
        case ISR_OP_MAC:
            // Rule 1.  A MAC reads OPSIZE beats from the GB read position, which then
            // moves on and wraps at the end of what the last WRVEC wrote.
            if (!gb_filled || !gb_len) {
                snprintf(buf, sizeof buf,
                         "ISR %u is a MAC with nothing having filled the GB", i);
                return buf;
            }
            if (gb_pos + sz > gb_len) {
                snprintf(buf, sizeof buf,
                         "ISR %u: MAC reads %u beats from beat %u of a %u-beat vector, "
                         "across its end", i, sz, gb_pos, gb_len);
                return buf;
            }
            gb_pos = (gb_pos + sz) % gb_len;
            undrained[t & 1u]++;
            break;
        case ISR_OP_RD_MAC:
            // Rule 3.  Two results in one word: one of them is simply lost.
            for (unsigned j = 0; j < nseen; j++)
                if (seen[j] == row) {
                    snprintf(buf, sizeof buf,
                             "ISR %u: a second RD_MAC lands on GPR word %u", i, row);
                    return buf;
                }
            if (nseen < sizeof seen / sizeof *seen) seen[nseen++] = row;
            undrained[t & 1u] = 0;
            break;
        default:
            break;
        }
    }

    // Rule 2.  The latch is cleared only by RD_MAC and by reset — per latch, so a
    // split-latch schedule has to drain BOTH.
    for (unsigned t = 0; t < 2; t++)
        if (undrained[t]) {
            snprintf(buf, sizeof buf,
                     "%u MAC(s) into latch %u are never drained by a RD_MAC; that "
                     "accumulator would be inherited by whatever runs next",
                     undrained[t], t);
            return buf;
        }
    // Rule 1 again: the next program's vector must start at beat 0.
    if (gb_pos) {
        snprintf(buf, sizeof buf,
                 "the program ends with the GB read position at beat %u of %u; the "
                 "MACs stopped part-way through the vector", gb_pos, gb_len);
        return buf;
    }
    // Rule 4.  done rises when the fetcher ACCEPTS the last ISR.
    if (emu_isr_produces_result(f(&p->w[p->n - 1], ISR_F_OPCODE)))
        return "the last ISR produces a result, and done rises when the fetcher "
               "accepts it — its result would still be in flight.  Append an EOS.";
    return NULL;
}

// -------------------------------------------------------------- launching ---
const char *pim_exec_run(pim_exec *e, const pim_prog *p, pim_launch *out)
{
    const char *bad;
    uint64_t t0;

    if (!e || !p) return "pim_exec_run: null argument";
    if (out) memset(out, 0, sizeof *out);

    if (p->n > e->max_isrs)
        return ex_err(e, "the program is %u ISRs and this engine accepts %u",
                      p->n, e->max_isrs);
    if ((bad = pim_prog_verify(p, e->verify_flags)))
        return ex_err(e, "refusing to launch: %s", bad);

    // Before IMEM, not just before the doorbell — see the note at the top.
    if ((bad = require_idle(e, "load the program"))) return bad;

    if ((bad = pim_axi_write_ctx(e->ctx, EMU_AXI_IMEM, p->w,
                             (size_t)p->n * EMU_WORD_BYTES)))
        return ex_err(e, "IMEM load: %s", bad);

    if ((bad = require_idle(e, "ring the doorbell"))) return bad;

    if (out) out->status_before = cfr_rd(e, CFR_STATUS);
    cfr_wr(e, CFR_PROG_LEN, p->n);
    if (cfr_rd(e, CFR_PROG_LEN) != p->n)
        return ex_err(e, "PROG_LEN read back %u, not %u",
                      cfr_rd(e, CFR_PROG_LEN), p->n);
    // This read is non-posted, so the posted PROG_LEN write above is at the device
    // before the doorbell below.  It is an ordering point, not a check.
    (void)cfr_rd(e, CFR_PROG_LEN);

    t0 = now_us();
    cfr_wr(e, CFR_CTRL, CFR_CTRL_DOORBELL);

    for (;;) {
        uint32_t st = cfr_rd(e, CFR_STATUS);
        if (out) out->polls++;
        if (st & CFR_STATUS_DONE) { if (out) out->saw_done = true; break; }
        if (now_us() - t0 > (uint64_t)e->timeout_ms * 1000ull) break;
    }
    if (!(cfr_rd(e, CFR_STATUS) & CFR_STATUS_DONE))
        return ex_err(e, "the program did not finish in %u ms (%u polls, "
                         "STATUS=%#08x).  The dispatcher is still running it, or it "
                         "never started.", e->timeout_ms, out ? out->polls : 0,
                      cfr_rd(e, CFR_STATUS));
    if (out) {
        uint32_t lo, hi;

        out->us           = now_us() - t0;
        out->status_after = cfr_rd(e, CFR_STATUS);
        // RUN_CYC stops counting when done rises, so after done LO and HI are
        // stable and two reads give one value.
        lo = cfr_rd(e, CFR_RUN_CYC_LO);
        hi = cfr_rd(e, CFR_RUN_CYC_HI);
        out->run_cyc = CFR_RUN_CYC(lo, hi);
    }
    // A TIMEOUT IS THE ERROR IT IS.  This used to return success and let the caller
    // decide from a poisoned result word; nothing does that any more, so a launch
    // that never raised done has to say so here or nothing says it at all.
    //
    // `done` IS A HELD LEVEL and require_idle above accepts it as idle, so a stale
    // one from the previous run can satisfy the very first poll.  That is a real gap
    // and it is not closed here: closing it needs to know whether the doorbell
    // clears the bit, which has not been measured.  What this does catch is the
    // launch that hangs, which is the failure that actually strands a caller.
    return NULL;
}

const char *pim_exec_scrub(pim_exec *e, uint32_t gpr_word)
{
    const pim_geometry *g;
    pim_isr  storage[16 + 1];
    pim_prog prog;
    const char *bad;

    if (!e) return "pim_exec_scrub: e is NULL";
    g = pim_geom_ctx(e->ctx);
    if (g->nch + 1u > sizeof storage / sizeof *storage)
        return ex_err(e, "%u channels needs %u ISRs and the scrub buffer holds %zu",
                      g->nch, g->nch + 1, sizeof storage / sizeof *storage);

    pim_prog_init(&prog, storage, sizeof storage / sizeof *storage);
    for (uint32_t ch = 0; ch < g->nch; ch++) {
        // 1-hot, like every RD_MAC: a multicast would have two channels writing one
        // word.  RD_MAC broadcasts to every BANK though, so nch of these clear all
        // 16*nch latches.
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << ch);
        struct emu_isr      isr;

        s.opsize = 0;
        s.row    = gpr_word + ch;
        if ((bad = emu_isr_build(&isr, &s)))
            return ex_err(e, "scrub RD_MAC ch%u refused: %s", ch, bad);
        if ((bad = pim_prog_push(&prog, (const pim_isr *)&isr)))
            return ex_err(e, "%s", bad);
    }
    {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_EOS, (1u << g->nch) - 1u);
        struct emu_isr      isr;
        if ((bad = emu_isr_build(&isr, &s))) return ex_err(e, "scrub EOS: %s", bad);
        if ((bad = pim_prog_push(&prog, (const pim_isr *)&isr))) return ex_err(e, "%s", bad);
    }
    return pim_exec_run(e, &prog, NULL);
}

// ONE READ, NOT SIXTEEN.  VIOL_ANY0/1 carry a bit per bank, two channels to a
// register, so the summary costs one access and only a nonzero answer needs the
// per-bank walk.
const char *pim_exec_violations(pim_exec *e, unsigned ch, uint32_t *any)
{
    uint32_t v;

    if (!e || !any) return "pim_exec_violations: null argument";
    v = cfr_rd(e, VIOL_ANY_REG(ch));
    if (v == 0xFFFFFFFFu)
        return ex_err(e, "the CFR reads all-ones at the violation summary — the BAR "
                         "mapping is stale, or this is a v1 image whose statistics "
                         "are still at their own window");
    *any = (v >> VIOL_ANY_SHIFT(ch)) & 0xFFFFu;
    return NULL;
}

const char *pim_exec_violation_detail(pim_exec *e, unsigned ch, pim_viol *out)
{
    const pim_geometry *g;

    if (!e || !out) return "pim_exec_violation_detail: null argument";
    memset(out, 0, sizeof *out);
    g = pim_geom_ctx(e->ctx);
    if (ch >= g->nch) return ex_err(e, "no such channel: %u", ch);

    for (unsigned b = 0; b < EMU_NBANKS; b++) {
        uint32_t base = VIOL_BANK_BASE(ch, b);
        uint32_t cnt  = cfr_rd(e, base + VIOL_CNT);
        uint32_t mx   = cfr_rd(e, base + VIOL_MAX);

        out->sticky    |= cfr_rd(e, base + VIOL_STICKY);
        out->act_fill  += VIOL_CNT_ACT(cnt);
        out->pre_drain += VIOL_CNT_PRE(cnt);
        if (VIOL_MAX_ACT(mx) > out->worst_act_fill)
            out->worst_act_fill = VIOL_MAX_ACT(mx);
        if (VIOL_MAX_PRE(mx) > out->worst_pre_drain)
            out->worst_pre_drain = VIOL_MAX_PRE(mx);
    }
    return NULL;
}

// ONE REGISTER FOR THE WHOLE BOARD.  v1 cleared each channel's own block; v2.0's
// VIOL_CTRL[0] wipes every channel and every bank at once.
const char *pim_exec_clear_violations(pim_exec *e)
{
    if (!e) return "pim_exec_clear_violations: e is NULL";
    cfr_wr(e, VIOL_CTRL, 1u);
    return NULL;
}
