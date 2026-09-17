// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_internal.h — shared between the runtime's translation units, not exported.
//
// The layering the public header describes is enforced here by what each file is
// allowed to touch:
//   pim_dev.c    the only file that opens fds, mmaps, or calls pwrite/pread
//   pim_alloc.c  the only file that hands out row indices
//   pim_prog.c   the only file that builds a 256-bit ISR word or rings a doorbell
//   pim_gemv.c   schedules; it names rows only through pim_row_of()
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_INTERNAL_H
#define PIM_INTERNAL_H

#include "pim.h"
#include "emu_regs.h"
#include "pim_platform.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define PIM_ERRLEN 512

// A run of free rows.  The allocator keeps these sorted by row and coalesced, so
// "largest run" is a walk and a free next to a free is never two entries.
// A run of free BYTES in the per-channel RoBaCo space.  Sorted by offset and
// coalesced, so "largest run" is a walk and a free next to a free is never two
// entries.  Everything here is a multiple of PIM_ROBACO_ROW.
struct pim_hole { uint64_t off, size; struct pim_hole *next; };

struct pim_dev {
    int                bar_fd, h2c, c2h;
    volatile uint8_t  *bar;

    struct pim_hole   *holes;        // free list over [0, bank_window * nbank)
    uint64_t           used;         // bytes, per channel

    uint32_t           gpr_vec_base, gpr_vec_words;
    uint32_t           gpr_res_base, gpr_res_words;

    uint32_t           max_isrs;
    size_t             max_xfer;

    struct emu_isr    *prog;         // max_isrs entries
    uint16_t          *res;          // one launch's result lanes
    float             *acc;          // fp32 partial-sum accumulator, chunk-outer
    uint8_t           *stage;        // weight upload staging, grown on demand
    size_t             stage_bytes;

    pim_stats          st;
    char               err[PIM_ERRLEN];
};

// Set the error string and return it.  Every failure path goes through this so a
// caller always has something to print.
const char *pim_err(pim_dev *d, const char *fmt, ...)
    __attribute__((format(printf, 2, 3)));

uint64_t pim_now_us(void);

// ---- pim_dev.c ---------------------------------------------------------------
uint32_t    pim_cfr_rd(pim_dev *d, uint32_t off);
void        pim_cfr_wr(pim_dev *d, uint32_t off, uint32_t v);
const char *pim_axi_write(pim_dev *d, uint64_t axi, const void *buf, size_t len);
const char *pim_axi_read (pim_dev *d, uint64_t axi, void *buf, size_t len);
const char *pim_gpr_write(pim_dev *d, uint32_t word, const void *src, uint32_t n);
const char *pim_gpr_read (pim_dev *d, uint32_t word, void *dst, uint32_t n);
const char *pim_require_idle(pim_dev *d, const char *what, uint32_t timeout_ms);
const char *pim_stage(pim_dev *d, size_t bytes, uint8_t **out);

// ---- pim_prog.c --------------------------------------------------------------
// A program under construction.  Nothing else in the runtime may write ISR words.
struct pim_prog {
    pim_dev        *dev;
    struct emu_isr *w;
    uint32_t        n, cap;

    // What the verifier tracks as words are appended, so a violation is reported
    // at the ISR that caused it rather than at the end.
    uint32_t        gb_opsize;       // OPSIZE of the live WRVEC; 0 = GB not loaded
    uint32_t        undrained;       // MACs since the last RD_MAC
    uint32_t        res_lo, res_hi;  // GPR result words used, for collision checks
    bool            res_any;
    const char     *bad;             // first violation, sticky
};

void        pim_prog_init(struct pim_prog *p, pim_dev *d);
const char *pim_emit_wrvec (struct pim_prog *p, uint32_t gpr_word, uint32_t opsize);
const char *pim_emit_mac   (struct pim_prog *p, uint32_t row, uint32_t col, uint32_t opsize);
const char *pim_emit_rd_mac(struct pim_prog *p, uint32_t gpr_word);
const char *pim_emit_eos   (struct pim_prog *p);
const char *pim_prog_verify(const struct pim_prog *p);
void        pim_prog_print(const struct pim_prog *p, FILE *out, unsigned max_words);

// Load, poison, ring, wait, and read [res_word, res_word + nres) back.
// `lanes` receives nres * PIM_LANES BF16 values.
const char *pim_launch(pim_dev *d, const struct pim_prog *p,
                       uint32_t res_word, uint32_t nres, uint16_t *lanes);

// ---- pim_alloc.c -------------------------------------------------------------
// The ONLY sanctioned path from a weight handle to an ISR ROW field.  The ISR
// carries no base and no bound; this is where the bound lives.
// The ONLY sanctioned path from a handle to an ISR's (ROW, COL).
const char *pim_place_of(pim_dev *d, const pim_weight *w, uint32_t group,
                         uint32_t chunk, uint32_t *row, uint32_t *col);
uint32_t    pim_chunk_opsize(const pim_weight *w, uint32_t chunk);

#endif // PIM_INTERNAL_H
