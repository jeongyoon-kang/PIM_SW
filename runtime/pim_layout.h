// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_layout — turn a host matrix into per-bank writes.
//
// THE PROBLEM THIS SOLVES
// A GEMV's all-bank MAC needs output column n in bank n%16 of channel (n/16)%C, and
// it needs the K-tile it is about to read to be one whole DRAM page there.  A host
// buffer is none of that: W is contiguous in memory, and the direct aperture gives
// each bank its own base address.  So SOMETHING has to scatter, and this is it.
//
// It is not a formatting detail.  Written the obvious way — one 2 KiB transfer per
// (bank, row) — a 64-row upload costs 13.9 ms.  Gathered so each bank's whole slice
// is ONE transfer it costs 0.4 ms [measured].  35x, and it decides whether loading
// a model is seconds or minutes.
//
// THE ADDRESS MODEL IS RoBaCo
//     robaco(row, bank, col) = row*32768 + bank*2048 + col*32
// One RoBaCo row is 32 KiB = 16 banks x 1 page = exactly what one all-bank MAC
// consumes.  Consecutive RoBaCo rows put each bank's bytes contiguous in that
// bank's own address space, which is why the gather above is possible at all.
//
// The MC used to decode BaRoCo, so a linear write through it landed entirely in one
// bank; that is why UPLOAD_PATH=direct.  It now decodes RoBaCo, which is this same
// model — so one linear transfer per RoBaCo row places every bank's page at once and
// the 16-transfer gather below stops being necessary.  The switch is a base address
// and a stride, not a layout change, exactly as planned.  Not taken yet: it wants a
// measurement first, and emu_mc --contig is what makes it.
//
// WHERE A COLUMN GOES
//     bank(n)    = n mod 16
//     channel(n) = (n / 16) mod C
//     supergroup = n / (16*C)          one MAC covers 16*C outputs
// Every channel uses the SAME (bank, row, col) for a given supergroup, because the
// ISR carries one ROW field and CH_MASK is fan-out only.  That is a hard constraint
// from the ISA, and pim_matrix_plan() makes it unbreakable by construction.
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_LAYOUT_H
#define PIM_LAYOUT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pim_platform.h"

#define PIM_ELEMS_PER_PAGE  1024u       // 2048 B / 2 B — one K-tile
#define PIM_ROBACO_ROW      32768u      // 16 banks x 2048 B

// How a [n x k] matrix sits in the banks.  n is the number of OUTPUTS (columns of
// the GEMV), k the dot-product length.
struct pim_matrix {
    uint32_t n, k;              // as asked for
    uint32_t npad;              // n rounded up to 16*nch  — the padding is zero
    uint32_t kpad;              // k rounded up to 1024
    uint32_t ngroups;           // npad / (16*nch)   supergroups
    uint32_t ntiles;            // kpad / 1024       K-tiles per supergroup
    uint32_t base_row;          // first DRAM row used, in every channel
    uint32_t nrows;             // ngroups * ntiles
    uint16_t last_opsize;       // beats in the final K-tile (64 unless k is short)
    unsigned nch, nbank;        // copied from the platform so place() is self-contained
};

// Plan the layout.  Fails if it would run past the 256 MiB a bank decodes, which is
// also exactly what the ISR's 17-bit ROW field can name.
const char *pim_matrix_plan(uint32_t n, uint32_t k,
                            uint32_t base_row, struct pim_matrix *out);

// Where output column `col`'s K-tile `tile` lives.  Every output of a supergroup
// lands at the same (row, col0) in a different (channel, bank) — that is what makes
// one MAC able to cover all of them.
void pim_matrix_place(const struct pim_matrix *m, uint32_t col, uint32_t tile,
                      unsigned *ch, unsigned *bank, uint32_t *row, uint32_t *col0);

// Scatter and write.  `w` is [n][k] bf16 row-major — row n is output n's weights,
// which is how torch stores nn.Linear.weight, so a caller can hand over the tensor
// unchanged.  Padding (npad-n columns, kpad-k elements) is written as ZERO, so a
// padded lane contributes nothing rather than whatever was in DRAM.
//
// One pwrite per (channel, bank).  `xfer` is the caller's transfer function so this
// file opens nothing; return 0 on success.
typedef int (*pim_xfer_fn)(void *ctx, uint64_t axi, const void *buf, size_t len);
const char *pim_matrix_upload(const struct pim_matrix *m,
                              const uint16_t *w, pim_xfer_fn xfer, void *ctx);

// Read the banks back and compare against `w`.  Byte-exact and arithmetic-free, so
// it verifies PLACEMENT for any data at all — including real weights, where a GEMV
// comparison would be arguing about rounding instead.
typedef int (*pim_read_fn)(void *ctx, uint64_t axi, void *buf, size_t len);
const char *pim_matrix_verify(const struct pim_matrix *m,
                              const uint16_t *w, pim_read_fn rd, void *ctx,
                              uint64_t *nbad);

// Bytes one channel's slice occupies, for reporting.
static inline uint64_t pim_matrix_bytes(const struct pim_matrix *m)
{ return (uint64_t)m->nrows * PIM_ROBACO_ROW; }

#endif // PIM_LAYOUT_H
