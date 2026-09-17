// SPDX-License-Identifier: MIT
// pim_layout — see pim_layout.h.
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pim_layout.h"
#include "emu_regs.h"

static char g_err[512];
static const char *fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#include <stdarg.h>
static const char *fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    return g_err;
}

static uint32_t roundup(uint32_t v, uint32_t m) { return (v + m - 1u) / m * m; }

const char *pim_matrix_plan(uint32_t n, uint32_t k,
                            uint32_t base_row, struct pim_matrix *out)
{
    if (!n || !k) return fail("matrix must have n>0 and k>0");
    memset(out, 0, sizeof *out);

    const uint32_t per_group = PIM_NBANK * PIM_NCH;      // outputs one MAC covers

    out->n     = n;
    out->k     = k;
    out->nch   = PIM_NCH;
    out->nbank = PIM_NBANK;
    out->npad  = roundup(n, per_group);
    out->kpad  = roundup(k, PIM_ELEMS_PER_PAGE);
    out->ngroups = out->npad / per_group;
    out->ntiles  = out->kpad / PIM_ELEMS_PER_PAGE;
    out->base_row = base_row;
    out->nrows    = out->ngroups * out->ntiles;

    // The final K-tile may be short.  OPSIZE counts BEATS (= columns), and a beat is
    // 16 elements, so a k that is not a multiple of 16 rounds UP and the caller
    // zero-pads the tail — the padded lanes contribute nothing.
    uint32_t tail = k - (out->ntiles - 1u) * PIM_ELEMS_PER_PAGE;
    out->last_opsize = (uint16_t)((tail + EMU_LANES_PER_WORD - 1u) / EMU_LANES_PER_WORD);

    // A bank decodes bank_window bytes, which is also exactly what ISR ROW reaches
    // (17 b x 2048 B).  Running past it is not a warning: the rows simply do not
    // exist, and the DMA would land in the undecoded gap.
    const uint64_t rows_avail = (uint64_t)PIM_BANK_WINDOW / EMU_ROW_BYTES;
    if ((uint64_t)base_row + out->nrows > rows_avail)
        return fail("[%u x %u] needs %u rows from %u, but a bank has only %" PRIu64
                    " (%" PRIu64 " MiB / 2048 B).  Split the matrix or move base_row.",
                    n, k, out->nrows, base_row, rows_avail, (uint64_t)(PIM_BANK_WINDOW >> 20));
    return NULL;
}

void pim_matrix_place(const struct pim_matrix *m, uint32_t col, uint32_t tile,
                      unsigned *ch, unsigned *bank, uint32_t *row, uint32_t *col0)
{
    const uint32_t per_group = m->nbank * m->nch;
    if (bank) *bank = col % m->nbank;
    if (ch)   *ch   = (col / m->nbank) % m->nch;
    // Same row in EVERY channel: the ISR has one ROW field and CH_MASK only decides
    // who executes, so a supergroup's pages must agree on where they are.
    if (row)  *row  = m->base_row + (col / per_group) * m->ntiles + tile;
    if (col0) *col0 = 0;            // one K-tile fills a whole page, so COL starts at 0
}

// One bank's whole slice, gathered.  Row r of the slice is the page for
// (supergroup r/ntiles, tile r%ntiles) of the output this bank holds in that group.
static void gather_bank(const struct pim_matrix *m, const uint16_t *w,
                        unsigned ch, unsigned bank, uint8_t *dst)
{
    const uint32_t per_group = m->nbank * m->nch;
    memset(dst, 0, (size_t)m->nrows * EMU_ROW_BYTES);

    for (uint32_t g = 0; g < m->ngroups; g++) {
        const uint32_t col = g * per_group + ch * m->nbank + bank;
        if (col >= m->n) continue;             // padding column: stays zero
        const uint16_t *src = w + (size_t)col * m->k;
        for (uint32_t t = 0; t < m->ntiles; t++) {
            uint32_t k0 = t * PIM_ELEMS_PER_PAGE;
            if (k0 >= m->k) break;             // padding tile: stays zero
            uint32_t cnt = m->k - k0;
            if (cnt > PIM_ELEMS_PER_PAGE) cnt = PIM_ELEMS_PER_PAGE;
            memcpy(dst + (size_t)(g * m->ntiles + t) * EMU_ROW_BYTES,
                   src + k0, (size_t)cnt * sizeof *src);
        }
    }
}

const char *pim_matrix_upload(const struct pim_matrix *m,
                              const uint16_t *w, pim_xfer_fn xfer, void *ctx)
{
    const size_t slice = (size_t)m->nrows * EMU_ROW_BYTES;
    uint8_t *buf = malloc(slice);
    if (!buf) return fail("out of memory staging %zu B per bank", slice);

    for (unsigned c = 0; c < m->nch; c++)
        for (unsigned b = 0; b < m->nbank; b++) {
            gather_bank(m, w, c, b, buf);
            // ONE transfer per bank.  Row by row this is 35x slower [measured].
            uint64_t axi = pim_bank(c, b) + (uint64_t)m->base_row * EMU_ROW_BYTES;
            const char *bad = pim_dma_check(axi, slice);
            if (bad) { free(buf); return fail("ch%u bank%u: %s", c, b, bad); }
            if (xfer(ctx, axi, buf, slice) != 0) {
                free(buf);
                return fail("ch%u bank%u: transfer of %zu B to 0x%011" PRIx64 " failed",
                            c, b, slice, axi);
            }
        }
    free(buf);
    return NULL;
}

const char *pim_matrix_verify(const struct pim_matrix *m,
                              const uint16_t *w, pim_read_fn rd, void *ctx,
                              uint64_t *nbad)
{
    const size_t slice = (size_t)m->nrows * EMU_ROW_BYTES;
    uint8_t *want = malloc(slice), *got = malloc(slice);
    if (!want || !got) { free(want); free(got); return fail("out of memory"); }

    uint64_t bad = 0;
    for (unsigned c = 0; c < m->nch; c++)
        for (unsigned b = 0; b < m->nbank; b++) {
            gather_bank(m, w, c, b, want);
            uint64_t axi = pim_bank(c, b) + (uint64_t)m->base_row * EMU_ROW_BYTES;
            memset(got, 0xA5, slice);
            if (rd(ctx, axi, got, slice) != 0) {
                free(want); free(got);
                return fail("ch%u bank%u: read of %zu B from 0x%011" PRIx64 " failed",
                            c, b, slice, axi);
            }
            for (size_t i = 0; i < slice; i += 2)
                if (memcmp(want + i, got + i, 2)) bad++;
        }
    free(want); free(got);
    if (nbad) *nbad = bad;
    return NULL;
}
