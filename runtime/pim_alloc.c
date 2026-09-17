// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_alloc.c — the rowset allocator, and weight placement.
//
// WHY THE UNIT IS A ROW AND NOT A BYTE
//   A MAC makes every bank in pu_mask read the same (ROW, COL).  So a region that
//   exists in only some banks is a region no MAC can read: it is not a smaller
//   allocation, it is an unusable one.  The unit is therefore ONE ROW INDEX HELD IN
//   ALL 16 BANKS — 2048 B x 16 = 32 KiB — and the device's address space is the
//   131072 row indices the ISR's 17-bit ROW field reaches, not 16 GiB of bytes.
//
// WHY THE LAYOUT IS WHAT IT IS
//   output o -> bank (o % 16)          RD_MAC returns lane i = bank i, so this is
//                                      the only split whose result comes back as a
//                                      contiguous GPR run instead of a gather
//   K-chunk  -> consecutive rows       COL + OPSIZE <= 64 forbids a MAC crossing a
//                                      row, so K is cut at 1024 whatever we do
//   COL = 0 always                     row-aligned.  Packing chunks densely would
//                                      save at most 1023 elements per output row
//                                      and would make every ROW a (row, col) pair
//                                      the allocator has to track — the ISR has no
//                                      base and no bound, so every extra degree of
//                                      freedom here is a way to read someone else's
//                                      tensor silently.
//
// WHY UPLOAD IS PER BANK AND NOT PER ROW
//   [measured 2026-08-10] writing 64 rows a row at a time is 1024 transfers of
//   2 KB and takes 13.9 ms.  Writing the same bytes as 16 contiguous per-bank
//   transfers takes 0.4 ms.  35x.  A layout whose per-bank slice is not contiguous
//   cannot load a model in reasonable time, which is why nrows is grouped by bank
//   on the way out.
//////////////////////////////////////////////////////////////////////////////////
#include "pim_internal.h"

#include <stdlib.h>

// ============================== rowset allocator ==============================
// First fit over a sorted, coalesced free list.  Weights are allocated once at
// model load and freed rarely, so nothing more clever earns its complexity — but
// coalescing is not optional: without it, load/free/load of different shapes
// fragments the row space and the failure is "out of memory" on a device that is
// half empty.
// One channel's RoBaCo space: nbank banks of bank_window bytes.  The same offset is
// used in every channel, so this is what the free list runs over and the real DRAM
// cost is PIM_NCH times it.  Both terms are compiled in from the platform conf.
#define PIM_CAPACITY ((uint64_t)PIM_BANK_WINDOW * PIM_NBANK)

static uint64_t roundup64(uint64_t v, uint64_t m) { return (v + m - 1u) / m * m; }

// BYTES IN, AN OFFSET OUT.  Granule and alignment are both one RoBaCo row: see the
// note on pim_alloc in pim.h for why neither is a parameter.
const char *pim_alloc(pim_dev *d, uint64_t nbytes, pim_buf *out)
{
    if (nbytes == 0)
        return pim_err(d, "pim_alloc(0) — an empty allocation has no address");

    const uint64_t want = roundup64(nbytes, PIM_ROBACO_ROW);

    for (struct pim_hole **pp = &d->holes; *pp; pp = &(*pp)->next) {
        struct pim_hole *h = *pp;
        // Every hole starts and ends on a row boundary — the whole space does, and
        // every split below preserves it — so there is no alignment gap to skip.
        if (h->size < want) continue;

        out->off  = h->off;
        out->size = want;
        d->used  += want;

        if (h->size == want) { *pp = h->next; free(h); return NULL; }
        h->off  += want;
        h->size -= want;
        return NULL;
    }
    uint64_t used, freeb, largest;
    pim_meminfo(d, &used, &freeb, &largest);
    return pim_err(d, "no run of %.1f MiB is free: %.1f used, %.1f free, largest "
                      "%.1f (all MiB, per channel).  A channel holds %.0f MiB "
                      "(%u banks x %.0f MiB), and there are %u channel(s).",
                   (double)want / 1048576.0, (double)used / 1048576.0,
                   (double)freeb / 1048576.0, (double)largest / 1048576.0,
                   (double)PIM_CAPACITY / 1048576.0, PIM_NBANK,
                   (double)PIM_BANK_WINDOW / 1048576.0, PIM_NCH);
}

const char *pim_free(pim_dev *d, pim_buf *b)
{
    if (!b || b->size == 0) return NULL;
    if (b->off % PIM_ROBACO_ROW || b->size % PIM_ROBACO_ROW)
        return pim_err(d, "pim_free of [%llu, %llu) is not row aligned — this handle "
                          "did not come from pim_alloc",
                       (unsigned long long)b->off,
                       (unsigned long long)(b->off + b->size));
    if (b->off + b->size > PIM_CAPACITY)
        return pim_err(d, "pim_free of [%llu, %llu) is outside the %llu byte space",
                       (unsigned long long)b->off,
                       (unsigned long long)(b->off + b->size),
                       (unsigned long long)PIM_CAPACITY);

    struct pim_hole **pp = &d->holes;
    while (*pp && (*pp)->off < b->off) {
        if ((*pp)->off + (*pp)->size > b->off)
            return pim_err(d, "pim_free of [%llu, %llu) overlaps a free run at "
                              "[%llu, %llu) — double free, or a corrupted handle",
                           (unsigned long long)b->off,
                           (unsigned long long)(b->off + b->size),
                           (unsigned long long)(*pp)->off,
                           (unsigned long long)((*pp)->off + (*pp)->size));
        pp = &(*pp)->next;
    }
    if (*pp && b->off + b->size > (*pp)->off)
        return pim_err(d, "pim_free of [%llu, %llu) overlaps a free run at [%llu, %llu)",
                       (unsigned long long)b->off,
                       (unsigned long long)(b->off + b->size),
                       (unsigned long long)(*pp)->off,
                       (unsigned long long)((*pp)->off + (*pp)->size));

    struct pim_hole *h = calloc(1, sizeof *h);
    if (!h) return pim_err(d, "out of memory freeing a buffer");
    h->off = b->off; h->size = b->size; h->next = *pp; *pp = h;
    d->used -= b->size;

    if (h->next && h->off + h->size == h->next->off) {          // coalesce right
        struct pim_hole *n = h->next;
        h->size += n->size; h->next = n->next; free(n);
    }
    for (struct pim_hole *q = d->holes; q && q->next; q = q->next)    // coalesce left
        if (q->off + q->size == q->next->off) {
            struct pim_hole *n = q->next;
            q->size += n->size; q->next = n->next; free(n);
            break;
        }
    b->off = b->size = 0;
    return NULL;
}

void pim_meminfo(const pim_dev *d, uint64_t *used, uint64_t *freeb,
                 uint64_t *largest_run)
{
    uint64_t f = 0, big = 0;
    for (const struct pim_hole *h = d->holes; h; h = h->next) {
        f += h->size;
        if (h->size > big) big = h->size;
    }
    if (used)        *used        = d->used;
    if (freeb)       *freeb       = f;
    if (largest_run) *largest_run = big;
}

// ============================== weights =======================================
uint32_t pim_chunk_opsize(const pim_weight *w, uint32_t chunk)
{
    return chunk + 1 == w->nchunks ? w->last_opsize : w->full_opsize;
}

// The ONLY path from a handle to an ISR's (ROW, COL).  Everything else calls this,
// so an out-of-range group or chunk is an error here instead of a read of whatever
// tensor happens to sit next door — the ISR carries no base and no bound.
//
// Placement.  groups_per_row groups share a page; within a page group g sits at
// column (g % gpr) * cols_per_group.  When a chunk is a whole page (k >= 1024) gpr
// is 1 and this reduces to one group per page per chunk, which is what a
// row-granular layout did.  When k is small it is where the 16x comes from.
const char *pim_place_of(pim_dev *d, const pim_weight *w, uint32_t group,
                         uint32_t chunk, uint32_t *row, uint32_t *col)
{
    if (group >= w->ngroups)
        return pim_err(d, "group %u is outside this weight (%u groups, n=%u)",
                       group, w->ngroups, w->n);
    if (chunk >= w->nchunks)
        return pim_err(d, "chunk %u is outside this weight (%u chunks, k=%u)",
                       chunk, w->nchunks, w->k);

    const uint32_t gpr = w->groups_per_row;
    const uint32_t cpg = w->kpad / PIM_LANES;              // columns per group
    // The allocator hands out ROW-ALIGNED byte offsets, so this division is exact
    // rather than a truncation that would quietly overlap the previous tensor.
    const uint32_t base_row = (uint32_t)(w->buf.off / PIM_ROBACO_ROW);

    uint32_t r = base_row + (group / gpr) * w->nchunks + chunk;
    uint32_t c = (gpr > 1) ? (group % gpr) * cpg : 0;

    // Both bounds matter and neither is checked downstream.
    if (c + pim_chunk_opsize(w, chunk) > PIM_COLS_PER_ROW)
        return pim_err(d, "group %u chunk %u would run from column %u for %u columns, "
                          "past the %u in a page — a MAC cannot cross a page and the "
                          "controller does not split one", group, chunk, c,
                       pim_chunk_opsize(w, chunk), PIM_COLS_PER_ROW);
    uint64_t at = (uint64_t)r * PIM_ROBACO_ROW + (uint64_t)c * PIM_COLSET_BYTES;
    if (at < w->buf.off || at >= w->buf.off + w->buf.size)
        return pim_err(d, "group %u chunk %u lands at byte %llu, outside this "
                          "weight's [%llu, %llu)", group, chunk,
                       (unsigned long long)at, (unsigned long long)w->buf.off,
                       (unsigned long long)(w->buf.off + w->buf.size));
    if (r > EMU_ISR_ROW_MAX)
        return pim_err(d, "row %u exceeds the ISR's 17-bit ROW field", r);
    *row = r; *col = c;
    return NULL;
}

static const char *weight_alloc(pim_dev *d, uint32_t n, uint32_t k, bool bias,
                                pim_weight *out)
{
    if (n == 0 || k == 0) return pim_err(d, "weight [%u][%u] has no elements", n, k);

    memset(out, 0, sizeof *out);
    out->n = n;
    out->k = k;
    out->has_bias = bias;
    if (bias) k += 1;                      // b becomes column k of W
    out->npad = (n + PIM_BANKS - 1) / PIM_BANKS * PIM_BANKS;   // whole groups
    out->kpad = (k + PIM_LANES - 1) / PIM_LANES * PIM_LANES;   // whole beats
    out->ngroups = out->npad / PIM_BANKS;
    out->nchunks = (out->kpad + PIM_ELEMS_PER_ROW - 1) / PIM_ELEMS_PER_ROW;
    out->full_opsize = PIM_BEATS_PER_ROW;
    // k is padded to a whole beat, NOT to a whole chunk: the final chunk carries a
    // shorter OPSIZE instead.  For k = 896 that is OPSIZE 56 rather than 64, which
    // the measured cost model (0.543 + 0.0128*L us) makes 7% cheaper per group.
    out->last_opsize = (uint16_t)((out->kpad - (out->nchunks - 1) * PIM_ELEMS_PER_ROW)
                                  / PIM_LANES);

    // How many groups share a page.  Only possible when a group fits in one page,
    // i.e. one chunk; with several chunks a group already owns whole pages.
    const uint32_t cpg = out->kpad / PIM_LANES;
    out->groups_per_row = (out->nchunks == 1 && cpg <= PIM_COLS_PER_ROW)
                          ? PIM_COLS_PER_ROW / cpg : 1;

    uint64_t nrows = ((uint64_t)out->ngroups + out->groups_per_row - 1)
                     / out->groups_per_row * out->nchunks;

    // The SHAPE decides how many bytes; the allocator decides where.  It knows
    // nothing of groups or chunks — it is handed a number.
    const char *e = pim_alloc(d, nrows * PIM_ROBACO_ROW, &out->buf);
    if (e) return e;
    if ((out->buf.off + out->buf.size - 1) / PIM_ROBACO_ROW > EMU_ISR_ROW_MAX) {
        pim_free(d, &out->buf);
        return pim_err(d, "weight [%u][%u] would end past the ISR's 17-bit ROW field",
                       n, k);
    }
    return NULL;
}

const char *pim_weight_alloc(pim_dev *d, uint32_t n, uint32_t k, pim_weight *out)
{ return weight_alloc(d, n, k, false, out); }
const char *pim_weight_alloc_bias(pim_dev *d, uint32_t n, uint32_t k, pim_weight *out)
{ return weight_alloc(d, n, k, true, out); }

const char *pim_weight_free(pim_dev *d, pim_weight *w)
{
    const char *e = pim_free(d, &w->buf);
    memset(w, 0, sizeof *w);
    return e;
}

// Build bank b's whole slice in one buffer, then push it with one transfer.
// Padding is written as zeros rather than left alone: a zero weight contributes
// exactly 0.0f to the sum in any order, so the padded columns and the padded
// output rows cannot perturb a result — but stale bytes from a previous tenant
// could, and would look like a hardware fault.
// Byte offset of (group, chunk) inside the per-bank staging block, i.e. relative
// to the weight's first page.  The same arithmetic pim_place_of does, in bytes.
static size_t place_off(const pim_weight *w, uint32_t g, uint32_t c)
{
    const uint32_t gpr = w->groups_per_row, cpg = w->kpad / PIM_LANES;
    const size_t row_rel = (size_t)(g / gpr) * w->nchunks + c;
    const size_t col     = (gpr > 1) ? (size_t)(g % gpr) * cpg : 0;
    return row_rel * EMU_ROW_BYTES + col * EMU_WORD_BYTES;
}

static void stage_bank(const pim_weight *w, const uint16_t *W, const uint16_t *bias,
                       unsigned b, uint8_t *blk)
{
    memset(blk, 0, (size_t)(w->buf.size / PIM_ROBACO_ROW) * EMU_ROW_BYTES);
    for (uint32_t g = 0; g < w->ngroups; g++) {
        uint32_t o = g * PIM_BANKS + b;                 // which output row
        if (o >= w->n) continue;                        // padded group: leave zeros
        for (uint32_t c = 0; c < w->nchunks; c++) {
            uint32_t k0 = c * PIM_ELEMS_PER_ROW;
            if (k0 >= w->k) break;
            uint32_t len = w->k - k0;
            if (len > PIM_ELEMS_PER_ROW) len = PIM_ELEMS_PER_ROW;
            memcpy(blk + place_off(w, g, c), W + (size_t)o * w->k + k0,
                   (size_t)len * sizeof(uint16_t));
        }
        // b_o goes in column k, which is chunk k/1024 at lane k%1024.  The vector's
        // element k is 1.0, so the MAC that reads this chunk adds b_o to the
        // accumulator — one more term, no special case anywhere downstream.
        if (w->has_bias) {
            uint32_t c = w->k / PIM_ELEMS_PER_ROW, off = w->k % PIM_ELEMS_PER_ROW;
            ((uint16_t *)(blk + place_off(w, g, c)))[off] = bias ? bias[o] : 0;
        }
    }
}

const char *pim_weight_upload(pim_dev *d, const pim_weight *w, const uint16_t *W,
                              const uint16_t *bias)
{
    if (w->has_bias && !bias)
        return pim_err(d, "this weight was allocated with a bias column; upload needs "
                          "the %u bias values", w->n);
    if (!w->has_bias && bias)
        return pim_err(d, "this weight has no bias column; allocate with "
                          "pim_weight_alloc_bias to store one");
    uint8_t *blk;
    const uint64_t base_row = w->buf.off / PIM_ROBACO_ROW;
    const size_t bytes = (size_t)(w->buf.size / PIM_ROBACO_ROW) * EMU_ROW_BYTES;
    const char *e = pim_stage(d, bytes, &blk);
    if (e) return e;

    for (unsigned b = 0; b < PIM_BANKS; b++) {
        stage_bank(w, W, bias, b, blk);
        e = pim_axi_write(d, pim_bank(0, b) + base_row * EMU_ROW_BYTES, blk, bytes);
        if (e) return e;
        // Cheap liveness probe between banks.  A 16-transfer upload that wedges the
        // device should say which bank it died on.
        e = pim_require_idle(d, "continue the weight upload", 1000);
        if (e) return e;
    }
    return NULL;
}

const char *pim_weight_verify(pim_dev *d, const pim_weight *w, const uint16_t *W,
                              const uint16_t *bias, uint64_t *n_bad)
{
    uint8_t *blk;
    const uint64_t base_row = w->buf.off / PIM_ROBACO_ROW;
    const size_t bytes = (size_t)(w->buf.size / PIM_ROBACO_ROW) * EMU_ROW_BYTES;
    uint8_t *want = malloc(bytes);
    if (!want) return pim_err(d, "out of memory verifying %zu B", bytes);
    const char *e = pim_stage(d, bytes, &blk);
    if (e) { free(want); return e; }

    uint64_t bad = 0;
    const char *first = NULL;
    static char msg[PIM_ERRLEN];

    for (unsigned b = 0; b < PIM_BANKS && !first; b++) {
        stage_bank(w, W, bias, b, want);
        e = pim_axi_read(d, pim_bank(0, b) + base_row * EMU_ROW_BYTES, blk, bytes);
        if (e) { free(want); return e; }
        const uint16_t *got = (const uint16_t *)blk;
        const uint16_t *exp = (const uint16_t *)want;
        for (size_t i = 0; i < bytes / 2; i++) {
            if (pim_bf16_same(got[i], exp[i])) continue;
            bad++;
            if (first) continue;
            size_t beat = (i % (EMU_ROW_BYTES / 2)) / PIM_LANES;
            size_t rel  = i / (EMU_ROW_BYTES / 2);
            snprintf(msg, sizeof msg,
                     "weight readback differs at bank %u, page %llu of this weight, "
                     "column %zu, lane %zu: wrote 0x%04x, read 0x%04x",
                     b, (unsigned long long)rel, beat, i % PIM_LANES, exp[i], got[i]);
            first = msg;
        }
    }
    free(want);
    if (n_bad) *n_bad = bad;
    if (first) { snprintf(d->err, PIM_ERRLEN, "%s", first); return d->err; }
    return NULL;
}
