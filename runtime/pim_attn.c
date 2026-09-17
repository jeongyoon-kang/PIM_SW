// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_attn.c — the KV cache, and the two attention matmuls.
//
// BOTH ARE GEMVs.  What makes them awkward is not the arithmetic, it is that the
// "matrix" grows by one row every step and the two of them want OPPOSITE layouts.
//
//   scores   s[t] = sum_d q[d] * K[t][d]     reduce over d,  index by t
//   output   o[d] = sum_t p[t] * V[t][d]     reduce over t,  index by d
//
// A MAC reduces along the beats INSIDE one bank's row and produces one value per
// bank.  So whatever is reduced has to lie along a row, and whatever indexes the
// output has to be the bank.  That forces:
//
//   K  POSITION-MAJOR.   position t -> bank t%16; the whole of K[t] (every kv head)
//                        lies along that row.  Appending one position is ONE
//                        contiguous write to ONE bank.
//   V  DIMENSION-MAJOR.  dimension D = h*head_dim + d -> bank D%16; positions run
//                        along the row.  V is stored TRANSPOSED.
//
// The transpose is what makes p.V possible at all, and it is also what makes it
// CHEAP: its reduction axis is the sequence length, so OPSIZE reaches 64 and the
// per-MAC overhead is amortised.  Scores are the opposite — head_dim is only 64,
// so OPSIZE is 4 and the fixed cost per MAC dominates.  Storing K position-major
// with COL packing at least stops that from also wasting the row: without it, 16
// positions would occupy a whole 32 KiB rowset to hold 128 B of data.
//
// THE PRICE OF THE TRANSPOSE, and how it is paid
//   Dimension-major means one new position touches hkv*head_dim different rows,
//   one element in each — and the slaves have no WSTRB, so a single element cannot
//   be written without destroying its beat.  The answer is to BUFFER: positions
//   accumulate on the host until a whole number of beats is ready, then go out as
//   one write per (bank, dimension-group).  Whatever has not been flushed is
//   folded into p.V on the host, which is a handful of multiply-adds and costs
//   nothing.  flush_gran trades write count against how much of the tail the host
//   carries.
//////////////////////////////////////////////////////////////////////////////////
#include "pim_internal.h"

#include <stdlib.h>

// ---- geometry ---------------------------------------------------------------
static uint32_t k_beats_per_pos(const pim_kv *c) { return c->hkv * c->head_dim / PIM_LANES; }
static uint32_t k_pos_per_row  (const pim_kv *c) { return PIM_BEATS_PER_ROW / k_beats_per_pos(c); }
static uint32_t v_groups       (const pim_kv *c) { return c->hkv * c->head_dim / PIM_BANKS; }

// K: position group G (16 consecutive positions, one per bank) -> (row, col).
static void k_place(const pim_kv *c, uint32_t G, uint32_t kvhead,
                    uint32_t *row, uint32_t *col)
{
    const uint32_t bpp = k_beats_per_pos(c), ppr = k_pos_per_row(c);
    *row = (uint32_t)(c->kbuf.off / PIM_ROBACO_ROW) + G / ppr;
    *col = (G % ppr) * bpp + kvhead * (c->head_dim / PIM_LANES);
}

// V: dimension group VG and position chunk cc -> row.  Group-major, so a group's
// chunks are consecutive — the same shape pim_weight uses, for the same reason.
static uint32_t v_row(const pim_kv *c, uint32_t VG, uint32_t cc)
{
    return (uint32_t)(c->vbuf.off / PIM_ROBACO_ROW) + VG * c->v_chunks + cc;
}

const char *pim_kv_alloc(pim_dev *d, uint32_t hkv, uint32_t head_dim,
                         uint32_t max_pos, uint32_t flush_gran, pim_kv *out)
{
    memset(out, 0, sizeof *out);
    if (!hkv || !head_dim || !max_pos) return pim_err(d, "kv cache has a zero dimension");
    if (head_dim % PIM_LANES)
        return pim_err(d, "head_dim %u must be a multiple of %u (a beat)", head_dim, PIM_LANES);
    out->hkv = hkv; out->head_dim = head_dim; out->max_pos = max_pos;
    out->flush_gran = flush_gran ? flush_gran : 64;
    if (out->flush_gran % PIM_LANES)
        return pim_err(d, "flush_gran %u must be a multiple of %u: a partial beat "
                          "cannot be written without WSTRB", out->flush_gran, PIM_LANES);

    const uint32_t bpp = hkv * head_dim / PIM_LANES;
    if (bpp > PIM_BEATS_PER_ROW)
        return pim_err(d, "hkv*head_dim = %u needs %u beats, more than one row (%u).  "
                          "K would have to be split across rows and this layout does "
                          "not do that.", hkv * head_dim, bpp, PIM_BEATS_PER_ROW);
    if (PIM_BEATS_PER_ROW % bpp)
        return pim_err(d, "hkv*head_dim = %u gives %u beats per position, which does "
                          "not divide the %u beats in a row", hkv * head_dim, bpp,
                       PIM_BEATS_PER_ROW);

    const uint32_t ngroups = (max_pos + PIM_BANKS - 1) / PIM_BANKS;   // 16 positions each
    const uint32_t ppr = PIM_BEATS_PER_ROW / bpp;
    const uint32_t krows = (ngroups + ppr - 1) / ppr;

    out->v_chunks = (max_pos + PIM_ELEMS_PER_ROW - 1) / PIM_ELEMS_PER_ROW;
    const uint32_t vrows = (hkv * head_dim / PIM_BANKS) * out->v_chunks;

    const char *e;
    // Page-aligned whole pages: the K layout packs positions by COL inside a page
    // and the V layout runs positions along one, so neither may straddle.
    if ((e = pim_alloc(d, (uint64_t)krows * PIM_ROBACO_ROW, &out->kbuf)))
        return e;
    if ((e = pim_alloc(d, (uint64_t)vrows * PIM_ROBACO_ROW, &out->vbuf))) {
        pim_free(d, &out->kbuf); return e;
    }

    out->pending = calloc((size_t)out->flush_gran * hkv * head_dim, sizeof *out->pending);
    if (!out->pending) { pim_kv_free(d, out); return pim_err(d, "out of memory"); }

    // Zero both caches.  A score MAC reads whole groups of 16 positions, so the
    // last group always contains slots past npos; if those held stale bytes from a
    // previous tenant they could be inf or NaN and would poison the whole beat.
    uint8_t *blk;
    const size_t kb = (size_t)krows * EMU_ROW_BYTES, vb = (size_t)vrows * EMU_ROW_BYTES;
    if ((e = pim_stage(d, kb > vb ? kb : vb, &blk))) { pim_kv_free(d, out); return e; }
    memset(blk, 0, kb > vb ? kb : vb);
    for (unsigned b = 0; b < PIM_BANKS; b++) {
        if ((e = pim_axi_write(d, pim_bank(0, b) + (out->kbuf.off / PIM_ROBACO_ROW) * EMU_ROW_BYTES,
                               blk, kb))) { pim_kv_free(d, out); return e; }
        if ((e = pim_axi_write(d, pim_bank(0, b) + (out->vbuf.off / PIM_ROBACO_ROW) * EMU_ROW_BYTES,
                               blk, vb))) { pim_kv_free(d, out); return e; }
    }
    return NULL;
}

const char *pim_kv_free(pim_dev *d, pim_kv *c)
{
    pim_free(d, &c->kbuf);
    pim_free(d, &c->vbuf);
    free(c->pending);
    memset(c, 0, sizeof *c);
    return NULL;
}

// ---- append ------------------------------------------------------------------
// k and v are both [hkv][head_dim] for ONE position.
//
// K goes straight out: one contiguous write of hkv*head_dim BF16 to bank t%16.
// V is buffered, because dimension-major means this position owns one element of
// each of hkv*head_dim different beats and none of them can be written alone.
const char *pim_kv_append(pim_dev *d, pim_kv *c, const uint16_t *k, const uint16_t *v)
{
    if (c->npos >= c->max_pos)
        return pim_err(d, "kv cache is full at %u positions", c->max_pos);
    const uint32_t t = c->npos, D = c->hkv * c->head_dim;
    uint32_t row, col;
    k_place(c, t / PIM_BANKS, 0, &row, &col);
    const char *e = pim_axi_write(d,
        pim_bank(0, t % PIM_BANKS) + (uint64_t)row * EMU_ROW_BYTES + col * EMU_WORD_BYTES,
        k, (size_t)D * sizeof *k);
    if (e) return e;

    memcpy(c->pending + (size_t)(t % c->flush_gran) * D, v, (size_t)D * sizeof *v);
    c->npos++;
    if (c->npos % c->flush_gran == 0) return pim_kv_flush(d, c);
    return NULL;
}

// Push whole beats of the buffered V out, transposed.  Only complete beats move;
// a partial beat stays on the host and pim_attn_output folds it in there.
const char *pim_kv_flush(pim_dev *d, pim_kv *c)
{
    const uint32_t D = c->hkv * c->head_dim;
    const uint32_t held = c->npos - c->flushed;           // buffered positions
    const uint32_t whole = held / PIM_LANES * PIM_LANES;  // that make full beats
    if (!whole) return NULL;

    const uint32_t t0 = c->flushed;
    if (t0 / PIM_ELEMS_PER_ROW != (t0 + whole - 1) / PIM_ELEMS_PER_ROW)
        return pim_err(d, "a flush of %u positions from %u would cross a chunk "
                          "boundary; flush_gran must divide %u", whole, t0,
                       PIM_ELEMS_PER_ROW);
    const uint32_t cc = t0 / PIM_ELEMS_PER_ROW;
    const uint32_t beat0 = (t0 % PIM_ELEMS_PER_ROW) / PIM_LANES;
    const uint32_t nbeat = whole / PIM_LANES;

    uint16_t *buf = malloc((size_t)nbeat * PIM_LANES * sizeof *buf);
    if (!buf) return pim_err(d, "out of memory flushing the kv cache");

    const char *e = NULL;
    for (uint32_t VG = 0; VG < v_groups(c) && !e; VG++) {
        const uint32_t row = v_row(c, VG, cc);
        for (unsigned b = 0; b < PIM_BANKS && !e; b++) {
            const uint32_t dim = VG * PIM_BANKS + b;      // this bank's dimension
            // Transpose: beat j, lane l  <-  position t0 + 16j + l
            for (uint32_t j = 0; j < nbeat; j++)
                for (unsigned l = 0; l < PIM_LANES; l++)
                    buf[j * PIM_LANES + l] =
                        c->pending[(size_t)(j * PIM_LANES + l) * D + dim];
            e = pim_axi_write(d, pim_bank(0, b) + (uint64_t)row * EMU_ROW_BYTES
                                 + (uint64_t)beat0 * EMU_WORD_BYTES,
                              buf, (size_t)nbeat * EMU_WORD_BYTES);
        }
    }
    free(buf);
    if (e) return e;

    // Slide whatever did not make a whole beat down to the front of the buffer.
    const uint32_t left = held - whole;
    if (left) memmove(c->pending, c->pending + (size_t)whole * D,
                      (size_t)left * D * sizeof *c->pending);
    c->flushed += whole;
    return NULL;
}

// ---- scores:  s[t] = sum_d q[d] * K[t][d] ------------------------------------
// EVERY QUERY HEAD IN ONE PROGRAM.  Per head the work is one vector load and two
// ISRs per group of 16 positions — about 19 us at S=512 — while a launch costs
// roughly 50 us of IMEM write, doorbell, poll and readback.  Called once per head
// that ratio is 4:1 against the compute [measured: 73 us/head].  Called once per
// LAYER the fixed cost is paid once for 32 heads.
//
// Each head needs its own WRVEC because each has its own q; what they share is the
// K cache and the doorbell.
const char *pim_attn_scores(pim_dev *d, const pim_kv *c, unsigned nq,
                            const uint32_t *kvhead, const uint16_t *const *q,
                            uint16_t *const *s)
{
    if (!c->npos) return pim_err(d, "the kv cache is empty");
    const uint32_t op = c->head_dim / PIM_LANES;             // 4 beats at head_dim 64
    const uint32_t G  = (c->npos + PIM_BANKS - 1) / PIM_BANKS;
    const char *e;

    if ((uint64_t)nq * op > d->gpr_vec_words)
        return pim_err(d, "%u heads x %u words of q exceed the %u reserved for vectors",
                       nq, op, d->gpr_vec_words);
    for (unsigned h = 0; h < nq; h++)
        if (kvhead[h] >= c->hkv) return pim_err(d, "query head %u maps to kv head %u of %u",
                                                h, kvhead[h], c->hkv);

    // All the q vectors first: one contiguous GPR write instead of nq of them.
    uint16_t *qb = calloc((size_t)nq * op * PIM_LANES, sizeof *qb);
    if (!qb) return pim_err(d, "out of memory");
    for (unsigned h = 0; h < nq; h++)
        memcpy(qb + (size_t)h * op * PIM_LANES, q[h], (size_t)c->head_dim * sizeof **q);
    e = pim_gpr_write(d, d->gpr_vec_base, qb, nq * op);
    free(qb);
    if (e) return e;

    // Split on HEAD boundaries: every head ends with its accumulators drained, so
    // nothing crosses a launch.
    const uint32_t per_head = 1 + 2 * G;
    if (per_head + 1 > d->max_isrs)
        return pim_err(d, "one head is %u ISRs at %u positions, over the %u cap",
                       per_head, c->npos, d->max_isrs);
    const uint32_t hstep = (d->max_isrs - 1) / per_head;

    for (unsigned h0 = 0; h0 < nq; h0 += hstep) {
        unsigned h1 = h0 + hstep > nq ? nq : h0 + hstep;
        struct pim_prog p;
        pim_prog_init(&p, d);
        for (unsigned h = h0; h < h1; h++) {
            if ((e = pim_emit_wrvec(&p, d->gpr_vec_base + h * op, op))) return e;
            for (uint32_t g = 0; g < G; g++) {
                uint32_t row, col;
                k_place(c, g, kvhead[h], &row, &col);
                if ((e = pim_emit_mac(&p, row, col, op))) return e;
                if ((e = pim_emit_rd_mac(&p, d->gpr_res_base + (h - h0) * G + g))) return e;
            }
        }
        if ((e = pim_emit_eos(&p))) return e;
        if ((e = pim_launch(d, &p, d->gpr_res_base, (h1 - h0) * G, d->res))) return e;
        for (unsigned h = h0; h < h1; h++)
            for (uint32_t g = 0; g < G; g++)
                for (unsigned b = 0; b < PIM_BANKS; b++) {
                    uint32_t t = g * PIM_BANKS + b;
                    if (t < c->npos)
                        s[h][t] = d->res[((size_t)(h - h0) * G + g) * PIM_LANES + b];
                }
    }
    return NULL;
}

// ---- output:  o[d] = sum_t p[t] * V[t][d] ------------------------------------
// Same batching, and the cheaper of the two: the reduction axis is the sequence,
// so OPSIZE reaches 64 and only head_dim/16 MACs are needed per head per chunk.
//
// The flushed positions are a GEMV against the transposed V; the buffered tail is
// folded in here on the host, into the same fp32 accumulator.
const char *pim_attn_output(pim_dev *d, const pim_kv *c, unsigned nq,
                            const uint32_t *kvhead, const uint16_t *const *p,
                            uint16_t *const *o)
{
    const uint32_t D = c->hkv * c->head_dim;
    const uint32_t ng = c->head_dim / PIM_BANKS;             // dimension groups per head
    const uint32_t C  = c->flushed ? (c->flushed + PIM_ELEMS_PER_ROW - 1) / PIM_ELEMS_PER_ROW : 0;
    const char *e;

    for (unsigned h = 0; h < nq; h++)
        if (kvhead[h] >= c->hkv) return pim_err(d, "query head %u maps to kv head %u of %u",
                                                h, kvhead[h], c->hkv);
    float *acc = calloc((size_t)nq * c->head_dim, sizeof *acc);
    if (!acc) return pim_err(d, "out of memory");

    for (uint32_t cc = 0; cc < C; cc++) {
        const uint32_t t0 = cc * PIM_ELEMS_PER_ROW;
        uint32_t len = c->flushed - t0;
        if (len > PIM_ELEMS_PER_ROW) len = PIM_ELEMS_PER_ROW;
        const uint32_t op = (len + PIM_LANES - 1) / PIM_LANES;

        if ((uint64_t)nq * op > d->gpr_vec_words) { free(acc);
            return pim_err(d, "%u heads x %u words of p exceed the vector area", nq, op); }
        uint16_t *pb = calloc((size_t)nq * op * PIM_LANES, sizeof *pb);
        if (!pb) { free(acc); return pim_err(d, "out of memory"); }
        for (unsigned h = 0; h < nq; h++)
            memcpy(pb + (size_t)h * op * PIM_LANES, p[h] + t0, (size_t)len * sizeof **p);
        e = pim_gpr_write(d, d->gpr_vec_base, pb, nq * op);
        free(pb);
        if (e) { free(acc); return e; }

        const uint32_t per_head = 1 + 2 * ng;
        const uint32_t hstep = (d->max_isrs - 1) / per_head;
        for (unsigned h0 = 0; h0 < nq; h0 += hstep) {
            unsigned h1 = h0 + hstep > nq ? nq : h0 + hstep;
            struct pim_prog pr;
            pim_prog_init(&pr, d);
            for (unsigned h = h0; h < h1; h++) {
                const uint32_t vg0 = kvhead[h] * c->head_dim / PIM_BANKS;
                if ((e = pim_emit_wrvec(&pr, d->gpr_vec_base + h * op, op))) { free(acc); return e; }
                for (uint32_t g = 0; g < ng; g++) {
                    if ((e = pim_emit_mac(&pr, v_row(c, vg0 + g, cc), 0, op))) { free(acc); return e; }
                    if ((e = pim_emit_rd_mac(&pr, d->gpr_res_base + (h - h0) * ng + g))) { free(acc); return e; }
                }
            }
            if ((e = pim_emit_eos(&pr))) { free(acc); return e; }
            if ((e = pim_launch(d, &pr, d->gpr_res_base, (h1 - h0) * ng, d->res))) { free(acc); return e; }
            for (unsigned h = h0; h < h1; h++)
                for (uint32_t g = 0; g < ng; g++)
                    for (unsigned b = 0; b < PIM_BANKS; b++)
                        acc[(size_t)h * c->head_dim + g * PIM_BANKS + b] +=
                            pim_bf16_to_f32(d->res[((size_t)(h - h0) * ng + g) * PIM_LANES + b]);
        }
    }

    // whatever has not been flushed
    for (unsigned h = 0; h < nq; h++)
        for (uint32_t t = c->flushed; t < c->npos; t++)
            for (uint32_t dd = 0; dd < c->head_dim; dd++)
                acc[(size_t)h * c->head_dim + dd] += pim_bf16_to_f32(p[h][t]) *
                    pim_bf16_to_f32(c->pending[(size_t)(t - c->flushed) * D
                                               + kvhead[h] * c->head_dim + dd]);

    for (unsigned h = 0; h < nq; h++)
        for (uint32_t dd = 0; dd < c->head_dim; dd++)
            o[h][dd] = pim_f32_to_bf16(acc[(size_t)h * c->head_dim + dd]);
    free(acc);
    return NULL;
}
