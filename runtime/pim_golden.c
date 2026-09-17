// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_golden.c — the device's arithmetic, on the host.
//
// This is not "close enough for a tolerance check".  It is meant to be BIT
// IDENTICAL to what PIM_EXACT returns, so that a mismatch is a fault report and
// not a discussion.  On a machine with no validity gate, a reference you can trust
// to the last bit is the only way to tell a broken kernel from a broken model.
//
// THE MODEL IS NOW TRANSCRIBED FROM THE RTL, not inferred.
//   ref/PIM_HW/src/bank_controller/mac_tree/ — see pim_mac_exact.c, which carries
//   the line citations.  A beat is BLOCK FLOATING POINT: all 16 products are
//   aligned to a shared max exponent and then summed as exact integers.  The adder
//   tree therefore cannot affect the value, and everything is lost in the
//   alignment shift instead.
//
//   This replaced a hand-fitted fp32 model that was right on about 3999 outputs in
//   4000.  The question that model could never answer — "is the tree adjacent-pair
//   or halves?" — turned out to be the wrong question entirely.
//
//////////////////////////////////////////////////////////////////////////////////
#include "pim.h"

uint64_t pim_now_us_impl(void);

#include <stdlib.h>
#include <string.h>

uint64_t pim_now_us_pub(void) { return pim_now_us_impl(); }

uint16_t pim_f32_to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    if (((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu))   // a NaN stays a NaN
        return (uint16_t)((u >> 16) | 0x40u);
    uint32_t lsb = (u >> 16) & 1u;                         // round to nearest even
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}

float pim_bf16_to_f32(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

// Compare as VALUES.  +0 and -0 are the same number and a bitwise test would call
// a zero row a mismatch.
bool pim_bf16_same(uint16_t a, uint16_t b)
{
    if (a == b) return true;
    return ((a | b) & 0x7FFFu) == 0;
}

// Distance on the BF16 number line, for reporting how far off something is.
// Mapping the sign bit to a negation makes the two halves join up at zero, so the
// answer is a count of representable values and not a bit-pattern difference.
unsigned pim_bf16_ulps(uint16_t a, uint16_t b)
{
    if (pim_bf16_same(a, b)) return 0;
    int32_t ia = (a & 0x8000u) ? -(int32_t)(a & 0x7FFFu) : (int32_t)a;
    int32_t ib = (b & 0x8000u) ? -(int32_t)(b & 0x7FFFu) : (int32_t)b;
    int32_t d = ia - ib;
    return (unsigned)(d < 0 ? -d : d);
}

uint16_t pim_mac_exact(const uint16_t *w, const uint16_t *x, unsigned k);

void pim_gemv_golden(const uint16_t *W, uint32_t n, uint32_t k,
                     const uint16_t *x, const uint16_t *bias, uint16_t *y)
{
    // The bias is folded exactly as the device folds it — an extra column of W and
    // an extra 1.0 on x — so it is one more lane of one more beat and nothing here
    // needs a special case beyond building that lane.
    const uint32_t ke = bias ? k + 1 : k;
    const uint32_t kb = (ke + PIM_LANES - 1) / PIM_LANES * PIM_LANES;   // whole beats
    uint16_t *wb = malloc((size_t)kb * 2 * sizeof *wb);
    if (!wb) { for (uint32_t o = 0; o < n; o++) y[o] = 0; return; }
    uint16_t *xb = wb + kb;

    // The padded tail must be ZERO in the operand buffers, not merely ignored: the
    // hardware has no lane mask, so a stale lane joins the exponent max and can
    // annihilate every real lane in its beat.
    memset(xb, 0, (size_t)kb * sizeof *xb);
    memcpy(xb, x, (size_t)k * sizeof *x);
    if (bias) xb[k] = pim_f32_to_bf16(1.0f);

    for (uint32_t o = 0; o < n; o++) {
        memset(wb, 0, (size_t)kb * sizeof *wb);
        memcpy(wb, W + (size_t)o * k, (size_t)k * sizeof *W);
        if (bias) wb[k] = bias[o];
        y[o] = pim_mac_exact(wb, xb, kb);
    }
    free(wb);
}
