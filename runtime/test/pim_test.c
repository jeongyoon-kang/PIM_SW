// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_test — the runtime's acceptance test, on real hardware.
//
//     ./pim_test                    # every case
//     ./pim_test --max-isrs 96      # force pim_gemv to split across launches
//     ./pim_test --shape 2048x2048  # one shape
//
// WHAT IT IS ACTUALLY CHECKING
//   Not "does a GEMV look about right".  Three specific claims, each of which the
//   design rests on:
//
//   1. PIM_EXACT is BIT IDENTICAL to pim_gemv_golden().  If this ever fails, the
//      measured summation order is wrong, or the accumulator does not stay wide
//      across ISRs, or the layout is off by a row.  It is the strongest single
//      statement this stack can make, and it costs one run.
//   2. PIM_FAST is within ONE ulp of exact, on every output.  That is the price of
//      the 1.4-1.6x schedule, and the test reports the distribution rather than a
//      pass/fail so a regression shows up as a shift, not as a cliff.
//   3. Padding does not perturb anything.  n not a multiple of 16 and k not a
//      multiple of 1024 or even of 16 are the normal case for real models
//      (Qwen2.5-0.5B is 896 and 4864 wide), so they are tested as the normal case.
//
//   It also reads every weight back through the aperture before computing, because
//   nothing in the hardware protects a resident weight, and a corrupted weight and
//   a broken kernel look identical from the outside.
//
// This uses ONLY the public header.  That is deliberate: if the test needs an
// internal, the public API is not finished.
//
// Exit: 0 all pass   1 a failure   2 usage
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim.h"

#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// ---- operands ----------------------------------------------------------------
// Transformer-shaped: weights are small, activations are order one.  The point is
// that the sums actually round; a vector chosen to be exact in BF16 would make
// every schedule agree and the test would prove nothing.
static uint32_t rs = 0x9E3779B9u;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
// [fixed] rnd() >> 8 leaves TWENTY-FOUR bits, so the divisor is 2^24.  With 2^23
// this returned [0, 2) and gauss() came out centred on +6 instead of 0 — which
// made every output a sum of same-signed terms and manufactured the very
// ill-conditioning the model scan below now creates on purpose.
static float unit(void) { return (float)(rnd() >> 8) / 16777216.0f; }
static float gauss(void)
{
    float s = 0.0f;
    for (int i = 0; i < 12; i++) s += unit();
    return s - 6.0f;
}

// ILL-CONDITIONED OPERANDS, on purpose.  Every product the same sign and roughly
// the same size, so a sequential chain's running sum climbs to K times a term and
// its rounding error climbs with it, while a balanced tree's does not.  This is
// what makes the wiring observable: on well-behaved data every candidate agrees
// and the scan learns nothing.
//   normal  what a transformer holds, and the case that decides the schedule
//   ill     all positive and the same size: makes the ACROSS-BEAT fp32 accumulator
//           lose precision, which is how the accumulator's width becomes visible
//   wide    mixed signs over 2^-10..2^10: makes the SIXTEEN products inside one
//           beat ill-conditioned among themselves, which is the only way the beat
//           reduction's wiring becomes visible.  Use it with k = 16, where the
//           output IS one beat reduction and nothing downstream can hide it.
enum opdist { OP_NORMAL, OP_ILL, OP_WIDE };
static enum opdist g_op = OP_NORMAL;
static float operand(void)
{
    switch (g_op) {
    case OP_ILL: return 0.5f + unit();
    case OP_WIDE: {
        int e = (int)(rnd() % 21) - 10;
        float f = ldexpf(1.0f + unit(), e);
        return (rnd() & 1) ? -f : f;
    }
    default: return gauss();
    }
}

struct shape { uint32_t n, k; const char *why; };

static const struct shape SHAPES[] = {
    {   16,   16, "smallest thing that exists: one group, one beat"       },
    {   64, 1024, "exactly one full K-chunk"                              },
    {  100,  896, "n and k both padded — Qwen2.5-0.5B's hidden size"      },
    {  128, 1000, "k not even a multiple of 16"                           },
    {  256, 2048, "C = 2, the first shape that has to chain"              },
    {  512, 4096, "C = 4"                                                 },
    {  480, 5632, "C = 6 with a short tail — TinyLlama's MLP width"       },
    { 2048, 2048, "a real q_proj"                                         },
};
#define NSHAPES (sizeof SHAPES / sizeof SHAPES[0])

struct verdict { unsigned exact_bad, fast_bad, hist[4]; unsigned max_ulp;
                 double max_abs, rms; };

// The same order as pim_gemv_golden, accumulated in DOUBLE instead of float.
//
// pim_gemv_golden accumulates in fp32 because the accumulator was measured to be
// wider than BF16 — but "wider than BF16" is not "exactly fp32", and the two models
// only part on a sum that lands within half a BF16 ulp of a rounding boundary.  So
// when EXACT and the fp32 golden disagree, this is the question worth asking: did
// the device keep MORE than 24 bits?
static uint16_t golden64_one(const uint16_t *w, const uint16_t *x, uint32_t k)
{
    double acc = 0.0;
    for (uint32_t i = 0; i < k; i++)
        acc += (double)pim_bf16_to_f32(w[i]) * (double)pim_bf16_to_f32(x[i]);
    return pim_f32_to_bf16((float)acc);
}

// The hybrid the hardware's shape suggests: multiplier lanes feed an adder that
// reduces a BEAT (16 lanes) in fp32, and the beat result goes into an accumulator
// that is WIDER than fp32.
//
// Both halves of that are forced by measurements that would otherwise contradict
// each other.  The constructed cancel vectors showed eps being destroyed when it
// met a huge lane INSIDE a beat, which an exact reduction would not do — so the
// beat reduction is narrow.  And a near-tie output matched an f64 accumulator and
// not an fp32 one — so what carries beats is wide.
static uint16_t golden_hybrid_one(const uint16_t *w, const uint16_t *x, uint32_t k)
{
    double acc = 0.0;
    for (uint32_t b = 0; b * PIM_LANES < k; b++) {
        float s = 0.0f;
        for (uint32_t i = 0; i < PIM_LANES; i++) {
            uint32_t j = b * PIM_LANES + i;
            if (j >= k) break;
            s += pim_bf16_to_f32(w[j]) * pim_bf16_to_f32(x[j]);
        }
        acc += (double)s;
    }
    return pim_f32_to_bf16((float)acc);
}

// ---- which adder-tree wiring is it? -----------------------------------------
// The accumulator is fp32 and the result is rounded once to BF16 at RD_MAC (per
// the designer).  What is NOT pinned down is how the 16 multiplier lanes are
// reduced before they reach it — and in floating point the wiring is not a detail,
// it is a different function.
//
// The constructed cancel vectors refuted two wirings, but they do not single one
// out: a level-1 pairing of (i, i+2), for instance, also cancels to zero on both.
// What DOES separate a chain from any balanced tree is ill-conditioned data — all
// products the same sign and the same size, so a chain's running sum grows to
// N times a term and its rounding error grows with it, while a tree's does not.
// That is what --model-scan generates.
enum { RED_CHAIN, RED_ADJ, RED_HALF, RED_EXACT, RED_N };
static const char *RED_NAME[RED_N] = {
    "sequential chain, fp32",
    "tree (2i, 2i+1), fp32",
    "tree (i, i+n/2), fp32",
    "beat reduced exactly, then fp32",
};

static float reduce_beat(const float *p, unsigned n, int how)
{
    float t[PIM_LANES];
    for (unsigned i = 0; i < PIM_LANES; i++) t[i] = i < n ? p[i] : 0.0f;
    switch (how) {
    case RED_CHAIN: { float s = 0.0f; for (unsigned i = 0; i < PIM_LANES; i++) s += t[i]; return s; }
    case RED_ADJ:
        for (unsigned m = PIM_LANES; m > 1; m /= 2)
            for (unsigned i = 0; i < m / 2; i++) t[i] = t[2 * i] + t[2 * i + 1];
        return t[0];
    case RED_HALF:
        for (unsigned m = PIM_LANES; m > 1; m /= 2)
            for (unsigned i = 0; i < m / 2; i++) t[i] = t[i] + t[i + m / 2];
        return t[0];
    default: { double s = 0.0; for (unsigned i = 0; i < PIM_LANES; i++) s += (double)t[i];
               return (float)s; }
    }
}

// products exact -> beat reduction (candidate) -> fp32 accumulator -> one RNE
static uint16_t model_one(const uint16_t *w, const uint16_t *x, uint32_t k, int how)
{
    float acc = 0.0f;
    for (uint32_t b = 0; b * PIM_LANES < k; b++) {
        float p[PIM_LANES];
        unsigned n = 0;
        for (unsigned i = 0; i < PIM_LANES; i++) {
            uint32_t j = b * PIM_LANES + i;
            if (j >= k) break;
            p[n++] = pim_bf16_to_f32(w[j]) * pim_bf16_to_f32(x[j]);
        }
        acc += reduce_beat(p, n, how);
    }
    return pim_f32_to_bf16(acc);
}

// How each candidate host model fares against the device, accumulated over every
// shape.  Only a model that is right on ALL of them is a golden.
static struct { uint64_t n, bad_f32, bad_f64, bad_hyb, bad[RED_N]; } g_scan;

static void scan(const uint16_t *W, const uint16_t *x, uint32_t n, uint32_t k,
                 const uint16_t *dev)
{
    for (uint32_t o = 0; o < n; o++) {
        const uint16_t *w = W + (size_t)o * k;
        float a32 = 0.0f;
        for (uint32_t i = 0; i < k; i++) a32 += pim_bf16_to_f32(w[i]) * pim_bf16_to_f32(x[i]);
        g_scan.n++;
        if (!pim_bf16_same(dev[o], pim_f32_to_bf16(a32)))       g_scan.bad_f32++;
        if (!pim_bf16_same(dev[o], golden64_one(w, x, k)))      g_scan.bad_f64++;
        if (!pim_bf16_same(dev[o], golden_hybrid_one(w, x, k))) g_scan.bad_hyb++;
        for (int m = 0; m < RED_N; m++)
            if (!pim_bf16_same(dev[o], model_one(w, x, k, m))) g_scan.bad[m]++;
    }
}

static void tally(struct verdict *v, const uint16_t *got, const uint16_t *want,
                  uint32_t n, bool is_exact)
{
    for (uint32_t i = 0; i < n; i++) {
        unsigned u = pim_bf16_ulps(got[i], want[i]);
        if (is_exact) { if (u) v->exact_bad++; continue; }
        if (u) v->fast_bad++;
        if (u > v->max_ulp) v->max_ulp = u;
        v->hist[u < 3 ? u : 3]++;
        // A ulp count is a RELATIVE measure, and on an output where the K-chunk
        // partials nearly cancel the relative error is unbounded while the absolute
        // error is not.  Both are reported, because only together do they say what
        // the schedule actually costs.
        double e = (double)pim_bf16_to_f32(got[i]) - (double)pim_bf16_to_f32(want[i]);
        if (e < 0) e = -e;
        if (e > v->max_abs) v->max_abs = e;
        double y = (double)pim_bf16_to_f32(want[i]);
        v->rms += y * y;
    }
}

static int run_shape(pim_dev *d, const struct shape *sh, bool verbose)
{
    const uint32_t n = sh->n, k = sh->k;
    uint16_t *W  = malloc((size_t)n * k * sizeof *W);
    uint16_t *x  = malloc((size_t)k * sizeof *x);
    uint16_t *yg = malloc((size_t)n * sizeof *yg);
    uint16_t *ye = malloc((size_t)n * sizeof *ye);
    uint16_t *yf = malloc((size_t)n * sizeof *yf);
    if (!W || !x || !yg || !ye || !yf) { fprintf(stderr, "out of memory\n"); return 1; }

    for (size_t i = 0; i < (size_t)n * k; i++) W[i] = pim_f32_to_bf16(operand() * 0.02f);
    for (uint32_t i = 0; i < k; i++)           x[i] = pim_f32_to_bf16(operand());

    pim_weight w;
    const char *e = pim_weight_alloc(d, n, k, &w);
    if (e) { printf("    alloc: %s\n", e); return 1; }

    if (verbose)
        printf("    layout: %u groups x %u chunks, %u group%s per page = %u pages "
               "(%.2f MiB), opsize %u..%u, padded to [%u][%u]\n",
               w.ngroups, w.nchunks, w.groups_per_row,
               w.groups_per_row == 1 ? "" : "s",
               (unsigned)(w.buf.size / PIM_ROBACO_ROW),
               (double)w.buf.size / 1048576.0,
               w.full_opsize, w.last_opsize, w.npad, w.kpad);

    int rc = 1;
    uint64_t nbad = 0;
    struct verdict v;
    memset(&v, 0, sizeof v);

    if ((e = pim_weight_upload(d, &w, W, NULL)))                 { printf("    upload: %s\n", e); goto out; }
    if ((e = pim_weight_verify(d, &w, W, NULL, &nbad)))          { printf("    readback: %s\n", e); goto out; }

    pim_gemv_golden(W, n, k, x, NULL, yg);

    if ((e = pim_gemv(d, &w, x, ye, PIM_EXACT)))           { printf("    EXACT: %s\n", e); goto out; }
    if ((e = pim_gemv(d, &w, x, yf, PIM_FAST)))            { printf("    FAST: %s\n", e); goto out; }

    tally(&v, ye, yg, n, true);
    tally(&v, yf, yg, n, false);
    v.rms = sqrt(v.rms / n);
    scan(W, x, n, k, ye);

    printf("    EXACT vs golden: %s",
           v.exact_bad ? "" : "bit identical on all outputs");
    if (v.exact_bad) {
        printf("%u / %u DIFFER", v.exact_bad, n);
        for (uint32_t i = 0; i < n; i++)
            if (!pim_bf16_same(ye[i], yg[i])) {
                printf("   first at y[%u]: device %.6g (0x%04x), golden %.6g (0x%04x)",
                       i, (double)pim_bf16_to_f32(ye[i]), ye[i],
                       (double)pim_bf16_to_f32(yg[i]), yg[i]);
                break;
            }
    }
    printf("\n    FAST  vs golden: %u / %u differ (%.1f%%), max %u ulp, "
           "max |err| %.3g against rms(y) %.3g  = %.2f%%\n",
           v.fast_bad, n, 100.0 * v.fast_bad / n, v.max_ulp,
           v.max_abs, v.rms, v.rms > 0 ? 100.0 * v.max_abs / v.rms : 0.0);

    // The claims, restated as tests.
    if (v.exact_bad) { printf("    -> FAIL: PIM_EXACT is not bit exact\n"); goto out; }
    // The bound that actually holds for chunk-outer is on the ABSOLUTE error: each
    // of the nchunks partials is rounded to BF16 on its way out, so the error is at
    // most nchunks * half a ulp OF A PARTIAL.  Judging it in ulps of the RESULT
    // fails on outputs where the partials cancel, and that is a property of the
    // data, not a defect.
    if (v.rms > 0 && v.max_abs > 0.02 * v.rms) {
        printf("    -> FAIL: PIM_FAST's worst absolute error is %.1f%% of rms(y)\n",
               100.0 * v.max_abs / v.rms);
        goto out;
    }
    printf("    -> pass\n");
    rc = 0;
out:
    pim_weight_free(d, &w);
    free(W); free(x); free(yg); free(ye); free(yf);
    return rc;
}

// q, k and v share one input vector.  Running them as one job is the whole reason
// pim_gemv_multi exists: it removes two doorbells and, more importantly, two of
// the three sets of vector loads — and a 64-beat WRVEC costs more than a 64-beat
// MAC.  What is checked here is that sharing does not perturb any of the three.
static int run_multi(pim_dev *d, bool timed)
{
    const uint32_t K = 2048;
    const uint32_t N[3] = { 2048, 256, 256 };            // q, and GQA-narrow k and v
    const char *nm[3] = { "q", "k", "v" };
    printf("\n  gemv_multi — q[%u][%u] + k[%u][%u] + v[%u][%u] sharing one x\n",
           N[0], K, N[1], K, N[2], K);

    uint16_t *W[3] = { 0 }, *yg[3] = { 0 }, *ye[3] = { 0 }, *yf[3] = { 0 };
    uint16_t *x = malloc((size_t)K * sizeof *x);
    pim_weight w[3];
    memset(w, 0, sizeof w);
    int rc = 1;
    const char *e;
    unsigned allocd = 0;

    if (!x) { fprintf(stderr, "out of memory\n"); goto out; }
    for (uint32_t i = 0; i < K; i++) x[i] = pim_f32_to_bf16(gauss());
    for (unsigned i = 0; i < 3; i++) {
        W[i]  = malloc((size_t)N[i] * K * sizeof **W);
        yg[i] = malloc((size_t)N[i] * sizeof **yg);
        ye[i] = malloc((size_t)N[i] * sizeof **ye);
        yf[i] = malloc((size_t)N[i] * sizeof **yf);
        if (!W[i] || !yg[i] || !ye[i] || !yf[i]) { fprintf(stderr, "out of memory\n"); goto out; }
        for (size_t q = 0; q < (size_t)N[i] * K; q++)
            W[i][q] = pim_f32_to_bf16(gauss() * 0.02f);
        if ((e = pim_weight_alloc(d, N[i], K, &w[i]))) { printf("    alloc %s: %s\n", nm[i], e); goto out; }
        allocd++;
        if ((e = pim_weight_upload(d, &w[i], W[i], NULL)))    { printf("    upload %s: %s\n", nm[i], e); goto out; }
        pim_gemv_golden(W[i], N[i], K, x, NULL, yg[i]);
    }

    const pim_weight *ws[3] = { &w[0], &w[1], &w[2] };
    uint16_t *outs_e[3] = { ye[0], ye[1], ye[2] };
    uint16_t *outs_f[3] = { yf[0], yf[1], yf[2] };

    pim_stats a, b;
    pim_get_stats(d, &a);
    if ((e = pim_gemv_multi(d, ws, 3, x, outs_e, PIM_EXACT))) { printf("    EXACT: %s\n", e); goto out; }
    pim_get_stats(d, &b);
    const uint64_t us_e = b.device_us - a.device_us;
    const uint64_t la_e = b.launches - a.launches;

    pim_get_stats(d, &a);
    if ((e = pim_gemv_multi(d, ws, 3, x, outs_f, PIM_FAST)))  { printf("    FAST: %s\n", e); goto out; }
    pim_get_stats(d, &b);
    const uint64_t us_f = b.device_us - a.device_us;
    const uint64_t la_f = b.launches - a.launches;

    unsigned bad = 0, maxu = 0;
    for (unsigned i = 0; i < 3; i++) {
        struct verdict v;
        memset(&v, 0, sizeof v);
        tally(&v, ye[i], yg[i], N[i], true);
        tally(&v, yf[i], yg[i], N[i], false);
        v.rms = sqrt(v.rms / N[i]);
        printf("    %s: EXACT ", nm[i]);
        if (!v.exact_bad) printf("bit identical");
        else {
            printf("%u/%u DIFFER", v.exact_bad, N[i]);
            unsigned shown = 0;
            for (uint32_t q = 0; q < N[i] && shown < 3; q++)
                if (!pim_bf16_same(ye[i][q], yg[i][q])) {
                    printf("  [y%u: dev %.6g want %.6g, %u ulp]", q,
                           (double)pim_bf16_to_f32(ye[i][q]),
                           (double)pim_bf16_to_f32(yg[i][q]),
                           pim_bf16_ulps(ye[i][q], yg[i][q]));
                    shown++;
                }
        }
        if (v.exact_bad) {
            unsigned m64 = 0;
            for (uint32_t q = 0; q < N[i]; q++)
                if (!pim_bf16_same(ye[i][q], yg[i][q]) &&
                    pim_bf16_same(ye[i][q], golden64_one(W[i] + (size_t)q * K, x, K)))
                    m64++;
            printf("  (device matches f64 on %u of them)", m64);
        }
        printf(",  FAST %u/%u differ, max |err| %.3g vs rms(y) %.3g\n",
               v.fast_bad, N[i], v.max_abs, v.rms);
        bad += v.exact_bad;
        if (v.rms > 0 && v.max_abs > 0.02 * v.rms) maxu = 1;   // absolute, see run_shape
    }
    if (timed)
        printf("    EXACT %"PRIu64" us in %"PRIu64" launch%s   FAST %"PRIu64" us in %"PRIu64
               " launch%s   -> %.2fx\n",
               us_e, la_e, la_e == 1 ? "" : "es", us_f, la_f, la_f == 1 ? "" : "es",
               us_f ? (double)us_e / (double)us_f : 0.0);
    if (bad)      { printf("    -> FAIL: sharing a vector perturbed an exact result\n"); goto out; }
    if (maxu) { printf("    -> FAIL: FAST's absolute error is over 2%% of rms(y)\n"); goto out; }
    printf("    -> pass\n");
    rc = 0;
out:
    for (unsigned i = 0; i < allocd; i++) pim_weight_free(d, &w[i]);
    for (unsigned i = 0; i < 3; i++) { free(W[i]); free(yg[i]); free(ye[i]); free(yf[i]); }
    free(x);
    return rc;
}

// A MAC has no bias input, so the bias becomes column k of W and element k of x
// becomes 1.0.  What is checked is that this is EXACT — the bias joins the fp32
// accumulator before the single rounding, so the device must agree with a golden
// that folds it the same way, bit for bit.  The cost in extra chunks is printed,
// because for a k that is already a multiple of 1024 it is not free.
static int run_bias(pim_dev *d, uint32_t n, uint32_t k)
{
    printf("\n  bias — [%u][%u] with b folded into column k\n", n, k);
    uint16_t *W = malloc((size_t)n * k * sizeof *W);
    uint16_t *bb = malloc((size_t)n * sizeof *bb);
    uint16_t *x = malloc((size_t)k * sizeof *x);
    uint16_t *yg = malloc((size_t)n * sizeof *yg), *ye = malloc((size_t)n * sizeof *ye);
    pim_weight w0, w1;
    int rc = 1;
    const char *e;
    unsigned got0 = 0, got1 = 0;
    if (!W || !bb || !x || !yg || !ye) { fprintf(stderr, "out of memory\n"); goto out; }
    for (size_t i = 0; i < (size_t)n * k; i++) W[i] = pim_f32_to_bf16(operand() * 0.02f);
    for (uint32_t i = 0; i < n; i++)           bb[i] = pim_f32_to_bf16(operand() * 0.5f);
    for (uint32_t i = 0; i < k; i++)           x[i] = pim_f32_to_bf16(operand());

    if ((e = pim_weight_alloc(d, n, k, &w0)))       { printf("    %s\n", e); goto out; }
    got0 = 1;
    if ((e = pim_weight_alloc_bias(d, n, k, &w1)))  { printf("    %s\n", e); goto out; }
    got1 = 1;
    printf("    without bias: %u chunks, opsize %u..%u, %.2f MiB\n",
           w0.nchunks, w0.full_opsize, w0.last_opsize, (double)w0.buf.size / 1048576.0);
    printf("    with bias:    %u chunks, opsize %u..%u, %.2f MiB  -> %+.2f (%+.1f%%)\n",
           w1.nchunks, w1.full_opsize, w1.last_opsize, (double)w1.buf.size / 1048576.0,
           ((double)w1.buf.size - (double)w0.buf.size) / 1048576.0,
           100.0 * ((double)w1.buf.size / (double)w0.buf.size - 1.0));

    if ((e = pim_weight_upload(d, &w1, W, bb)))     { printf("    %s\n", e); goto out; }
    if ((e = pim_gemv(d, &w1, x, ye, PIM_EXACT)))   { printf("    %s\n", e); goto out; }
    pim_gemv_golden(W, n, k, x, bb, yg);

    unsigned bad = 0;
    for (uint32_t i = 0; i < n; i++) if (!pim_bf16_same(ye[i], yg[i])) bad++;
    printf("    W.x + b vs golden: %s (%u of %u differ)\n",
           bad ? "DIFFER" : "bit identical", bad, n);
    // Refusing the mismatched pairing matters as much as computing it: uploading a
    // bias to a handle that has no column for it would silently drop the bias.
    if (!pim_weight_upload(d, &w0, W, bb)) { printf("    FAIL: accepted a bias on a "
                                                   "no-bias handle\n"); goto out; }
    if (!pim_weight_upload(d, &w1, W, NULL)) { printf("    FAIL: accepted a missing "
                                                     "bias on a bias handle\n"); goto out; }
    if (bad) { printf("    -> FAIL\n"); goto out; }
    printf("    -> pass\n");
    rc = 0;
out:
    if (got0) pim_weight_free(d, &w0);
    if (got1) pim_weight_free(d, &w1);
    free(W); free(bb); free(x); free(yg); free(ye);
    return rc;
}

// A whole decode: append positions one at a time, then ask for both attention
// matmuls at each of several lengths.  Appending one position at a time is the
// point — it is the case the layout has to survive, and it is where a transposed V
// could have been unaffordable.
static int run_attn(pim_dev *d, uint32_t hkv, uint32_t head_dim, uint32_t S,
                    uint32_t flush_gran)
{
    const uint32_t D = hkv * head_dim;
    printf("\n  attention — %u kv heads x head_dim %u, %u positions, flush every %u\n",
           hkv, head_dim, S, flush_gran);

    pim_kv c;
    const char *e;
    if ((e = pim_kv_alloc(d, hkv, head_dim, S, flush_gran, &c))) { printf("    %s\n", e); return 1; }
    printf("    K %u rows position-major (bank = t%%16), V %u rows transposed "
           "(bank = dim%%16, %u chunks)\n",
           (unsigned)(c.kbuf.size / PIM_ROBACO_ROW),
           (unsigned)(c.vbuf.size / PIM_ROBACO_ROW), c.v_chunks);

    uint16_t *K = malloc((size_t)S * D * sizeof *K);       // host mirror, [t][D]
    uint16_t *V = malloc((size_t)S * D * sizeof *V);
    uint16_t *q = malloc((size_t)head_dim * sizeof *q);
    uint16_t *p = malloc((size_t)S * sizeof *p);
    uint16_t *sd = malloc((size_t)S * sizeof *sd), *sg = malloc((size_t)S * sizeof *sg);
    uint16_t *od = malloc((size_t)head_dim * sizeof *od), *og = malloc((size_t)head_dim * sizeof *og);
    uint16_t *row = malloc((size_t)S * head_dim * sizeof *row);
    int rc = 1;
    if (!K || !V || !q || !p || !sd || !sg || !od || !og || !row) goto out;

    for (size_t i = 0; i < (size_t)S * D; i++) { K[i] = pim_f32_to_bf16(operand());
                                                 V[i] = pim_f32_to_bf16(operand()); }
    for (uint32_t i = 0; i < head_dim; i++) q[i] = pim_f32_to_bf16(operand());
    for (uint32_t i = 0; i < S; i++)        p[i] = pim_f32_to_bf16(operand() * 0.01f);

    uint64_t t_app = 0;
    for (uint32_t t = 0; t < S; t++) {
        uint64_t a = pim_now_us_pub();
        if ((e = pim_kv_append(d, &c, K + (size_t)t * D, V + (size_t)t * D))) {
            printf("    append %u: %s\n", t, e); goto out;
        }
        t_app += pim_now_us_pub() - a;
    }
    printf("    %u appends in %"PRIu64" us  (%.1f us each: one K write + a V flush "
           "every %u)\n", S, t_app, (double)t_app / S, flush_gran);

    unsigned bad_s = 0, bad_o = 0;
    double max_o = 0.0, rms_o = 0.0;
    uint32_t kvmap[1]; const uint16_t *qa[1], *pa[1]; uint16_t *sa[1], *oa[1];
    qa[0] = q; pa[0] = p; sa[0] = sd; oa[0] = od;
    for (uint32_t h = 0; h < hkv; h++) {
        kvmap[0] = h;
        if ((e = pim_attn_scores(d, &c, 1, kvmap, qa, sa))) { printf("    scores: %s\n", e); goto out; }
        for (uint32_t t = 0; t < S; t++)
            memcpy(row + (size_t)t * head_dim, K + (size_t)t * D + h * head_dim,
                   head_dim * sizeof *row);
        pim_gemv_golden(row, S, head_dim, q, NULL, sg);
        for (uint32_t t = 0; t < S; t++) if (!pim_bf16_same(sd[t], sg[t])) bad_s++;

        if ((e = pim_attn_output(d, &c, 1, kvmap, pa, oa))) { printf("    output: %s\n", e); goto out; }
        for (uint32_t dd = 0; dd < head_dim; dd++)          // V transposed, [d][t]
            for (uint32_t t = 0; t < S; t++)
                row[(size_t)dd * S + t] = V[(size_t)t * D + h * head_dim + dd];
        pim_gemv_golden(row, head_dim, S, p, NULL, og);
        for (uint32_t dd = 0; dd < head_dim; dd++) {
            if (!pim_bf16_same(od[dd], og[dd])) bad_o++;
            double e2 = (double)pim_bf16_to_f32(od[dd]) - (double)pim_bf16_to_f32(og[dd]);
            if (e2 < 0) e2 = -e2;
            if (e2 > max_o) max_o = e2;
            rms_o += (double)pim_bf16_to_f32(og[dd]) * (double)pim_bf16_to_f32(og[dd]);
        }
    }
    rms_o = sqrt(rms_o / (head_dim * hkv));
    printf("    q.K^T  %u / %u scores differ from the host model\n", bad_s, S * hkv);
    // p.V is judged on magnitude, not on a count.  When some positions are still on
    // the host the device half comes back already rounded to BF16 and the tail is
    // added after it, so there is one more rounding than a single fp32 sum — by
    // construction, not by accident.
    printf("    p.V    %u / %u outputs differ, max |err| %.3g vs rms %.3g = %.2f%%%s\n",
           bad_o, head_dim * hkv, max_o, rms_o, rms_o > 0 ? 100.0 * max_o / rms_o : 0.0,
           S % flush_gran ? "   (a host tail is live)" : "");
    if (bad_s) { printf("    -> FAIL: scores are not the same function\n"); goto out; }
    if (rms_o > 0 && max_o > 0.02 * rms_o) { printf("    -> FAIL: p.V is off by %.1f%%\n",
                                                    100.0 * max_o / rms_o); goto out; }
    printf("    -> pass\n");
    rc = 0;
out:
    pim_kv_free(d, &c);
    free(K); free(V); free(q); free(p); free(sd); free(sg); free(od); free(og); free(row);
    return rc;
}

// The allocator on its own: it decides where every weight lands, and a bug here is
// two tensors on the same rows, which reads as a hardware fault.
static int run_alloc(pim_dev *d)
{
    printf("\n  allocator\n");
    const uint64_t R = PIM_ROBACO_ROW;
    uint64_t used, freeb, largest, cap;
    pim_meminfo(d, &used, &freeb, &largest);
    cap = freeb;                       // a fresh device is entirely free
    if (used != 0 || largest != cap) {
        printf("    FAIL: fresh device is not empty (%llu used, %llu free, largest %llu)\n",
               (unsigned long long)used, (unsigned long long)freeb,
               (unsigned long long)largest);
        return 1;
    }
    printf("    capacity %.0f MiB per channel\n", (double)cap / 1048576.0);

    pim_buf a, b, c;
    const char *e;
    if ((e = pim_alloc(d, 2 * R, &a)) || (e = pim_alloc(d, 4 * R, &b)) ||
        (e = pim_alloc(d, 6 * R, &c))) { printf("    FAIL: %s\n", e); return 1; }
    if (a.off != 0 || b.off != 2 * R || c.off != 6 * R) {
        printf("    FAIL: bump order is %llu/%llu/%llu, expected 0/%llu/%llu\n",
               (unsigned long long)a.off, (unsigned long long)b.off,
               (unsigned long long)c.off, (unsigned long long)(2 * R),
               (unsigned long long)(6 * R));
        return 1;
    }

    // A request that is not a whole number of rows must be ROUNDED UP, not refused
    // and not silently truncated: the caller asked for bytes and the hardware's unit
    // is a row.
    pim_buf odd;
    if ((e = pim_alloc(d, 1, &odd))) { printf("    FAIL: %s\n", e); return 1; }
    if (odd.size != R || odd.off % R) {
        printf("    FAIL: 1-byte request gave off %llu size %llu, expected a whole row\n",
               (unsigned long long)odd.off, (unsigned long long)odd.size);
        return 1;
    }
    pim_free(d, &odd);

    // Free the middle, then a hole-sized request must reuse it rather than extend.
    if ((e = pim_free(d, &b))) { printf("    FAIL: %s\n", e); return 1; }
    pim_buf r;
    if ((e = pim_alloc(d, 4 * R, &r))) { printf("    FAIL: %s\n", e); return 1; }
    if (r.off != 2 * R) {
        printf("    FAIL: hole at %llu not reused (got %llu)\n",
               (unsigned long long)(2 * R), (unsigned long long)r.off);
        return 1;
    }

    // EVERY offset is row aligned.  Placement divides by the row size and would
    // overlap the previous tensor if this ever stopped holding — there is no
    // alignment argument any more precisely because it is not optional.
    pim_buf al;
    if ((e = pim_alloc(d, 3 * R + 17, &al))) { printf("    FAIL: %s\n", e); return 1; }
    if (al.off % R || al.size != 4 * R) {
        printf("    FAIL: alloc landed at %llu size %llu — not row aligned/rounded\n",
               (unsigned long long)al.off, (unsigned long long)al.size);
        return 1;
    }

    // Free everything and check it coalesced back to one run.
    pim_free(d, &al); pim_free(d, &r); pim_free(d, &a); pim_free(d, &c);
    pim_meminfo(d, &used, &freeb, &largest);
    if (used != 0 || largest != cap) {
        printf("    FAIL: not coalesced (%llu used, largest run %llu of %llu)\n",
               (unsigned long long)used, (unsigned long long)largest,
               (unsigned long long)cap);
        return 1;
    }

    // Asking for one byte more than exists must be refused, not wrapped.
    pim_buf too;
    if (!pim_alloc(d, cap + 1, &too)) {
        printf("    FAIL: an over-capacity request succeeded\n"); return 1;
    }
    if ((e = pim_alloc(d, cap, &r))) { printf("    FAIL: exact-capacity: %s\n", e); return 1; }
    pim_free(d, &r);

    printf("    bump, rounding, hole reuse, row alignment, coalescing, overflow — pass\n");
    return 0;
}

static void usage(const char *p)
{
    fprintf(stderr,
"Usage: %s [--max-isrs N] [--shape NxK] [--seed N]\n"
"\n"
"Acceptance test for the runtime, on real hardware.  Checks that PIM_EXACT is\n"
"bit identical to the host reference, that PIM_FAST is within one BF16 ulp, and\n"
"that padded shapes behave like unpadded ones.\n"
"\n"
"  --max-isrs N   cap a single program (default 2000).  A small value forces\n"
"                 pim_gemv to split one GEMV across several doorbells, which is\n"
"                 the path a large model takes — worth exercising deliberately.\n"
"  --shape NxK    run one shape instead of the built-in list\n"
"  --seed N       operand seed\n"
"  --ill          all-positive, same-magnitude operands.  Makes a sequential\n"
"                 chain's fp32 error large and a tree's small, which is the only\n"
"                 way the accumulator's width becomes observable.\n"
"  --wide         mixed-sign operands spanning 2^-10..2^10.  With --shape Nx16 the\n"
"                 output IS one beat reduction, so this shows the tree wiring.\n"
"\n"
"Exit: 0 pass, 1 fail, 2 usage\n", p);
}

int main(int argc, char **argv)
{
    pim_config cfg = { 0 };
    struct shape one = { 0, 0, "as asked" };
    bool single = false;

    static const struct option lo[] = {
        { "max-isrs", required_argument, NULL, 1 },
        { "shape",    required_argument, NULL, 2 },
        { "seed",     required_argument, NULL, 3 },
        { "ill",      no_argument,       NULL, 4 },
        { "wide",     no_argument,       NULL, 5 },
        { "help",     no_argument,       NULL, 'h' }, { 0, 0, 0, 0 }
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 1: cfg.max_isrs = (uint32_t)strtoul(optarg, NULL, 0); break;
        case 2:
            if (!strcmp(optarg, "multi")) { one.n = 1; one.k = 1; single = true; break; }
            if (sscanf(optarg, "%ux%u", &one.n, &one.k) != 2 || !one.n || !one.k) {
                fprintf(stderr, "--shape wants NxK, e.g. 2048x2048\n"); return 2;
            }
            single = true; break;
        case 3: rs = (uint32_t)strtoul(optarg, NULL, 0); if (!rs) rs = 1; break;
        case 4: g_op = OP_ILL; break;
        case 5: g_op = OP_WIDE; break;
        case 'h': usage(argv[0]); return 0;
        default: usage(argv[0]); return 2;
        }
    }

    pim_dev *d;
    const char *e = pim_open(&cfg, &d);
    if (e) { fprintf(stderr, "pim_open: %s\n", e); return 1; }

    printf("runtime — acceptance test\n");
    {
        // Capacity is a property of the image, not a constant — ask the device.
        uint64_t used, freeb, largest;
        pim_meminfo(d, &used, &freeb, &largest);
        printf("  device space %.2f GiB per channel, in %u B rows\n",
               (double)(used + freeb) / 1073741824.0, PIM_ROBACO_ROW);
    }
    if (cfg.max_isrs) printf("  program cap %u ISRs — GEMVs larger than this split\n",
                             cfg.max_isrs);

    unsigned fails = 0, ran = 0;
    if (run_alloc(d)) fails++;
    ran++;

    const struct shape *list = single ? &one : SHAPES;
    const unsigned count = (single && one.n == 1) ? 0 : (single ? 1 : NSHAPES);
    for (unsigned i = 0; i < count; i++) {
        printf("\n  [%u][%u] — %s\n", list[i].n, list[i].k, list[i].why);
        if (run_shape(d, &list[i], true)) fails++;
        ran++;
    }
    if (!single) {
        if (run_bias(d, 100, 896))  fails++;            // k % 1024 != 0: one more beat
        ran++;
        if (run_bias(d, 256, 2048)) fails++;            // k % 1024 == 0: a whole chunk
        ran++;
    }
    if (!single) {
        if (run_attn(d, 4, 64, 512, 64)) fails++;       // clean: S % flush_gran == 0
        ran++;
        if (run_attn(d, 4, 64, 300, 64)) fails++;       // leaves a live tail on the host
        ran++;
    }
    if (!single || one.n == 1) { if (run_multi(d, true)) fails++; ran++; }

    printf("\n  which host model is the device?  (PIM_EXACT, %"PRIu64" outputs)\n",
           g_scan.n);
    printf("    fp32 all the way            %"PRIu64" wrong\n", g_scan.bad_f32);
    printf("    f64 all the way             %"PRIu64" wrong\n", g_scan.bad_f64);
    printf("    fp32 per beat, f64 across   %"PRIu64" wrong\n", g_scan.bad_hyb);
    printf("    -- fp32 accumulator, beat reduced by:\n");
    for (int m = 0; m < RED_N; m++)
        printf("       %-34s %"PRIu64" wrong%s\n", RED_NAME[m], g_scan.bad[m],
               g_scan.bad[m] ? "" : "   <-- consistent");

    pim_stats st;
    pim_get_stats(d, &st);
    printf("\n  %"PRIu64" launches, %"PRIu64" ISRs, %"PRIu64" us on the device\n",
           st.launches, st.isrs, st.device_us);
    printf("  H2C %.1f MiB in %"PRIu64" us (%.0f MB/s)   C2H %.2f MiB in %"PRIu64" us (%.0f MB/s)\n",
           (double)st.h2c_bytes / 1048576.0, st.h2c_us,
           st.h2c_us ? (double)st.h2c_bytes / st.h2c_us : 0.0,
           (double)st.c2h_bytes / 1048576.0, st.c2h_us,
           st.c2h_us ? (double)st.c2h_bytes / st.c2h_us : 0.0);

    printf("\n================================================================\n");
    printf("%u of %u cases passed\n%s\n", ran - fails, ran, fails ? "FAIL" : "PASS");
    pim_close(d);
    return fails ? 1 : 0;
}
