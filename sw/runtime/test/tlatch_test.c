// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// tlatch_test — one vector load per output group, or one per PAIR of groups.
//
//     ./tlatch_test [--groups N] [--k N] [--reps N] [--bdf BDF]
//
// WHAT THE SECOND ACCUMULATOR LATCH (ISR[35], "T") IS ACTUALLY FOR.
//
// A GB-sourced MAC rewinds the GB read pointer, so one WRVEC can feed many MACs
// [measured, emu_chain --test rewind].  That is free while K fits one DRAM page.
// It stops being free the moment K exceeds 1024: each output group must then chain
// its K-chunks in the accumulator, and RD_MAC is read-CLEARING, so a second group
// cannot be interleaved without destroying the first group's running sum.  With one
// latch the only exact schedule reloads the whole vector once PER GROUP.
//
// With two, the vector is loaded once and the groups differ only in ROW and T:
//
//     SINGLE                          DUAL
//     WRVEC(chunk 0)                  WRVEC(chunk 0)
//     MAC(group A, chunk 0)           MAC(group A, chunk 0, T=0)
//     WRVEC(chunk 1)                  MAC(group B, chunk 0, T=1)
//     MAC(group A, chunk 1)           WRVEC(chunk 1)
//     RD_MAC(A)                       MAC(group A, chunk 1, T=0)
//     WRVEC(chunk 0)   <-- again      MAC(group B, chunk 1, T=1)
//     MAC(group B, chunk 0)           RD_MAC(A), RD_MAC(B)
//     WRVEC(chunk 1)   <-- again
//     MAC(group B, chunk 1)
//     RD_MAC(B)
//
// BOTH ARE EXACT.  Every output still accumulates all of its chunks in one latch and
// is rounded once at its RD_MAC, so this is not a speed-for-accuracy trade — the two
// schedules must agree bit for bit with each other AND with the host reference.  The
// only thing that differs is how many times the vector crosses into the GB.
//
// AND THAT IS WHAT IS MEASURED: WRVEC count, and time.  A 64-beat WRVEC costs more
// than a MAC, so on a real GEMV the vector loads are most of the program.
//
// Exit: 0 pass   1 something disagreed
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pim/pim.h>
#include <pim/pim_exec.h>
#include <pim/pim_gemv.h>
#include <pim/pim_addr.h>

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }

static uint32_t rng = 424242u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

static unsigned differ(const uint16_t *a, const uint16_t *b, uint32_t n)
{ unsigned d = 0; for (uint32_t i = 0; i < n; i++) if (a[i] != b[i]) d++; return d; }

int main(int argc, char **argv)
{
    const char *bdf = NULL;
    uint32_t groups = 4, k = 2048, reps = 20;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--groups") && i+1 < argc) groups = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--k")      && i+1 < argc) k      = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--reps")   && i+1 < argc) reps   = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--bdf")    && i+1 < argc) bdf    = argv[++i];
        else { printf("usage: %s [--groups N] [--k N] [--reps N] [--bdf BDF]\n", argv[0]); return 2; }
    }

    pim_ctx *c; pim_exec *e; const char *bad;
    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }

    // allow_t_latch: this program is the evidence, so it is the one caller entitled
    // to set it.  Everything else gets pim_prog_verify's refusal.
    pim_exec_config ec = { .bdf = bdf, .set_timing = true, .allow_t_latch = true };
    if ((bad = pim_exec_open(c, &ec, &e))) { printf("SKIPPED: %s\n", bad); pim_close(c); return 0; }

    const pim_geometry *g = pim_geom_ctx(c);
    uint32_t n = g->nbank * g->nch * groups;
    pim_gemv_w w;
    if ((bad = pim_gemv_alloc(c, n, k, &w))) { printf("alloc: %s\n", bad); return 1; }

    printf("%u ch x %u bank -> %u outputs per supergroup\n", g->nch, g->nbank,
           g->nbank * g->nch);
    printf("n=%u (%u supergroups), k=%u (%u chunks), %u reps\n\n",
           n, w.ngroups, k, w.nchunks, reps);

    uint16_t *W  = malloc((size_t)n * k * 2), *x = malloc((size_t)k * 2);
    uint16_t *y1 = malloc((size_t)n * 2), *y2 = malloc((size_t)n * 2);
    uint16_t *gold = malloc((size_t)n * 2);
    void *xg = pim_alloc_ctx(c, (size_t)w.kpad * 2, PIM_MEM_GPR);
    void *yg = pim_alloc_ctx(c, (size_t)w.ngroups * g->nch * 32, PIM_MEM_GPR);
    if (!W || !x || !y1 || !y2 || !gold || !xg || !yg) {
        printf("allocation: %s\n", pim_last_error_ctx(c)); return 1; }

    for (size_t i = 0; i < (size_t)n * k; i++) W[i] = rnd_bf16();
    for (size_t i = 0; i < k; i++)             x[i] = rnd_bf16();
    if ((bad = pim_gemv_upload(c, &w, W))) { printf("upload: %s\n", bad); return 1; }
    pim_gemv_golden(W, n, k, x, gold);

    uint32_t sw, snw; pim_addr_gpr_words_ctx(c, yg, g->nch * 32, &sw, &snw);
    pim_exec_clear_violations(e);
    if ((bad = pim_exec_scrub(e, sw))) { printf("scrub: %s\n", bad); return 1; }

    struct { const char *name; pim_acc_mode mode; uint16_t *y; uint64_t us;
             pim_gemv_stat st; } r[] = {
        { "SINGLE (1 latch) ", PIM_ACC_SINGLE, y1, 0, {0} },
        { "DUAL   (2 latch) ", PIM_ACC_DUAL,   y2, 0, {0} },
    };

    printf("  %-18s %5s %7s %11s   %s\n", "", "ISRs", "WRVECs", "us/launch", "vs golden");
    for (unsigned i = 0; i < 2; i++) {
        for (uint32_t rep = 0; rep < reps; rep++) {
            if ((bad = pim_gemv_ex(c, e, &w, x, xg, yg, r[i].y, r[i].mode, &r[i].st)))
                { printf("  %s: %s\n", r[i].name, bad); return 1; }
            r[i].us += r[i].st.launch_us;
        }
        unsigned d = differ(r[i].y, gold, n);
        printf("  %-18s %5u %7u %11.1f   %u/%u differ\n", r[i].name,
               r[i].st.nisr, r[i].st.nwrvec, (double)r[i].us / reps, d, n);
        CHECK(d == 0, "%s is not bit-exact against the host reference", r[i].name);
    }

    // The two schedules compute the same thing by different routes, so they must
    // agree with EACH OTHER too — a stronger statement than each matching the host,
    // and the one that would catch a latch that quietly aliases.
    CHECK(differ(y1, y2, n) == 0, "the two schedules disagree with each other on %u "
          "of %u outputs", differ(y1, y2, n), n);

    {
        double saved = 100.0 * (1.0 - (double)r[1].st.nwrvec / r[0].st.nwrvec);
        double faster = 100.0 * (1.0 - (r[1].us / (double)reps) / (r[0].us / (double)reps));
        printf("\n  vector loads: %u -> %u  (%.0f%% fewer)\n",
               r[0].st.nwrvec, r[1].st.nwrvec, saved);
        printf("  launch time : %.1f -> %.1f us  (%.1f%% %s)\n",
               r[0].us / (double)reps, r[1].us / (double)reps,
               faster < 0 ? -faster : faster, faster < 0 ? "slower" : "faster");
        if (w.nchunks == 1)
            printf("\n  NOTE: k=%u is one chunk, where a single latch can already share\n"
                   "        one WRVEC across every group (WRVEC | MAC | RD_MAC | MAC ...).\n"
                   "        The saving only exists when K spans several chunks — try --k 4096.\n", k);
        if (w.ngroups == 1)
            printf("\n  NOTE: one supergroup, so DUAL has no second group to pair with\n"
                   "        and degenerates to SINGLE.  Try --groups 4.\n");
    }

    for (uint32_t ch = 0; ch < g->nch; ch++) {
        pim_viol v;
        if (!pim_exec_violation_detail(e, ch, &v))
            printf("\nch%u violations: RCD_RD %u, RECOVERY_WR %u, CCD_WR %u",
                   ch, v.rcd_rd, v.recovery_wr, v.ccd_wr);
        CHECK(!v.ccd_wr, "ch%u raised CCD_WR, which nothing here should", ch);
    }
    printf("\n\n%s\n", fail ? "FAIL" : "all checks passed");

    pim_free_ctx(c, yg); pim_free_ctx(c, xg); pim_gemv_free(c, &w);
    pim_exec_close(e); pim_close(c);
    return fail != 0;
}
