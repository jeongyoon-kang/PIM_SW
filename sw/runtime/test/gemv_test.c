// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// gemv_test — one GEMV tile, end to end, on the stack in sw/.
//
//     ./gemv_test [--n N] [--k N] [--dual] [--max-isrs N] [--bdf BDF]
//
//     allocate weights (DRAM) and a vector + result buffer (GPR)
//     -> permute the matrix on the host, one transfer through the MC
//     -> tile the vector into the GPR
//     -> build a program, load IMEM, ring the doorbell
//     -> read the result words back and check every lane against a host reference
//
// TWO PHASES, AND THE FIRST ONE NEEDS NOTHING.
//   phase 0  the placement arithmetic, on a geometry built by hand.  No board, no
//            module.  This is the part most likely to be wrong and the part a board
//            checks least clearly — a mis-permuted weight comes back as a wrong
//            number in one lane, which is a terrible way to learn about it.
//   phase 1  the real thing.  Skipped, loudly, if /dev/pim or the card is absent.
//
// WHY THE COMPARISON IS BIT-EXACT AND THE OPERANDS ARE RANDOM.
//   The device accumulates in fp32 strictly in lane order and rounds once, at
//   RD_MAC [measured 2026-08-10, emu_chain --test order/acc].  pim_gemv_golden()
//   is that same order.  So there is no tolerance to argue about: any disagreement
//   is a defect, and random distinct weights make every mis-placement change the
//   answer.  "Nice" operands would only have hidden which lane went wrong.
//
// Exit: 0 pass (or phase 1 skipped)   1 fail
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pim/pim.h>
#include <pimrt/pim_exec.h>
#include <pimrt/pim_gemv.h>
#include <pim/pim_addr.h>

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

// ---- BF16, restated here so the test does not lean on the thing under test ----
static uint16_t f2b(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }
static float    b2f(uint16_t h){ uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }

static uint32_t rng = 12345u;
static uint16_t rnd_bf16(void)
{
    rng = rng * 1103515245u + 12345u;
    // Small magnitudes, so a k = 2048 dot product cannot reach fp32's exponent
    // range and the only rounding in play is the single one at the end.
    return f2b((float)((int)((rng >> 16) & 0xFF) - 128) / 64.0f);
}

// ================================ phase 0 ====================================
// A geometry built by hand, so the placement can be walked without a driver.
static pim_geometry mk(uint32_t nch)
{
    pim_geometry g = {0};
    g.nch = nch; g.nbank = 16; g.row_bytes = 2048;
    g.map = PIM_MAP_RO_CH_BA_CO;
    g.unit_bytes = (uint64_t)nch * 16 * 2048;
    g.mem[PIM_MEM_DRAM].base        = 0x020400000000ull;
    g.mem[PIM_MEM_DRAM].bytes       = (uint64_t)nch * 0x100000000ull;
    g.mem[PIM_MEM_DRAM].hugepage_bytes = 2u << 20;
    g.mem[PIM_MEM_DRAM].nr_hugepages   = g.mem[PIM_MEM_DRAM].bytes / (2u << 20);
    g.mem[PIM_MEM_GPR].base         = 0x020200000000ull;
    g.mem[PIM_MEM_GPR].bytes        = 4u << 20;
    g.mem[PIM_MEM_GPR].hugepage_bytes  = 4096;
    g.mem[PIM_MEM_GPR].nr_hugepages    = (4u << 20) / 4096;
    pim_geom_derive(&g);
    return g;
}

static pim_tensor mkw(const pim_geometry *g, uint32_t groups, uint32_t k)
{
    pim_tensor w = {0};
    w.nout = g->nbank * g->nch * groups;
    w.noutpad = w.nout;
    w.nred = k;
    w.nredpad = (k + 15) / 16 * 16;
    w.nchunks = (w.nredpad + 1023) / 1024;
    w.last_beats = (w.nredpad - (w.nchunks - 1) * 1024) / 16;
    w.ngroups = groups;
    w.nch = g->nch;
    w.nbank = g->nbank;
    return w;
}

static void phase0(uint32_t nch, uint32_t groups, uint32_t k)
{
    pim_geometry g = mk(nch);
    pim_tensor  ww = mkw(&g, groups, k);
    uint64_t span = (uint64_t)ww.ngroups * ww.nchunks * g.unit_bytes;
    uint64_t collisions = 0, misdecoded = 0, overrun = 0;
    unsigned char *hit = calloc(span / 2, 1);   // one byte per BF16 slot

    if (!hit) { printf("  (out of memory)\n"); fail++; return; }

    for (uint32_t j = 0; j < ww.nout; j++)
        for (uint32_t i = 0; i < k; i++) {
            uint64_t off = pim_tensor_offset(&g, &ww, j, i);
            pim_coord co;

            if (off + 2 > span) { overrun++; continue; }
            if (hit[off / 2]++) collisions++;

            // Decoded INDEPENDENTLY, by the same function the runtime will use to
            // fill the ISR's ROW field.  If the placement and the decode disagree,
            // the weights are somewhere the hardware will not look.  The row must
            // carry the SUPERGROUP as well as the chunk — that is the part the
            // single-tile version could not check at all.
            pim_geom_decode(&g, g.mem[PIM_MEM_DRAM].base + off, &co);
            if (co.bank     != j % 16                                       ||
                co.ch       != (j / 16) % nch                               ||
                co.row      != (j / (16 * nch)) * ww.nchunks + i / 1024     ||
                co.col_byte != (i % 1024) / 16 * 32 + (i % 16) * 2)
                misdecoded++;
        }

    printf("  ch%u g%-2u k=%-5u  n=%-4u units=%-3u span=%-9"PRIu64" "
           "collisions %"PRIu64", mis-decoded %"PRIu64", overrun %"PRIu64"\n",
           nch, groups, k, ww.nout, ww.ngroups * ww.nchunks, span,
           collisions, misdecoded, overrun);
    CHECK(!collisions, "two weights share a byte");
    CHECK(!misdecoded, "a weight decodes to the wrong (row, ch, bank, col)");
    CHECK(!overrun,    "a weight lands past the end of the allocation");
    free(hit);
}

static void phase0_golden(void)
{
    // W = [[1,1,1,1],[2,2,2,2]], x = [1,2,3,4] -> y = [10, 20], exact in BF16.
    uint16_t W[8], x[4], y[2];
    for (int i = 0; i < 4; i++) { W[i] = f2b(1.0f); W[4 + i] = f2b(2.0f); x[i] = f2b((float)(i + 1)); }
    pim_gemv_golden(W, 2, 4, x, y);
    printf("  golden [[1..],[2..]] . [1,2,3,4] = %g, %g\n", b2f(y[0]), b2f(y[1]));
    CHECK(b2f(y[0]) == 10.0f && b2f(y[1]) == 20.0f, "the host reference is wrong");

    // THE CASE THAT SEPARATES THIS REFERENCE FROM AN fp32 SUM.  A beat is block
    // floating point: every lane is right-shifted to the beat's MAX exponent before
    // anything is added, and a shift of 24 or more kills the lane outright.  So
    // [+1, -1, 2^-30] is exactly zero on silicon — the eps is annihilated during
    // ALIGNMENT, before the cancelling pair ever meets.  An fp32 loop returns eps.
    //
    // If this ever prints nonzero, the reference has been replaced by a float model
    // and every "bit-exact" claim in this file is empty.
    {
        uint16_t Wc[16] = {0}, xc[16] = {0}, y1;
        Wc[0] = Wc[1] = Wc[2] = f2b(1.0f);
        xc[0] = f2b(1.0f); xc[1] = f2b(-1.0f); xc[2] = f2b(0x1p-30f);
        y1 = pim_mac_exact(Wc, xc, 16);
        printf("  block-FP: [+1, -1, 2^-30] -> %04x (%g)   [an fp32 sum gives %g]\n",
               y1, b2f(y1), (double)0x1p-30f);
        CHECK((y1 & 0x7FFFu) == 0,
              "eps survived alignment — the reference is an fp32 sum, not the RTL");
    }
}

// ---- phase 0b: the program the launch would run, without a launch ----------
//
// THE ISR ENCODING IS WHERE A MISTAKE IS A WRONG NUMBER.  The validity gate was
// pulled out of the fetch path for timing on 2026-07-29, so a program with a field
// in the wrong place still decodes to a plausible instruction and executes.  A
// board tells you about that as one wrong lane, hours later.  This tells you now.
static void phase0_program(uint32_t nch, uint32_t groups, uint32_t k,
                           pim_acc_mode mode)
{
    pim_geometry g = mk(nch);
    pim_tensor   w = mkw(&g, groups, k);
    uint32_t    *rows;
    pim_isr     *storage;
    pim_prog     prog;
    const char  *bad;
    uint32_t     xword = 1000, yword = 5000;
    uint32_t     nunits = w.ngroups * w.nchunks;
    uint32_t     want_isr = pim_gemv_nisr(&w, w.ngroups, mode);
    uint32_t     nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;

    rows    = malloc(nunits * sizeof *rows);
    storage = malloc(want_isr * sizeof *storage);
    if (!rows || !storage) { fail++; return; }
    for (uint32_t i = 0; i < nunits; i++) rows[i] = 40 + i * 7;   /* scattered */

    pim_prog_init(&prog, storage, want_isr);
    bad = pim_gemv_program(&g, &w, 0, w.ngroups, xword, yword, rows, mode, &prog);
    CHECK(!bad, "ch%u g%u k=%u: %s", nch, groups, k, bad ? bad : "");
    if (bad) { free(rows); free(storage); return; }
    CHECK(prog.n == want_isr, "%u ISRs built, %u predicted", prog.n, want_isr);

    // CHECK PROPERTIES, NOT A SEQUENCE.  A walker that tracks "which chunk am I on"
    // has to mirror the builder's loop structure, and then it stops being an
    // independent check — it agrees with the builder because it IS the builder.
    // These are statements about the finished program instead:
    //
    //   every unit's DRAM row appears as a MAC row EXACTLY ONCE, with the T its
    //     supergroup implies and the OPSIZE its chunk implies;
    //   every result word appears as an RD_MAC row EXACTLY ONCE, 1-hot on its
    //     channel and with its supergroup's T;
    //   every MAC's OPSIZE matches the WRVEC that last filled the GB.
    //
    // Between them those pin the whole schedule, including the part that matters
    // here: a MAC whose ROW belongs to group B must carry B's latch.
    uint32_t nwrvec = 0, nmac = 0, nrd = 0, neos = 0, wrong = 0, gb_op = 0;
    uint8_t *mac_seen = calloc(nunits, 1), *rd_seen = calloc(w.ngroups * nch, 1);
    if (!mac_seen || !rd_seen) { fail++; free(rows); free(storage); return; }

    for (uint32_t i = 0; i < prog.n; i++) {
        pim_isr_info d;
        pim_isr_decode(&prog.w[i], &d);

        if (d.opcode == PIM_OP_WRVEC) {
            if (d.ch_mask != (1u << nch) - 1u || d.t) wrong++;
            if (d.row < xword || (d.row - xword) % 64) wrong++;
            gb_op = d.opsize;
            nwrvec++;
        } else if (d.opcode == PIM_OP_MAC) {
            uint32_t u = nunits;
            for (uint32_t q = 0; q < nunits; q++) if (rows[q] == d.row) { u = q; break; }
            if (u == nunits) { wrong++; nmac++; continue; }   /* a row nothing owns */
            uint32_t grp = u / w.nchunks, ck2 = u % w.nchunks;
            uint32_t want = (ck2 == w.nchunks - 1) ? w.last_beats : 64;
            if (mac_seen[u]++)                       wrong++;   /* twice */
            if (d.t != grp % nlatch)                 wrong++;   /* wrong latch */
            if (d.opsize != want || d.opsize != gb_op) wrong++;
            if (d.col || d.pu_mask != 0xFFFFu || d.gb_mc_mask != 0xFFFFu ||
                d.ch_mask != (1u << nch) - 1u)       wrong++;
            nmac++;
        } else if (d.opcode == PIM_OP_RD_MAC) {
            uint32_t idx = d.row - yword;
            if (d.row < yword || idx >= w.ngroups * nch) { wrong++; nrd++; continue; }
            uint32_t grp = idx / nch, ch = idx % nch;
            if (rd_seen[idx]++)                      wrong++;
            if (d.t != grp % nlatch)                 wrong++;
            if (d.opsize || d.pu_mask || d.gb_mc_mask ||
                d.ch_mask != (1u << ch))             wrong++;
            nrd++;
        } else if (d.opcode == PIM_OP_EOS) neos++;
        else wrong++;
    }
    for (uint32_t u = 0; u < nunits; u++)          if (!mac_seen[u]) wrong++;
    for (uint32_t u = 0; u < w.ngroups * nch; u++) if (!rd_seen[u])  wrong++;
    free(mac_seen); free(rd_seen);

    printf("  ch%u g%-2u k=%-5u %-6s  %3u ISAs  %2u WRVEC  %2u MAC  %u EOS  "
           "-> %u field(s) wrong\n", nch, groups, k,
           mode == PIM_ACC_DUAL ? "DUAL" : "SINGLE",
           prog.n, nwrvec, nmac, neos, wrong);
    CHECK(!wrong, "%u ISR field(s) decoded wrong", wrong);
    CHECK(nwrvec == pim_gemv_nwrvec(&w, w.ngroups, mode), "%u WRVECs, %u predicted",
          nwrvec, pim_gemv_nwrvec(&w, w.ngroups, mode));
    CHECK(neos == 1, "the program does not end in exactly one EOS");
    CHECK(!(bad = pim_prog_verify(&prog, PIM_VERIFY_ALLOW_T)),
          "the verifier rejected a good program: %s", bad ? bad : "");
    free(rows); free(storage);
}

// The verifier has to REFUSE things, and a verifier nobody has seen refuse anything
// is indistinguishable from `return NULL`.
static void phase0_verifier(void)
{
    pim_geometry g = mk(2);
    pim_tensor   w = mkw(&g, 1, 2048);
    uint32_t rows[2] = { 40, 41 };
    pim_isr  storage[64];
    pim_prog prog;
    const char *bad;

    struct { const char *what; int drop; } cases[] = {
        { "a program ending in RD_MAC (no EOS)", 1 },
        { "a MAC never drained by a RD_MAC",     3 },   /* drops EOS + both RD_MACs */
    };

    for (unsigned t = 0; t < sizeof cases / sizeof *cases; t++) {
        pim_prog_init(&prog, storage, sizeof storage / sizeof *storage);
        if (pim_gemv_program(&g, &w, 0, w.ngroups, 1000, 5000, rows, PIM_ACC_SINGLE, &prog)) { fail++; return; }
        prog.n -= cases[t].drop;
        bad = pim_prog_verify(&prog, 0);
        printf("  %-38s -> %s\n", cases[t].what, bad ? "refused" : "ACCEPTED (BUG)");
        if (bad) printf("     \"%.70s...\"\n", bad);
        CHECK(bad != NULL, "the verifier accepted %s", cases[t].what);
    }

    // Two RD_MACs on the same GPR word: one result is simply lost.
    pim_prog_init(&prog, storage, sizeof storage / sizeof *storage);
    if (pim_gemv_program(&g, &w, 0, w.ngroups, 1000, 5000, rows, PIM_ACC_SINGLE, &prog)) { fail++; return; }
    prog.w[prog.n - 2] = prog.w[prog.n - 3];        /* clone the first RD_MAC */
    bad = pim_prog_verify(&prog, 0);
    printf("  %-38s -> %s\n", "two RD_MACs on one GPR word",
           bad ? "refused" : "ACCEPTED (BUG)");
    if (bad) printf("     \"%.70s...\"\n", bad);
    CHECK(bad != NULL, "the verifier accepted a duplicated RD_MAC word");

    // A MAC whose OPSIZE does not match the WRVEC that filled the GB.  Rebuild with
    // a short last chunk and then lengthen its MAC.
    {
        pim_tensor s = w;
        s = mkw(&g, 1, 1500);
        pim_prog_init(&prog, storage, sizeof storage / sizeof *storage);
        if (pim_gemv_program(&g, &s, 0, s.ngroups, 1000, 5000, rows, PIM_ACC_SINGLE, &prog)) { fail++; return; }
        // ISR 2 is the second WRVEC, ISR 3 its MAC.  Swap in the first MAC, whose
        // OPSIZE is 64 rather than 30.
        prog.w[3] = prog.w[1];
        bad = pim_prog_verify(&prog, 0);
        printf("  %-38s -> %s\n", "MAC OPSIZE != the WRVEC that filled GB",
               bad ? "refused" : "ACCEPTED (BUG)");
        if (bad) printf("     \"%.70s...\"\n", bad);
        CHECK(bad != NULL, "the verifier accepted a MAC/WRVEC OPSIZE mismatch");
    }
}

// ================================ phase 1 ====================================
static int run_one(pim_ctx *c, pim_exec *e, uint32_t n, uint32_t k,
                   pim_acc_mode mode)
{
    pim_tensor w;
    size_t xb, yb;
    uint16_t *W = NULL, *x = NULL, *y = NULL, *gold = NULL;
    void *xg = NULL, *yg = NULL;
    const char *bad;
    uint64_t nbad = 0;
    unsigned wrong = 0;
    int rc = 1;

    if ((bad = pim_gemv_alloc(c, n, k, &w))) { printf("  alloc: %s\n", bad); return 1; }
    // The caller no longer derives these; getting nredpad*2 or groups*nch*32 wrong
    // reads back as someone else's data rather than as an error.
    pim_gemv_gpr_bytes(&w, mode, pim_exec_max_isrs(e), &xb, &yb);

    W    = malloc((size_t)n * k * 2);
    x    = malloc((size_t)k * 2);
    y    = malloc((size_t)n * 2);
    gold = malloc((size_t)n * 2);
    xg   = pim_alloc_ctx(c, xb, PIM_MEM_GPR);
    yg   = pim_alloc_ctx(c, yb, PIM_MEM_GPR);
    if (!W || !x || !y || !gold || !xg || !yg) {
        printf("  allocation failed: %s\n", pim_last_error_ctx(c)); goto out;
    }
    for (size_t i = 0; i < (size_t)n * k; i++) W[i] = rnd_bf16();
    for (size_t i = 0; i < k; i++)             x[i] = rnd_bf16();

    if ((bad = pim_gemv_upload(c, &w, W)))     { printf("  upload: %s\n", bad); goto out; }
    if ((bad = pim_gemv_verify(c, &w, W, &nbad))) { printf("  readback: %s\n", bad); goto out; }
    pim_gemv_stat st;
    if ((bad = pim_gemv_ex(c, e, &w, x, xg, yg, y, mode, &st)))
        { printf("  gemv: %s\n", bad); goto out; }

    pim_gemv_golden(W, n, k, x, gold);
    for (uint32_t j = 0; j < n; j++)
        if (y[j] != gold[j]) {
            if (wrong < 3)
                printf("    out %-5u (group %u, ch %u, bank %2u): device %04x (%g)  "
                       "host %04x (%g)\n", j, j / (w.nbank * w.nch),
                       (j / w.nbank) % w.nch, j % w.nbank,
                       y[j], b2f(y[j]), gold[j], b2f(gold[j]));
            wrong++;
        }
    printf("  n=%-6u k=%-5u -> pad %ux%u, %u grp x %u chunk | %4u ISA %4u WRVEC "
           "%2u launch %7.1f us | %u/%u exact\n",
           n, k, w.noutpad, w.nredpad, w.ngroups, w.nchunks, st.nisr, st.nwrvec,
           st.nlaunch, (double)st.launch_us, n - wrong, n);
    if (wrong) fail++;
    rc = wrong ? 1 : 0;

out:
    if (yg) pim_free_ctx(c, yg);
    if (xg) pim_free_ctx(c, xg);
    free(gold); free(y); free(x); free(W);
    pim_tensor_free(c, &w);
    return rc;
}

int main(int argc, char **argv)
{
    const char *bdf = NULL;
    uint32_t opt_n = 0, opt_k = 0, max_isrs = 0;
    pim_acc_mode mode = PIM_ACC_SINGLE;
    pim_ctx  *c = NULL;
    pim_exec *e = NULL;
    const char *bad;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--n")        && i+1 < argc) opt_n    = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--k")        && i+1 < argc) opt_k    = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--max-isrs") && i+1 < argc) max_isrs = (uint32_t)strtoul(argv[++i],0,0);
        else if (!strcmp(argv[i], "--bdf")      && i+1 < argc) bdf      = argv[++i];
        else if (!strcmp(argv[i], "--dual"))                   mode     = PIM_ACC_DUAL;
        else { printf("usage: %s [--n N] [--k N] [--dual] [--max-isrs N] [--bdf BDF]\n",
                      argv[0]); return 2; }
    }

    printf("phase 0 — placement arithmetic (no board, no module)\n");
    for (uint32_t nch = 1; nch <= 4; nch <<= 1) {
        phase0(nch, 1, 1024);    /* one full chunk                       */
        phase0(nch, 1, 2048);    /* two chunks, both full                */
        phase0(nch, 1, 1500);    /* two chunks, the last short + padding */
        phase0(nch, 1, 16);      /* one beat                             */
        phase0(nch, 3, 2048);    /* several supergroups                  */
    }
    phase0_golden();

    printf("\nphase 0b — the program, encoded and decoded (still no board)\n");
    for (uint32_t nch = 1; nch <= 4; nch <<= 1)
        for (uint32_t gr = 1; gr <= 3; gr += 2) {
            phase0_program(nch, gr, 1024, PIM_ACC_SINGLE);
            phase0_program(nch, gr, 2048, PIM_ACC_SINGLE);
            phase0_program(nch, gr, 2048, PIM_ACC_DUAL);
            phase0_program(nch, gr, 1500, PIM_ACC_DUAL);
        }
    printf("\nphase 0c — what the verifier must refuse\n");
    phase0_verifier();
    if (fail) { printf("\nFAIL (phase 0)\n"); return 1; }

    printf("\nphase 1 — on the board\n");
    if ((bad = pim_open(NULL, &c))) {
        printf("  SKIPPED: %s\n", bad);
        printf("\nphase 0 passed; phase 1 needs pim.ko and the card.\n");
        return 0;
    }
    {
        pim_exec_config ec = { .bdf = bdf, .max_isrs = max_isrs,
                               .allow_t_latch = (mode == PIM_ACC_DUAL) };
        if ((bad = pim_exec_open(c, &ec, &e))) {
            printf("  SKIPPED: %s\n", bad);
            pim_close(c);
            printf("\nphase 0 passed; phase 1 needs the control plane.\n");
            return 0;
        }
    }
    {
        const pim_geometry *g = pim_geom_ctx(c);
        printf("  %u ch x %u bank, unit %llu KiB, n = %u outputs per tile\n",
               g->nch, g->nbank, (unsigned long long)(g->unit_bytes >> 10),
               g->nch * g->nbank);
    }
    pim_exec_clear_violations(e);

    // BEFORE ANYTHING ELSE.  The accumulator latch survives across processes, and
    // whatever ran on this board last may have left a partial sum in it — a MAC
    // accumulates INTO what is there, so the first GEMV would silently add its work
    // to someone else's.  One doorbell of RD_MACs clears all 16*nch of them.
    {
        void *sg = pim_alloc_ctx(c, 4096, PIM_MEM_GPR);
        uint32_t word, nw;
        if (!sg) { printf("  scrub buffer: %s\n", pim_last_error_ctx(c)); }
        else if ((bad = pim_addr_gpr_words_ctx(c, sg, pim_geom_ctx(c)->nch * 32, &word, &nw)))
            printf("  scrub: %s\n", bad);
        else if ((bad = pim_exec_scrub(e, word)))
            printf("  scrub: %s\n", bad);
        else
            printf("  accumulators scrubbed (%u channel(s) drained into GPR word %u)\n",
                   pim_geom_ctx(c)->nch, word);
        if (sg) pim_free_ctx(c, sg);
    }

    {
        uint32_t per = pim_geom_ctx(c)->nbank * pim_geom_ctx(c)->nch;
        if (opt_n || opt_k) {
            run_one(c, e, opt_n ? opt_n : per, opt_k ? opt_k : 1024, mode);
        } else {
            // Each shape is the FIRST to exercise something: one beat, one full
            // chunk, chunk chaining, a short tail chunk, several supergroups, an n
            // that needs padding, and the smallest GEMV there is.
            run_one(c, e, per,         16,   mode);
            run_one(c, e, per,         1024, mode);
            run_one(c, e, per,         2048, mode);
            run_one(c, e, per,         1500, mode);
            run_one(c, e, per * 5,     2048, mode);
            run_one(c, e, per * 3 + 7, 1500, mode);
            run_one(c, e, 1,           1024, mode);
        }
    }

    // ROUTINE VERSUS NOT.  RCD_RD is the timing model reporting that the memory was
    // slower than its allowance, and RECOVERY_WR comes from the MC write path — both
    // are expected here and neither touches the answer [measured 2026-08-21; see
    // pim_timing in pim_exec.h].  CCD_WR and ewmul_drop are not expected at all, so
    // they are the ones worth failing on.
    for (uint32_t ch = 0; ch < pim_geom_ctx(c)->nch; ch++) {
        pim_viol v;
        if (pim_exec_violation_detail(e, ch, &v)) continue;
        printf("  ch%u violations: RCD_RD %u (worst overrun %u cy), RECOVERY_WR %u, "
               "CCD_RD %u  [routine]\n", ch, v.rcd_rd, v.worst_rcd_rd,
               v.recovery_wr, v.ccd_rd);
        CHECK(!v.ccd_wr, "ch%u raised %u CCD_WR violation(s), which nothing here "
              "should do", ch, v.ccd_wr);
        CHECK(!(v.sticky & (1u << 8)), "ch%u dropped an EWMUL, which nothing here "
              "should issue", ch);
    }

    pim_exec_close(e);
    pim_close(c);
    printf("\n%s\n", fail ? "FAIL" : "all checks passed");
    return fail != 0;
}
