// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// lower_test.c — the two-pass code generator produces the same program as the
// one-pass one, ISR for ISR, on a simulated card.
//
// WHAT THIS IS EVIDENCE OF.  pim_gemv_program() is the path that has run on the
// board and come back bit-exact (2026-08-21, ch2).  So "pim_gemv_logical() lowered
// equals pim_gemv_program()" transfers that evidence to the new path without
// needing the board — every ISR word is compared, not just the count.
//
// AND IT IS A REAL COMPARISON, not a tautology: the two builders share no code.  One
// resolves up front and writes the RD_MAC-per-channel expansion out by hand; the
// other emits one logical drain, records what its ROW means, and lets
// pim_prog_lower() do the expansion and the resolving.  If the split hint were
// wrong, or a word index off by one, or the fanout mapped to the wrong ISR, the
// words would differ here.
//
// THE ALLOCATIONS ARE REAL — real pim_alloc, real pool, real address arithmetic —
// against the simulated driver in fake_drv.h.  That matters because the whole point
// of the second pass is to ask where things landed, and a fake resolver would be
// asking nothing.
//
//     make run        # needs no board and no module
//////////////////////////////////////////////////////////////////////////////////
#include "fake_drv.h"

#include "pimrt/pim_gemv.h"
#include "pimrt/pim_logical.h"
#include "pimrt/pim_matvec.h"
#include "pimrt/pim_exec.h"
#include "pim/pim_addr.h"

#include "emu_regs.h"

static int fail;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  FAIL: "); printf(__VA_ARGS__); \
                                          printf("\n"); fail++; } }while(0)

// A GEMV shape's weight allocation.  THE TILING COMES FROM pim_tensor_plan(), PIM_LAYOUT_OUT_MAJOR, not from
// a copy of its arithmetic — a test that re-derives what it is testing would agree
// with itself and with nothing else.
static const char *plan(pim_ctx *c, uint32_t n, uint32_t k, pim_tensor *w)
{
    const pim_geometry *g = pim_geom_ctx(c);

    pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, n, k, w);
    w->base = pim_alloc_ctx(c, pim_tensor_bytes(g, w), PIM_MEM_DRAM);
    if (!w->base) return "weight allocation failed";
    // pim_gemv_upload() stamps this after the transfer; there is no transfer here
    // (the fds are -1), so the test stamps it the same way.  Doing it by hand is
    // the point of the refusal check further down.
    return pim_tag_set_ctx(c, w->base, pim_tensor_tag(g, w));
}

// ------------------------------------------------------------------------------
// A growable tensor, grown one step at a time so its units sit in several
// allocations: every MAC of a matvec over it must name the card row of the unit that
// holds its (group, chunk), and the column of the group's slot in that row.
static void grow_mac_case(pim_ctx *c, pim_layout layout, uint32_t width, uint32_t ntok,
                          uint32_t out_count, uint32_t red_off, uint32_t red_len,
                          const char *what)
{
    const pim_geometry *g = pim_geom_ctx(c);
    enum { CAP = 4096 };
    static pim_isr  li[CAP], lo[CAP];
    static pim_ref  lr[CAP];
    static pim_atom la[CAP / 2];
    pim_logical lp;
    pim_prog    prog;
    pim_tensor  t;
    uint32_t    step, beats, nck, nmac = 0, badrow = 0, badcol = 0;
    uint64_t    vtag = 0x7465737476656374ull;       /* any nonzero value */
    size_t      xbytes, ybytes;
    void       *x = NULL, *y = NULL;
    const char *bad;

    pim_tensor_plan_growable(g, layout, width, &t);
    step = (layout == PIM_LAYOUT_RED_MAJOR) ? 1024u : g->nch * g->nbank * t.pack;
    for (uint32_t n = step; n < ntok + step; n += step)
        if ((bad = pim_tensor_grow(c, &t, n))) { CHECK(0, "%s: grow: %s", what, bad); goto out; }
    if (layout != PIM_LAYOUT_RED_MAJOR && !out_count) out_count = pim_tensor_groups(&t, ntok);
    if (layout == PIM_LAYOUT_RED_MAJOR && !out_count) out_count = t.ngroups;

    beats  = pim_matvec_beats(red_off, red_len);
    xbytes = (size_t)beats * 32;
    ybytes = (size_t)out_count * g->nch * 32;
    x = pim_alloc_ctx(c, xbytes, PIM_MEM_GPR);
    y = pim_alloc_ctx(c, ybytes, PIM_MEM_GPR);
    CHECK(x && y, "%s: GPR allocation", what);
    if (!x || !y) goto out;
    pim_tag_set_ctx(c, x, vtag);

    pim_logical_init(&lp, li, CAP, lr, CAP, la, CAP / 2, g->nch, 1);
    if ((bad = pim_matvec_logical(g, &t, 0, out_count, red_off, red_len, x, xbytes, vtag,
                                  y, ybytes, PIM_ACC_SINGLE, &lp))) {
        CHECK(0, "%s: build: %s", what, bad); goto out;
    }
    pim_prog_init(&prog, lo, CAP);
    if ((bad = pim_prog_lower_ctx(c, &lp, &prog, 0))) {
        CHECK(0, "%s: lower: %s", what, bad); goto out;
    }

    // The builder's order: group outer, chunk inner.
    nck = (red_off + red_len + 1023) / 1024 - red_off / 1024;
    for (uint32_t i = 0; i < prog.n; i++) {
        pim_isr_info in;
        uint32_t grp, ck, unit, want_col;
        pim_unit un;

        pim_isr_decode(&prog.w[i], &in);
        if (in.opcode != PIM_OP_MAC) continue;
        grp  = nmac / nck;
        ck   = red_off / 1024 + nmac % nck;
        unit = pim_tensor_unit(&t, grp, ck);
        nmac++;
        if (pim_addr_unit_ctx(c, pim_tensor_unit_addr(g, &t, unit), g->unit_bytes, 0, &un)
            || in.row != un.row)
            badrow++;
        want_col = ((ck == red_off / 1024) ? (red_off % 1024) / 16 : 0)
                 + pim_tensor_row_off(&t, grp) / 16;
        if (in.col != want_col) badcol++;
    }
    CHECK(nmac == out_count * nck, "%s: %u MAC(s), expected %u", what, nmac, out_count * nck);
    CHECK(badrow == 0, "%s: %u of %u MAC(s) name a row that is not their unit's", what,
          badrow, nmac);
    CHECK(badcol == 0, "%s: %u of %u MAC(s) have the wrong column", what, badcol, nmac);
    printf("  %-26s %3u unit(s) in %2u allocation(s): %4u MAC(s), rows %s, columns %s\n",
           what, pim_tensor_nunits(&t), t.npages, nmac, badrow ? "WRONG" : "ok",
           badcol ? "WRONG" : "ok");
out:
    if (x) pim_free_ctx(c, x);
    if (y) pim_free_ctx(c, y);
    pim_tensor_free(c, &t);
}

// ------------------------------------------------------------------------------
// Q.K^T with Q laid out like a row of K: one WRVEC for every head, then per output
// group one MAC per head in row order.  Checked after lowering and verifying: the one
// WRVEC's length, every MAC's row, column, length and latch, and the drain count.
static void heads_case(pim_ctx *c, uint32_t width, uint32_t head_len, uint32_t ntok,
                       pim_acc_mode mode, const char *what)
{
    const pim_geometry *g = pim_geom_ctx(c);
    enum { CAP = 8192 };
    static pim_isr  li[CAP], lo[CAP];
    static pim_ref  lr[CAP];
    static pim_atom la[CAP / 2];
    uint32_t    nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
    uint32_t    nhead = width / head_len, head_beats = head_len / 16;
    uint32_t    out_count, step, nwrvec = 0, wrvec_len = 0, nmac = 0, nrd = 0;
    uint32_t    badrow = 0, badcol = 0, badlen = 0, badt = 0;
    uint64_t    vtag = 0x7465737476656374ull;
    size_t      xbytes, ybytes;
    void       *x = NULL, *y = NULL;
    pim_logical lp;
    pim_prog    prog;
    pim_tensor  t;
    const char *bad;

    pim_tensor_plan_growable(g, PIM_LAYOUT_OUT_PACKED, width, &t);
    step = g->nch * g->nbank * t.pack;
    for (uint32_t n = step; n < ntok + step; n += step)
        if ((bad = pim_tensor_grow(c, &t, n))) { CHECK(0, "%s: grow: %s", what, bad); goto out; }
    out_count = pim_tensor_groups(&t, ntok);

    xbytes = (size_t)width * 2;
    ybytes = (size_t)out_count * nhead * g->nch * 32;
    x = pim_alloc_ctx(c, xbytes, PIM_MEM_GPR);
    y = pim_alloc_ctx(c, ybytes, PIM_MEM_GPR);
    CHECK(x && y, "%s: GPR allocation", what);
    if (!x || !y) goto out;
    pim_tag_set_ctx(c, x, vtag);

    pim_logical_init(&lp, li, CAP, lr, CAP, la, CAP / 2, g->nch, nlatch);
    if ((bad = pim_matvec_heads_logical_part(g, &t, 0, out_count, 0, nhead, head_len,
                                             x, xbytes, vtag, y, ybytes, mode, &lp)) ||
        (bad = pim_matvec_eos(g, &lp))) {
        CHECK(0, "%s: build: %s", what, bad); goto out;
    }
    CHECK(lp.nisr == pim_matvec_heads_nisr(out_count, nhead, mode),
          "%s: %u logical ISRs, pim_matvec_heads_nisr says %u", what, lp.nisr,
          pim_matvec_heads_nisr(out_count, nhead, mode));
    pim_prog_init(&prog, lo, CAP);
    if ((bad = pim_prog_lower_ctx(c, &lp, &prog,
                                  mode == PIM_ACC_DUAL ? PIM_LOWER_ALLOW_T : 0))) {
        CHECK(0, "%s: lower: %s", what, bad); goto out;
    }
    bad = pim_prog_verify(&prog, mode == PIM_ACC_DUAL ? PIM_VERIFY_ALLOW_T : 0);
    CHECK(!bad, "%s: pim_prog_verify: %s", what, bad ? bad : "");

    for (uint32_t i = 0; i < prog.n; i++) {
        pim_isr_info in;
        pim_isr_decode(&prog.w[i], &in);
        if (in.opcode == PIM_OP_WRVEC) { nwrvec++; wrvec_len = in.opsize; }
        if (in.opcode == PIM_OP_RD_MAC) nrd++;
        if (in.opcode == PIM_OP_MAC) {
            uint32_t grp = nmac / nhead, h = nmac % nhead;
            pim_unit un;
            nmac++;
            if (pim_addr_unit_ctx(c, pim_tensor_unit_addr(g, &t, pim_tensor_unit(&t, grp, 0)),
                                  g->unit_bytes, 0, &un) || in.row != un.row)
                badrow++;
            if (in.col != pim_tensor_row_off(&t, grp) / 16 + h * head_beats) badcol++;
            if (in.opsize != head_beats) badlen++;
            if (in.t != h % nlatch) badt++;
        }
    }
    CHECK(nwrvec == 1 && wrvec_len == nhead * head_beats,
          "%s: %u WRVEC(s) of %u beats, expected one of %u", what, nwrvec, wrvec_len,
          nhead * head_beats);
    CHECK(nmac == out_count * nhead, "%s: %u MAC(s), expected %u", what, nmac,
          out_count * nhead);
    CHECK(nrd == out_count * nhead * g->nch, "%s: %u RD_MAC(s), expected %u", what, nrd,
          out_count * nhead * g->nch);
    CHECK(!badrow && !badcol && !badlen && !badt,
          "%s: of %u MACs, %u wrong row, %u wrong column, %u wrong length, %u wrong latch",
          what, nmac, badrow, badcol, badlen, badt);
    printf("  %-34s 1 WRVEC of %2u beats, %4u MAC(s) of %u beats, %4u RD_MAC(s): %s\n",
           what, wrvec_len, nmac, head_beats, nrd,
           (badrow || badcol || badlen || badt) ? "WRONG" : "ok");
out:
    if (x) pim_free_ctx(c, x);
    if (y) pim_free_ctx(c, y);
    pim_tensor_free(c, &t);
}

// ------------------------------------------------------------------------------
// The GB read position (pim_prog_verify, rule 1): MACs may take one vector in
// segments, but not across its end, and the vector must be read in whole passes
// before the next WRVEC and before the program ends.
static void check_gb_wrap(void)
{
    struct { const char *what; int accept; uint32_t n; struct { uint32_t op, len; } s[8]; }
    cases[] = {
        { "two MACs take one vector in halves", 1, 6,
          { { ISR_OP_WRVEC, 8 }, { ISR_OP_MAC, 4 }, { ISR_OP_RD_MAC, 0 },
            { ISR_OP_MAC, 4 }, { ISR_OP_RD_MAC, 0 }, { ISR_OP_EOS, 0 } } },
        { "the program ends half-way through", 0, 4,
          { { ISR_OP_WRVEC, 8 }, { ISR_OP_MAC, 4 }, { ISR_OP_RD_MAC, 0 },
            { ISR_OP_EOS, 0 } } },
        { "a WRVEC half-way through the last", 0, 7,
          { { ISR_OP_WRVEC, 8 }, { ISR_OP_MAC, 4 }, { ISR_OP_RD_MAC, 0 },
            { ISR_OP_WRVEC, 8 }, { ISR_OP_MAC, 8 }, { ISR_OP_RD_MAC, 0 },
            { ISR_OP_EOS, 0 } } },
        { "a MAC across the end of the vector", 0, 6,
          { { ISR_OP_WRVEC, 8 }, { ISR_OP_MAC, 6 }, { ISR_OP_RD_MAC, 0 },
            { ISR_OP_MAC, 6 }, { ISR_OP_RD_MAC, 0 }, { ISR_OP_EOS, 0 } } },
    };

    printf("\nthe GB read position (verify rule 1)\n");
    for (unsigned k = 0; k < sizeof cases / sizeof *cases; k++) {
        pim_isr  w[8];
        pim_prog p;
        uint32_t rdword = 0;
        const char *bad;

        pim_prog_init(&p, w, 8);
        for (uint32_t i = 0; i < cases[k].n; i++) {
            struct emu_isr_spec s = emu_isr_default_ch(cases[k].s[i].op, 1u);
            struct emu_isr      e;
            s.opsize = cases[k].s[i].len;
            if (cases[k].s[i].op == ISR_OP_MAC) {
                s.row = 100; s.pu_mask = 0xFFFF; s.gb_mc_mask = 0xFFFF;
            } else if (cases[k].s[i].op == ISR_OP_RD_MAC) {
                s.row = 1000 + rdword++;
            }
            emu_isr_build(&e, &s);
            pim_prog_push(&p, (const pim_isr *)&e);
        }
        bad = pim_prog_verify(&p, 0);
        CHECK(cases[k].accept ? !bad : bad != NULL, "%s: %s", cases[k].what,
              bad ? bad : "accepted");
        printf("  %-36s -> %s\n", cases[k].what, bad ? "refused" : "accepted");
        if (bad) printf("     \"%.66s...\"\n", bad);
    }
}

// ------------------------------------------------------------------------------
// Build both ways and compare.  `mode` and the group range are the caller's, so the
// hoisted-WRVEC path (nchunks == 1) and the chained one are both reachable.
static void compare(pim_ctx *c, uint32_t n, uint32_t k, pim_acc_mode mode,
                    const char *what)
{
    const pim_geometry *g = pim_geom_ctx(c);
    pim_tensor  w;
    const char *bad;
    void       *xg, *yg;
    size_t      xb, yb;
    uint32_t    nlog, nlow, nold;
    pim_isr    *si, *so, *sl;
    pim_ref    *sr;
    pim_atom   *sa, atoms[64];
    uint32_t   *rows;
    pim_logical lp;
    pim_prog    from_logical, from_direct;
    uint32_t    xword, xn, yword, yn, natom = 0;

    printf("\n%s  n=%u k=%u %s\n", what, n, k,
           mode == PIM_ACC_DUAL ? "DUAL" : "SINGLE");

    if ((bad = plan(c, n, k, &w))) { CHECK(0, "%s", bad); return; }

    // The GPR operands, sized the way pim_gemv_gpr_bytes would for one launch of
    // the whole thing.
    xb = (size_t)w.nredpad * 2;
    yb = (size_t)w.ngroups * g->nch * 32;
    xg = pim_alloc_ctx(c, xb, PIM_MEM_GPR);
    yg = pim_alloc_ctx(c, yb, PIM_MEM_GPR);
    if (!xg || !yg) { CHECK(0, "GPR allocation failed"); return; }
    // pim_gemv_upload_x() stamps this after the transfer; there is none here.
    pim_tag_set_ctx(c, xg, pim_gemv_xtag(&w));

    // ---- pass 1: no addresses are looked at -------------------------------
    nlog = pim_gemv_logical_nisr(&w, w.ngroups, mode);
    si = calloc(nlog, sizeof *si);
    sr = calloc(nlog, sizeof *sr);
    sa = calloc(w.ngroups + 1, sizeof *sa);
    pim_logical_init(&lp, si, nlog, sr, nlog, sa, w.ngroups + 1, g->nch,
                     mode == PIM_ACC_DUAL ? 2u : 1u);
    bad = pim_gemv_logical(g, &w, 0, w.ngroups, xg, xb, yg, yb, mode, &lp);
    CHECK(!bad, "pim_gemv_logical: %s", bad ? bad : "");
    if (bad) return;
    CHECK(lp.nisr == nlog, "logical pushed %u ISRs, predicted %u", lp.nisr, nlog);

    // ---- pass 2: now the addresses matter ---------------------------------
    nlow = pim_lower_isr_count(&lp);
    sl = calloc(nlow, sizeof *sl);
    pim_prog_init(&from_logical, sl, nlow);
    bad = pim_prog_lower_ctx(c, &lp, &from_logical,
                             mode == PIM_ACC_DUAL ? PIM_LOWER_ALLOW_T : 0);
    CHECK(!bad, "pim_prog_lower: %s", bad ? bad : "");
    if (bad) return;

    // ---- the one-pass builder, resolved by hand ---------------------------
    nold = pim_gemv_nisr(&w, w.ngroups, mode);
    so   = calloc(nold, sizeof *so);
    rows = calloc((size_t)w.ngroups * w.nchunks, sizeof *rows);
    for (uint32_t u = 0; u < w.ngroups * w.nchunks; u++) {
        pim_unit un;
        bad = pim_addr_unit_ctx(c, w.base, (size_t)w.ngroups * w.nchunks * g->unit_bytes,
                                u, &un);
        CHECK(!bad, "pim_addr_unit(%u): %s", u, bad ? bad : "");
        if (bad) return;
        rows[u] = (uint32_t)un.row;
    }
    CHECK(!pim_addr_gpr_words_ctx(c, xg, xb, &xword, &xn), "x words");
    CHECK(!pim_addr_gpr_words_ctx(c, yg, yb, &yword, &yn), "y words");

    pim_prog_init(&from_direct, so, nold);
    bad = pim_gemv_program(g, &w, 0, w.ngroups, xword, yword, rows, mode,
                           &from_direct);
    CHECK(!bad, "pim_gemv_program: %s", bad ? bad : "");
    if (bad) return;

    // ---- they must be the same program ------------------------------------
    printf("  logical %u ISR -> lowered %u ISR   |   direct %u ISR\n",
           lp.nisr, from_logical.n, from_direct.n);
    CHECK(from_logical.n == from_direct.n,
          "lowered %u ISRs, the direct builder made %u", from_logical.n,
          from_direct.n);
    if (from_logical.n != from_direct.n) return;

    {
        uint32_t nbad = 0, first_bad = 0;
        for (uint32_t i = 0; i < from_direct.n; i++)
            if (memcmp(&from_logical.w[i], &from_direct.w[i], sizeof(pim_isr))) {
                if (!nbad) first_bad = i;
                nbad++;
            }
        CHECK(nbad == 0, "%u ISR(s) differ, first at %u", nbad, first_bad);
        if (nbad) {
            pim_isr_info a, b;
            pim_isr_decode(&from_logical.w[first_bad], &a);
            pim_isr_decode(&from_direct.w[first_bad],  &b);
            printf("     lowered: %-7s opsize %-3u row %-6u ch %#x t %u\n",
                   pim_isr_opname(a.opcode), a.opsize, a.row, a.ch_mask, a.t);
            printf("     direct : %-7s opsize %-3u row %-6u ch %#x t %u\n",
                   pim_isr_opname(b.opcode), b.opsize, b.row, b.ch_mask, b.t);
        } else {
            printf("  every ISR word identical\n");
        }
    }

    // The lowered program must also pass the engine's own gate.
    bad = pim_prog_verify(&from_logical, mode == PIM_ACC_DUAL ? PIM_VERIFY_ALLOW_T : 0);
    CHECK(!bad, "pim_prog_verify on the lowered program: %s", bad ? bad : "");

    // ---- atoms land where the accumulation regions actually are -----------
    bad = pim_lower_atoms(&lp, atoms, 64, &natom);
    if (!bad && natom <= 64) {
        uint32_t nlatch = (mode == PIM_ACC_DUAL) ? 2u : 1u;
        uint32_t want   = (w.ngroups + nlatch - 1) / nlatch;
        CHECK(natom == want, "%u accumulation region(s), expected %u", natom, want);
        for (uint32_t i = 0; i < natom; i++) {
            pim_isr_info in;
            CHECK(atoms[i].last < from_logical.n, "atom %u runs past the program", i);
            if (atoms[i].last >= from_logical.n) break;
            pim_isr_decode(&from_logical.w[atoms[i].last], &in);
            CHECK(in.opcode == PIM_OP_RD_MAC,
                  "atom %u ends on %s, not RD_MAC", i, pim_isr_opname(in.opcode));
            pim_isr_decode(&from_logical.w[atoms[i].first], &in);
            CHECK(in.opcode == PIM_OP_MAC || in.opcode == PIM_OP_WRVEC,
                  "atom %u starts on %s", i, pim_isr_opname(in.opcode));
        }
        printf("  %u accumulation region(s), each ending on its RD_MAC\n", natom);
    }

    pim_free_ctx(c, xg); pim_free_ctx(c, yg); pim_free_ctx(c, w.base);
    free(si); free(sr); free(sa); free(sl); free(so); free(rows);
}

// ------------------------------------------------------------------------------
// The lowerer must refuse a program built for a different board.
static void check_refusals(pim_ctx *c)
{
    pim_logical lp;
    pim_isr     isr[4];
    pim_ref     ref[4];
    pim_atom    atom[4];
    pim_prog    out;
    pim_isr     obuf[8];
    const char *bad;

    printf("\nwhat the lowerer must refuse\n");

    pim_logical_init(&lp, isr, 4, ref, 4, atom, 4, pim_geom_ctx(c)->nch + 1, 1);
    pim_prog_init(&out, obuf, 8);
    bad = pim_prog_lower_ctx(c, &lp, &out, 0);
    CHECK(bad != NULL, "a program built for the wrong channel count was accepted");
    if (bad) printf("  wrong channel count -> refused\n     \"%.68s...\"\n", bad);

    // A GPR reference past the end of its allocation.
    {
        void *gp = pim_alloc_ctx(c, 4096, PIM_MEM_GPR);
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u);
        struct emu_isr      e;
        pim_isr    i2[2]; pim_ref r2[2]; pim_atom a2[2];
        pim_prog   o2; pim_isr ob2[8];

        s.opsize = 0; s.row = 0;
        emu_isr_build(&e, &s);
        pim_logical_init(&lp, i2, 2, r2, 2, a2, 2, pim_geom_ctx(c)->nch, 1);
        pim_logical_push_ref(&lp, (const pim_isr *)&e, PIM_REF_GPR_WORD,
                             PIM_SPLIT_NONE, gp, 4096, 9999, 0);
        pim_prog_init(&o2, ob2, 8);
        bad = pim_prog_lower_ctx(c, &lp, &o2, 0);
        CHECK(bad != NULL, "a GPR word past the allocation was accepted");
        if (bad) printf("  GPR word past the allocation -> refused\n     \"%.68s...\"\n", bad);
        pim_free_ctx(c, gp);
    }
}

int main(void)
{
    pim_ctx *c = mkctx(2);          /* the board-verified topology */
    const pim_geometry *g = pim_geom_ctx(c);

    printf("two-pass code generation vs the one-pass builder\n");
    printf("%u ch x %u bank, unit %llu KiB\n", g->nch, g->nbank,
           (unsigned long long)(g->unit_bytes >> 10));

    compare(c, 2048, 1024, PIM_ACC_SINGLE, "hoisted WRVEC (one chunk)");
    compare(c, 2048, 4096, PIM_ACC_SINGLE, "chained chunks");
    compare(c, 2048, 4096, PIM_ACC_DUAL,   "chained chunks, two latches");
    compare(c, 1024, 1500, PIM_ACC_SINGLE, "short final chunk");
    compare(c,   32, 1024, PIM_ACC_SINGLE, "one group");
    compare(c, 4096, 8192, PIM_ACC_DUAL,   "large");

    check_refusals(c);

    {   /* an allocation nobody stamped, and one stamped for another shape */
        pim_tensor wa, wb;  pim_isr i4[64];  pim_ref r4[64];  pim_atom a4[8];
        pim_isr o4[128]; pim_prog p4; pim_logical l4;
        void *x4, *y4;  const char *bad;

        pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, 64, 1024, &wa);
        wa.base = pim_alloc_ctx(c, pim_tensor_bytes(g, &wa), PIM_MEM_DRAM);   /* 표식 없음 */
        x4 = pim_alloc_ctx(c, wa.nredpad * 2, PIM_MEM_GPR);
        pim_tag_set_ctx(c, x4, pim_gemv_xtag(&wa));
        y4 = pim_alloc_ctx(c, wa.ngroups * g->nch * 32, PIM_MEM_GPR);
        pim_logical_init(&l4, i4, 64, r4, 64, a4, 8, g->nch, 1);
        pim_gemv_logical(g, &wa, 0, wa.ngroups, x4, wa.nredpad * 2,
                         y4, wa.ngroups * g->nch * 32, PIM_ACC_SINGLE, &l4);
        pim_prog_init(&p4, o4, 128);
        bad = pim_prog_lower_ctx(c, &l4, &p4, 0);
        CHECK(bad != NULL, "an unstamped weight allocation was accepted");
        if (bad) printf("  unstamped weight allocation -> refused\n     \"%.66s...\"\n", bad);

        /* now stamp it for a DIFFERENT shape */
        pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, 64, 2048, &wb);
        pim_tag_set_ctx(c, wa.base, pim_tensor_tag(g, &wb));
        bad = pim_prog_lower_ctx(c, &l4, &p4, 0);
        CHECK(bad != NULL, "a weight allocation laid out for another shape was accepted");
        if (bad) printf("  stamped for another shape -> refused\n     \"%.66s...\"\n", bad);

        pim_tag_set_ctx(c, wa.base, pim_tensor_tag(g, &wa));
        pim_prog_init(&p4, o4, 128);
        CHECK(!pim_prog_lower_ctx(c, &l4, &p4, 0), "the correct stamp was rejected");
        pim_free_ctx(c, x4); pim_free_ctx(c, y4); pim_free_ctx(c, wa.base);
    }

    {   /* a vector buffer nobody padded */
        pim_tensor w5;  pim_isr i5[64];  pim_ref r5[64];  pim_atom a5[8];
        pim_isr o5[128]; pim_prog p5; pim_logical l5;
        void *x5, *y5;  const char *bad;

        pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, 32, 1000, &w5);          /* k=1000 -> nredpad 1008: 8 lane 이 패딩 */
        w5.base = pim_alloc_ctx(c, pim_tensor_bytes(g, &w5), PIM_MEM_DRAM);
        pim_tag_set_ctx(c, w5.base, pim_tensor_tag(g, &w5));
        x5 = pim_alloc_ctx(c, w5.nredpad * 2, PIM_MEM_GPR);      /* 표식 없음 */
        y5 = pim_alloc_ctx(c, w5.ngroups * g->nch * 32, PIM_MEM_GPR);
        pim_logical_init(&l5, i5, 64, r5, 64, a5, 8, g->nch, 1);
        pim_gemv_logical(g, &w5, 0, w5.ngroups, x5, w5.nredpad * 2,
                         y5, w5.ngroups * g->nch * 32, PIM_ACC_SINGLE, &l5);
        pim_prog_init(&p5, o5, 128);
        bad = pim_prog_lower_ctx(c, &l5, &p5, 0);
        CHECK(bad != NULL, "an unpadded vector buffer was accepted");
        if (bad) printf("  unpadded vector buffer -> refused\n     \"%.66s...\"\n", bad);
        pim_free_ctx(c, x5); pim_free_ctx(c, y5); pim_free_ctx(c, w5.base);
    }

    {   /* a DUAL schedule lowered without evidence must be refused */
        pim_tensor w2;  pim_isr i3[64];  pim_ref r3[64];  pim_atom a3[8];
        pim_isr o3[128]; pim_prog p3; pim_logical l3;
        void *x3, *y3;  const char *bad;

        printf("\n");
        pim_tensor_plan(g, PIM_LAYOUT_OUT_MAJOR, 64, 1024, &w2);
        w2.base = pim_alloc_ctx(c, pim_tensor_bytes(g, &w2), PIM_MEM_DRAM);
        pim_tag_set_ctx(c, w2.base, pim_tensor_tag(g, &w2));
        x3 = pim_alloc_ctx(c, w2.nredpad * 2, PIM_MEM_GPR);
        pim_tag_set_ctx(c, x3, pim_gemv_xtag(&w2));
        y3 = pim_alloc_ctx(c, w2.ngroups * g->nch * 32, PIM_MEM_GPR);
        pim_logical_init(&l3, i3, 64, r3, 64, a3, 8, g->nch, 2);
        pim_gemv_logical(g, &w2, 0, w2.ngroups, x3, w2.nredpad * 2,
                         y3, w2.ngroups * g->nch * 32, PIM_ACC_DUAL, &l3);
        pim_prog_init(&p3, o3, 128);
        bad = pim_prog_lower_ctx(c, &l3, &p3, 0);
        CHECK(bad != NULL, "a two-latch schedule was lowered without evidence");
        if (bad) printf("  two latches without evidence -> refused\n     \"%.66s...\"\n", bad);
        pim_prog_init(&p3, o3, 128);
        CHECK(!pim_prog_lower_ctx(c, &l3, &p3, PIM_LOWER_ALLOW_T),
              "PIM_LOWER_ALLOW_T did not let it through");
        pim_free_ctx(c, x3); pim_free_ctx(c, y3); pim_free_ctx(c, w2.base);
    }

    printf("\ngrowable KV tensors\n");
    grow_mac_case(c, PIM_LAYOUT_RED_MAJOR,  512,  2500, 0, 0,   2500, "S.V, V 1B, 3 chunks");
    grow_mac_case(c, PIM_LAYOUT_RED_MAJOR,  1024, 1500, 4, 0,   1500, "S.V, V 3B, one head");
    grow_mac_case(c, PIM_LAYOUT_OUT_PACKED, 512,  300,  0, 64,  64,   "q.K, K 1B, head 1");
    grow_mac_case(c, PIM_LAYOUT_OUT_PACKED, 1024, 200,  0, 896, 128,  "q.K, K 3B, head 7");

    printf("\nQ laid out like a row of K: every head behind one WRVEC\n");
    heads_case(c, 512,  64,  300, PIM_ACC_SINGLE, "K 1B (8x64, 2 tokens a row)");
    heads_case(c, 512,  64,  300, PIM_ACC_DUAL,   "K 1B, two latches");
    heads_case(c, 1024, 128, 200, PIM_ACC_SINGLE, "K 3B/Qwen (8x128)");
    heads_case(c, 1024, 128, 200, PIM_ACC_DUAL,   "K 3B/Qwen, two latches");
    check_gb_wrap();

    printf("\n%s\n", fail ? "FAILED" : "all checks passed");
    return fail ? 1 : 0;
}
