// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// fake_drv.h — a simulated /dev/pim, so the allocator can be tested with no board
// and no module loaded.
//
// WHAT IT REPLACES IS EXACTLY pim_dev.c.  A test that includes this links the REAL
// pim_alloc.c / pim_geom.c / pim_addr.c and supplies the three things pim_dev.c
// would have: the two ioctls and the error string.  That is the whole reason those
// files are separate — the seam is a file boundary, so it can be cut.
//
// WHAT IT IS FOR.  The failure it exists to catch is ALIASING: two live allocations
// resolving to one PIM address.  Every other property survives an off-by-one index;
// that one does not, and it is the hardest failure to work backwards to from a wrong
// GEMV result.
//
// The bitmap here is deliberately dumber than the real driver's — contiguous run
// first, then singles — because the point is to exercise libpim, not to re-test the
// kernel's allocator.
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_FAKE_DRV_H
#define PIM_FAKE_DRV_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "pim_internal.h"

// ---------------- fake driver: one bitmap per region -------------------------
#define NR_DRAM   256u          /* x 2 MiB = 512 MiB */
#define NR_GPR   1024u          /* x 4 KiB =   4 MiB */
#define NR_MAX   1024u

static unsigned char blk[PIM_NMEM][NR_MAX];
static uint64_t rbase [PIM_NMEM] = { 0x020400000000ull, 0x020200000000ull };
static uint64_t rhugepage[PIM_NMEM] = { 2u << 20,          4096u };
static uint64_t rn    [PIM_NMEM] = { NR_DRAM,           NR_GPR };
static unsigned ioctl_allocs[PIM_NMEM], ioctl_frees[PIM_NMEM], out[PIM_NMEM];

const char *pim_set_err(pim_ctx *c, const char *fmt, ...)
{
    char *b = c ? c->err : NULL; static char s[PIM_ERRBUF];
    va_list ap; if (!b) b = s;
    va_start(ap, fmt); vsnprintf(b, PIM_ERRBUF, fmt, ap); va_end(ap);
    return b;
}
const char *pim_last_error_ctx(const pim_ctx *c)
{ return (c && c->err[0]) ? c->err : "(no error recorded)"; }

/* Also from pim_dev.c, which this test does not link (it owns the fd and the ioctls
 * we are replacing).  The runtime reads geometry through it. */
const pim_geometry *pim_geom_ctx(const pim_ctx *c) { return c ? &c->geom : NULL; }

/* Also from pim_dev.c: the process-wide default context.  mkctx() installs the
 * simulated one here, so a test can drive the short-form API (pim_alloc(n, where),
 * pim_addr_unit(p, ...)) against the fake driver exactly as an application would. */
static pim_ctx *fake_default;
pim_ctx *pim_default(void) { return fake_default; }

int pim_ioctl_alloc(pim_ctx *c, pim_mem m, uint32_t n, uint64_t vbase,
                    struct pim_extent *ext, uint32_t cap, uint32_t *nex)
{
    (void)c; (void)vbase;
    unsigned got = 0, k = 0; unsigned i, j, N = (unsigned)rn[m];
    ioctl_allocs[m]++;

    for (i = 0; i + n <= N; i++) {                    /* contiguous run first */
        for (j = 0; j < n && !blk[m][i+j]; j++) {}
        if (j == n) {
            for (j = 0; j < n; j++) blk[m][i+j] = 1;
            ext[0].addr = rbase[m] + (uint64_t)i * rhugepage[m];
            ext[0].nr_hugepages = n; *nex = 1; out[m] += n; return 0;
        }
    }
    for (i = 0; i < N && got < n; i++) {              /* singles fallback */
        if (blk[m][i]) continue;
        if (k && ext[k-1].addr + (uint64_t)ext[k-1].nr_hugepages*rhugepage[m]
                 == rbase[m] + (uint64_t)i*rhugepage[m]) ext[k-1].nr_hugepages++;
        else { if (k >= cap) return -1;
               ext[k].addr = rbase[m] + (uint64_t)i*rhugepage[m];
               ext[k].nr_hugepages = 1; k++; }
        blk[m][i] = 1; got++;
    }
    if (got < n) { for (unsigned q = 0; q < k; q++)
                       for (j = 0; j < ext[q].nr_hugepages; j++)
                           blk[m][(ext[q].addr - rbase[m])/rhugepage[m] + j] = 0;
                   return -1; }
    *nex = k; out[m] += n; return 0;
}

int pim_ioctl_free(pim_ctx *c, pim_mem m, const struct pim_extent *ext,
                   uint32_t nex, uint64_t vb)
{
    (void)c; (void)vb; ioctl_frees[m]++;
    for (uint32_t i = 0; i < nex; i++)
        for (uint32_t j = 0; j < ext[i].nr_hugepages; j++) {
            uint64_t idx = (ext[i].addr - rbase[m])/rhugepage[m] + j;
            if (idx >= rn[m] || !blk[m][idx]) { printf("  BAD FREE %s idx %"PRIu64"\n",
                    m ? "gpr" : "dram", idx); return -1; }
            blk[m][idx] = 0; out[m]--;
        }
    return 0;
}

// ---------------- simulated card memory --------------------------------------
// WHY A TEST NEEDS BYTES AND NOT JUST A BITMAP.  The bitmap above answers "do two
// allocations overlap".  It cannot answer "did this land where the layout says it
// should", and that is the question pim_tensor's whole design rests on: an append
// writing one element too far is a wrong number rather than a crash, and no amount
// of address arithmetic checked against itself would notice.  So the fake grows a
// backing store and a pim_memcpy_ctx that honours pim_resolve, and a host test can
// then read the card back and compare.
//
// POISONED, NOT ZEROED, and lazily.  Zero-filled storage would make
// PIM_ALLOC_F_ZERO vacuous — every test would pass whether or not it cleared
// anything.  Poison is also what the real board gives you, since nothing zeroes a
// region between users.  Per-hugepage allocation keeps that honest without paying
// for the 512 MiB the fake DRAM region nominally spans; a test touches a handful.
#define FAKE_POISON  0xA5

static unsigned char *card[PIM_NMEM][NR_MAX];

// Host pointer for one AXI address, and how many bytes remain in its hugepage.
static unsigned char *card_at(pim_mem m, uint64_t axi, size_t *room)
{
    uint64_t off, idx, within;

    if (axi < rbase[m]) return NULL;
    off = axi - rbase[m];
    idx = off / rhugepage[m];
    within = off % rhugepage[m];
    if (idx >= rn[m]) return NULL;
    if (!card[m][idx]) {
        card[m][idx] = malloc(rhugepage[m]);
        if (!card[m][idx]) return NULL;
        memset(card[m][idx], FAKE_POISON, rhugepage[m]);
    }
    *room = (size_t)(rhugepage[m] - within);
    return card[m][idx] + within;
}

// Replaces pim_memcpy.c, and deliberately goes through pim_resolve_ctx rather than
// reading rec->gran_addr[] directly: a test that used the same private path as the
// code under test would agree with it by construction.
const char *pim_memcpy_ctx(pim_ctx *c, void *dst, const void *src, size_t n,
                           pim_dir dir, unsigned flags)
{
    const void    *devp = (dir == PIM_TO_DEV) ? dst : src;
    unsigned char *hostp = (unsigned char *)((dir == PIM_TO_DEV) ? (void *)src : dst);
    size_t done = 0;

    (void)flags;
    if (!c) return "pim_memcpy_ctx: no context";

    while (done < n) {
        pim_loc     loc;
        size_t      run, room, step;
        unsigned char *cp;
        const char *bad = pim_resolve_ctx(c, (const char *)devp + done, 1, &loc, &run);

        if (bad) return bad;
        if (run > n - done) run = n - done;
        cp = card_at(loc.mem, loc.axi, &room);
        if (!cp) return "fake card: address outside the simulated region";
        step = run < room ? run : room;
        if (dir == PIM_TO_DEV) memcpy(cp, hostp + done, step);
        else                   memcpy(hostp + done, cp, step);
        done += step;
    }
    return NULL;
}

const char *pim_memcpy(void *dst, const void *src, size_t n, pim_dir dir, unsigned f)
{ return pim_memcpy_ctx(pim_default(), dst, src, n, dir, f); }

// The rest of pim_memcpy.c's surface.  pim_exec.c reaches the control plane through
// these — BAR2, not a pim_alloc pointer — so they take a raw AXI address and skip
// the ledger entirely.  The fake serves them from the same backing store, which is
// what lets lower_test build a program and an exec engine with no board.
const char *pim_axi_allow_ctx(pim_ctx *c, uint64_t base, uint64_t bytes)
{ (void)c; (void)base; (void)bytes; return NULL; }

static const char *fake_axi(uint64_t axi, void *host, size_t n, bool to_dev)
{
    size_t done = 0;

    while (done < n) {
        size_t room, step;
        unsigned char *cp = NULL;

        for (int m = 0; m < PIM_NMEM && !cp; m++)
            if (axi + done >= rbase[m] &&
                axi + done <  rbase[m] + rn[m] * rhugepage[m])
                cp = card_at((pim_mem)m, axi + done, &room);
        if (!cp) return "fake card: AXI address outside every simulated region";
        step = (n - done) < room ? (n - done) : room;
        if (to_dev) memcpy(cp, (unsigned char *)host + done, step);
        else        memcpy((unsigned char *)host + done, cp, step);
        done += step;
    }
    return NULL;
}

const char *pim_axi_write_ctx(pim_ctx *c, uint64_t axi, const void *src, size_t n)
{ (void)c; return fake_axi(axi, (void *)src, n, true); }

const char *pim_axi_read_ctx(pim_ctx *c, uint64_t axi, void *dst, size_t n)
{ (void)c; return fake_axi(axi, dst, n, false); }

// What is on the card right now at `axi`, whatever wrote it.  For a test that wants
// to look without going through a pointer it is checking.
__attribute__((unused))
static unsigned char fake_peek(pim_mem m, uint64_t axi)
{
    size_t room;
    unsigned char *p = card_at(m, axi, &room);
    return p ? *p : 0;
}

// ---------------- harness ----------------------------------------------------
static pim_ctx *mkctx(unsigned nch)
{
    pim_ctx *c = calloc(1, sizeof *c);
    c->fd = c->h2c = c->c2h = -1;
    c->max_xfer = 4u<<20; c->keep_free_hugepages = 2;
    c->geom.nch = nch; c->geom.nbank = 16; c->geom.row_bytes = 2048;
    c->geom.map = PIM_MAP_RO_CH_BA_CO;
    c->geom.unit_bytes = (uint64_t)nch*16*2048;
    for (int m = 0; m < PIM_NMEM; m++) {
        c->geom.mem[m].base        = rbase[m];
        c->geom.mem[m].hugepage_bytes = rhugepage[m];
        c->geom.mem[m].nr_hugepages   = rn[m];
        c->geom.mem[m].bytes       = rn[m] * rhugepage[m];
    }
    pim_geom_derive(&c->geom);
    pthread_mutex_init(&c->lock, NULL);
    fake_default = c;
    return c;
}


#endif // PIM_FAKE_DRV_H
