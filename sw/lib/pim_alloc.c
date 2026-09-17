// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_alloc.ctx — everything that owns a pim_alloc_rec.  Design §7, §9.4.
//
// ONE FILE BECAUSE IT IS ONE DATA STRUCTURE.  An allocation's gran_addr[] IS the
// page table: pim_alloc fills it, pim_resolve reads it, pim_free hands it back.  The
// three tiers underneath it are the same story at different scales, and splitting
// them across files only meant the reader had to hold three of them open at once:
//
//   §1  host address space   PROT_NONE reservations.  What a pim pointer IS.
//   §2  card memory          hugepages from the driver, cut into granules.
//   §3  the record table      pointer -> pim_alloc_rec, binary search on vbase.
//   §4  the public API       pim_alloc / pim_free / pim_usable / pim_where.
//   §5  translation          pim_resolve — the map between §1 and §2.
//
// The central claim of the design is implemented in §5, and it is this:
//
//     the caller sees ONE contiguous range; the card holds it in pieces; the pieces
//     are invisible to compute and cost only extra transfers.
//
// A lookup is a shift and an index — no syscall, which is the entire payoff of
// sizing the driver's DRAM hugepage to preserve the address bits (design §4).
//
// BOTH POOLS SHARE ALL OF IT.  A GPR allocation is the same record with a different
// granule and a different region tag; nothing below asks which pool it is in, it
// reads hugepage_bytes and gran out of the geometry.  For the GPR those are equal, so
// grans_per_hugepage is 1 and the inner bitmap is one bit wide — a DEGENERATE case, not
// a special case.  The reservations cannot overlap each other whichever pool they
// came from, because the thing keeping them apart is the kernel's VMA allocator and
// it does not know about pools either.
//
// WHAT IS NOT HERE, AND WHY THAT MATTERS.  Nothing in this file opens a device or
// issues an ioctl — it reaches the driver only through pim_ioctl_alloc/free, which
// live in pim_dev.ctx.  That is the seam lib/test/pool_test.ctx cuts along: it links
// this file and simulates those two calls, so the allocator is tested with no board
// and no module loaded.  Keep it that way.
//
// LOCKING.  §2 and §3 assume ctx->lock is held; §4 and §5 take it.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

//////////////////////////////////////////////////////////////////////////////////
// §1  HOST ADDRESS SPACE, never host memory (design §7.3)
//
// This is the trick that lets pim_alloc return a plain void *.  mmap with PROT_NONE
// reserves a range of this process's address space and nothing else:
//
//   physical memory   0.  A PROT_NONE fault has no success path, so demand paging
//                     never populates anything — the consumption is not deferred,
//                     it is impossible.
//   page tables       0, for the same reason.
//   what it costs     one VMA (tens of bytes, one slot of vm.max_map_count).
//
// And it buys, in exchange for that one VMA:
//
//   uniqueness        the kernel's VMA allocator will not place a later mapping
//                     inside this range, so two pim_allocs cannot collide and a
//                     pim address cannot collide with the heap.  The uniqueness
//                     problem is handed to the subsystem whose job it already is.
//   loud misuse       dereferencing a card pointer is a SIGSEGV at the offending
//                     line, not silent corruption of whatever host object happened
//                     to live at a made-up address.
//
// Same family of trick as guard pages, CUDA's UVA reservations and ASan's shadow.
//////////////////////////////////////////////////////////////////////////////////

// Reserve `bytes` at an `align`-aligned address.  mmap only promises page
// alignment, so over-reserve by one alignment and hand the slack back; what is left
// is still a VMA, so the protection is unaffected.
//
// WHY ALIGN AT ALL.  When an allocation lands on whole hugepages (the common case) the
// host address and the card address then agree in their low hugepage_shift bits, and a
// pointer printed in a log can be read as a card coordinate directly.  It does NOT
// survive unit-level scattering — units of one allocation can sit at different
// offsets within their hugepages — so treat it as a convenience, never as a fact to
// compute with.  pim_resolve() is the fact.
static void *vspace_reserve(size_t bytes, size_t align)
{
    long   pg = sysconf(_SC_PAGESIZE);
    size_t span, head, tail;
    uintptr_t base;
    char *raw;

    if (pg > 0 && align < (size_t)pg)
        align = (size_t)pg;
    if (!bytes)
        return NULL;

    span = bytes + align;
    raw  = mmap(NULL, span, PROT_NONE,
                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED)
        return NULL;

    base = ((uintptr_t)raw + align - 1) & ~(uintptr_t)(align - 1);
    head = (size_t)(base - (uintptr_t)raw);
    tail = span - head - bytes;

    if (head)
        munmap(raw, head);
    if (tail)
        munmap((char *)base + bytes, tail);

    return (void *)base;
}

static void vspace_release(void *ptr, size_t bytes)
{
    if (ptr && bytes)
        munmap(ptr, bytes);
}

//////////////////////////////////////////////////////////////////////////////////
// §2  CARD MEMORY — hugepages from the driver, cut into granules (design §7.2, §7.4)
//
// THE SAME ALGORITHM AS THE DRIVER, ONE LAYER DOWN.  The driver keeps a bitmap over
// (region -> hugepages); this keeps a bitmap over (hugepage -> granules).  What differs is
// the policy, and it differs for a concrete reason:
//
//   the driver   prefers a contiguous RUN, because a run is one pwrite later.
//   this         prefers an already-PARTIAL hugepage, because a hugepage with even one
//                granule in use cannot be handed back — so filling partials first is
//                what keeps whole hugepages free and returnable.
//
// The GPR's degeneracy has one observable consequence here: a GPR allocation always
// costs an ioctl the first time (there is no partial hugepage to fill), which is fine
// while GPR objects are long-lived by construction — a vector is uploaded once and
// re-read by many WRVECs.  If a profile ever shows GPR churn, the fix is to grow by
// several pages at a time in pool_grow(), and nothing else changes.
//
// WHY THERE IS NO SEPARATE "CONTIGUOUS RUN" PASS.  Granules are taken in ascending
// index order within a hugepage, so whatever adjacency exists is already picked up
// consecutively and collapses into one extent at transfer time.  A best-run search
// would only pay off in a hugepage fragmented by mixed-size frees.
//
// EVERY FUNCTION IN THIS SECTION ASSUMES ctx->lock IS HELD.
//////////////////////////////////////////////////////////////////////////////////

// A pool is kept sorted by address: giving a granule back is a binary search, and
// trim can coalesce neighbours without a second sort.
static int hugepage_find(const pim_pool *pool, uint64_t addr)
{
    uint32_t lo = 0, hi = pool->n;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (pool->v[mid].addr == addr) return (int)mid;
        if (pool->v[mid].addr <  addr) lo = mid + 1;
        else                        hi = mid;
    }
    return -1;
}

static int pool_insert(pim_ctx *ctx, pim_mem where, uint64_t addr)
{
    pim_pool *pool = &ctx->pool[where];
    uint32_t lo = 0, hi = pool->n;

    if (pool->n == pool->cap) {
        uint32_t cap = pool->cap ? pool->cap * 2 : 16;
        pim_hugepage *v = realloc(pool->v, (size_t)cap * sizeof *v);
        if (!v) return -1;
        pool->v = v;
        pool->cap = cap;
    }
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (pool->v[mid].addr < addr) lo = mid + 1;
        else                       hi = mid;
    }
    memmove(&pool->v[lo + 1], &pool->v[lo], (size_t)(pool->n - lo) * sizeof *pool->v);
    pool->v[lo].addr       = addr;
    pool->v[lo].used       = 0;
    pool->v[lo].free_grans = ctx->geom.mem[where].grans_per_hugepage;
    pool->n++;
    return 0;
}

static uint32_t grab(pim_ctx *ctx, pim_mem where, pim_hugepage *blk, uint32_t need, uint64_t *out)
{
    const pim_region *region = &ctx->geom.mem[where];
    uint32_t took = 0, gi;

    for (gi = 0; gi < region->grans_per_hugepage && took < need; gi++) {
        if (blk->used & (1ull << gi))
            continue;
        blk->used |= 1ull << gi;
        blk->free_grans--;
        out[took++] = blk->addr + (uint64_t)gi * region->gran;
    }
    return took;
}

// `partial_only` is the fill-partial-first pass.
static uint32_t sweep(pim_ctx *ctx, pim_mem where, uint32_t need, uint64_t *out,
                      bool partial_only)
{
    uint32_t gpb = ctx->geom.mem[where].grans_per_hugepage, got = 0, i;

    for (i = 0; i < ctx->pool[where].n && got < need; i++) {
        pim_hugepage *blk = &ctx->pool[where].v[i];
        if (!blk->free_grans)
            continue;
        if (partial_only && blk->free_grans == gpb)
            continue;
        got += grab(ctx, where, blk, need - got, out + got);
    }
    return got;
}

static void pool_give(pim_ctx *ctx, pim_mem where, const uint64_t *addr, uint32_t ngrans);

static int pool_grow(pim_ctx *ctx, pim_mem where, uint32_t nhugepages)
{
    uint64_t bb = ctx->geom.mem[where].hugepage_bytes;
    struct pim_extent *ext;
    uint32_t nex = 0, i, j;

    ext = calloc(nhugepages, sizeof *ext);
    if (!ext) { errno = ENOMEM; return -1; }

    // vbase is passed as 0 — "not registered".  In this two-tier design a BLOCK
    // REQUEST is not an ALLOCATION: one request can serve several later allocations
    // and one allocation can draw on hugepages from several requests, so there is no
    // single reservation that honestly describes it.  The field stays in the ABI
    // because the day the tiers change (or isolation arrives, design §11) there
    // will be, and adding it then would be an ABI break.  See sw/README.md.
    if (pim_ioctl_alloc(ctx, where, nhugepages, 0, ext, nhugepages, &nex) < 0) {
        free(ext);
        return -1;
    }

    for (i = 0; i < nex; i++)
        for (j = 0; j < ext[i].nr_hugepages; j++)
            if (pool_insert(ctx, where, ext[i].addr + (uint64_t)j * bb) < 0) {
                // The hugepages are ours whether or not we can remember them, so hand
                // back the ones we could not record rather than leaking.
                struct pim_extent rest = {
                    .addr      = ext[i].addr + (uint64_t)j * bb,
                    .nr_hugepages = ext[i].nr_hugepages - j,
                };
                pim_ioctl_free(ctx, where, &rest, 1, 0);
                for (uint32_t k = i + 1; k < nex; k++)
                    pim_ioctl_free(ctx, where, &ext[k], 1, 0);
                free(ext);
                errno = ENOMEM;
                return -1;
            }

    free(ext);
    return 0;
}

// out[] receives ngrans addresses.  All or nothing: a partial take would leave the
// caller holding granules it never asked for and cannot describe.
static int pool_take(pim_ctx *ctx, pim_mem where, uint32_t ngrans, uint64_t *out)
{
    uint32_t gpb = ctx->geom.mem[where].grans_per_hugepage;
    uint32_t got;

    got = sweep(ctx, where, ngrans, out, true);                        // 1. partials first
    if (got < ngrans)
        got += sweep(ctx, where, ngrans - got, out + got, false);      // 2. anything left

    if (got < ngrans) {                                          // 3. ask the driver
        uint32_t need = ngrans - got;
        uint32_t want = (need + gpb - 1) / gpb;
        if (pool_grow(ctx, where, want) < 0) {
            pool_give(ctx, where, out, got);
            return -1;
        }
        got += sweep(ctx, where, need, out + got, false);
    }

    if (got < ngrans) {
        pool_give(ctx, where, out, got);
        errno = ENOMEM;
        return -1;
    }
    return 0;
}

static void pool_give(pim_ctx *ctx, pim_mem where, const uint64_t *addr, uint32_t ngrans)
{
    const pim_region *region = &ctx->geom.mem[where];
    uint32_t i;

    for (i = 0; i < ngrans; i++) {
        uint64_t base = addr[i] - (addr[i] - region->base) % region->hugepage_bytes;
        uint32_t gi    = (uint32_t)((addr[i] - base) / region->gran);
        int k = hugepage_find(&ctx->pool[where], base);

        if (k < 0)                      // not ours; a double free upstream
            continue;
        if (!(ctx->pool[where].v[k].used & (1ull << gi)))
            continue;
        ctx->pool[where].v[k].used &= ~(1ull << gi);
        ctx->pool[where].v[k].free_grans++;
    }
}

// Return fully-empty hugepages to the driver, keeping `keep_free_hugepages` of them
// (design §7.4).  Returning eagerly makes an alloc/free loop bounce on ioctl;
// hoarding starves anything else on the card.  `all` overrides the threshold and is
// what teardown uses.
//
// Exported within the library for lib/test/pool_test.ctx, which drains both pools at
// the end to prove nothing leaked.
void pim_pool_trim(pim_ctx *ctx, pim_mem where, bool all)
{
    uint32_t gpb = ctx->geom.mem[where].grans_per_hugepage;
    uint64_t bb  = ctx->geom.mem[where].hugepage_bytes;
    pim_pool *pool  = &ctx->pool[where];
    uint32_t nfree = 0, keep, togo, nex = 0, i, kept = 0;
    struct pim_extent *ext;
    bool *drop;

    for (i = 0; i < pool->n; i++)
        if (pool->v[i].free_grans == gpb)
            nfree++;

    keep = all ? 0 : ctx->keep_free_hugepages;
    if (nfree <= keep)
        return;
    togo = nfree - keep;

    ext  = calloc(togo, sizeof *ext);
    drop = calloc(pool->n ? pool->n : 1, 1);
    if (!ext || !drop) {                 // trimming is optional; failing is not fatal
        free(ext);
        free(drop);
        return;
    }

    // High addresses first, so what we keep is the low, likely-adjacent end — and
    // so neighbours coalesce into one extent as we walk.
    for (i = pool->n; i-- > 0 && togo; ) {
        uint64_t rec = pool->v[i].addr;

        if (pool->v[i].free_grans != gpb)
            continue;
        drop[i] = true;
        if (nex && ext[nex - 1].addr == rec + bb) {
            ext[nex - 1].addr = rec;
            ext[nex - 1].nr_hugepages++;
        } else {
            ext[nex].addr      = rec;
            ext[nex].nr_hugepages = 1;
            nex++;
        }
        togo--;
    }

    // If the driver refuses the free we simply keep the hugepages: drop[] is dropped on
    // the floor and the pool is unchanged.
    if (nex && pim_ioctl_free(ctx, where, ext, nex, 0) == 0) {
        for (i = 0; i < pool->n; i++)
            if (!drop[i])
                pool->v[kept++] = pool->v[i];
        pool->n = kept;
    }
    free(ext);
    free(drop);
}

//////////////////////////////////////////////////////////////////////////////////
// §3  THE RECORD TABLE — which allocation a pointer is inside
//
// SORTED BY vbase, so resolving a pointer that points INTO an allocation (which is
// the normal case — `w + 1024`) is a binary search rather than a scan.  A list would
// be fine for tens of allocations and quietly quadratic for thousands.
//
// ONE REGISTRY FOR BOTH POOLS: see the file banner.
//
// Caller holds ctx->lock.
//////////////////////////////////////////////////////////////////////////////////
static int rec_insert(pim_ctx *ctx, pim_alloc_rec *rec)
{
    uint32_t lo = 0, hi = ctx->nallocs;

    if (ctx->nallocs == ctx->allocs_cap) {
        uint32_t cap = ctx->allocs_cap ? ctx->allocs_cap * 2 : 16;
        pim_alloc_rec **v = realloc(ctx->allocs, (size_t)cap * sizeof *v);
        if (!v) return -1;
        ctx->allocs = v;
        ctx->allocs_cap = cap;
    }
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if ((uintptr_t)ctx->allocs[mid]->vbase < (uintptr_t)rec->vbase) lo = mid + 1;
        else                                                        hi = mid;
    }
    memmove(&ctx->allocs[lo + 1], &ctx->allocs[lo],
            (size_t)(ctx->nallocs - lo) * sizeof *ctx->allocs);
    ctx->allocs[lo] = rec;
    ctx->nallocs++;
    return 0;
}

// The last allocation whose vbase is <= ptr, then a range check.  Sorted and
// non-overlapping (the kernel's VMA allocator guarantees the second — design §7.3),
// so that one candidate is the only one it can be.
//
static pim_alloc_rec *rec_lookup(const pim_ctx *ctx, const void *ptr)
{
    uintptr_t va = (uintptr_t)ptr;
    uint32_t lo = 0, hi = ctx->nallocs;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if ((uintptr_t)ctx->allocs[mid]->vbase <= va) lo = mid + 1;
        else                                          hi = mid;
    }
    if (!lo) return NULL;
    {
        pim_alloc_rec *rec = ctx->allocs[lo - 1];
        uintptr_t base = (uintptr_t)rec->vbase;
        return (va >= base && va < base + rec->bytes) ? rec : NULL;
    }
}

static pim_alloc_rec *rec_remove(pim_ctx *ctx, const void *ptr)   // exact vbase match
{
    uint32_t i;

    for (i = 0; i < ctx->nallocs; i++)
        if (ctx->allocs[i]->vbase == ptr) {
            pim_alloc_rec *rec = ctx->allocs[i];
            memmove(&ctx->allocs[i], &ctx->allocs[i + 1],
                    (size_t)(ctx->nallocs - i - 1) * sizeof *ctx->allocs);
            ctx->nallocs--;
            return rec;
        }
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////
// §4  THE PUBLIC ALLOCATOR
//////////////////////////////////////////////////////////////////////////////////

// PIM_ALLOC_F_ZERO.  One host buffer of zeroes, pushed over the allocation in
// chunks.
//
// NOT ONE TRANSFER OF `bytes`.  A host buffer the size of the allocation would be
// the obvious shape and it is the wrong one: the KV tensors this exists for are
// hundreds of megabytes, and asking the host for that much just to send zeroes
// would fail on exactly the allocations that need it most.  A fixed buffer costs
// nothing and pim_memcpy already splits each chunk across whatever granule runs it
// spans, so the transfer count is the same either way.
//
// AFTER THE LOCK IS DROPPED, because pim_memcpy_ctx resolves the pointer and takes
// ctx->lock itself.  That is safe here only because the allocation is not yet
// visible to the caller: the pointer has not been returned, so nothing else can
// reach it however many threads are running.
#define ZERO_CHUNK  (256u << 10)

static const char *zero_fill(pim_ctx *ctx, void *ptr, size_t bytes)
{
    size_t      chunk = bytes < ZERO_CHUNK ? bytes : ZERO_CHUNK;
    void       *zeroes = calloc(1, chunk);
    const char *bad = NULL;
    size_t      off;

    if (!zeroes) { errno = ENOMEM; return "out of host memory for the zero buffer"; }
    for (off = 0; off < bytes && !bad; off += chunk) {
        size_t n = bytes - off < chunk ? bytes - off : chunk;
        bad = pim_memcpy_ctx(ctx, (char *)ptr + off, zeroes, n, PIM_TO_DEV, 0);
    }
    free(zeroes);
    return bad;
}

void *pim_alloc_ctx(pim_ctx *ctx, size_t nbytes, pim_mem where)
{
    return pim_alloc_ex_ctx(ctx, nbytes, where, 0);
}

void *pim_alloc_ex_ctx(pim_ctx *ctx, size_t nbytes, pim_mem where, unsigned flags)
{
    pim_alloc_rec *rec;
    const pim_region *region;
    uint64_t gran;
    size_t   bytes;
    uint32_t ngrans;

    if (!ctx || !nbytes)          { errno = EINVAL; return NULL; }
    if (flags & ~(PIM_ALLOC_F_ZERO | PIM_ALLOC_F_CONTIG)) {
        pim_set_err(ctx, "pim_alloc_ex: flags %#x has bits this version does not "
                         "define; they are reserved and must be zero", flags);
        errno = EINVAL;
        return NULL;
    }
    if (flags & PIM_ALLOC_F_CONTIG) {
        // Refused rather than forwarded.  The driver's PIM_ALLOC_CONTIG makes
        // HUGEPAGES adjacent, which is not what this flag promises: the pool still
        // hands out whichever granules inside them are free, so a caller told "yes"
        // could still get a scattered gran_addr[] and a transfer per granule.  The
        // honest version is a run search in pool_take(); until that exists, saying
        // no is the only answer that cannot mislead.
        pim_set_err(ctx, "PIM_ALLOC_F_CONTIG is reserved and not implemented: it "
                         "would need a consecutive-run search in the granule pool, "
                         "not the driver's hugepage-level flag");
        errno = ENOTSUP;
        return NULL;
    }
    if ((unsigned)where >= PIM_NMEM) {
        pim_set_err(ctx, "pim_alloc: %u is not a memory region", (unsigned)where);
        errno = EINVAL;
        return NULL;
    }
    region = &ctx->geom.mem[where];

    // ROUNDED UP TO THAT POOL'S GRANULE.  For DRAM that is the broadcast unit
    // (design §7.4): an all-bank operation touches the whole unit, so two
    // allocations sharing one could not both be operands — the rounding buys safety
    // at the price of internal fragmentation, exactly as CUDA's VMM granularity
    // does.  For the GPR it is the 4 KiB page, and there is no operand-shape reason
    // behind it at all (design §9.4) — only that v1 declines to guess at a
    // size-class layer before any size distribution has been measured.
    gran   = region->gran;
    bytes  = (nbytes + gran - 1) & ~(size_t)(gran - 1);
    if (bytes < nbytes || bytes > region->bytes) {           // overflow, or hopeless
        pim_set_err(ctx, "%zu bytes rounds to %zu, past the %llu KiB in the %s pool",
                    nbytes, bytes, (unsigned long long)(region->bytes >> 10),
                    where == PIM_MEM_GPR ? "gpr" : "dram");
        errno = ENOMEM;
        return NULL;
    }
    ngrans = (uint32_t)(bytes / gran);

    // HEADER PLUS TABLE IN ONE BLOCK.  gran_addr[] is a flexible array member, so
    // sizeof *rec is the header alone (24 B) and the ngrans entries are asked for on
    // top of it — see the layout diagram at pim_alloc_rec in pim_internal.h.
    rec = calloc(1, sizeof *rec + (size_t)ngrans * sizeof rec->gran_addr[0]);
    if (!rec) { errno = ENOMEM; return NULL; }
    rec->bytes    = bytes;
    rec->mem      = where;
    rec->nr_grans = ngrans;

    // FILLS gran_addr[] WITH PIM PHYSICAL ADDRESSES, one per granule, and they need
    // not be consecutive — the pool hands out whatever granules are free.  The host
    // side reserved just below is contiguous regardless; that asymmetry is what the
    // table exists to record.
    pthread_mutex_lock(&ctx->lock);
    if (pool_take(ctx, where, ngrans, rec->gran_addr) < 0) {
        pthread_mutex_unlock(&ctx->lock);
        pim_set_err(ctx, "the %s pool has no room for %u granule(s) of %llu KiB",
                    where == PIM_MEM_GPR ? "gpr" : "dram", ngrans,
                    (unsigned long long)(gran >> 10));
        free(rec);
        errno = ENOMEM;
        return NULL;
    }

    // Reserve the host address space only once the card memory is in hand — the
    // reservation is the cheap half and undoing it is a munmap either way, but this
    // way the failure that actually happens (pool full) never leaves a VMA behind.
    rec->vbase = vspace_reserve(bytes, (size_t)region->hugepage_bytes);
    if (!rec->vbase || rec_insert(ctx, rec) < 0) {
        pool_give(ctx, where, rec->gran_addr, ngrans);
        pthread_mutex_unlock(&ctx->lock);
        vspace_release(rec->vbase, bytes);
        pim_set_err(ctx, "out of host address space for a %zu byte reservation", bytes);
        free(rec);
        errno = ENOSPC;
        return NULL;
    }
    pthread_mutex_unlock(&ctx->lock);

    // THE CONTRACT IS THE POINT.  A caller that asked for zeroes and got a live
    // pointer to memory that is not zero has no way to find out, so a failure here
    // must take the allocation with it rather than degrade to pim_alloc's promise.
    if (flags & PIM_ALLOC_F_ZERO) {
        const char *bad = zero_fill(ctx, rec->vbase, bytes);
        if (bad) {
            char keep[PIM_ERRBUF];
            snprintf(keep, sizeof keep, "%s", bad);
            pim_free_ctx(ctx, rec->vbase);
            pim_set_err(ctx, "PIM_ALLOC_F_ZERO could not clear %zu bytes: %s",
                        bytes, keep);
            errno = EIO;
            return NULL;
        }
    }
    return rec->vbase;
}

void pim_free_ctx(pim_ctx *ctx, void *ptr)
{
    pim_alloc_rec *rec;

    if (!ctx || !ptr) return;

    pthread_mutex_lock(&ctx->lock);
    rec = rec_remove(ctx, ptr);
    if (rec) {
        pool_give(ctx, rec->mem, rec->gran_addr, rec->nr_grans);
        pim_pool_trim(ctx, rec->mem, false);
    }
    pthread_mutex_unlock(&ctx->lock);

    if (!rec) {
        pim_set_err(ctx, "pim_free(%p): not a live allocation of this context", ptr);
        return;
    }

    // TABLE ENTRY FIRST, munmap SECOND (design §7.3).  The moment the range is
    // unmapped it can be handed to any other mmap — malloc's included — and if a
    // stale table entry were still there, that reused address would resolve to card
    // memory this allocation no longer owns.
    vspace_release(rec->vbase, rec->bytes);
    free(rec);
}

size_t pim_usable_ctx(const pim_ctx *ctx, const void *ptr)
{
    // The cast is to take the lock, nothing else; the state is only read.
    pim_ctx *mut = (pim_ctx *)ctx;
    pim_alloc_rec *rec;
    size_t left;

    if (!ctx || !ptr) return 0;
    pthread_mutex_lock(&mut->lock);
    rec = rec_lookup(ctx, ptr);
    left = rec ? rec->bytes - (size_t)((const char *)ptr - (const char *)rec->vbase)
               : 0;
    pthread_mutex_unlock(&mut->lock);
    return left;
}

bool pim_where_ctx(const pim_ctx *ctx, const void *ptr, pim_mem *out)
{
    pim_ctx *mut = (pim_ctx *)ctx;
    pim_alloc_rec *rec;

    if (!ctx || !ptr) return false;
    pthread_mutex_lock(&mut->lock);
    rec = rec_lookup(ctx, ptr);
    if (rec && out) *out = rec->mem;
    pthread_mutex_unlock(&mut->lock);
    return rec != NULL;
}

void pim_meminfo_ctx(const pim_ctx *ctx, pim_mem where, uint64_t *allocated,
                 uint64_t *pooled, uint64_t *hugepages_held)
{
    pim_ctx *mut = (pim_ctx *)ctx;
    uint64_t alloc_b = 0, held_b;
    uint32_t i;

    if (!ctx || (unsigned)where >= PIM_NMEM) return;
    pthread_mutex_lock(&mut->lock);
    for (i = 0; i < ctx->nallocs; i++)
        if (ctx->allocs[i]->mem == where)
            alloc_b += ctx->allocs[i]->bytes;
    held_b = (uint64_t)ctx->pool[where].n * ctx->geom.mem[where].hugepage_bytes;
    if (allocated)   *allocated   = alloc_b;
    if (pooled)      *pooled      = held_b - alloc_b;
    if (hugepages_held) *hugepages_held = ctx->pool[where].n;
    pthread_mutex_unlock(&mut->lock);
}

// ------------------------------------------------------------- the tag ------
// Stored, never read by this library.  See pim.h for what it is for.
const char *pim_tag_set_ctx(pim_ctx *ctx, void *ptr, uint64_t tag)
{
    pim_alloc_rec *rec;

    if (!ctx || !ptr) return "pim_tag_set: null argument";
    pthread_mutex_lock(&ctx->lock);
    rec = rec_lookup(ctx, ptr);
    if (rec) rec->tag = tag;
    pthread_mutex_unlock(&ctx->lock);
    return rec ? NULL : pim_set_err(ctx, "pim_tag_set(%p): not a live allocation", ptr);
}

uint64_t pim_tag_get_ctx(const pim_ctx *ctx, const void *ptr)
{
    pim_ctx *mut = (pim_ctx *)ctx;
    pim_alloc_rec *rec;
    uint64_t tag;

    if (!ctx || !ptr) return 0;
    pthread_mutex_lock(&mut->lock);
    rec = rec_lookup(ctx, ptr);
    tag = rec ? rec->tag : 0;
    pthread_mutex_unlock(&mut->lock);
    return tag;
}

//////////////////////////////////////////////////////////////////////////////////
// §5  TRANSLATION — the card address of `ptr`, and how far the card stays contiguous
//////////////////////////////////////////////////////////////////////////////////
const char *pim_resolve_ctx(pim_ctx *ctx, const void *ptr, size_t len,
                        pim_loc *loc, size_t *run)
{
    const pim_region *region;
    pim_alloc_rec *rec;
    size_t off, within, avail;
    uint64_t addr;
    uint32_t gi;

    if (!ctx || !ptr) return pim_set_err(ctx, "pim_resolve: null argument");

    pthread_mutex_lock(&ctx->lock);
    rec = rec_lookup(ctx, ptr);
    if (!rec) {
        pthread_mutex_unlock(&ctx->lock);
        return pim_set_err(ctx, "pim_resolve(%p): not inside a live allocation", ptr);
    }
    off = (size_t)((const char *)ptr - (const char *)rec->vbase);
    if (len > rec->bytes - off) {
        size_t have = rec->bytes - off;
        pthread_mutex_unlock(&ctx->lock);
        return pim_set_err(ctx, "pim_resolve: %zu bytes from %p but only %zu remain "
                              "in that allocation", len, ptr, have);
    }

    region      = &ctx->geom.mem[rec->mem];
    gi      = (uint32_t)(off >> region->gran_shift);
    within = off & (size_t)(region->gran - 1);
    addr   = rec->gran_addr[gi] + within;

    // Extend the run across granule boundaries for as long as consecutive granules
    // are consecutive on the card, so a contiguously-allocated buffer resolves to
    // ONE run and one transfer — which is what the driver's contiguous-run-first
    // policy was for.  In the GPR that is the normal case: the region is linear, so
    // a run of pages from one extent is one span of addresses.
    avail = (size_t)region->gran - within;
    while (avail < len && gi + 1 < rec->nr_grans &&
           rec->gran_addr[gi + 1] == rec->gran_addr[gi] + region->gran) {
        avail += region->gran;
        gi++;
    }

    if (loc) {
        loc->mem = rec->mem;
        loc->axi = addr;
        loc->off = addr - region->base;
    }
    if (run) *run = avail < len ? avail : len;
    pthread_mutex_unlock(&ctx->lock);
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////
// §6  TEARDOWN
//
// One call for pim_close() to make, because the order matters and it is this file's
// business to know it: the records go first (they own the reservations), the pool
// second (it owns the hugepages).  Both are best effort — close(fd) reclaims everything
// this process holds anyway (design §6.2).  That is the guarantee; this is tidiness.
//
// Caller holds ctx->lock.
//////////////////////////////////////////////////////////////////////////////////
void pim_mem_fini(pim_ctx *ctx)
{
    uint32_t i;

    for (i = 0; i < ctx->nallocs; i++) {
        vspace_release(ctx->allocs[i]->vbase, ctx->allocs[i]->bytes);
        free(ctx->allocs[i]);
    }
    free(ctx->allocs);
    ctx->allocs  = NULL;
    ctx->nallocs = ctx->allocs_cap = 0;

    for (int where = 0; where < PIM_NMEM; where++) {
        pim_pool_trim(ctx, (pim_mem)where, true);
        free(ctx->pool[where].v);
        ctx->pool[where].v = NULL;
        ctx->pool[where].n = ctx->pool[where].cap = 0;
    }
}
