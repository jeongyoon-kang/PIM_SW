// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_dev.c — the file descriptors, and the one place that talks to the driver.
//
// THREE DESCRIPTORS AND NO MAPPING.
//   /dev/pim              the ledgers, one per region.  ioctl only — no read, no
//                         write, no mmap; see the driver's pim_file.c for why each
//                         is absent.
//   /dev/qdma...-MM-0/1   the data path.  On a QDMA MM queue the AXI address IS the
//                         file offset, so a card address goes straight into pwrite —
//                         and that is as true of the GPR as of DRAM.  The GPR is a
//                         different span of the same AXI space on the same queue,
//                         which is why adding it needed no new transport at all.
//
// There is deliberately NO control-BAR mapping here.  That decision (design §6.1,
// §9.1) is what keeps this library out of the register plane entirely — and it has a
// cost worth stating: the old runtime detected a board that had stopped answering by
// reading a status register and seeing all-ones.  This stack cannot, so a dead board
// surfaces as EIO from QDMA and nothing sooner.  See sw/README.md.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DEV_DEFAULT        "/dev/pim"
#define H2C_DEFAULT        "/dev/qdma01000-MM-0"
#define C2H_DEFAULT        "/dev/qdma01000-MM-1"
#define MAX_XFER_DEFAULT   (4u << 20)
#define KEEP_FREE_DEFAULT  2u

// The ABI's whole promise is that both sides lay these out identically.  Asserting
// the sizes here and in the module's pim_main.c turns a field reordered on one side
// into a build failure instead of a wrong number at run time.
_Static_assert(sizeof(struct pim_region_info) == 40,  "pim_region_info layout changed: bump PIM_ABI_VERSION");
_Static_assert(sizeof(struct pim_info)        == 112, "pim_info layout changed: bump PIM_ABI_VERSION");
_Static_assert(sizeof(struct pim_extent)      == 16,  "pim_extent layout changed: bump PIM_ABI_VERSION");
_Static_assert(sizeof(struct pim_alloc_req)   == 32,  "pim_alloc_req layout changed: bump PIM_ABI_VERSION");
_Static_assert(sizeof(struct pim_free_req)    == 32,  "pim_free_req layout changed: bump PIM_ABI_VERSION");

// Errors raised before a context exists have nowhere else to live.  Not reentrant,
// and open is not a hot path.
static char open_err[PIM_ERRBUF];

const char *pim_set_err(pim_ctx *c, const char *fmt, ...)
{
    char *buf = c ? c->err : open_err;
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(buf, PIM_ERRBUF, fmt, ap);
    va_end(ap);
    return buf;
}

const char *pim_last_error_ctx(const pim_ctx *c)
{
    return (c && c->err[0]) ? c->err : NULL;
}

const pim_geometry *pim_geom_ctx(const pim_ctx *c)
{
    return c ? &c->geom : NULL;
}

// ------------------------------------------------------------ the driver ----
static const char *region_name(pim_mem m)
{
    return m == PIM_MEM_GPR ? "gpr" : "dram";
}

int pim_ioctl_alloc(pim_ctx *c, pim_mem m, uint32_t nr_hugepages, uint64_t vbase,
                    struct pim_extent *ext, uint32_t cap, uint32_t *nex)
{
    struct pim_alloc_req req;

    memset(&req, 0, sizeof req);
    req.region     = (uint32_t)m;
    req.nr_hugepages  = nr_hugepages;
    req.nr_extents = cap;
    req.vbase      = vbase;
    req.extents    = (uint64_t)(uintptr_t)ext;

    if (ioctl(c->fd, PIM_IOC_ALLOC, &req) < 0) {
        pim_set_err(c, "ALLOC %s x%u: %s", region_name(m), nr_hugepages,
                    strerror(errno));
        return -1;
    }
    *nex = req.nr_extents;
    return 0;
}

int pim_ioctl_free(pim_ctx *c, pim_mem m, const struct pim_extent *ext,
                   uint32_t nex, uint64_t vbase)
{
    struct pim_free_req req;

    memset(&req, 0, sizeof req);
    req.region     = (uint32_t)m;
    req.nr_extents = nex;
    req.vbase      = vbase;
    req.extents    = (uint64_t)(uintptr_t)ext;

    if (ioctl(c->fd, PIM_IOC_FREE, &req) < 0) {
        pim_set_err(c, "FREE %s, %u extent(s): %s", region_name(m), nex,
                    strerror(errno));
        return -1;
    }
    return 0;
}

// ------------------------------------------------------------------ open ----
static const char *bail(pim_ctx *c, const char *msg)
{
    // Copy out before the context dies: the caller gets a reason, not a dangling
    // pointer into freed memory.
    if (c) {
        if (c->err[0]) snprintf(open_err, sizeof open_err, "%s", c->err);
        if (c->fd  >= 0) close(c->fd);
        if (c->h2c >= 0) close(c->h2c);
        if (c->c2h >= 0) close(c->c2h);
        free(c);
    }
    return msg ? msg : open_err;
}

const char *pim_open(const pim_config *cfg, pim_ctx **out)
{
    pim_config      d;
    struct pim_info info;
    pim_ctx        *c;
    const char     *bad;
    int             m;

    if (!out)
        return pim_set_err(NULL, "pim_open: out is NULL");
    *out = NULL;

    memset(&d, 0, sizeof d);
    if (cfg) d = *cfg;
    if (!d.dev)  d.dev  = DEV_DEFAULT;
    if (!d.h2c)  d.h2c  = H2C_DEFAULT;
    if (!d.c2h)  d.c2h  = C2H_DEFAULT;
    if (!d.max_xfer)         d.max_xfer         = MAX_XFER_DEFAULT;
    if (!d.keep_free_hugepages) d.keep_free_hugepages = KEEP_FREE_DEFAULT;

    c = calloc(1, sizeof *c);
    if (!c) return pim_set_err(NULL, "out of memory");
    c->fd = c->h2c = c->c2h = -1;
    c->max_xfer         = d.max_xfer;
    c->keep_free_hugepages = d.keep_free_hugepages;
    pthread_mutex_init(&c->lock, NULL);

    c->fd = open(d.dev, O_RDWR | O_CLOEXEC);
    if (c->fd < 0) {
        pim_set_err(c, "open(%s): %s — is pim.ko loaded?  See sw/drv/Makefile",
                    d.dev, strerror(errno));
        return bail(c, NULL);
    }

    memset(&info, 0, sizeof info);
    if (ioctl(c->fd, PIM_IOC_GET_INFO, &info) < 0) {
        pim_set_err(c, "GET_INFO: %s", strerror(errno));
        return bail(c, NULL);
    }
    // A stale library against a fresh module is the one mismatch that would
    // otherwise look like data corruption instead of a version problem.
    if (info.abi_version != PIM_ABI_VERSION) {
        pim_set_err(c, "ABI mismatch: this libpim speaks %u, the loaded module "
                       "speaks %u", PIM_ABI_VERSION, info.abi_version);
        return bail(c, NULL);
    }

    c->geom.nch        = info.nch;
    c->geom.nbank      = info.nbank;
    c->geom.row_bytes  = info.row_bytes;
    c->geom.map        = (pim_map)info.addr_map;
    c->geom.unit_bytes = info.unit_bytes;
    for (m = 0; m < PIM_NMEM; m++) {
        c->geom.mem[m].base        = info.region[m].base;
        c->geom.mem[m].bytes       = info.region[m].bytes;
        c->geom.mem[m].hugepage_bytes = info.region[m].hugepage_bytes;
        c->geom.mem[m].nr_hugepages   = info.region[m].nr_hugepages;
    }
    pim_geom_derive(&c->geom);

    bad = pim_geom_check(&c->geom);
    if (bad) {
        pim_set_err(c, "the loaded module reports an unusable geometry: %s", bad);
        return bail(c, NULL);
    }

    // RoChBaCo ONLY, for now, and by decision rather than by omission: the whole
    // stack above assumes a broadcast unit is nch consecutive RoBaCo rows, which is
    // what the interleaved map gives.  ChRoBaCo puts 4 GiB between a supergroup's
    // channel replicas and every layout question above changes with it.  (The GPR is
    // unaffected either way — it is not interleaved.)
    if (c->geom.map != PIM_MAP_RO_CH_BA_CO) {
        pim_set_err(c, "the module reports ChRoBaCo; this stack implements RoChBaCo "
                       "only.  Reload with pim_addr_map=1 and set the board's "
                       "MODE_CTRL to match.");
        return bail(c, NULL);
    }

    c->h2c = open(d.h2c, O_WRONLY | O_CLOEXEC);
    if (c->h2c < 0) {
        pim_set_err(c, "open(%s): %s — run scripts/qdma_queues.sh setup",
                    d.h2c, strerror(errno));
        return bail(c, NULL);
    }
    c->c2h = open(d.c2h, O_RDONLY | O_CLOEXEC);
    if (c->c2h < 0) {
        pim_set_err(c, "open(%s): %s — run scripts/qdma_queues.sh setup",
                    d.c2h, strerror(errno));
        return bail(c, NULL);
    }

    *out = c;
    return NULL;
}

void pim_close(pim_ctx *c)
{
    if (!c) return;

    pthread_mutex_lock(&c->lock);
    pim_mem_fini(c);
    pthread_mutex_unlock(&c->lock);

    pthread_mutex_destroy(&c->lock);
    if (c->fd  >= 0) close(c->fd);      // the driver reclaims whatever is left
    if (c->h2c >= 0) close(c->h2c);
    if (c->c2h >= 0) close(c->c2h);
    free(c);
}

// A FENCE, NOT A CACHE FLUSH (design §9.3).  There is no CPU mapping of card memory
// in this design, so no host cache can hold a copy of it and there is nothing to
// flush; what a caller actually needs is "everything I submitted has landed".
// pim_memcpy is synchronous, so today that is already true on return and this
// returns immediately.  It exists now so that code written correctly against a
// synchronous library stays correct when an asynchronous path appears.
const char *pim_sync_ctx(pim_ctx *c)
{
    (void)c;
    return NULL;
}

//////////////////////////////////////////////////////////////////////////////////
// THE DEFAULT CONTEXT
//
// One per process, opened on demand.  Everything below is a one-line forward to the
// _ctx form; the only real code is the lazy open, and the only reason it is here
// rather than in pim_alloc.c is that opening is this file's business.
//
// WHY IT IS A SINGLETON AND NOT A CURRENT-CONTEXT POINTER.  CUDA's thread-local
// "current device" buys the ability to switch, and pays for it with a whole class of
// bug where a call lands on the wrong one.  There is one board and design §11 says
// one process owns it, so there is nothing to switch between; a caller that really
// wants a second context can still open one and use the _ctx calls.
//
// pim_shutdown() exists for a process that wants the driver's hugepages back before
// it exits.  Not calling it is fine — close(fd) reclaims everything (design §6.2).
//////////////////////////////////////////////////////////////////////////////////
static pim_ctx        *g_ctx;
static const char     *g_open_err;      // why the lazy open failed, if it did
static pim_config      g_cfg;
static bool            g_cfg_set, g_tried, g_atfork;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// AFTER fork() THE CHILD MUST NOT USE THE PARENT'S CONTEXT.  The driver's ledger is
// per-fd (design §6.2), so a shared fd would give two processes one ledger and each
// would free hugepages the other still holds.  pim.h has always said "a child must
// call pim_open() for itself"; with a default context that rule cannot be left to
// the caller, because the child reaches the context by doing nothing at all.
//
// So the child drops it: the fds are closed (a separate descriptor from the
// parent's, so the parent is unaffected — the driver's release() runs only when the
// last one goes) and the lazy open is armed again, giving the child its own context
// and its own ledger on next use.  The pim_ctx itself is leaked rather than freed:
// free() is not async-signal-safe and a forked child is expected to exec or exit.
//
// Pointers the parent got from pim_alloc() are NOT valid in the child.  They name
// hugepages in the parent's ledger, which the child no longer has a route to.
static void after_fork_in_child(void)
{
    if (g_ctx) {
        if (g_ctx->fd  >= 0) close(g_ctx->fd);
        if (g_ctx->h2c >= 0) close(g_ctx->h2c);
        if (g_ctx->c2h >= 0) close(g_ctx->c2h);
    }
    pthread_mutex_init(&g_lock, NULL);      // the parent may have held it
    g_ctx = NULL;
    g_open_err = NULL;
    g_tried = false;
}

static void lazy_open(void)
{
    pthread_mutex_lock(&g_lock);
    if (!g_atfork) {
        pthread_atfork(NULL, NULL, after_fork_in_child);
        g_atfork = true;
    }
    if (!g_tried) {
        g_tried = true;
        g_open_err = pim_open(g_cfg_set ? &g_cfg : NULL, &g_ctx);
    }
    pthread_mutex_unlock(&g_lock);
}

// CONFIGURATION THAT ARRIVES TOO LATE IS AN ERROR, NOT A SHRUG.  The context opens
// on first use, so a pim_init(cfg) after that could only be ignored — and a caller
// who asked for a different device or a different trim threshold and silently got
// the defaults has no way to notice.
const char *pim_init(const pim_config *cfg)
{
    bool already;

    pthread_mutex_lock(&g_lock);
    already = g_tried;
    if (cfg && !already) { g_cfg = *cfg; g_cfg_set = true; }
    pthread_mutex_unlock(&g_lock);

    if (cfg && already)
        return "pim_init: the default context is already open; configuration would "
               "be ignored.  Call pim_init() before the first allocation, or use "
               "pim_open() for a context of your own.";

    lazy_open();
    return g_ctx ? NULL : (g_open_err ? g_open_err : "pim_init: open failed");
}

pim_ctx *pim_default(void)
{
    lazy_open();
    return g_ctx;
}

// Does not re-arm the lazy open: a context that has been shut down stays shut down.
// Re-opening would need every pointer handed out before it to be invalid, and
// nothing tracks that.
void pim_shutdown(void)
{
    pthread_mutex_lock(&g_lock);
    pim_close(g_ctx);
    g_ctx = NULL;
    g_tried = true;
    g_open_err = "pim_shutdown() was called; the default context is gone";
    pthread_mutex_unlock(&g_lock);
}

// A failed open has no context to hold the message, so it is reported from here.
const char *pim_last_error(void)
{
    pim_ctx *c = pim_default();
    return c ? pim_last_error_ctx(c) : (g_open_err ? g_open_err : "(no error)");
}

const pim_geometry *pim_geom(void)         { return pim_geom_ctx(pim_default()); }

void *pim_alloc(size_t nbytes, pim_mem where)
{
    pim_ctx *c = pim_default();
    if (!c) { errno = ENODEV; return NULL; }
    return pim_alloc_ctx(c, nbytes, where);
}

void   pim_free  (void *p)                     { pim_free_ctx(pim_default(), p); }
size_t pim_usable(const void *p)               { return pim_usable_ctx(pim_default(), p); }
bool   pim_where (const void *p, pim_mem *out) { return pim_where_ctx(pim_default(), p, out); }

void pim_meminfo(pim_mem where, uint64_t *allocated, uint64_t *pooled,
                 uint64_t *hugepages_held)
{ pim_meminfo_ctx(pim_default(), where, allocated, pooled, hugepages_held); }

const char *pim_resolve(const void *p, size_t len, pim_loc *loc, size_t *run)
{
    pim_ctx *c = pim_default();
    return c ? pim_resolve_ctx(c, p, len, loc, run) : pim_last_error();
}

const char *pim_memcpy(void *dst, const void *src, size_t n, pim_dir dir,
                       unsigned flags)
{
    pim_ctx *c = pim_default();
    return c ? pim_memcpy_ctx(c, dst, src, n, dir, flags) : pim_last_error();
}

const char *pim_tag_set(void *p, uint64_t tag)
{ return pim_tag_set_ctx(pim_default(), p, tag); }

uint64_t pim_tag_get(const void *p)
{ return pim_tag_get_ctx(pim_default(), p); }

const char *pim_sync(void)
{
    pim_ctx *c = pim_default();
    return c ? pim_sync_ctx(c) : pim_last_error();
}
