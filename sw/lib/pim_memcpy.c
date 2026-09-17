// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_memcpy.c — moving bytes along the map (design §8).
//
// TWO KINDS OF SCATTERING, TWO OWNERS, AND ONLY ONE OF THEM IS OURS.
//
//   host side    The source buffer's physical pages are scattered too, and that is
//                entirely the QDMA driver's problem: hand it a user virtual address
//                and it pins the pages and builds the scatter-gather chain itself.
//                512 physical fragments is still ONE pwrite.  We implement none of
//                that and must not try to.
//
//   card side    pwrite names a single starting offset, so one call covers one
//                contiguous run of card addresses.  A scattered allocation is
//                therefore one call per run — which is all that the design's
//                "non-contiguity costs transfers, not correctness" amounts to in
//                code.
//
// AND THE COST IS SMALL.  A DRAM run is at least one broadcast unit (128 KiB) and
// usually a whole hugepage (2 MiB); a 2 MiB PCIe transfer is hundreds of microseconds
// while a pwrite's syscall and descriptor setup is a few.  ~1 %.  That number is the
// reason there is no compaction anywhere in this stack.
//
// CARD-TO-CARD IS NOT CHECKED FOR.  `dir` names which side is the card and the other
// side is taken to be host memory; passing two PIM pointers is a caller bug that
// surfaces as EFAULT out of pwrite, because a PROT_NONE reservation has no pages to
// pin.  It fails loudly either way, so this file does not carry a registry lookup to
// improve the wording — which is what keeps it dependent on the PUBLIC API alone.
//
// THE GPR NEEDS NOTHING SPECIAL HERE (design §9.4).  It is a linear span of the same
// AXI address space on the same MM queue, so a GPR pointer walks the same loop and
// simply finds one run where DRAM might find several.  The only thing that would
// have justified a separate path is a separate transport, and there is not one.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim_internal.h"

#include <errno.h>
#include <string.h>
#include <unistd.h>

// The transport loop, shared by pim_memcpy and the raw-window calls.  `axi` walks
// with `host`, so a caller that has already split its range by extent hands one run
// at a time and this only splits further for max_xfer.
static const char *xfer(pim_ctx *c, pim_dir dir, uint64_t axi, char *host, size_t n,
                        size_t reported_at, size_t reported_of)
{
    while (n) {
        size_t  want = n > c->max_xfer ? c->max_xfer : n;
        ssize_t k;

        if (dir == PIM_TO_DEV)
            k = pwrite(c->h2c, host, want, (off_t)axi);
        else
            k = pread(c->c2h, host, want, (off_t)axi);

        if (k < 0)
            return pim_set_err(c, "%s %zu B at card %#llx (offset %zu of %zu): %s",
                               dir == PIM_TO_DEV ? "pwrite" : "pread", want,
                               (unsigned long long)axi, reported_at, reported_of,
                               strerror(errno));
        if (k == 0)
            return pim_set_err(c, "%s returned 0 at card %#llx — the queue is not "
                                  "accepting; check qdma_queues.sh status",
                               dir == PIM_TO_DEV ? "pwrite" : "pread",
                               (unsigned long long)axi);
        host        += (size_t)k;
        axi         += (uint64_t)k;
        n           -= (size_t)k;
        reported_at += (size_t)k;
    }
    return NULL;
}

// ---------------------------------------------------------- raw windows -----
const char *pim_axi_allow_ctx(pim_ctx *c, uint64_t base, uint64_t bytes)
{
    if (!c) return pim_set_err(NULL, "pim_axi_allow: ctx is NULL");
    c->raw_base  = base;
    c->raw_bytes = bytes;
    return NULL;
}

static const char *raw_ok(pim_ctx *c, uint64_t axi, size_t n)
{
    if (!c->raw_bytes)
        return pim_set_err(c, "no raw AXI window is open; pim_axi_allow() first "
                              "(the compute runtime does this at pim_exec_open)");
    if (axi < c->raw_base || axi - c->raw_base > c->raw_bytes ||
        (uint64_t)n > c->raw_bytes - (axi - c->raw_base))
        return pim_set_err(c, "%#llx + %zu is outside the allowed window "
                              "[%#llx, %#llx).  A DMA to an address that decodes "
                              "nowhere latches the H2C engine.",
                           (unsigned long long)axi, n,
                           (unsigned long long)c->raw_base,
                           (unsigned long long)(c->raw_base + c->raw_bytes));
    return NULL;
}

const char *pim_axi_write_ctx(pim_ctx *c, uint64_t axi, const void *src, size_t n)
{
    const char *bad;

    if (!c || !src) return pim_set_err(c, "pim_axi_write: null argument");
    if (!n) return NULL;
    if ((bad = raw_ok(c, axi, n))) return bad;
    return xfer(c, PIM_TO_DEV, axi, (char *)(uintptr_t)src, n, 0, n);
}

const char *pim_axi_read_ctx(pim_ctx *c, uint64_t axi, void *dst, size_t n)
{
    const char *bad;

    if (!c || !dst) return pim_set_err(c, "pim_axi_read: null argument");
    if (!n) return NULL;
    if ((bad = raw_ok(c, axi, n))) return bad;
    return xfer(c, PIM_FROM_DEV, axi, dst, n, 0, n);
}

const char *pim_memcpy_ctx(pim_ctx *c, void *dst, const void *src, size_t n,
                       pim_dir dir, unsigned flags)
{
    const char *card_p;
    char       *host;
    size_t      done = 0;

    if (!c)                return pim_set_err(NULL, "pim_memcpy: ctx is NULL");
    if (!dst || !src)      return pim_set_err(c, "pim_memcpy: null pointer");
    if (flags)             return pim_set_err(c, "pim_memcpy: flags must be 0 "
                                                 "(none are defined yet)");
    if (dir != PIM_TO_DEV && dir != PIM_FROM_DEV)
        return pim_set_err(c, "pim_memcpy: dir is neither PIM_TO_DEV nor "
                              "PIM_FROM_DEV");
    if (!n) return NULL;

    // The cast off const is safe: for PIM_TO_DEV the host side is src and is only
    // read, and pread only ever writes through the PIM_FROM_DEV branch's dst.
    card_p = (dir == PIM_TO_DEV) ? (const char *)dst : (const char *)src;
    host   = (dir == PIM_TO_DEV) ? (char *)(uintptr_t)(const void *)src : (char *)dst;

    while (done < n) {
        pim_loc  loc;
        size_t   run;
        uint64_t addr;
        const char *bad = pim_resolve_ctx(c, card_p + done, n - done, &loc, &run);

        if (bad)
            return bad;
        addr = loc.axi;

        bad = xfer(c, dir, addr, host + done, run, done, n);
        if (bad)
            return bad;
        done += run;
    }
    return NULL;
}
