// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_internal.h — libpim's own state.  Not installed, not includable from outside.
//
// THE LAYERS, AND WHICH FILE OWNS EACH
//   pim_alloc.c    ALLOCATION, all of it.  Host address space (PROT_NONE, §7.3),
//                  the card-memory pool (§7.2, §7.4, §9.4), the record table, and the
//                  translation between them.  One file because it is one data
//                  structure: an allocation's granule table IS the page table.
//   pim_geom.c     address arithmetic.  Pure functions, no state, no context.
//   pim_memcpy.c   moving bytes along that map, over QDMA (design §8).
//   pim_dev.c      the descriptors and the error string.  THE ONLY FILE THAT TOUCHES
//                  THE DRIVER — which is the seam lib/test/pool_test.c cuts along.
//
// Nothing in libpim knows what a command is, or what a WRVEC does with a GPR word.
// That is the runtime, one layer up.
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_INTERNAL_H
#define PIM_INTERNAL_H

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pim/pim.h"
#include "uapi/pim_ioctl.h"

#define PIM_ERRBUF 320

// ---------------------------------------------------------------- pool -------
// One hugepage from the driver, cut into granules.
//
// `used` is a bitmap and grans_per_hugepage is at most 64 — pim_geom_check() refuses a
// geometry where it is not, because everything here would then need a real bitmap
// and nothing yet asks for a hugepage that big.  It is 16 for DRAM (2 MiB / 128 KiB on
// four channels) and 1 for the GPR, where the driver's page IS libpim's granule.
typedef struct {
    uint64_t addr;           // AXI address of granule 0 of this hugepage
    uint64_t used;           // bit i set: granule i is allocated
    uint32_t free_grans;
} pim_hugepage;

// ONE POOL PER REGION, same code, different constants — which is the whole claim of
// design §9.4.  The GPR's pool degenerates to one granule per hugepage and that is not
// a special case anywhere below; it just makes the inner bitmap one bit wide.
typedef struct {
    pim_hugepage *v;
    uint32_t   n, cap;
} pim_pool;

// ------------------------------------------------------------ allocation -----
// One allocation.  The granule table IS the page table (design §7.2): entry i is the
// PIM PHYSICAL ADDRESS of the i-th granule, and the host side is always vbase +
// i*gran.  Translation is a shift and an index, with no syscall — which is the
// reason the driver's DRAM hugepage had to be big enough to preserve the RoChBaCo
// address bits (design §4).
//
// TWO ADDRESS SPACES MEET IN THIS STRUCT, AND MIXING THEM UP IS THE EASIEST
// MISTAKE TO MAKE HERE.  The PIM memory is on an FPGA board behind PCIe; the CPU
// cannot load or store it, and every byte reaches it through QDMA pread/pwrite.
//
//   vbase        HOST VIRTUAL ADDRESS.  What pim_alloc returns.  A PROT_NONE
//                reservation — a name for the allocation, backed by nothing.
//   gran_addr[]  PIM PHYSICAL ADDRESSES, one per granule.  What pwrite takes as its
//                file offset.  (Older comments in this tree call these "card
//                addresses"; same thing.)
//   the struct   ordinary host heap.  It is malloc'd memory like any other.
//
// The host side of an allocation is ALWAYS one contiguous range; the PIM side need
// not be.  That asymmetry is the whole point of the table, and design §7.2's claim.
//
// ------------------------------------------------------------------------------
// gran_addr[] HAS NO SIZE IN ITS DECLARATION — it is a C99 flexible array member, so
// sizeof(pim_alloc_rec) is 24 and EXCLUDES it.  The array's length is chosen when
// the record is malloc'd, by asking for the header plus the elements in ONE hugepage:
//
//     rec = calloc(1, sizeof *rec + (size_t)ngrans * sizeof rec->gran_addr[0]);
//
// For ngrans = 3 that is 24 + 24 = 48 bytes laid out as
//
//   off  0        8        16   20        24       32       40      48
//       +--------+--------+----+--------+--------+--------+--------+
//       | vbase  | bytes  |mem |nr_grans|gran[0] |gran[1] |gran[2] |
//       +--------+--------+----+--------+--------+--------+--------+
//       |<------- sizeof *a = 24 ------>|<---- ngrans * 8 = 24 --->|
//                                        ^
//                                        a->gran_addr starts here
//
// so a->gran_addr is not a stored pointer — it IS offset 24 of the record.
//
// WHY ONE ALLOCATION AND NOT TWO (a header plus a separate uint64_t array):
//   - one malloc and one free, so one failure path instead of two;
//   - a->gran_addr[u] is one memory reference, not a pointer chased twice.  That
//     read is on the transfer path — pim_resolve() does it for every extent;
//   - nr_grans and the array cannot get out of step, because there is no second
//     object to get out of step with.
//
// AND IT IS WHY nr_grans MUST BE STORED.  A flexible array does not know its own
// length and sizeof cannot report it, so the bound has to be written down; see
// pim_resolve()'s `u + 1 < a->nr_grans`.
typedef struct {
    void     *vbase;         // host VA.  PROT_NONE reservation, never dereferenced
    size_t    bytes;         // request rounded up to a whole granule
    pim_mem   mem;           // which pool it came from
    uint32_t  nr_grans;      // length of gran_addr[] — sizeof cannot tell you
    uint64_t  tag;           // opaque; see pim_tag_set().  0 = untagged
    uint64_t  gran_addr[];   // [nr_grans] PIM physical addresses.  Flexible array
                             // member: NOT counted by sizeof(pim_alloc_rec)
} pim_alloc_rec;

// ------------------------------------------------------------------ ctx ------
struct pim_ctx {
    int      fd;             // /dev/pim — the ledgers
    int      h2c, c2h;       // QDMA MM queues.  The AXI address IS the file offset,
                             // for DRAM and the GPR alike.
    uint32_t max_xfer;
    uint32_t keep_free_hugepages;

    pim_geometry geom;
    pim_pool     pool[PIM_NMEM];

    // Allocations from BOTH pools in one registry, sorted by vbase so a pointer
    // anywhere inside one is a binary search away.  One registry rather than two
    // because the kernel's VMA allocator already guarantees the reservations do not
    // overlap each other — whichever pool they belong to.
    pim_alloc_rec **allocs;
    uint32_t        nallocs, allocs_cap;

    // The one window pim_axi_write/read may touch, installed by the compute
    // runtime at pim_exec_open().  Zero means "nothing is allowed", which is the
    // state an application that never runs a program stays in.
    uint64_t raw_base, raw_bytes;

    pthread_mutex_t lock;    // covers pool[] + allocs.  NOT the fds, NOT err.
    char err[PIM_ERRBUF];
};

// pim_dev.c
const char *pim_set_err(pim_ctx *c, const char *fmt, ...);
int         pim_ioctl_alloc(pim_ctx *c, pim_mem m, uint32_t nr_hugepages, uint64_t vbase,
                            struct pim_extent *ext, uint32_t cap, uint32_t *nex);
int         pim_ioctl_free (pim_ctx *c, pim_mem m, const struct pim_extent *ext,
                            uint32_t nex, uint64_t vbase);

// pim_alloc.c — the allocator's whole cross-file surface.  Everything else about an
// allocation, the record table and the pool included, is static in there; the rest of
// libpim reaches an allocation through the PUBLIC API (pim_resolve, pim_usable,
// pim_where) like any application would.  Caller holds c->lock for both of these.
void            pim_pool_trim (pim_ctx *ctx, pim_mem where, bool all);  // test/pool_test.c
void            pim_mem_fini  (pim_ctx *ctx);                       // pim_dev.c

#endif // PIM_INTERNAL_H
