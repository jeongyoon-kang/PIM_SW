/* SPDX-License-Identifier: MIT */
/*
 * pim.h — what an application links against.  Allocation and data movement.
 *
 * THE MODEL (design §3.1).  The PIM memory is on a card across PCIe.  It is not in
 * this process's address space, the CPU cannot load or store it, and every byte that
 * reaches it goes over QDMA.  This is the discrete-accelerator model — cudaMalloc
 * and cudaMemcpy, not malloc and a pointer you can dereference.
 *
 * WHAT pim_alloc RETURNS (design §7.3).  A real void *, from a real mmap — but a
 * PROT_NONE one.  The library reserves ADDRESS SPACE it will never back with memory,
 * so the value it hands you is a coordinate that
 *
 *   - is a plain pointer: index it, do arithmetic on it, print it with %p;
 *   - cannot collide with your heap, your stack, or another pim_alloc, because the
 *     kernel's VMA allocator is the thing keeping them apart;
 *   - SIGSEGVs the instant you dereference it, instead of quietly corrupting
 *     whatever host object happened to sit at a made-up address.
 *
 * The cost is one VMA per allocation group.  Zero physical memory, zero page table
 * entries — a PROT_NONE fault has no success path, so nothing is ever populated.
 *
 * ERRORS, AND WHY THEY ARE NOT UNIFORM.
 *   pim_alloc()  returns NULL and sets errno.  It is malloc's contract, and it is
 *                what makes `uint16_t *w = pim_alloc(ctx, n * 2, PIM_MEM_DRAM);` read normally.
 *   everything   returns NULL on success, or a human-readable reason.  The string
 *   else         belongs to the library and stays valid until the next call on that
 *                context.  pim_last_error() repeats the most recent one.
 *
 * NOT THREAD SAFE PER CONTEXT, SAFE ACROSS THEM.  One pim_ctx must be driven by one
 * thread at a time; the internal pool has a lock, but the QDMA file descriptors and
 * the error string do not.  Two threads that each open their own context are fine,
 * and so are two processes (design §6.2) — the driver keeps a separate ledger per
 * fd and reclaims it on close, including after a kill -9.
 *
 * FORK.  A context never crosses fork().  The driver's ledger is per-fd (design
 * §6.2), so a shared fd would give two processes one ledger.
 *
 *   default context   handled for you.  The child drops the inherited one (its fds
 *                     are closed, the parent's are untouched) and opens its own on
 *                     next use.
 *   pim_open contexts the child must open its own; the inherited pim_ctx * is dead.
 *
 * EITHER WAY, POINTERS DO NOT SURVIVE THE FORK.  A pointer from pim_alloc() names
 * hugepages in the parent's ledger, and the child has no route to them.
 *
 * EXEC.  All three fds are O_CLOEXEC, so an exec'd program starts clean.
 *
 * DO NOT LINK THIS ALONGSIDE emulator_top/runtime/pim.h.  Both export pim_alloc and
 * pim_free with incompatible signatures.  The old tree is superseded by this one;
 * see sw/README.md.
 */
#ifndef PIM_PIM_H
#define PIM_PIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "pim_geometry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================== context ==================================== */
typedef struct pim_ctx pim_ctx;

/* Defined in full further down; declared here so the default-context calls below
 * can be grouped where an application will look for them. */
typedef struct pim_loc pim_loc;
typedef enum { PIM_TO_DEV = 0, PIM_FROM_DEV = 1 } pim_dir;

typedef struct {
	/* NULL for each of these takes the default in the comment. */
	const char *dev;	/* "/dev/pim"               — the hugepage ledger  */
	const char *h2c;	/* "/dev/qdma01000-MM-0"    — host to card      */
	const char *c2h;	/* "/dev/qdma01000-MM-1"    — card to host      */

	/* Largest single pread/pwrite.  0 -> 4 MiB.  Also the granularity at which
	 * a long transfer is re-checked for liveness, which is what turns "it died
	 * somewhere in 250 MiB" into "it died at this offset". */
	uint32_t max_xfer;

	/* Trim hysteresis (design §7.4).  Fully empty hugepages kept in each pool
	 * rather than returned to the driver.  0 -> 2.  Returning eagerly makes an
	 * alloc/free loop bounce on ioctl; keeping too many starves anything else
	 * on the card.  Applies to both pools. */
	uint32_t keep_free_hugepages;
} pim_config;

/* ============================== the default context =========================
 * AN APPLICATION DOES NOT NEED TO KNOW WHAT A pim_ctx IS.  Everything below this
 * block comes in two spellings:
 *
 *     pim_alloc    (nbytes, where)          uses the process's default context
 *     pim_alloc_ctx(ctx, nbytes, where)     uses the one you name
 *
 * and the short one is what application code should call.  The long one exists for
 * two callers: the compute runtime, which is handed a context by whoever set the
 * board up, and lib/test, which drives a simulated one.
 *
 * WHY A SINGLETON IS HONEST HERE.  The execution engine — dispatcher, IMEM,
 * doorbell, the accumulator latches that outlive a process — is a singleton on this
 * hardware, and design §11 (2026-08-21) already states that one process owns the
 * board.  A per-process context matches that; threading a handle through every layer
 * implied a multiplicity the hardware does not have.
 *
 * INITIALISATION IS LAZY BUT NOT SILENT.  The first short-form call opens the
 * default context with all-default configuration.  Call pim_init() first if you want
 * to choose the configuration, or simply to find out WHY opening failed: the sentence
 * it returns names the cause ("is pim.ko loaded?", an ABI mismatch, an unusable
 * geometry), whereas a failed pim_alloc() can only set errno to ENODEV.
 *
 * NOT THREAD SAFE, same as any single context (see below).  The lazy open itself is
 * serialised, so racing first calls cannot open twice. */
const char *pim_init(const pim_config *cfg);   /* optional; NULL cfg = defaults */
void        pim_shutdown(void);                /* releases the default context */

/* The default context itself, opened on demand.  NULL if opening failed — ask
 * pim_last_error().  This is the bridge for APIs that still take a context, the
 * compute runtime's among them. */
pim_ctx    *pim_default(void);

const char *pim_last_error(void);
const pim_geometry *pim_geom(void);

void  *pim_alloc(size_t nbytes, pim_mem where);
void   pim_free (void *p);
size_t pim_usable(const void *p);
bool   pim_where (const void *p, pim_mem *out);
void   pim_meminfo(pim_mem where, uint64_t *allocated, uint64_t *pooled,
                   uint64_t *hugepages_held);
const char *pim_resolve(const void *p, size_t len, pim_loc *loc, size_t *run);
const char *pim_memcpy(void *dst, const void *src, size_t n, pim_dir dir,
                       unsigned flags);
const char *pim_sync(void);

/* ============================== explicit contexts ==========================
 * Below here every call names its context.  pim_open/pim_close are how you get one
 * that is not the default; the default is opened for you and closed by
 * pim_shutdown().
 *
 * cfg may be NULL for all defaults.  Opens the ledger, reads the geometry, checks
 * the §4.1 invariant, and opens the two QDMA queues. */
const char *pim_open (const pim_config *cfg, pim_ctx **out);
void        pim_close(pim_ctx *c);
const char *pim_last_error_ctx(const pim_ctx *c);

/* The topology the loaded driver reported.  Valid until pim_close(). */
const pim_geometry *pim_geom_ctx(const pim_ctx *c);

/* ============================== allocation =================================
 * TWO POOLS, ONE ALLOCATOR (design §9.4).  Where an object goes is the caller's
 * choice and it is explicit, because the two are not interchangeable:
 *
 *   PIM_MEM_DRAM   several GiB.  Weight matrices, KV cache.  Interleaved across
 *                  channels and banks, so a request rounds up to a BROADCAST UNIT
 *                  (geom->unit_bytes, 128 KiB on four channels).  It has to: an
 *                  all-bank operation touches the whole unit, so two allocations
 *                  sharing one could not both be operands.
 *   PIM_MEM_GPR    4 MiB.  The staging memory between the host and each channel's
 *                  global buffer — WRVEC multicasts a vector out of it into the GBs,
 *                  RD_MAC unicasts MAC results back into it.  Linear, so a request
 *                  rounds up to a 4 KiB page and nothing more.
 *
 * THE GPR IS MEMORY, NOT SCRATCH.  A vector is uploaded once and re-read by many
 * WRVECs; a result buffer accumulates where its caller put it.  Objects with
 * different birth and death dates are live in there at the same time and only the
 * application knows their lifetimes — which is malloc's problem statement, so it
 * gets malloc's answer.  In particular THERE IS NO LAUNCH-BOUNDARY INVALIDATION:
 * contents persist exactly as long as the allocation does.
 *
 *     uint16_t *w = pim_alloc(c, n * k * 2, PIM_MEM_DRAM);   // weight matrix
 *     uint16_t *v = pim_alloc(c, k * 2,     PIM_MEM_GPR);    // vector
 *     uint16_t *y = pim_alloc(c, n * 2,     PIM_MEM_GPR);    // result buffer
 *
 * NULL and errno on failure — ENOMEM when that pool is full, ENOSPC when the host is
 * out of address space or VMAs.  Ask pim_usable() for what you actually got. */
void  *pim_alloc_ctx(pim_ctx *c, size_t nbytes, pim_mem where);
void   pim_free_ctx (pim_ctx *c, void *p);

/* Bytes actually reserved for p — nbytes rounded up to that pool's granule.  0 if p
 * is not a live allocation of this context. */
size_t pim_usable_ctx(const pim_ctx *c, const void *p);

/* Which pool p came from.  false if p is not a live allocation. */
bool   pim_where_ctx(const pim_ctx *c, const void *p, pim_mem *out);

/* ============================== the layout tag =============================
 * A NOTE THE OWNER ATTACHES TO AN ALLOCATION.  libpim stores it and never looks
 * inside — it is a uint64_t, not a type.
 *
 * WHAT IT IS FOR.  How bytes are arranged inside a PIM allocation is a CONTRACT
 * BETWEEN THREE THINGS THAT RUN AT DIFFERENT TIMES: whoever sized the allocation,
 * whoever filled it, and whoever later builds commands that read it.  Nothing in
 * the hardware or in this library checks that they agreed — a matrix written
 * linearly into an allocation a kernel expects in bank-major order produces a
 * number, not an error, because every MAC simply reads a row that holds something
 * else.
 *
 * So the filler stamps what it wrote, and the code generator states what it
 * assumes; pim_prog_lower() compares them.  A value of 0 means "nobody said",
 * which is what a raw pim_memcpy leaves behind and therefore what a kernel that
 * requires a layout will refuse.
 *
 * WHY libpim DOES NOT INTERPRET IT.  This library knows about hugepages, granules
 * and addresses; it does not know what a MAC is, and a layout is a statement about
 * operations.  Keeping the value opaque is what lets the tag exist here — where the
 * allocation is — without the allocator learning what a GEMV looks like.
 *
 * The tag dies with the allocation.  It is not inherited, not checked on free. */
const char *pim_tag_set(void *p, uint64_t tag);
uint64_t    pim_tag_get(const void *p);
const char *pim_tag_set_ctx(pim_ctx *c, void *p, uint64_t tag);
uint64_t    pim_tag_get_ctx(const pim_ctx *c, const void *p);

/* Per-pool accounting, all in bytes of card memory. */
void   pim_meminfo_ctx(const pim_ctx *c, pim_mem where, uint64_t *allocated,
                   uint64_t *pooled, uint64_t *hugepages_held);

/* ============================== translation ================================
 * Where a PIM pointer actually is, and how many bytes stay contiguous from there.
 * This is libpim's page table, exposed.  (The design doc calls it pim_translate.)
 *
 * It is here because the compute runtime (design §9) is built on it and must not
 * reach into the library's internals to get it: an operation names card addresses
 * and GPR words, and this is the only sanctioned way to turn a pointer into either.
 *
 *   .mem   which pool — so a caller that must have one or the other can say so
 *   .axi   the absolute AXI address.  What pwrite/pread take, in BOTH pools; the
 *          GPR is on the same MM queue at a different address, which is why
 *          pim_memcpy needs no GPR-specific path at all.
 *   .off   the offset within the region.  For the GPR this is what the ISA cares
 *          about: word index = off / PIM_GPR_WORD_BYTES.
 *
 * `run` is bounded by the end of the allocation and by the next discontiguity — for
 * a scattered DRAM allocation that is hugepage_bytes at most.  Walk it in a loop.
 *
 * Not const: a failure is reported through the context's error string, the same as
 * every other call here. */
struct pim_loc {
    pim_mem  mem;
    uint64_t axi;
    uint64_t off;
};

const char *pim_resolve_ctx(pim_ctx *c, const void *p, size_t len,
                        pim_loc *loc, size_t *run);

/* ============================== data movement ==============================
 * Synchronous.  Returns after every piece has landed (design §9.3) — which is what
 * makes it safe to launch an operation on the region immediately afterwards.
 *
 * Exactly one of dst/src is a PIM pointer, chosen by dir; the other is ordinary host
 * memory and may be anything malloc or mmap gave you.  The host side's physical
 * scattering is the QDMA driver's problem and is handled inside one call (design
 * §8.1) — only the CARD side is split here, once per contiguous extent.
 *
 * IT TAKES A GPR POINTER TOO, with no separate API and no separate path.  The GPR is
 * a linear span of the same AXI space reachable over the same MM queue, so the only
 * thing that differs is that the extent walk finds one run instead of several. */
const char *pim_memcpy_ctx(pim_ctx *c, void *dst, const void *src, size_t n,
                       pim_dir dir, unsigned flags);   /* flags: 0 today */

/* ============================== raw windows ================================
 * A transfer to an AXI address that is NOT under a ledger.
 *
 * IT EXISTS FOR ONE THING: the program a launch runs.  IMEM is a single buffer the
 * dispatcher fetches from word 0, owned by whoever holds the engine — so there is
 * nothing for an allocator to hand out and pim_alloc has no answer for it.  The GPR
 * and DRAM go through pim_alloc/pim_memcpy and must keep doing so; this is not a
 * back door to them.
 *
 * pim_axi_allow() must name the window first, and until it does every call here is
 * refused.  libpim does not know where IMEM is and must not guess — the compute
 * runtime does (hwdef/emu_regs.h) and installs it at pim_exec_open().  The guard is
 * not bureaucracy: a DMA to an address that decodes nowhere returns EIO AND LATCHES
 * THE H2C ENGINE, after which every transfer fails until the queue is torn down. */
const char *pim_axi_allow_ctx(pim_ctx *c, uint64_t base, uint64_t bytes);
const char *pim_axi_write_ctx(pim_ctx *c, uint64_t axi, const void *src, size_t n);
const char *pim_axi_read_ctx (pim_ctx *c, uint64_t axi, void *dst, size_t n);

/* A fence, not a cache flush (design §9.3).  There is no CPU mapping of card memory,
 * so there is no cache to flush; what this waits for is every transfer and operation
 * this context has submitted.  Today pim_memcpy is synchronous and this is a no-op
 * that returns NULL — it exists so that the day an async path appears, callers that
 * were already correct stay correct. */
const char *pim_sync_ctx(pim_ctx *c);

#ifdef __cplusplus
}
#endif
#endif /* PIM_PIM_H */
