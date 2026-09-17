/* SPDX-License-Identifier: MIT */
/*
 * pim_ioctl.h — the /dev/pim contract.  COMPILED BY BOTH SIDES.
 *
 * This is the only file that sw/drv and sw/lib share, and it is the reason the two
 * can be written independently.  Everything the driver does is here; everything
 * libpim may assume about the driver is here.  Nothing else crosses.
 *
 * WHAT THE DRIVER IS  (design §6.1, §9.4)
 *   A ledger — now TWO of them, one per device memory region.  Each owns a
 *   contiguous span of CARD AXI ADDRESS SPACE, cuts it into fixed hugepages, hands them
 *   out per-fd, and takes them all back when the fd closes, including after a kill.
 *
 *     PIM_REGION_DRAM   several GiB, 2 MiB hugepages.  Weight matrices and KV cache.
 *                    RoChBaCo interleaved — libpim cuts it further, into broadcast
 *                    units, because that is what one all-bank operation consumes.
 *     PIM_REGION_GPR    4 MiB, 4 KiB hugepages.  Vectors and result buffers.  Linear, no
 *                    interleave, so a hugepage is also libpim's granule.
 *
 *   THE SAME CODE RUNS BOTH.  Only the constants differ (design §9.4): bitmap ledger,
 *   two-stage first-fit, per-fd ownership, reclaim on release.  A third region would
 *   be a table entry.
 *
 * WHY THE GPR IS AN ALLOCATOR'S PROBLEM AND NOT A SCRATCHPAD.  A vector is uploaded
 * once and re-read by many WRVECs; a result buffer accumulates where its caller put
 * it.  So objects with different birth and death dates are alive in there at the
 * same time and only the application knows their lifetimes.  That is malloc's
 * problem statement, and it gets malloc's answer.
 *
 * WHAT THE DRIVER IS NOT
 *   It does not know what a broadcast unit is, it does not know RoChBaCo, it does not
 *   know what a GPR word is, it does not move data, and it does not touch a control
 *   register.  Data goes over QDMA (design §8) and geometry is interpreted by libpim
 *   (design §5).  The geometry fields below are CARRIED, not USED.
 *
 * ABI RULE
 *   Fields are explicitly sized and 8-byte aligned, and every struct is padded to a
 *   multiple of 8, so a 32-bit userspace sees the same layout as the kernel and
 *   .compat_ioctl can be the same handler.  Add fields only at the end, and bump
 *   PIM_ABI_VERSION when you do.
 */
#ifndef _UAPI_PIM_IOCTL_H
#define _UAPI_PIM_IOCTL_H

#include <linux/types.h>
#include <linux/ioctl.h>

#define PIM_DEV_PATH        "/dev/pim"

/* Bumped whenever a struct below grows or changes meaning.  pim_open() compares it
 * against the value it was compiled with and refuses a mismatch — a stale libpim
 * against a fresh module is the one failure that would otherwise look like data
 * corruption instead of a version problem.
 *
 *   1  DRAM hugepages only
 *   2  two regions (design §9.4): the ledger became an array and the requests grew
 *      a `mem` selector
 */
#define PIM_ABI_VERSION     2u

/* ---------------------------------------------------------------- regions ---
 * NAMED "REGION" AND NOT "MEMORY KIND" ON PURPOSE.  To this driver they are two
 * spans of card address space with different hugepage sizes and nothing else; that one
 * holds weights and the other is staging for WRVEC is entirely libpim's knowledge.
 * The public API in include/pim/pim.h calls the same numbers PIM_MEM_DRAM and
 * PIM_MEM_GPR, which is the vocabulary a caller thinks in. */
#define PIM_REGION_DRAM	0u
#define PIM_REGION_GPR	1u
#define PIM_NREGION	2u

/*
 * One region's ledger, as reported.  `hugepage_bytes` is the DRIVER's granule and is
 * all the driver knows; what libpim cuts a hugepage into afterwards is libpim's
 * business and differs between the two regions:
 *
 *   DRAM   hugepage 2 MiB -> 16 broadcast units of 128 KiB (4 channels).  The two-level
 *          split exists because the unit is what an operation addresses and the
 *          hugepage is what preserves the RoChBaCo address bits (design §4).
 *   GPR    hugepage 4 KiB -> 1.  Linear memory, nothing to preserve, so there is no
 *          second level and libpim just rounds requests to the page (design §9.4).
 */
struct pim_region_info {
	__u64	base;		/* AXI address of hugepage 0                     */
	__u64	bytes;		/* size of the region                         */
	__u64	hugepage_bytes;	/* the driver's allocation granule            */
	__u64	nr_hugepages;	/* bytes / hugepage_bytes                        */
	__u64	nr_free;	/* ADVISORY.  Stale the instant it is read.   */
};

/* ---------------------------------------------------------------- geometry ---
 * Everything the driver was told at insmod, plus what it derived.  libpim asks once
 * at open and never again; these do not change while a module is loaded.
 *
 * unit_bytes IS DERIVED, not a parameter: nch * nbank * row_bytes.  One broadcast
 * unit is what a single all-bank operation across every channel consumes, and it is
 * the granule libpim allocates DRAM in (design §3.2, §7.4).  The driver computes it
 * only so that both sides get the same number from the same arithmetic.  It has
 * nothing to do with the GPR.
 */
struct pim_info {
	__u32	abi_version;	/* PIM_ABI_VERSION of the loaded module */
	__u32	flags;		/* PIM_INFO_* below                     */
	__u32	nch;		/* channels                             */
	__u32	nbank;		/* banks per channel                    */
	__u32	row_bytes;	/* row buffer — 2048 on this emulator    */
	__u32	addr_map;	/* PIM_MAP_* — carried, for libpim      */
	__u64	unit_bytes;	/* nch * nbank * row_bytes  (DRAM only) */
	struct pim_region_info region[PIM_NREGION];
};

/* The channel address map the DRAM region was programmed for.  The driver never
 * decodes an address, so this is a value it passes through from its module parameter
 * to whoever does.  THE GPR IS NOT INTERLEAVED and this does not apply to it. */
#define PIM_MAP_CHROBACO	0u	/* channel is the top field, one channel every bytes/nch */
#define PIM_MAP_ROCHBACO	1u	/* channel sits between row and bank, one channel every 32 KiB */

#define PIM_INFO_HAS_OWNERS	0x1u	/* module tracks per-hugepage owners (/proc/pim) */

/* ------------------------------------------------------------------ extent ---
 * A run of consecutive hugepages in one region.  The unit of both allocation results
 * and frees.
 *
 * WHY EXTENTS AND NOT AN INDEX ARRAY.  The driver's first-fit tries for one
 * contiguous run before it falls back to picking hugepages singly (design §6.3), and
 * the whole point of that try is that a contiguous result collapses to ONE pwrite
 * later.  An index array would throw that information away at the boundary and make
 * libpim rediscover it.  It is worth just as much in the GPR, where a linear region
 * means a contiguous run is always one transfer.
 */
struct pim_extent {
	__u64	addr;		/* AXI address of the first hugepage of the run */
	__u32	nr_hugepages;	/* consecutive hugepages, >= 1                  */
	__u32	__pad;
};

/* --------------------------------------------------------------- ALLOC/FREE ---
 * nr_extents is in-capacity on the way down and out-count on the way back.
 *
 * SIZING IT.  nr_extents == nr_hugepages is ALWAYS enough — the worst case is every
 * hugepage landing in its own run.  A caller that sizes it that way can never see
 * -ENOSPC.
 *
 * IF IT IS TOO SMALL the call allocates NOTHING, returns -ENOSPC, and leaves
 * nr_extents holding the count that would have been needed.  Half-succeeding would
 * hand back an allocation the caller cannot describe well enough to free.
 *
 * vbase IS OPTIONAL (design §7.3).  It is the host address-space reservation libpim
 * made.  The driver only records it, for /proc/pim and for a cheap overlap check
 * inside one fd; 0 means "not registered" and nothing behaves differently.
 */
struct pim_alloc_req {
	__u32	region;		/* in: PIM_REGION_*                          */
	__u32	nr_hugepages;	/* in                                        */
	__u32	nr_extents;	/* in: capacity.  out: extents written.      */
	__u32	flags;		/* in: PIM_ALLOC_*                           */
	__u64	vbase;		/* in, optional: host VA reservation, or 0   */
	__u64	extents;	/* in: user pointer to struct pim_extent[]   */
};

/* Refuse rather than fall back to a scattered result.  For a caller that would
 * rather fail than pay the extra transfers; NOT the default, because availability
 * is the reason the fallback exists (design §6.3). */
#define PIM_ALLOC_CONTIG	0x1u

/* Frees are described the same way they were handed out.  Every extent must name
 * hugepages this fd actually owns in that region; the call frees all of them or none. */
struct pim_free_req {
	__u32	region;		/* in: PIM_REGION_*                          */
	__u32	nr_extents;	/* in                                        */
	__u32	flags;		/* in: must be 0                             */
	__u32	__pad;
	__u64	extents;	/* in: user pointer to struct pim_extent[]   */
	__u64	vbase;		/* in, optional: the vbase given to ALLOC    */
};

/* ------------------------------------------------------------------ ioctls ---
 * The magic exists to catch an ioctl aimed at the wrong fd, not to reserve a number
 * globally — this is an out-of-tree module and its numbers only have to be unique
 * on its own file descriptor.
 */
#define PIM_IOC_MAGIC		'P'
#define PIM_IOC_GET_INFO	_IOR (PIM_IOC_MAGIC, 0x01, struct pim_info)
#define PIM_IOC_ALLOC		_IOWR(PIM_IOC_MAGIC, 0x02, struct pim_alloc_req)
#define PIM_IOC_FREE		_IOW (PIM_IOC_MAGIC, 0x03, struct pim_free_req)

/* 0x04.. are reserved for the execution path — the engine lease and the doorbell.
 * Design §9.1 leaves open whether it goes through the kernel at all: plan A puts
 * ACQUIRE/LAUNCH/RELEASE here, plan B puts the command ring in card memory and this
 * file never grows.  Nothing may be assigned in that range until that is decided.
 *
 * NOTE THAT ALLOCATION IS NOT PART OF THAT QUESTION.  The GPR is memory and is
 * managed here; what a WRVEC does with a GPR word is the execution layer's problem
 * and needs nothing from this file. */

#endif /* _UAPI_PIM_IOCTL_H */
