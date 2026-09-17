/* SPDX-License-Identifier: MIT */
/*
 * pim_drv.h — internal to the module.  Nothing here crosses to userspace; the
 * contract is ../include/uapi/pim_ioctl.h and only that.
 */
#ifndef _PIM_DRV_H
#define _PIM_DRV_H

#include <linux/fs.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/types.h>

#include "uapi/pim_ioctl.h"

#define PIM_DRV_NAME	"pim"
#define pim_err(fmt, ...)	pr_err (PIM_DRV_NAME ": " fmt, ##__VA_ARGS__)
#define pim_info_msg(fmt, ...)	pr_info(PIM_DRV_NAME ": " fmt, ##__VA_ARGS__)

/*
 * A ledger.  ONE PER REGION, and the plural is the point (design §9.4): the DRAM
 * pool and the GPR pool are the same machine with different constants, so this is a
 * struct that gets instantiated rather than a set of globals that gets copied.
 *
 *   name          for /proc and for messages, so an error says which pool ran out
 *   base/bytes    the span of card AXI space this ledger owns
 *   hugepage_bytes   its granule.  2 MiB for DRAM, 4 KiB for the GPR
 *   bitmap        one bit per hugepage, 1 = allocated.  THE ledger; everything else is
 *                 policy on top of it
 *   owner         per hugepage, the owning fd's slot id.  Debug only, 0 = free
 *
 * `lock` covers bitmap, owner and nr_free together.  A mutex, not a spinlock: every
 * path that takes it is process context (open, release, ioctl) and may sleep in
 * copy_*_user while NOT holding it.  When interrupt-driven completion arrives
 * (design §11) it will need its own lock, not this one.
 */
struct pim_ledger {
	const char		*name;
	struct mutex		lock;
	unsigned long		*bitmap;
	u16			*owner;
	u64			nr_free;
	u64			base, bytes, hugepage_bytes, nr_hugepages;
};

extern struct pim_ledger pim_led[PIM_NREGION];
extern struct pim_info   pim_info_tmpl;		/* the parts that never change */

/*
 * Per-fd state (design §6.2).  One per open(), reclaimed in release() — which the
 * kernel calls on close, on exit, and on kill -9 alike.  That is where the "a leak
 * is structurally impossible" claim comes from.
 *
 * SINGLE PROCESS IS THE CURRENT ASSUMPTION for the execution engine, and this is
 * unaffected by it: per-fd ownership is the crash-reclaim mechanism, not a
 * multi-process feature.  Two fds still get separate ledgers because that costs
 * nothing and removes a whole class of question.
 *
 * `hugepages[]` is this fd's own bitmap over the same index space as each ledger's.  It
 * costs nr_hugepages/8 bytes (1 KiB DRAM + 128 B GPR) and buys two things a list of
 * extents would not: release is one scan, and FREE can prove an extent belongs to
 * the caller before it touches the global bitmap.
 */
struct pim_file {
	struct mutex		lock;		/* covers hugepages, nr_held, vbases */
	unsigned long		*hugepages[PIM_NREGION];
	u64			nr_held[PIM_NREGION];
	u16			slot;		/* identity in led->owner */
	pid_t			pid;		/* for /proc only */
	struct list_head	vbases;		/* struct pim_vbase, /proc only */
	struct list_head	node;		/* on pim_files */
};

/* Recorded, never acted on (design §7.3).  libpim's host address-space reservation
 * for one allocation.  It exists so /proc/pim can be read next to a debugger's view
 * of the same process, and so the field does not have to be added to the ABI later. */
struct pim_vbase {
	struct list_head	node;
	u64			vbase;
	u64			bytes;
};

extern struct list_head	pim_files;
extern struct mutex	pim_files_lock;

/* ---- pim_chunk.c — the allocator (design §6.3), shared by every region.  All of
 * these take the ledger's lock themselves; callers must not hold it. ---- */
int  pim_hugepages_alloc(struct pim_file *f, unsigned mem, u32 nr_hugepages, u32 flags,
		      struct pim_extent *out, u32 cap, u32 *nr_out);
int  pim_hugepages_free (struct pim_file *f, unsigned mem,
		      const struct pim_extent *ext, u32 nr);
void pim_hugepages_reclaim(struct pim_file *f);	/* every region, everything held */

/* AXI address <-> hugepage index, the only place the conversion lives */
static inline u64 pim_hugepage_addr(const struct pim_ledger *l, u64 idx)
{ return l->base + idx * l->hugepage_bytes; }

static inline bool pim_addr_to_hugepage(const struct pim_ledger *l, u64 addr, u64 *idx)
{
	u64 off;

	if (addr < l->base)
		return false;
	off = addr - l->base;
	if (off % l->hugepage_bytes)
		return false;
	*idx = off / l->hugepage_bytes;
	return *idx < l->nr_hugepages;
}

/* ---- pim_file.c ---- */
extern const struct file_operations pim_fops;

#endif /* _PIM_DRV_H */
