// SPDX-License-Identifier: MIT
/*
 * pim_chunk.c — the hugepage allocator (design §6.3), one instance per region (§9.4).
 *
 * A bitmap and two-stage first-fit, and that is deliberately all of it.  The three
 * ideas the design separates:
 *
 *   the bitmap    is the LEDGER.  Not an optimisation — any policy needs one.  A
 *                 free list would do as a ledger but cannot answer "are these n
 *                 adjacent", which is the whole of stage 1.
 *   first-fit     is the SEARCH.  Scan from the front, take the first that fits.
 *   contiguous    is the POLICY.  A scattered allocation costs one pwrite per run
 *   run first     later (design §8.1), so try for one run before picking singles.
 *
 * Stage 2 is what makes the allocator's promise true: if the total free count is
 * enough, the allocation SUCCEEDS.  There is no external fragmentation to fail on,
 * because nothing here ever needs a specific size of hole.  Stage 1 only decides how
 * many transfers the caller will pay for.
 *
 * NOTHING BELOW MENTIONS DRAM OR THE GPR.  It reads its constants out of the ledger
 * it was handed, which is exactly the claim design §9.4 makes: same code, different
 * numbers.  The DRAM pool runs it over 8192 hugepages of 2 MiB and the GPR pool over
 * 1024 hugepages of 4 KiB.
 *
 * SCALE.  16 GiB of card memory is 8192 hugepages — a 1 KiB bitmap; the GPR is 128
 * bytes.  A linear scan of that is nothing, and allocation is a slow path by
 * construction (libpim batches into it, design §7.1).
 */
#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/kernel.h>

#include "pim_drv.h"

/* Lock order everywhere in this module: f->lock, then a ledger's lock.  Only one
 * ledger is ever held at a time, so the regions cannot deadlock against each other. */

static void commit_extent(struct pim_file *f, unsigned mem, u64 idx, u32 run)
{
	struct pim_ledger *l = &pim_led[mem];
	u32 i;

	bitmap_set(l->bitmap, idx, run);
	bitmap_set(f->hugepages[mem], idx, run);
	if (l->owner)
		for (i = 0; i < run; i++)
			l->owner[idx + i] = f->slot;
	l->nr_free       -= run;
	f->nr_held[mem]  += run;
}

static void release_extent(struct pim_file *f, unsigned mem, u64 idx, u32 run)
{
	struct pim_ledger *l = &pim_led[mem];
	u32 i;

	bitmap_clear(l->bitmap, idx, run);
	bitmap_clear(f->hugepages[mem], idx, run);
	if (l->owner)
		for (i = 0; i < run; i++)
			l->owner[idx + i] = 0;
	l->nr_free       += run;
	f->nr_held[mem]  -= run;
}

/*
 * out[] receives up to `cap` extents; *nr_out receives how many were used, or — on
 * -ENOSPC — how many would have been needed.
 *
 * NOTHING IS COMMITTED UNTIL THE WHOLE REQUEST IS KNOWN TO FIT.  A partial success
 * would hand back an allocation the caller cannot fully describe, and therefore
 * cannot fully free; it would leak by construction.
 */
int pim_hugepages_alloc(struct pim_file *f, unsigned mem, u32 nr_hugepages, u32 flags,
		     struct pim_extent *out, u32 cap, u32 *nr_out)
{
	struct pim_ledger *l;
	unsigned long total, idx;
	u32 nex = 0, i;
	u64 got = 0;
	int ret = 0;

	if (mem >= PIM_NREGION)
		return -EINVAL;
	l = &pim_led[mem];
	total = (unsigned long)l->nr_hugepages;

	if (!nr_hugepages || nr_hugepages > total)
		return -EINVAL;
	if (flags & ~PIM_ALLOC_CONTIG)
		return -EINVAL;

	mutex_lock(&f->lock);
	mutex_lock(&l->lock);

	if (l->nr_free < nr_hugepages) {
		ret = -ENOMEM;
		goto out;
	}

	/* ---- stage 1: one contiguous run, first-fit ---- */
	idx = bitmap_find_next_zero_area(l->bitmap, total, 0, nr_hugepages, 0);
	if (idx < total) {
		if (cap < 1) {
			nex = 1;
			ret = -ENOSPC;
			goto out;
		}
		out[0].addr      = pim_hugepage_addr(l, idx);
		out[0].nr_hugepages = nr_hugepages;
		out[0].__pad     = 0;
		nex = 1;
		commit_extent(f, mem, idx, nr_hugepages);
		goto out;
	}

	if (flags & PIM_ALLOC_CONTIG) {
		ret = -ENOSPC;
		goto out;
	}

	/* ---- stage 2: singles, coalesced into runs as they come ---- */
	idx = 0;
	while (got < nr_hugepages) {
		unsigned long start, run = 0;

		idx = find_next_zero_bit(l->bitmap, total, idx);
		if (idx >= total)
			break;
		start = idx;
		while (got < nr_hugepages && idx < total &&
		       !test_bit(idx, l->bitmap)) {
			idx++;
			run++;
			got++;
		}
		if (nex < cap) {
			out[nex].addr      = pim_hugepage_addr(l, start);
			out[nex].nr_hugepages = (u32)run;
			out[nex].__pad     = 0;
		}
		nex++;
	}

	/* nr_free said there was room, so this can only fire if the ledger and the
	 * counter disagree — which would be a bug in this file, not a full region. */
	if (got < nr_hugepages) {
		pim_err("%s: bitmap and nr_free disagree: wanted %u, found %llu, free %llu\n",
			l->name, nr_hugepages, got, l->nr_free);
		ret = -ENOMEM;
		goto out;
	}
	if (nex > cap) {
		ret = -ENOSPC;		/* nothing was committed; nex says how many */
		goto out;
	}
	for (i = 0; i < nex; i++) {
		u64 bidx;

		(void)pim_addr_to_hugepage(l, out[i].addr, &bidx);
		commit_extent(f, mem, bidx, out[i].nr_hugepages);
	}

out:
	mutex_unlock(&l->lock);
	mutex_unlock(&f->lock);
	*nr_out = nex;
	return ret;
}

/*
 * Validate every extent before freeing any of them.  A free that half-succeeds is
 * worse than one that fails: the caller has no way to find out where it stopped.
 *
 * Ownership is checked against THIS FD's bitmap, so one process cannot free
 * another's hugepages even though it can name their addresses.  (That is a mistake
 * check, not an isolation boundary — card addresses are public in this design,
 * design §11.)
 */
int pim_hugepages_free(struct pim_file *f, unsigned mem,
		    const struct pim_extent *ext, u32 nr)
{
	struct pim_ledger *l;
	u32 i, j;
	int ret = 0;

	if (mem >= PIM_NREGION)
		return -EINVAL;
	l = &pim_led[mem];

	mutex_lock(&f->lock);
	mutex_lock(&l->lock);

	for (i = 0; i < nr; i++) {
		u64 idx;

		if (!ext[i].nr_hugepages ||
		    !pim_addr_to_hugepage(l, ext[i].addr, &idx) ||
		    idx + ext[i].nr_hugepages > l->nr_hugepages) {
			ret = -EINVAL;
			goto out;
		}
		for (j = 0; j < ext[i].nr_hugepages; j++) {
			if (!test_bit(idx + j, f->hugepages[mem])) {
				ret = -EPERM;	/* not ours, or freed twice */
				goto out;
			}
		}
	}

	for (i = 0; i < nr; i++) {
		u64 idx;

		(void)pim_addr_to_hugepage(l, ext[i].addr, &idx);
		release_extent(f, mem, idx, ext[i].nr_hugepages);
	}

out:
	mutex_unlock(&l->lock);
	mutex_unlock(&f->lock);
	return ret;
}

/*
 * Everything this fd still holds, in every region, from release().  This is the
 * function that makes a leak impossible: the kernel calls release() on close, on
 * exit, and on kill -9, so there is no way for a process to keep card memory it is
 * no longer alive to use.
 */
void pim_hugepages_reclaim(struct pim_file *f)
{
	unsigned mem;

	mutex_lock(&f->lock);
	for (mem = 0; mem < PIM_NREGION; mem++) {
		struct pim_ledger *l = &pim_led[mem];
		unsigned long idx, total = (unsigned long)l->nr_hugepages;

		if (!f->hugepages[mem])
			continue;
		mutex_lock(&l->lock);
		for (idx = 0; idx < total; idx++)
			if (test_bit(idx, f->hugepages[mem]))
				release_extent(f, mem, idx, 1);
		mutex_unlock(&l->lock);
	}
	mutex_unlock(&f->lock);
}
