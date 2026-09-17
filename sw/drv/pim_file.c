// SPDX-License-Identifier: MIT
/*
 * pim_file.c — the file_operations (design §6.5).
 *
 * THREE ENTRY POINTS AND NO MORE, and the omissions carry as much design as the
 * inclusions:
 *
 *   .read/.write   absent.  Data moves over QDMA's own char devices (design §8);
 *                  a second path to the same bytes would be a second thing to keep
 *                  correct.  That is as true of the GPR as of DRAM — both are AXI
 *                  addresses on the same MM queue.
 *   .mmap          absent.  Card memory is not in the host's address space
 *                  (design §3.1), and the decision not to expose the control BAR to
 *                  userspace removed the last would-be caller (design §6.1).
 *   .llseek        absent.  There is no position.
 *
 * .release IS THE IMPORTANT ONE.  The kernel calls it on close, on exit, and on
 * kill -9, so "hugepages are always reclaimed, in every region" is guaranteed by the fd
 * lifetime rather than by anyone remembering to free.
 */
#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#include "pim_drv.h"

static atomic_t pim_slot_seq = ATOMIC_INIT(0);

static void free_file(struct pim_file *f)
{
	unsigned mem;

	for (mem = 0; mem < PIM_NREGION; mem++)
		bitmap_free(f->hugepages[mem]);
	kfree(f);
}

static int pim_dev_open(struct inode *ino, struct file *filp)
{
	struct pim_file *f;
	unsigned mem;

	f = kzalloc(sizeof(*f), GFP_KERNEL);
	if (!f)
		return -ENOMEM;

	for (mem = 0; mem < PIM_NREGION; mem++) {
		f->hugepages[mem] = bitmap_zalloc((unsigned int)pim_led[mem].nr_hugepages,
					       GFP_KERNEL);
		if (!f->hugepages[mem]) {
			free_file(f);
			return -ENOMEM;
		}
	}

	f->pid  = current->tgid;
	/* Advisory identity for /proc and led->owner.  Wraps, and 0 is reserved for
	 * "free", so this is a label rather than a handle — nothing looks a file up
	 * by it. */
	f->slot = (u16)((atomic_inc_return(&pim_slot_seq) & 0x7fff) + 1);
	mutex_init(&f->lock);
	INIT_LIST_HEAD(&f->vbases);

	mutex_lock(&pim_files_lock);
	list_add_tail(&f->node, &pim_files);
	mutex_unlock(&pim_files_lock);

	filp->private_data = f;
	return 0;
}

static int pim_dev_release(struct inode *ino, struct file *filp)
{
	struct pim_file *f = filp->private_data;
	struct pim_vbase *v, *tmp;
	unsigned mem;

	mutex_lock(&pim_files_lock);
	list_del(&f->node);
	mutex_unlock(&pim_files_lock);

	for (mem = 0; mem < PIM_NREGION; mem++)
		if (f->nr_held[mem])
			pim_info_msg("reclaiming %llu %s hugepage(s) from pid %d\n",
				     f->nr_held[mem], pim_led[mem].name, f->pid);
	pim_hugepages_reclaim(f);

	list_for_each_entry_safe(v, tmp, &f->vbases, node) {
		list_del(&v->node);
		kfree(v);
	}
	free_file(f);
	return 0;
}

/* Recorded only (design §7.3).  Nothing in this module reads vbase to decide
 * anything; it is here so /proc/pim can be lined up against a debugger's view of
 * the same process, and so that libpim's own bookkeeping bugs surface as an overlap
 * warning instead of as wrong data much later. */
static void record_vbase(struct pim_file *f, u64 vbase, u64 bytes)
{
	struct pim_vbase *v, *n;

	if (!vbase || !bytes)
		return;

	n = kzalloc(sizeof(*n), GFP_KERNEL);
	if (!n)
		return;			/* debug aid; losing it is not a failure */
	n->vbase = vbase;
	n->bytes = bytes;

	mutex_lock(&f->lock);
	list_for_each_entry(v, &f->vbases, node) {
		if (vbase < v->vbase + v->bytes && v->vbase < vbase + bytes) {
			pim_err("pid %d registered overlapping vbase %#llx+%#llx over %#llx+%#llx — libpim bug\n",
				f->pid, vbase, bytes, v->vbase, v->bytes);
			break;
		}
	}
	list_add_tail(&n->node, &f->vbases);
	mutex_unlock(&f->lock);
}

static void forget_vbase(struct pim_file *f, u64 vbase)
{
	struct pim_vbase *v, *tmp;

	if (!vbase)
		return;
	mutex_lock(&f->lock);
	list_for_each_entry_safe(v, tmp, &f->vbases, node) {
		if (v->vbase == vbase) {
			list_del(&v->node);
			kfree(v);
			break;
		}
	}
	mutex_unlock(&f->lock);
}

static long ioc_get_info(void __user *arg)
{
	struct pim_info info = pim_info_tmpl;
	unsigned mem;

	for (mem = 0; mem < PIM_NREGION; mem++) {
		mutex_lock(&pim_led[mem].lock);
		info.region[mem].nr_free = pim_led[mem].nr_free;
		mutex_unlock(&pim_led[mem].lock);
	}
	return copy_to_user(arg, &info, sizeof(info)) ? -EFAULT : 0;
}

static long ioc_alloc(struct pim_file *f, void __user *arg)
{
	struct pim_alloc_req req;
	struct pim_extent *ext;
	u32 kcap, nex = 0;
	long ret;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (req.region >= PIM_NREGION)
		return -EINVAL;
	if (!req.nr_hugepages || req.nr_hugepages > pim_led[req.region].nr_hugepages)
		return -EINVAL;
	if (!req.nr_extents || !req.extents)
		return -EINVAL;

	/* A request can never need more extents than it has hugepages, so there is no
	 * reason to allocate for a larger claimed capacity. */
	kcap = min_t(u32, req.nr_extents, req.nr_hugepages);
	ext  = kvmalloc_array(kcap, sizeof(*ext), GFP_KERNEL);
	if (!ext)
		return -ENOMEM;

	ret = pim_hugepages_alloc(f, req.region, req.nr_hugepages, req.flags, ext, kcap, &nex);
	if (ret) {
		req.nr_extents = nex;	/* on -ENOSPC: what would have been needed */
		if (copy_to_user(arg, &req, sizeof(req)))
			ret = -EFAULT;
		goto out;
	}

	if (copy_to_user(u64_to_user_ptr(req.extents), ext, nex * sizeof(*ext))) {
		pim_hugepages_free(f, req.region, ext, nex);
		ret = -EFAULT;
		goto out;
	}
	req.nr_extents = nex;
	if (copy_to_user(arg, &req, sizeof(req))) {
		pim_hugepages_free(f, req.region, ext, nex);
		ret = -EFAULT;
		goto out;
	}
	record_vbase(f, req.vbase,
		     (u64)req.nr_hugepages * pim_led[req.region].hugepage_bytes);
out:
	kvfree(ext);
	return ret;
}

static long ioc_free(struct pim_file *f, void __user *arg)
{
	struct pim_free_req req;
	struct pim_extent *ext;
	long ret;

	if (copy_from_user(&req, arg, sizeof(req)))
		return -EFAULT;
	if (req.region >= PIM_NREGION || req.flags || req.__pad)
		return -EINVAL;
	if (!req.nr_extents || req.nr_extents > pim_led[req.region].nr_hugepages ||
	    !req.extents)
		return -EINVAL;

	ext = kvmalloc_array(req.nr_extents, sizeof(*ext), GFP_KERNEL);
	if (!ext)
		return -ENOMEM;
	if (copy_from_user(ext, u64_to_user_ptr(req.extents),
			   req.nr_extents * sizeof(*ext))) {
		ret = -EFAULT;
		goto out;
	}

	ret = pim_hugepages_free(f, req.region, ext, req.nr_extents);
	if (!ret)
		forget_vbase(f, req.vbase);
out:
	kvfree(ext);
	return ret;
}

static long pim_dev_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct pim_file *f = filp->private_data;
	void __user *p = (void __user *)arg;

	switch (cmd) {
	case PIM_IOC_GET_INFO:	return ioc_get_info(p);
	case PIM_IOC_ALLOC:	return ioc_alloc(f, p);
	case PIM_IOC_FREE:	return ioc_free(f, p);
	default:		return -ENOTTY;
	}
}

const struct file_operations pim_fops = {
	.owner		= THIS_MODULE,
	.open		= pim_dev_open,
	.release	= pim_dev_release,
	.unlocked_ioctl	= pim_dev_ioctl,
	/* Every struct in the ABI is explicitly sized and 8-byte aligned, so a
	 * 32-bit caller produces the same layout and the same handler serves it. */
	.compat_ioctl	= compat_ptr_ioctl,
};
