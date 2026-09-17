// SPDX-License-Identifier: MIT
/*
 * pim_main.c — module parameters, init, /dev/pim, /proc/pim.
 *
 * WHERE GEOMETRY COMES FROM (design §6.4).  From insmod.  The board cannot report
 * its own channel count, so the truth has to be injected somewhere, and putting it
 * in ONE place — the module's parameters — is what stops the driver and libpim from
 * being built for different topologies.  libpim never reads a conf file; it asks
 * GET_INFO.  Changing channel count is an insmod line, not a rebuild of anything.
 *
 *     insmod pim.ko pim_channels=4 pim_banks=16 pim_row_size=2048 \
 *                   pim_base=0x020400000000 pim_ch_span=0x100000000 \
 *                   pim_gpr_base=0x020200000000 pim_gpr_size=0x400000
 *
 * The values are visible afterwards at /sys/module/pim/parameters/.
 *
 * TWO REGIONS, ONE MACHINE (design §9.4).  The GPR is 4 MiB of staging between the
 * host and each channel's global buffer: WRVEC multicasts from it into the GBs,
 * RD_MAC unicasts MAC results back into it.  That makes it MEMORY, not scratch — a
 * vector is uploaded once and re-read by many WRVECs, a result buffer accumulates
 * where its caller put it, and objects with different lifetimes are live in there at
 * once.  So it gets a ledger of its own, with the same code and different constants.
 *
 * THE DRIVER STILL DOES NOT UNDERSTAND ANY OF IT.  It multiplies the DRAM numbers
 * once to get the broadcast unit, checks the one invariant its own hugepage sizing has
 * to satisfy (design §4.1), and then only reports.  It does not know that a GPR word
 * is 32 bytes or what WRVEC does with one.
 *
 * WHEN THE HARDWARE GROWS AN INFO REGISTER (design §11) only pim_init() changes:
 * the parameters become defaults and the register wins.  GET_INFO and everything
 * above it stay exactly as they are.
 *
 * CARVING OUT A REGION LATER.  If the command ring ends up in card memory (design
 * §9.1 plan B) it needs space the allocator can never hand out.  Do NOT do that by
 * moving pim_base past it: the DRAM base is what libpim's row decoding counts from,
 * so shifting it would shift every ROW field by the size of the hole.  It becomes a
 * first-hugepage index instead — the base stays where the aperture starts.
 */
#include <linux/atomic.h>
#include <linux/bitmap.h>
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/mutex.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/slab.h>

#include "pim_drv.h"

/* ------------------------------------------------------------- parameters --- */
static unsigned int pim_channels = 4;
static unsigned int pim_banks    = 16;
static unsigned int pim_row_size = 2048;
static unsigned int pim_addr_map = PIM_MAP_ROCHBACO;
static unsigned long long pim_base       = 0x020400000000ULL;	/* MC s_axi aperture */
static unsigned long long pim_ch_span    = 0x100000000ULL;	/* 4 GiB per channel */
static unsigned long long pim_hugepage_size = 2ULL << 20;		/* design §4 */
/* The GPR lives at the bottom of BAR2's AXI window and the CFR begins immediately
 * after it — CFR offset 0 is the DOORBELL.  Nothing here can overrun into it (the
 * ledger never hands out a hugepage past `bytes`), but that is why the size is a
 * parameter and not something anyone should round up "to be safe". */
static unsigned long long pim_gpr_base = 0x020200000000ULL;	/* BAR2 AXI + OFF_GPR */
static unsigned long long pim_gpr_size = 4ULL << 20;		/* 4 MiB */
static unsigned long long pim_gpr_page = 4096;			/* design §9.4 */
static bool pim_track_owners = true;

module_param(pim_channels, uint, 0444);
MODULE_PARM_DESC(pim_channels, "channels on the loaded bitstream (power of two)");
module_param(pim_banks, uint, 0444);
MODULE_PARM_DESC(pim_banks, "banks per channel (power of two)");
module_param(pim_row_size, uint, 0444);
MODULE_PARM_DESC(pim_row_size, "row buffer in bytes (power of two)");
module_param(pim_addr_map, uint, 0444);
MODULE_PARM_DESC(pim_addr_map, "0 = ChRoBaCo, 1 = RoChBaCo.  DRAM only; carried to libpim, not used here");
module_param(pim_base, ullong, 0444);
MODULE_PARM_DESC(pim_base, "AXI address of the first managed DRAM byte");
module_param(pim_ch_span, ullong, 0444);
MODULE_PARM_DESC(pim_ch_span, "bytes of DRAM per channel; region = channels * this");
module_param(pim_hugepage_size, ullong, 0444);
MODULE_PARM_DESC(pim_hugepage_size, "DRAM granule; a power of two and a multiple of the broadcast unit");
module_param(pim_gpr_base, ullong, 0444);
MODULE_PARM_DESC(pim_gpr_base, "AXI address of the GPR region (BAR2 AXI base + OFF_GPR)");
module_param(pim_gpr_size, ullong, 0444);
MODULE_PARM_DESC(pim_gpr_size, "bytes of GPR.  The CFR starts immediately after it");
module_param(pim_gpr_page, ullong, 0444);
MODULE_PARM_DESC(pim_gpr_page, "GPR granule; a power of two, at least one 32 B word");
module_param(pim_track_owners, bool, 0444);
MODULE_PARM_DESC(pim_track_owners, "keep a per-hugepage owner table for /proc/pim");

/* -------------------------------------------------------------- the ledgers --- */
struct pim_ledger pim_led[PIM_NREGION];
struct pim_info   pim_info_tmpl;
LIST_HEAD(pim_files);
DEFINE_MUTEX(pim_files_lock);

/* Same assertions as libpim's pim_dev.c.  If one side's layout moves, both stop
 * building rather than one of them reading the other's fields at the wrong offset. */
static_assert(sizeof(struct pim_region_info)  == 40, "pim_region_info layout changed: bump PIM_ABI_VERSION");
static_assert(sizeof(struct pim_info)      == 112, "pim_info layout changed: bump PIM_ABI_VERSION");
static_assert(sizeof(struct pim_extent)    == 16, "pim_extent layout changed: bump PIM_ABI_VERSION");
static_assert(sizeof(struct pim_alloc_req) == 32, "pim_alloc_req layout changed: bump PIM_ABI_VERSION");
static_assert(sizeof(struct pim_free_req)  == 32, "pim_free_req layout changed: bump PIM_ABI_VERSION");

static struct proc_dir_entry *pim_proc;

static bool is_pow2_u64(u64 v)
{
	return v && !(v & (v - 1));
}

/*
 * Everything that must hold before a single hugepage is handed out.
 *
 * The invariant that matters is the DRAM one (design §4.1): the broadcast unit
 * divides the hugepage.  Given that, no all-bank operation can span two hugepages, so a
 * scattered allocation is invisible to compute and shows up only as extra
 * transfers.  If a future topology breaks it, the fix is a bigger hugepage — not a
 * smarter allocator, and certainly not letting it slide.
 *
 * THE GPR HAS NO SUCH INVARIANT because it has no interleave: it is linear memory
 * and its only constraints are that a page is a whole number of 32 B words (the
 * ISA's granule) and that the region is a whole number of pages.
 *
 * The power-of-two demands on the DRAM side are not decoration.  libpim translates a
 * pointer to a card address with a shift (pim_geometry.h), and the channel field of
 * RoChBaCo is a bit field in the first place — a non-power-of-two channel count is
 * not a geometry this hardware can have.
 */
static int check_geometry(u64 unit)
{
	if (!pim_channels || pim_channels > 8 || !is_pow2_u64(pim_channels)) {
		pim_err("pim_channels=%u must be a power of two in 1..8\n", pim_channels);
		return -EINVAL;
	}
	if (!is_pow2_u64(pim_banks) || !is_pow2_u64(pim_row_size) || pim_row_size < 32) {
		pim_err("pim_banks=%u and pim_row_size=%u must be powers of two, row >= 32\n",
			pim_banks, pim_row_size);
		return -EINVAL;
	}
	if (pim_addr_map > PIM_MAP_ROCHBACO) {
		pim_err("pim_addr_map=%u is neither 0 (ChRoBaCo) nor 1 (RoChBaCo)\n",
			pim_addr_map);
		return -EINVAL;
	}
	if (!is_pow2_u64(pim_hugepage_size)) {
		pim_err("pim_hugepage_size=%#llx must be a power of two\n", pim_hugepage_size);
		return -EINVAL;
	}
	if (pim_hugepage_size % unit) {
		pim_err("THE INVARIANT FAILS: broadcast unit %#llx does not divide hugepage %#llx.  Raise pim_hugepage_size.\n",
			unit, pim_hugepage_size);
		return -EINVAL;
	}
	if (!is_pow2_u64(pim_gpr_page) || pim_gpr_page < 32) {
		pim_err("pim_gpr_page=%#llx must be a power of two of at least 32 B (one word)\n",
			pim_gpr_page);
		return -EINVAL;
	}
	return 0;
}

static int ledger_init(unsigned mem, const char *name, u64 base, u64 bytes, u64 hugepage)
{
	struct pim_ledger *l = &pim_led[mem];

	if (base % hugepage) {
		pim_err("%s: base %#llx is not aligned to its %#llx hugepage\n",
			name, base, hugepage);
		return -EINVAL;
	}
	if (!bytes || bytes % hugepage) {
		pim_err("%s: region %#llx is empty or not a whole number of %#llx hugepages\n",
			name, bytes, hugepage);
		return -EINVAL;
	}

	l->name        = name;
	l->base        = base;
	l->bytes       = bytes;
	l->hugepage_bytes = hugepage;
	l->nr_hugepages   = bytes / hugepage;
	l->nr_free     = l->nr_hugepages;
	mutex_init(&l->lock);

	l->bitmap = bitmap_zalloc((unsigned int)l->nr_hugepages, GFP_KERNEL);
	if (!l->bitmap)
		return -ENOMEM;
	if (pim_track_owners) {
		l->owner = kvcalloc(l->nr_hugepages, sizeof(*l->owner), GFP_KERNEL);
		if (!l->owner) {
			bitmap_free(l->bitmap);
			l->bitmap = NULL;
			return -ENOMEM;
		}
	}

	pim_info_tmpl.region[mem].base        = base;
	pim_info_tmpl.region[mem].bytes       = bytes;
	pim_info_tmpl.region[mem].hugepage_bytes = hugepage;
	pim_info_tmpl.region[mem].nr_hugepages   = l->nr_hugepages;
	pim_info_tmpl.region[mem].nr_free     = l->nr_hugepages;

	pim_info_msg("%-4s %#018llx + %llu KiB = %llu hugepages of %llu KiB\n",
		     name, base, bytes >> 10, l->nr_hugepages, hugepage >> 10);
	return 0;
}

static void ledger_fini(unsigned mem)
{
	kvfree(pim_led[mem].owner);
	bitmap_free(pim_led[mem].bitmap);
	pim_led[mem].owner  = NULL;
	pim_led[mem].bitmap = NULL;
}

/* -------------------------------------------------------------- /proc/pim --- */
static int pim_proc_show(struct seq_file *m, void *v)
{
	struct pim_info info = pim_info_tmpl;
	struct pim_file *f;
	u64 freeb[PIM_NREGION];
	unsigned mem;

	for (mem = 0; mem < PIM_NREGION; mem++) {
		mutex_lock(&pim_led[mem].lock);
		freeb[mem] = pim_led[mem].nr_free;
		mutex_unlock(&pim_led[mem].lock);
	}

	seq_printf(m, "abi        %u\n", info.abi_version);
	seq_printf(m, "geometry   %u ch x %u bank, row %u B, unit %llu KiB, map %s\n",
		   info.nch, info.nbank, info.row_bytes, info.unit_bytes >> 10,
		   info.addr_map == PIM_MAP_ROCHBACO ? "RoChBaCo" : "ChRoBaCo");
	seq_puts(m, "\nregion  base               size        hugepage    hugepages     free\n");
	for (mem = 0; mem < PIM_NREGION; mem++)
		seq_printf(m, "%-6s  %#018llx  %7llu KiB %6llu KiB %8llu %8llu\n",
			   pim_led[mem].name, info.region[mem].base,
			   info.region[mem].bytes >> 10,
			   info.region[mem].hugepage_bytes >> 10,
			   info.region[mem].nr_hugepages, freeb[mem]);

	seq_puts(m, "\n  pid  slot     dram      gpr  vbase\n");
	/* A ledger lock is released above and f->lock taken below, never nested.  The
	 * allocator's order is f->lock then a ledger's; taking them the other way
	 * round here is exactly how this would deadlock. */
	mutex_lock(&pim_files_lock);
	list_for_each_entry(f, &pim_files, node) {
		struct pim_vbase *vb;
		bool first = true;

		mutex_lock(&f->lock);
		seq_printf(m, "%5d %5u %8llu %8llu", f->pid, f->slot,
			   f->nr_held[PIM_REGION_DRAM], f->nr_held[PIM_REGION_GPR]);
		list_for_each_entry(vb, &f->vbases, node) {
			seq_printf(m, "%s%#llx+%#llx", first ? "  " : ", ",
				   vb->vbase, vb->bytes);
			first = false;
		}
		seq_puts(m, first ? "  -\n" : "\n");
		mutex_unlock(&f->lock);
	}
	mutex_unlock(&pim_files_lock);
	return 0;
}

/* ------------------------------------------------------------------ device --- */
static struct miscdevice pim_misc = {
	.minor	= MISC_DYNAMIC_MINOR,
	.name	= PIM_DRV_NAME,
	.fops	= &pim_fops,
	/* Lab hardware, and every tool in this tree runs unprivileged.  Tighten it
	 * with a udev rule rather than by editing this if that changes. */
	.mode	= 0666,
};

static int __init pim_init(void)
{
	u64 unit;
	int ret;

	unit = (u64)pim_channels * pim_banks * pim_row_size;

	ret = check_geometry(unit);
	if (ret)
		return ret;

	pim_info_tmpl.abi_version = PIM_ABI_VERSION;
	pim_info_tmpl.flags       = pim_track_owners ? PIM_INFO_HAS_OWNERS : 0;
	pim_info_tmpl.nch         = pim_channels;
	pim_info_tmpl.nbank       = pim_banks;
	pim_info_tmpl.row_bytes   = pim_row_size;
	pim_info_tmpl.addr_map    = pim_addr_map;
	pim_info_tmpl.unit_bytes  = unit;

	pim_info_msg("%u ch x %u bank, row %u B, unit %llu KiB, map %s\n",
		     pim_channels, pim_banks, pim_row_size, unit >> 10,
		     pim_addr_map == PIM_MAP_ROCHBACO ? "RoChBaCo" : "ChRoBaCo");

	ret = ledger_init(PIM_REGION_DRAM, "dram", pim_base,
			  (u64)pim_channels * pim_ch_span, pim_hugepage_size);
	if (ret)
		return ret;
	ret = ledger_init(PIM_REGION_GPR, "gpr", pim_gpr_base, pim_gpr_size,
			  pim_gpr_page);
	if (ret)
		goto err_dram;

	ret = misc_register(&pim_misc);
	if (ret) {
		pim_err("misc_register failed: %d\n", ret);
		goto err_gpr;
	}

	pim_proc = proc_create_single(PIM_DRV_NAME, 0444, NULL, pim_proc_show);
	if (!pim_proc)
		pim_info_msg("/proc/pim unavailable; continuing without it\n");
	return 0;

err_gpr:
	ledger_fini(PIM_REGION_GPR);
err_dram:
	ledger_fini(PIM_REGION_DRAM);
	return ret;
}

static void __exit pim_exit(void)
{
	unsigned mem;

	/* Every open fd pins this module, so by the time rmmod gets here there are
	 * no files left and nothing to reclaim.  That is the whole reason the
	 * per-fd reclaim in release() is enough. */
	if (pim_proc)
		remove_proc_entry(PIM_DRV_NAME, NULL);
	misc_deregister(&pim_misc);
	for (mem = 0; mem < PIM_NREGION; mem++)
		ledger_fini(mem);
	pim_info_msg("unloaded\n");
}

module_init(pim_init);
module_exit(pim_exit);

MODULE_LICENSE("Dual MIT/GPL");
MODULE_DESCRIPTION("PIM emulator device-memory allocator (DRAM + GPR)");
MODULE_VERSION("0.2");
