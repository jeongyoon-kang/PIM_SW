/* SPDX-License-Identifier: MIT */
/*
 * pim_geometry.h — the address arithmetic, as pure functions.
 *
 * WHERE THE NUMBERS COME FROM.  From the driver, at open, over PIM_IOC_GET_INFO —
 * which got them from its module parameters (design §6.4).  NOT from a -D and not
 * from a conf file.  That is the single most important difference between this
 * stack and hwdef/: hwdef compiles the topology in, so switching channel count means
 * rebuilding every binary; here it means a different insmod line.
 *
 *   hwdef/pim_platform.h   PIM_NCH is a macro.  pim_decode() answers for the one
 *                          topology the binary was built for.
 *   this file              nch is a field.  pim_geom_decode() answers for the
 *                          topology the loaded module reported.
 *
 * Both compute the same thing and MUST agree when pointed at the same board.  They
 * are not checked against each other, because nothing here includes hwdef — a probe
 * that wants to check them can, and lib/test/geom_test.c is that probe.
 *
 * TWO REGIONS (design §9.4), and only one of them has any geometry at all:
 *
 *   PIM_MEM_DRAM   RoChBaCo interleaved.  An address decodes to (row, ch, bank,
 *                  col), which is what a command names.  Allocated in broadcast
 *                  units, because that is what one all-bank operation consumes.
 *   PIM_MEM_GPR    LINEAR.  4 MiB of staging between the host and the channels'
 *                  global buffers.  There is nothing to decode: an offset is an
 *                  offset, and word index = offset / 32 is the only conversion the
 *                  ISA needs.  pim_geom_decode() does not apply to it and refuses.
 *
 * NO STATE, NO DEVICE.  Everything below is arithmetic on a pim_geometry the caller
 * already holds.  That is what lets the compute runtime (design §9.1) do its whole
 * address translation without a syscall.
 */
#ifndef PIM_GEOMETRY_H
#define PIM_GEOMETRY_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Where an allocation lives.  These are the driver's region indices under the names
 * a caller thinks in; pim_dev.c static-asserts that the two agree. */
typedef enum {
	PIM_MEM_DRAM = 0,	/* weights, KV cache — several GiB, interleaved */
	PIM_MEM_GPR  = 1,	/* vectors, result buffers — 4 MiB, linear      */
} pim_mem;
#define PIM_NMEM  2

/* The channel address map.  DRAM ONLY — the GPR is not interleaved.
 *
 * ONLY ROCHBACO IS SUPPORTED BY THIS STACK TODAY.  ChRoBaCo is decoded here because
 * the arithmetic is three lines and leaving it out would make the two maps look more
 * different than they are, but nothing above has been written for it. */
typedef enum {
	PIM_MAP_CH_RO_BA_CO = 0,	/* [Ch | Row | Bank | Col] — channel every bytes/nch */
	PIM_MAP_RO_CH_BA_CO = 1,	/* [Row | Ch | Bank | Col] — channel every 32 KiB    */
} pim_map;

/*
 * One region, as libpim sees it.
 *
 * TWO GRANULES, AND THE DIFFERENCE BETWEEN THEM IS THE DESIGN.
 *   hugepage   what the DRIVER hands out.  2 MiB for DRAM, 4 KiB for the GPR.
 *   gran    what LIBPIM hands out.  For DRAM the broadcast unit (128 KiB on four
 *           channels), because that is the smallest thing an all-bank operation can
 *           address.  For the GPR the hugepage itself — linear memory has no such
 *           constraint, so there is no second level and a request just rounds to a
 *           page (design §9.4).
 * grans_per_hugepage is 16 for DRAM and 1 for the GPR, and the same pool code runs both.
 */
typedef struct {
	uint64_t base;			/* AXI address of hugepage 0                 */
	uint64_t bytes;			/* size of the region                     */
	uint64_t hugepage_bytes;		/* the driver's granule                   */
	uint64_t gran;			/* libpim's granule                       */
	uint64_t nr_hugepages;
	uint32_t gran_shift;		/* log2(gran)                             */
	uint32_t grans_per_hugepage;
} pim_region;

typedef struct {
	uint32_t nch;		/* channels                                        */
	uint32_t nbank;		/* banks per channel                               */
	uint32_t row_bytes;	/* row buffer — 2048                                */
	pim_map  map;		/* DRAM only                                       */
	uint64_t unit_bytes;	/* broadcast unit: nch * nbank * row_bytes         */

	pim_region mem[PIM_NMEM];
} pim_geometry;

/* One channel's whole page set: nbank * row_bytes.  32 KiB here, and exactly what a
 * single all-bank operation in ONE channel consumes.  Multiply by nch for what a
 * broadcast across every channel consumes, which is unit_bytes. */
static inline uint64_t pim_robaco_row(const pim_geometry *g)
{ return (uint64_t)g->nbank * g->row_bytes; }

/* A GPR byte offset as the ISA sees it.  WRVEC and RD_MAC name a WORD, and a word is
 * one 256-bit beat — the same 32 B everywhere else in this hardware. */
#define PIM_GPR_WORD_BYTES  32u
static inline uint64_t pim_gpr_word_of(const pim_geometry *g, uint64_t axi)
{ return (axi - g->mem[PIM_MEM_GPR].base) / PIM_GPR_WORD_BYTES; }

/* THE INVARIANT THE DRAM SIDE RESTS ON (design §4.1).
 *
 *     unit_bytes divides hugepage_bytes, and hugepages are hugepage_bytes-aligned.
 *
 * Given that, no broadcast operation can ever straddle a hugepage boundary — so the
 * scattering of hugepages across the card is invisible to compute and shows up only as
 * the number of transfers a memcpy costs.  If this stops holding, the fix is a
 * bigger hugepage, not a cleverer allocator.
 *
 * The GPR has no counterpart: it is linear, so nothing about it can straddle
 * anything.  What is checked there is only that a page is a whole number of 32 B
 * words, because that is the granule WRVEC addresses in.
 *
 * Returns NULL when g is usable, otherwise the reason.  Called once at open; a
 * caller that skips it gets wrong answers rather than a diagnosis. */
const char *pim_geom_check(const pim_geometry *g);

/* Fill in gran / gran_shift / grans_per_hugepage from the rest.  pim_open() does this;
 * exposed so a test can build a geometry without a board. */
void pim_geom_derive(pim_geometry *g);

/* ---------------------------------------------------------------- decode ---
 * Where a DRAM address physically sits.  `a` is an absolute AXI address inside
 * [mem[PIM_MEM_DRAM].base, +bytes).  NOT VALID FOR GPR ADDRESSES.
 *
 * The four coordinates are what a PIM command names.  row and col are what the ISA
 * calls ROW and COL; `col_byte` is the byte offset inside the 2048 B page, which is
 * col * 32 for a 32 B column.
 *
 * This is the same arithmetic as hwdef's pim_decode_as(), parameterised instead of
 * compiled in.  See the note at the top of this file about why there are two.
 *
 * NAMED pim_geom_decode AND NOT pim_decode ON PURPOSE.  hwdef exports a pim_decode()
 * with a different signature, and colliding with it would make a translation unit
 * that includes both fail to compile.  Which sounds like a useful guard until you
 * want the one program that SHOULD include both: a probe that checks this
 * arithmetic against the compiled-in version that was validated on the board.  The
 * guard against mixing two topology sources belongs in a comment and a review, not
 * in an accidental symbol clash that forbids the check.
 */
typedef struct {
	uint64_t row;		/* DRAM row index within the bank     */
	uint32_t ch;		/* channel                            */
	uint32_t bank;		/* bank within the channel            */
	uint32_t col_byte;	/* byte offset within the 2048 B page */
} pim_coord;

static inline void pim_geom_decode(const pim_geometry *g, uint64_t a, pim_coord *out)
{
	const pim_region *r = &g->mem[PIM_MEM_DRAM];
	uint64_t off  = a - r->base;
	uint64_t rbr  = pim_robaco_row(g);		/* nbank * row_bytes */
	uint64_t within, rowf;
	uint32_t ch;

	if (g->map == PIM_MAP_RO_CH_BA_CO) {
		rowf   = off / g->unit_bytes;
		ch     = (uint32_t)((off / rbr) % g->nch);
		within = off % rbr;
	} else {
		uint64_t ch_span = r->bytes / g->nch;
		ch     = (uint32_t)(off / ch_span);
		off   %= ch_span;
		rowf   = off / rbr;
		within = off % rbr;
	}
	out->row      = rowf;
	out->ch       = ch;
	out->bank     = (uint32_t)(within / g->row_bytes);
	out->col_byte = (uint32_t)(within % g->row_bytes);
}

/* The inverse.  Mostly for tests: a decode that is not checked against an encode is
 * only half a claim. */
static inline uint64_t pim_geom_encode(const pim_geometry *g, const pim_coord *c)
{
	const pim_region *r = &g->mem[PIM_MEM_DRAM];
	uint64_t rbr    = pim_robaco_row(g);
	uint64_t within = (uint64_t)c->bank * g->row_bytes + c->col_byte;

	if (g->map == PIM_MAP_RO_CH_BA_CO)
		return r->base + c->row * g->unit_bytes + (uint64_t)c->ch * rbr + within;

	return r->base + (uint64_t)c->ch * (r->bytes / g->nch)
	                + c->row * rbr + within;
}

#ifdef __cplusplus
}
#endif
#endif /* PIM_GEOMETRY_H */
