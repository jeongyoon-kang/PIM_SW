/* SPDX-License-Identifier: MIT */
/*
 * pim_tensor.h — a matrix that lives on the card, and the contract for how it got
 * there.
 *
 * WHAT THE HARDWARE COMPUTES, ONCE, FOR EVERYTHING:
 *
 *     y[0 .. nout)  =  M[nout][nred] . v[0 .. nred)
 *
 * A linear layer, a Q.K^T, an S.V — all three are that, and they differ in nothing
 * the ISR encoder can see.  So there is one tensor type and one program builder;
 * what varies is which axis of the model the two axes carry.
 *
 *     linear    out = the output row      red = the input feature
 *     Q . K^T   out = the KEY TOKEN       red = the head's dimension
 *     S . V     out = the head's dimension  red = the VALUE TOKEN
 *
 * THE AXES ARE NOT A CHOICE.  The accumulator hangs off the bank, so the bank axis
 * is the output axis; a MAC reduces along the beats of one DRAM row, so the row axis
 * is the reduction axis.  Everything below follows from those two sentences.
 *
 * ============================== the layout ==================================
 * Inside a broadcast unit the placement is the same for every layout: the output
 * picks (channel, bank), the reduction picks the column.  Which unit holds MAC group
 * `group`'s reduction chunk `chunk` follows the growth axis, so that the units a
 * growing tensor gains come after the ones it already has:
 *
 *     OUT_MAJOR, OUT_PACKED   unit = group / pack * nchunks + chunk     group outer
 *     RED_MAJOR               unit = chunk * ngroups + group            chunk outer
 *
 * The layout also names the HOST array's index order — which axis you walk to
 * read the bytes you are about to send:
 *
 *     PIM_LAYOUT_OUT_MAJOR   src is [nout][nred]   one output's run is contiguous
 *     PIM_LAYOUT_RED_MAJOR   src is [nred][nout]   one reduction step's run is
 *     PIM_LAYOUT_OUT_PACKED  src is [nout][nred], and outputs shorter than half a
 *                            row share one (t->pack of them, t->slot elements
 *                            each).  The formula is the same one with pack > 1.
 *
 * It is carried in the tensor and mixed into the tag: bytes placed by an OUT_MAJOR
 * walk and read by a program built for a RED_MAJOR one give a plausible wrong
 * number, and the tag is what catches it.
 *
 * ============================== growth =====================================
 * A KV cache grows one token at a time, and WHICH AXIS grows is the layout:
 *
 *     K is OUT_PACKED  a new token is a new OUTPUT   -> grows along `out`
 *     V is RED_MAJOR   a new token is a new REDUCTION STEP -> grows along `red`
 *
 * A KV tensor is planned with pim_tensor_plan_growable: it starts with no room along
 * its growth axis, and pim_tensor_grow adds room by pim_alloc'ing the units it needs
 * and recording them in the tensor's unit table.  Units already written stay where
 * they are.  Room comes in whole units:
 *
 *     along `out`   one unit = nch * nbank * pack outputs   (K: 32 or 64 tokens on ch2)
 *     along `red`   one chunk = 1024 reductions, ngroups units (V: 1024 tokens)
 *
 * pim_tensor_append() always advances the growth axis, so a caller writes the same
 * call for both and the asymmetry stays here.  And the asymmetry is large, falling
 * straight out of which axis is held fixed:
 *
 *     OUT_MAJOR  `out` fixed  -> (bank, ch, group) fixed, the reduction run walks
 *                one DRAM row.  ONE CONTIGUOUS TRANSFER per token, as long as
 *                nred fits a row.
 *     RED_MAJOR  `red` fixed  -> the outputs walk across every bank of every
 *                channel, at row_bytes apart.  ONE TRANSFER PER OUTPUT, of two
 *                bytes for a single token.
 *
 * That second line is the cost of attention on this machine and it is not a defect
 * to be fixed by a better loop: the bank axis is row_bytes apart in the address map,
 * so any write that spans banks and not whole rows is a scatter.  What it responds
 * to is BATCHING — append `count` tokens at once and each piece becomes 2*count
 * bytes while the number of pieces stays nout.
 *
 * ============================== the zero contract ==========================
 * A launch names `red_len`, and OPSIZE covers whole beats, so the final beat of any
 * launch reads PAST the write frontier.  For OUT_MAJOR that is harmless twice over:
 * the reduction length is a fixed model dimension and already beat-aligned, and a
 * garbage OUTPUT is confined to its own bank's accumulator and discarded by the
 * host.  For RED_MAJOR it is not harmless at all — the frontier lands wherever the
 * sequence happens to be.
 *
 * IS A ZEROED VECTOR OPPOSITE THOSE LANES ENOUGH?  Not quite, and the reason is
 * narrower than it looks.
 *
 * MEASURED 2026-09-17, runtime/test/tensor_board on the ch2 image.  The tail was
 * filled with six values and the same MAC run against a golden from pim_mac_exact:
 *
 *     1.5e18                                 0 of 32 outputs differ
 *     3.4e38, the largest finite BF16        0 of 32
 *     +Inf                                  32 of 32, every output NaN
 *     -Inf                                  32 of 32, every output NaN
 *     NaN                                   32 of 32, every output NaN
 *     0xA5A5, what an unwritten page reads   0 of 32
 *
 * SO THE MECHANISM IS NOT THE BLOCK EXPONENT.  A lane that is merely enormous —
 * right up to the largest finite BF16 — is neutralised by the zero beside it, which
 * says the exponent is taken after the multiply and not before.  What survives a
 * zero multiplier is the IEEE special: 0 * Inf is NaN, and one NaN poisons its
 * bank's entire accumulation.
 *
 * THE INVARIANT STANDS, for that reason instead.  A BF16 lane is Inf or NaN when
 * its exponent field is all ones: 256 of the 65536 patterns.  But the argument does
 * not rest on that ratio, because unwritten DRAM is not uniform — it holds whatever
 * the last user left.  It rests on the shape of the failure.  V's reduction axis is
 * the SEQUENCE, so fifteen times in sixteen the final beat straddles the frontier,
 * on every head of every layer of every token; one NaN takes its bank's whole
 * accumulation, and o_proj spreads that across the entire layer output.  You cannot
 * know what is there and once is enough.
 *
 * So a RED_MAJOR tensor carries an invariant:
 *
 *     M[out][red] == 0   for   frontier <= red < roundup(frontier, 16)
 *
 * DELIBERATELY THE WEAK FORM.  Above that beat the contents are free, because no
 * MAC with this `red_len` reaches them — and the difference is the difference
 * between clearing fifteen elements and clearing half a gigabyte.  Three calls keep
 * it: append establishes it as it goes (see the lazy clear below), pim_tensor_alloc
 * with PIM_ALLOC_F_ZERO establishes it at red = 0, and pim_tensor_truncate restores
 * it when the frontier moves BACKWARDS — a generation restart, a cache crop, a
 * rejected speculative token.  Nothing else may move the frontier back.
 */
#ifndef PIM_TENSOR_H
#define PIM_TENSOR_H

#include <stdint.h>

#include "pim/pim.h"
#include "pim/pim_geometry.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Elements of BF16 in one beat and in one DRAM row.  Both are facts about the
 * datapath, not parameters: a beat is 16 lanes and a row is EMU_MAX_OPSIZE beats. */
#define PIM_TENSOR_ELEMS_PER_BEAT  16u
#define PIM_TENSOR_ELEMS_PER_ROW   1024u

/* WHICH AXIS THE HOST ARRAY'S OUTER INDEX IS.  See the header comment: this does
 * NOT change pim_tensor_offset, only how upload/append walk `src` — and which axis
 * append advances. */
typedef enum {
	/* src is [nout][nred].  Linear weights (torch's nn.Linear.weight order, so a
	 * tensor goes straight in) and the K cache, whose new token is a new output. */
	PIM_LAYOUT_OUT_MAJOR = 0,
	/* src is [nred][nout].  The V cache, whose new token is a new reduction step
	 * and therefore a scatter across every bank. */
	PIM_LAYOUT_RED_MAJOR = 1,
	/* src is [nout][nred], as OUT_MAJOR, and outputs short enough to share a bank
	 * row do.  Each output gets `slot` elements of a row — nred rounded up to a
	 * power of two — and `pack` = 1024 / slot neighbouring outputs sit side by side
	 * in one row.  Output o then starts o * slot elements into the allocation, so
	 * consecutive outputs are consecutive on the card.  The K cache of a model whose
	 * H_kv*D is below 1024; at 1024 and above it places exactly like OUT_MAJOR. */
	PIM_LAYOUT_OUT_PACKED = 2,
} pim_layout;

typedef struct {
	void      *base;       /* PIM_MEM_DRAM, every unit in one allocation.  NULL
	                        * for a growable tensor, whose units are in `units`   */
	pim_layout layout;
	uint32_t   nout, nred; /* as asked for                                       */

	/* PADDED TO WHAT THE HARDWARE ADDRESSES.  `nout` to a whole supergroup,
	 * because one all-bank all-channel MAC covers exactly nbank*nch outputs and
	 * there is no lane mask anywhere on the MAC path — a partial supergroup is not
	 * expressible.  `nred` only to a whole BEAT, not to a whole row: the last
	 * chunk carries a shorter OPSIZE instead, which is real work saved. */
	uint32_t   noutpad, nredpad;
	uint32_t   nchunks;    /* ceil(nredpad / 1024)                               */
	uint32_t   last_beats; /* beats in the final chunk, 1..64                    */
	uint32_t   ngroups;    /* noutpad / (nbank * nch): MAC groups, one MAC each  */
	uint32_t   nch, nbank; /* copied from the geometry it was planned against    */

	/* Outputs per bank row and the elements each owns in it.  pack is 1 and slot
	 * is 1024 except for an OUT_PACKED tensor with nred under 1024; noutpad is
	 * then padded to whole rows, nbank * nch * pack outputs. */
	uint32_t   pack, slot;

	uint64_t   tag;        /* pim_tensor_tag(), stamped on the allocation        */

	/* ---- growth state.  Meaningful once anything has been appended. --------- */

	/* How far along the growth axis has been written.  The launch length a caller
	 * passes is its business, but this is what the zero invariant is stated
	 * against, and pim_tensor_truncate needs it to know what to clear. */
	uint32_t   frontier;

	/* RED_MAJOR lazy clear: reduction chunks cleared so far.  Clearing all of a
	 * KV cache up front costs a full pass of DMA over hundreds of megabytes —
	 * about 630 ms for a 3B model at 8K — and a generation that stops after two
	 * hundred tokens paid all of it for nothing.  Clearing chunk by chunk on first
	 * entry costs the same in total and only what is used. */
	uint32_t   zeroed_chunks;

	/* ---- growable storage: used when `growable` is 1, and `base` is then NULL.
	 * units[u] is where broadcast unit u lives — a host pointer into one of the
	 * allocations in pages[], which pim_tensor_grow made and pim_tensor_free frees. */
	uint32_t   growable;
	uint32_t   nunits, units_cap;
	void     **units;
	uint32_t   npages, pages_cap;
	void     **pages;
} pim_tensor;

/* ============================== planning ===================================
 * THE TILING DECISION, WITH NOTHING ALLOCATED.  Pure arithmetic on the geometry: no
 * context, no pool, no board.  `out->base` is left NULL.
 *
 * Split from the allocation because they are different decisions with different
 * lifetimes — how a shape tiles is the same in every layer of a model, while where
 * a tensor lives belongs to whoever owns it. */
void   pim_tensor_plan (const pim_geometry *g, pim_layout layout,
                        uint32_t nout, uint32_t nred, pim_tensor *out);

/* Bytes of PIM_MEM_DRAM a planned shape needs. */
size_t pim_tensor_bytes(const pim_geometry *g, const pim_tensor *t);

/* plan + pim_alloc_ex + stamp the tag, in one call.  `aflags` goes to
 * pim_alloc_ex_ctx; PIM_ALLOC_F_ZERO there establishes the zero invariant eagerly
 * for a caller that would rather not pay it in the middle of a generation. */
const char *pim_tensor_alloc(pim_ctx *c, pim_layout layout,
                             uint32_t nout, uint32_t nred, unsigned aflags,
                             pim_tensor *out);
void        pim_tensor_free (pim_ctx *c, pim_tensor *t);

/* ============================== growth =====================================
 * A tensor with no room yet along its growth axis — `out` for OUT_MAJOR and
 * OUT_PACKED, `red` for RED_MAJOR.  `fixed` is the other axis.  Nothing is
 * allocated; the first pim_tensor_grow does that. */
void        pim_tensor_plan_growable(const pim_geometry *g, pim_layout layout,
                                     uint32_t fixed, pim_tensor *out);

/* Make room for at least `n` along the growth axis.  The missing units are
 * pim_alloc'd as one allocation, tagged, and appended to the unit table.  Asking for
 * no more than the tensor has does nothing; on failure the tensor is unchanged. */
const char *pim_tensor_grow(pim_ctx *c, pim_tensor *t, uint32_t n);

/* Room along the growth axis: what grow has provided, or the planned size. */
uint32_t    pim_tensor_room(const pim_tensor *t);

/* Units the tensor holds. */
static inline uint32_t pim_tensor_nunits(const pim_tensor *t)
{ return t->pack ? t->ngroups / t->pack * t->nchunks : 0; }

/* The host pointer of broadcast unit `u`, however the tensor is stored.  NULL past
 * the last unit. */
void       *pim_tensor_unit_addr(const pim_geometry *g, const pim_tensor *t, uint32_t u);

/* ============================== addressing =================================
 * Where M[out][red] sits, as a byte offset into the tensor's units laid end to end.
 * Unit offset / unit_bytes is where it actually is: pim_tensor_unit_addr() of that
 * unit, plus the remainder.
 *
 * PUBLIC ON PURPOSE.  This permutation is the part most likely to be wrong and the
 * part a board cannot check cheaply, so a host-only test walks it directly.  Pure
 * arithmetic; touches nothing.
 *
 *     slot = out % pack           q = out / pack
 *     bank = q % nbank            ch = (q/nbank) % nch
 *     group = q / (nbank*nch) * pack + slot           chunk(red) = red / 1024
 *     unit = pim_tensor_unit(t, group, chunk)
 *     element in the row = slot * t->slot + red % 1024
 *
 * With pack = 1 this is bank = out % nbank, group = out / (nbank*nch) and the
 * element is red % 1024.
 *
 * Every channel of a supergroup uses the SAME (row, col): the ISA carries one ROW
 * field and CH_MASK is fan-out only, so that part is structural. */
uint64_t pim_tensor_offset(const pim_geometry *g, const pim_tensor *t,
                           uint32_t out, uint32_t red);

/* Which broadcast unit holds MAC group `group`'s reduction chunk `chunk`.
 *
 * THE ONE PLACE THIS FORMULA LIVES, and that is not tidiness.  It is the contract
 * between where the host PUTS an element (pim_tensor_offset, at upload) and where a
 * MAC LOOKS for it (the program builder, at launch) — different functions, in
 * different files, at different times.  If they ever disagree every MAC reads a row
 * its data is not in, and nothing catches it: not the encoder, not the hardware.
 * The result is a number, not an error.  Written once, they cannot disagree. */
static inline uint32_t pim_tensor_unit(const pim_tensor *t, uint32_t group,
                                       uint32_t chunk)
{
	if (t->layout == PIM_LAYOUT_RED_MAJOR)
		return chunk * (t->ngroups / t->pack) + group / t->pack;
	return group / t->pack * t->nchunks + chunk;
}

/* Where MAC group `group` starts inside its row, in elements: the slot it owns.
 * The program builder adds it to COL.  0 unless several outputs share a row. */
static inline uint32_t pim_tensor_row_off(const pim_tensor *t, uint32_t group)
{ return group % t->pack * t->slot; }

/* The output that bank `bank` of channel `ch` computes in MAC group `group`.
 * Results come back per (group, channel, bank); this puts each one at its place
 * in output order.  With pack > 1 one MAC covers every pack-th output, and the
 * pack MAC groups of one row together cover a contiguous range. */
static inline uint32_t pim_tensor_output(const pim_tensor *t, uint32_t group,
                                         uint32_t ch, uint32_t bank)
{ return ((group / t->pack * t->nch + ch) * t->nbank + bank) * t->pack + group % t->pack; }

/* How many MAC groups, from group 0, cover outputs [0, n).  Whole rows take pack
 * groups each; in a partly filled last row only the slots that hold an output are
 * computed, so n = 10 with pack 2 takes 2 groups and n = 65 takes 3. */
static inline uint32_t pim_tensor_groups(const pim_tensor *t, uint32_t n)
{
	uint32_t per = t->nch * t->nbank * t->pack, rem = n % per;
	return n / per * t->pack + (rem < t->pack ? rem : t->pack);
}

/* A fingerprint of everything that decides where an element goes: the layout, the
 * padded shape, the topology.  Stamped on the allocation by pim_tensor_alloc (on
 * every allocation, by pim_tensor_grow) and compared by pim_prog_lower, so a program
 * built for one arrangement cannot read an allocation filled with another.
 *
 * For a growable tensor the size along the growth axis is left out, so the tag does
 * not change as it grows.  A unit's contents do not depend on that size.
 *
 * The layout is mixed in because two tensors of the same shape and different
 * layouts hold transposed contents, and the hardware cannot tell them apart. */
uint64_t pim_tensor_tag(const pim_geometry *g, const pim_tensor *t);

/* ============================== filling ====================================
 * UPLOAD — the whole tensor at once, for something that does not grow.  `src` is
 * nout x nred BF16 in the tensor's layout order, unpadded; the padding this adds is
 * zero, which contributes exactly 0.0f to a sum in any order and so cannot perturb
 * a real output.  That is a statement about the arithmetic, not a tolerance. */
const char *pim_tensor_upload(pim_ctx *c, pim_tensor *t, const void *src);

/* APPEND — `count` steps along the GROWTH AXIS, starting at `first`.
 *
 *     OUT_MAJOR   the growth axis is `out`.  src is [count][nred].
 *     RED_MAJOR   the growth axis is `red`.  src is [count][nout].
 *
 * Either way src is [step][the other axis] and contiguous, which is the shape the
 * host already has a new token in.
 *
 * `first` need not equal t->frontier — a caller re-writing a rejected speculative
 * token appends over it — but appending PAST the frontier with a gap would leave
 * unwritten elements below it, so that is refused.
 *
 * It writes within the room the tensor has.  A growable tensor is given more with
 * pim_tensor_grow first.
 *
 * For RED_MAJOR this clears the reduction chunk `first` lands in, if it has not
 * been cleared before.  That is the zero invariant being established one chunk
 * ahead of the frontier rather than all at once. */
const char *pim_tensor_append(pim_ctx *c, pim_tensor *t, uint32_t first,
                              uint32_t count, const void *src);

/* TRUNCATE — move the frontier BACK to `pos` and restore the zero invariant.
 *
 * The one call that has to exist for the invariant to close, and the one a caller
 * forgets.  Three things in transformers move a frontier backwards: Cache.reset()
 * between generations, crop(), and a rejected speculative token.  After any of
 * them the elements above `pos` hold the previous generation's values, and the
 * final beat of the next launch reads them.
 *
 * RED_MAJOR clears [pos, frontier) across every output.  pos == 0 is free: it
 * retracts the lazy-clear bookkeeping instead, so the next append re-clears the
 * chunk it touches.  OUT_MAJOR has nothing to clear and only moves the frontier. */
const char *pim_tensor_truncate(pim_ctx *c, pim_tensor *t, uint32_t pos);

#ifdef __cplusplus
}
#endif
#endif /* PIM_TENSOR_H */
