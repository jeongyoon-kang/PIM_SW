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
 * THE DEVICE FORMULA IS THE SAME FOR BOTH LAYOUTS, and that is the surprise worth
 * stating plainly, because "K is row-major and V is column-major" suggests two
 * addressing schemes and there is only one.  pim_tensor_offset(g, t, out, red) does
 * not branch on the layout at all.
 *
 * What the layout names is the HOST array's index order — which axis you walk to
 * read the bytes you are about to send:
 *
 *     PIM_LAYOUT_OUT_MAJOR   src is [nout][nred]   one output's run is contiguous
 *     PIM_LAYOUT_RED_MAJOR   src is [nred][nout]   one reduction step's run is
 *
 * It is carried in the tensor, and mixed into the tag, because the two are NOT
 * interchangeable even though the addressing is: bytes placed by an OUT_MAJOR walk
 * and read by a program built for a RED_MAJOR one are transposed, silently, into a
 * plausible wrong number.
 *
 * ============================== growth =====================================
 * A KV cache grows one token at a time, and WHICH AXIS grows is the layout:
 *
 *     K is OUT_MAJOR   a new token is a new OUTPUT   -> grows along `out`
 *     V is RED_MAJOR   a new token is a new REDUCTION STEP -> grows along `red`
 *
 * pim_tensor_append() always advances the growth axis, so a caller writes the same
 * call for both and the asymmetry stays here.  And the asymmetry is large, falling
 * straight out of the one formula by which axis is held fixed:
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
} pim_layout;

typedef struct {
	void      *base;       /* PIM_MEM_DRAM, ngroups * nchunks broadcast units    */
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
	uint32_t   ngroups;    /* noutpad / (nbank * nch)                            */
	uint32_t   nch, nbank; /* copied from the geometry it was planned against    */

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

/* ============================== addressing =================================
 * Where M[out][red] sits inside the allocation.
 *
 * PUBLIC ON PURPOSE.  This permutation is the part most likely to be wrong and the
 * part a board cannot check cheaply, so a host-only test walks it directly.  Pure
 * arithmetic; touches nothing.
 *
 *     bank(out) = out % nbank     ch(out) = (out/nbank) % nch
 *     group(out) = out / (nbank*nch)      chunk(red) = red / 1024
 *     unit = group * nchunks + chunk
 *
 * Every channel of a supergroup uses the SAME (row, col): the ISA carries one ROW
 * field and CH_MASK is fan-out only, so that part is structural. */
uint64_t pim_tensor_offset(const pim_geometry *g, const pim_tensor *t,
                           uint32_t out, uint32_t red);

/* Which broadcast unit holds output group `group`'s reduction chunk `chunk`.
 *
 * THE ONE PLACE THIS FORMULA LIVES, and that is not tidiness.  It is the contract
 * between where the host PUTS an element (pim_tensor_offset, at upload) and where a
 * MAC LOOKS for it (the program builder, at launch) — different functions, in
 * different files, at different times.  If they ever disagree every MAC reads a row
 * its data is not in, and nothing catches it: not the encoder, not the hardware.
 * The result is a number, not an error.  Written once, they cannot disagree. */
static inline uint32_t pim_tensor_unit(const pim_tensor *t, uint32_t group,
                                       uint32_t chunk)
{ return group * t->nchunks + chunk; }

/* A fingerprint of everything that decides where an element goes: the layout, the
 * padded shape, the topology.  Stamped on the allocation by pim_tensor_alloc and
 * compared by pim_prog_lower, so a program built for one arrangement cannot read an
 * allocation filled with another.
 *
 * THE LAYOUT IS MIXED IN EVEN THOUGH THE ADDRESSING DOES NOT DEPEND ON IT.  Two
 * tensors of the same shape and different layouts occupy the same bytes and hold
 * transposed contents; the hardware cannot tell them apart and this is the only
 * thing that can. */
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
