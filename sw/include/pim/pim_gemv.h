/* SPDX-License-Identifier: MIT */
/*
 * pim_gemv.h — one GEMV tile, end to end.
 *
 * y[0..n) = W[n][k] . x[0..k)   in BF16, on the device.
 *
 * WHAT ONE TILE MEANS HERE: n = 16 * nch outputs, which is exactly what a single
 * all-bank, all-channel MAC covers.  Output j lives in bank j % 16 of channel
 * j / 16, and every channel uses the SAME (row, col) — the ISA carries one ROW field
 * and CH_MASK is fan-out only, so that is structural rather than a choice.  More
 * outputs means more supergroups, which is a loop around everything below and is not
 * implemented yet.
 *
 * k IS NOT LIMITED.  A DRAM page holds 1024 BF16, so k is cut into ceil(k/1024)
 * chunks and the program chains them: WRVEC, MAC, WRVEC, MAC, ... and one RD_MAC at
 * the end.  The accumulator is fp32 and survives across ISRs [measured 2026-08-10],
 * so the whole dot product is rounded ONCE, at RD_MAC.  This is the group-outer
 * schedule; it is bit-exact against pim_gemv_golden().
 *
 * WHY THE WEIGHT UPLOAD IS ONE TRANSFER.  Under RoChBaCo a linear run of unit_bytes
 * (nch * 32 KiB) covers one DRAM row in every bank of every channel — which is
 * precisely the operand of one all-bank MAC.  So placing a matrix is a HOST-SIDE
 * PERMUTATION into a staging buffer followed by a single pim_memcpy, not a scatter
 * of one transfer per (channel, bank).  The old direct-aperture path needed 16
 * transfers per channel because each bank had its own base address; going through
 * the MC removes that entirely.
 *
 * THE VECTOR AND THE RESULTS LIVE IN THE GPR, and the caller allocates them, because
 * they are memory with the caller's lifetime (design §9.4) — a vector uploaded once
 * and re-read by many launches is the normal case, and hiding the allocation would
 * make that impossible to express.
 */
#ifndef PIM_GEMV_H
#define PIM_GEMV_H

#include <stdint.h>

#include "pim.h"
#include "pim_exec.h"
#include "pim_logical.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void    *w;           /* PIM_MEM_DRAM: ngroups * nchunks broadcast units     */
    uint32_t n, k;        /* as asked for                                        */
    uint32_t npad;        /* n rounded up to a whole supergroup                  */
    uint32_t kpad;        /* k rounded up to a whole beat (16 BF16)              */
    uint32_t nchunks;     /* ceil(kpad / 1024)                                   */
    uint32_t last_beats;  /* beats in the final chunk, 1..64                     */
    uint32_t ngroups;     /* npad / (16 * nch) — supergroups                     */
    uint32_t nch, nbank;  /* copied from the geometry                            */
} pim_gemv_w;

/* ANY n AND ANY k.  Both are padded with ZEROS to what the hardware addresses:
 *
 *   n -> npad,  a whole supergroup (16 * nch outputs).  The padded outputs get zero
 *        weights, compute a zero dot product, and are dropped on the way back.
 *   k -> kpad,  a whole beat (16 BF16).  k is deliberately NOT padded to a whole
 *        1024 — the final chunk carries a shorter OPSIZE instead.
 *
 * Zero weights contribute exactly 0.0f to a sum in any order, so neither padding
 * perturbs a real output — that is a statement about the arithmetic, not a
 * tolerance.  What they cost is memory and a little work: ask npad and nchunks. */
/* THE TILING DECISION, WITH NOTHING ALLOCATED.  Pure arithmetic on the geometry —
 * no context, no pool, no board.  `out->w` is left NULL; the caller allocates.
 *
 * SPLIT FROM pim_gemv_alloc BECAUSE THE TWO ARE DIFFERENT DECISIONS.  How a shape
 * tiles depends on (n, k, geometry) and is the same in every layer of a model; WHERE
 * the weights live is per-tensor and belongs to whoever owns their lifetime.  The
 * GPR side of this API already worked that way — pim_gemv_gpr_bytes() hands back a
 * size and the caller calls pim_alloc() — and the weight side was the odd one out.
 *
 * It is also what lets a shape be planned once and reused: pim_gemv_logical() needs
 * a plan, not an allocation. */
void        pim_gemv_plan  (const pim_geometry *g, uint32_t n, uint32_t k,
                            pim_gemv_w *out);

/* Bytes of PIM_MEM_DRAM a planned shape needs. */
size_t      pim_gemv_bytes (const pim_geometry *g, const pim_gemv_w *w);

/* plan + pim_alloc in one call, for a caller that does not want to own the two
 * steps.  Sets out->w; pim_gemv_free releases it. */
const char *pim_gemv_alloc (pim_ctx *c, uint32_t n, uint32_t k, pim_gemv_w *out);
void        pim_gemv_free  (pim_ctx *c, pim_gemv_w *w);

/* W is [n][k] row-major BF16 — torch's nn.Linear.weight order, so a tensor goes
 * straight in.  Permuted on the host and sent as ONE transfer. */
const char *pim_gemv_upload(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W);

/* Read it back through the same window and compare.  Nothing in the hardware
 * protects resident weights, and a corrupted weight is indistinguishable from a bad
 * kernel from the outside. */
const char *pim_gemv_verify(pim_ctx *c, const pim_gemv_w *w, const uint16_t *W,
                            uint64_t *n_bad);

/* Where W[j][i] sits inside the weight allocation.
 *
 * PUBLIC ON PURPOSE: this permutation is the part most likely to be wrong and the
 * part a board cannot check cheaply, so a host-only test walks it directly.  It is
 * pure arithmetic and touches nothing.
 *
 *     bank(j) = j % 16     channel(j) = (j/16) % nch     supergroup(j) = j / (16*nch)
 *     unit    = supergroup * nchunks + chunk
 *
 * Every channel of a supergroup uses the SAME (row, col) — the ISA carries one ROW
 * field and CH_MASK is fan-out only — so that part is structural, not a choice. */
uint64_t pim_gemv_offset(const pim_geometry *g, const pim_gemv_w *w,
                         uint32_t j, uint32_t i);

/* Which broadcast unit holds output group `group`'s K-chunk `chunk`.
 *
 * THE ONE PLACE THIS FORMULA LIVES, and it is worth a function for a reason that is
 * not tidiness.  It is the contract between where the host PUTS a weight
 * (pim_gemv_offset, at upload) and where a MAC LOOKS for it (the program builders,
 * at launch).  Those are different functions, in different files, running at
 * different times — and if they ever disagree, every MAC reads a row that is not
 * the one its weights are in.  Nothing checks that: not the encoder, not
 * pim_prog_verify, not the hardware.  The result is a number, not an error.
 *
 * Written once, they cannot disagree. */
static inline uint32_t pim_gemv_unit(const pim_gemv_w *w, uint32_t group,
                                     uint32_t chunk)
{ return group * w->nchunks + chunk; }

/* The layout tag pim_gemv_upload() stamps and the program builders assume.  It is a
 * fingerprint of everything that decides WHERE a weight goes: the padded shape and
 * the topology.  Two allocations agree only if a MAC built for one would read the
 * right bytes out of the other. */
uint64_t pim_gemv_tag(const pim_geometry *g, const pim_gemv_w *w);

/* THE VECTOR HAS A LAYOUT CONTRACT TOO, and it is sharper than the matrix's.
 *
 * A WRVEC reads whole 16-lane beats, and there is no lane mask anywhere on the MAC
 * path.  So the elements from k to kpad are not "unused" — they are multiplied,
 * they join their beat's exponent maximum, and a stale one can annihilate every
 * real lane beside it or, if its bytes decode as Inf or NaN, poison the result.
 * ZEROING THEM IS NOT AN OPTIMISATION, IT IS THE CONTRACT (pim_mac_exact.c,
 * precondition 1).
 *
 * pim_gemv_upload_x() pads and stamps; pim_gemv_xtag() is what the WRVECs assume.
 * A GPR buffer filled by a plain pim_memcpy is refused at lowering, because there
 * is no way to tell from the bytes whether the tail was zeroed on purpose. */
uint64_t    pim_gemv_xtag(const pim_gemv_w *w);
const char *pim_gemv_upload_x(pim_ctx *c, const pim_gemv_w *w, void *xgpr,
                              const uint16_t *x);

/* ============================== eager all-bank =============================
 * VERSION 1 OF THE LAYOUT QUESTION: pad everything to what the hardware addresses,
 * and the host's array becomes the device's.
 *
 * WHAT MAKES IT WORK.  A MAC covers every bank of every channel, and each of those
 * banks contributes ONE output whose reduction runs along ONE row buffer.  So the
 * hardware's natural unit is "one output, one row" — and if a caller lays its matrix
 * out that way too, at a stride of a whole row rather than a tight k, the two
 * agree byte for byte:
 *
 *     device offset of W[j][i]  ==  (j * 1024 + i) * 2        for k <= 1024
 *
 * Verified numerically for every (j, i) by lib/test/eager_test, and it is not a
 * coincidence: under RoChBaCo the bank index is the field just below the row, so
 * consecutive outputs land in consecutive banks and consecutive channels with no
 * gap — offset j is exactly j row buffers in.
 *
 * SO THE UPLOAD IS ONE pim_memcpy.  No permutation, no staging buffer, and
 * pim_gemv_offset() is not consulted at all.  The allocation and the host array are
 * even the SAME SIZE (npad * 2048 bytes either way).
 *
 * WHAT IT COSTS, AND WHAT IT DOES NOT.  A row is 1024 elements and k may be far
 * less, so the HOST array carries the same padding the device already had — the
 * device side was never smaller, because an allocation reserves whole rows whatever
 * k is.  COMPUTE IS UNAFFECTED: OPSIZE still covers only ceil(k/16) beats, so a
 * MAC reads the real elements and stops.  Only the host's memory grows.
 *
 * WHY k <= 1024.  Past one row the reduction is cut into chunks, and chunks of
 * DIFFERENT outputs interleave in the allocation — output 0's second chunk sits
 * after output 31's first.  No amount of padding makes that linear, so
 * pim_gemv_upload_eager() refuses it and the permuting pim_gemv_upload() is the
 * path for those.
 *
 * ZEROS ARE STILL REQUIRED IN ONE PLACE: elements k .. ceil(k/16)*16-1 of each row,
 * the tail of the final beat, for the reason in pim_mac_exact.c's precondition 1.
 * The rest of the row is never read and may hold anything.
 */

/* Elements per row of the host array — the stride pim_gemv_upload_eager expects. */
uint32_t pim_gemv_host_stride(const pim_gemv_w *w);

/* Whether this shape can use the eager form at all (k <= one row buffer). */
bool     pim_gemv_is_eager(const pim_gemv_w *w);

/* W is [npad][pim_gemv_host_stride(w)] BF16, row-major.  Sent as one transfer with
 * no permutation, then stamped.  Rows n..npad-1 are never read as outputs and may
 * hold anything. */
const char *pim_gemv_upload_eager(pim_ctx *c, const pim_gemv_w *w,
                                  const uint16_t *W);

/* THE OTHER WAY TO SPEND THE SAME PADDING: one transfer PER OUTPUT ROW, carrying
 * only the beats a MAC will read.
 *
 *   pim_gemv_upload_eager   ONE pwrite of npad * 1024 elements.  Moves the row
 *                           padding too — for k = 128 that is 8x the real bytes.
 *   pim_gemv_upload_rows    n pwrites of kpad elements each.  Moves only what is
 *                           read, and skips rows n..npad-1 entirely because no
 *                           output is taken from them.
 *
 * MEASURED, ch2, 2026-08-26 (runtime/test/eager_board):
 *
 *     n=2048 k=128     permuting  8225 us    one DMA  5891 us    per row  26067 us
 *     n=2048 k=1024    permuting 19769 us    one DMA  5903 us    per row  30035 us
 *     n=32   k=128     permuting   154 us    one DMA   110 us    per row    479 us
 *
 * ONE DMA WINS EVERYWHERE, and it is not close.  The row form moves an EIGHTH of
 * the bytes at k = 128 and still takes four times as long, because a pwrite costs
 * about 13 us of descriptor setup whatever it carries — 2048 of them is 26 ms
 * before a single byte moves.  Fewer and fatter beats less and thinner by a wide
 * margin on this transport.
 *
 * THE PERMUTING FORM'S COST IS THE PERMUTATION, not the transfer: at k = 1024 both
 * send the same 4 MiB and it takes 14 ms longer, which is the scattered host-side
 * write pattern rather than anything on the card.  So where the eager form applies
 * it is 3.3x faster than the path that has always been used.
 *
 * pim_gemv_upload_rows is kept anyway: it is what an APPEND wants.  A KV cache adds
 * one row per token, and there "one transfer of the whole matrix" is not on the
 * menu — the choice is this or nothing.
 *
 * THE PROGRAM IS THE SAME EITHER WAY.  Both leave the identical bytes in the
 * identical places, and OPSIZE already covers only ceil(k/16) beats, so no opcode
 * changes — the choice is purely how the bytes get there.
 *
 * W is [n][kpad] BF16, row-major: beat-padded, NOT row-padded.  The tail from k to
 * kpad must be zero for the reason in pim_mac_exact.c's precondition 1; everything
 * past kpad is never read and is not sent. */
const char *pim_gemv_upload_rows(pim_ctx *c, const pim_gemv_w *w,
                                 const uint16_t *W);

/* Convenience for a caller whose matrix is tight [n][k]: copy it into the padded
 * shape, zeroing what has to be zero.  A caller that already owns its data in the
 * padded layout — a KV cache appended to one row at a time, say — does not need
 * this and should not pay for it. */
void pim_gemv_pack_eager(const pim_gemv_w *w, const uint16_t *W, uint16_t *out);

/* ============================== schedules =================================
 * WHAT THE SECOND ACCUMULATOR LATCH IS FOR.
 *
 * A WRVEC loads the vector into the global buffer, and a GB-sourced MAC REWINDS the
 * GB read pointer when it starts [measured, emu_chain --test rewind] — so one vector
 * load can feed many MACs.  That is free for K <= 1024, where the program is
 * WRVEC once and then a (MAC, RD_MAC) pair per output group.
 *
 * IT STOPS BEING FREE THE MOMENT K EXCEEDS ONE PAGE.  Then each output group has to
 * chain its chunks in the accumulator, and RD_MAC is read-CLEARING, so a second
 * group cannot be interleaved without destroying the first group's running sum.
 * With one latch the only exact schedule is group-outer:
 *
 *     for each group:  for each chunk: WRVEC(chunk), MAC(group, chunk)
 *                      RD_MAC
 *
 * and that reloads the whole vector once PER GROUP — ngroups * nchunks WRVECs for
 * what is physically one vector.  A 64-beat WRVEC costs more than a MAC, so on a
 * real GEMV that is most of the time.
 *
 * TWO LATCHES REMOVE HALF OF IT.  Issue the WRVEC once and then two MACs that differ
 * only in ROW and T:
 *
 *     for each pair of groups:
 *         for each chunk:  WRVEC(chunk)
 *                          MAC(group A, chunk, T=0)
 *                          MAC(group B, chunk, T=1)
 *         RD_MAC(T=0), RD_MAC(T=1)
 *
 * Each output still accumulates every one of its chunks in its own latch and is
 * rounded ONCE at its RD_MAC — so this is not a speed-for-accuracy trade.  BOTH
 * schedules below are bit-exact against pim_gemv_golden(); the only difference is
 * how many times the vector is loaded.
 *
 *   PIM_ACC_SINGLE   latch 0 only.   ngroups * nchunks WRVECs.
 *   PIM_ACC_DUAL     both latches.   ceil(ngroups/2) * nchunks WRVECs.
 *
 * DUAL needs the hardware to decode ISR[35]; see pim_exec_config.allow_t_latch.
 *
 * TWO CASES WHERE DUAL BUYS NOTHING, and both are worth knowing before reaching for
 * it:
 *   nchunks == 1   K fits one page, so there is no accumulator to protect between
 *                  groups.  BOTH schedules hoist a single WRVEC to the top and take
 *                  a (MAC, RD_MAC) pair per group off it.  One vector load either
 *                  way — pim_gemv_nwrvec() says so.
 *   ngroups == 1   there is no second group to pair with, so DUAL is SINGLE.
 */
typedef enum { PIM_ACC_SINGLE = 0, PIM_ACC_DUAL = 1 } pim_acc_mode;

typedef struct {
    uint64_t  launch_us;    /* summed over every doorbell this GEMV rang       */
    uint32_t  polls;
    uint32_t  nlaunch;      /* doorbells — more than one when IMEM is the limit */
    uint32_t  nisr;         /* ISRs, summed over launches                      */
    uint32_t  nwrvec;       /* vector loads, summed — the number DUAL halves   */
    bool      saw_done;     /* of the LAST launch; reported, never believed    */
} pim_gemv_stat;

/* Build the program a launch would run, from coordinates the caller has already
 * resolved.  chunk_row[] is the ISR ROW field for each K-chunk — pim_addr_unit()
 * produces them.
 *
 * SEPARATE FROM pim_gemv() SO IT CAN BE CHECKED WITHOUT A BOARD.  The ISR encoding
 * is where a mistake is a wrong NUMBER rather than an error — the validity gate was
 * pulled out of the fetch path for timing and has not come back — so being able to
 * walk the words on a host, with no card and no module, is worth a separate entry
 * point.  pim_gemv() resolves the coordinates and calls this. */
const char *pim_gemv_program(const pim_geometry *g, const pim_gemv_w *w,
                             uint32_t group_first, uint32_t group_count,
                             uint32_t xword, uint32_t yword,
                             const uint32_t *unit_row, pim_acc_mode mode,
                             pim_prog *out);

/* ============================== the two-pass form ==========================
 * THE SAME PROGRAM, BUILT WITHOUT KNOWING WHERE ANYTHING IS.
 *
 * pim_gemv_program() above takes coordinates the caller has already resolved —
 * unit_row[], xword, yword.  This one takes the ALLOCATIONS instead and records
 * what each address field means, leaving pim_prog_lower() to look them up.  See
 * pim_logical.h for why that is worth a second entry point:
 *
 *   - the RD_MAC-per-channel expansion stops being written out here.  This emits
 *     ONE logical drain per output group and marks it PIM_SPLIT_PER_CHANNEL; the
 *     lowerer knows a multicast RD_MAC is illegal, so no kernel has to.
 *   - the result depends on (n, k, geometry) and nothing else, so a transformer
 *     builds it once per SHAPE and lowers it once per layer's weights.
 *
 * IT NEEDS NO CONTEXT, NO ALLOCATOR AND NO BOARD.  The pointers are recorded, never
 * dereferenced or resolved.  w->w may even be a placeholder as long as the pointer
 * that eventually gets lowered names an allocation of the right size.
 *
 * Sizing: ask pim_gemv_nisr() for the ISR count as before — logical ISRs are FEWER,
 * because a per-channel drain is one here and nch after lowering — and use
 * pim_lower_isr_count() for the lowered buffer. */
const char *pim_gemv_logical(const pim_geometry *g, const pim_gemv_w *w,
                             uint32_t group_first, uint32_t group_count,
                             const void *xgpr, size_t xbytes,
                             const void *ygpr, size_t ybytes,
                             pim_acc_mode mode, pim_logical *out);

/* Logical ISR count — what pim_gemv_logical() will push.  Differs from
 * pim_gemv_nisr() only in that each group's drain counts once instead of nch times. */
uint32_t pim_gemv_logical_nisr(const pim_gemv_w *w, uint32_t group_count,
                               pim_acc_mode mode);

/* How many ISRs and how many vector loads a schedule costs, without building it.
 * The WRVEC count is the number the whole two-latch question is about. */
/* What `group_count` supergroups cost in one program.  Pass w->ngroups for the
 * whole GEMV; pim_gemv_ex() passes whatever fits one doorbell. */
uint32_t pim_gemv_nisr  (const pim_gemv_w *w, uint32_t group_count, pim_acc_mode mode);
uint32_t pim_gemv_nwrvec(const pim_gemv_w *w, uint32_t group_count, pim_acc_mode mode);

/* How many supergroups fit in one program of at most `max_isrs` ISRs, and therefore
 * how many launches the whole GEMV takes.  A model-sized GEMV does not fit in one:
 * IMEM holds 16383 ISRs and a [151936 x 1024] lm_head needs 14246 of them, so the
 * next size up does not.  pim_gemv_ex() splits by supergroup automatically; this is
 * how a caller finds out that it will, and how it sizes the result buffer. */
uint32_t pim_gemv_groups_per_launch(const pim_gemv_w *w, pim_acc_mode mode,
                                    uint32_t max_isrs);

/* The two GPR allocations a launch needs.  The vector is the whole padded k; the
 * result buffer only has to hold ONE launch's supergroups, because pim_gemv_ex()
 * drains it between launches.
 *
 * EXISTS SO A CALLER NEED NOT REDERIVE IT.  kpad*2 and groups*nch*32 are easy to get
 * wrong in a way that reads back as someone else's data rather than as an error. */
void pim_gemv_gpr_bytes(const pim_gemv_w *w, pim_acc_mode mode, uint32_t max_isrs,
                        size_t *xbytes, size_t *ybytes);

/* One launch.
 *   x     [k] BF16 in host memory.  Tiled into xgpr here.
 *   xgpr  a PIM_MEM_GPR allocation of at least kpad*2 bytes.
 *   ygpr  a PIM_MEM_GPR allocation — ask pim_gemv_gpr_bytes().  RD_MAC writes one
 *         32 B word per (supergroup, channel), and CH_MASK must be 1-hot so there is
 *         one RD_MAC per channel.  Only ONE launch's worth is needed: the buffer is
 *         drained between launches.
 *   y     [n] BF16, host memory, receives the answer.
 *
 * xgpr is rewritten every call.  A caller with a resident vector can skip that by
 * passing the same buffer and the same x; nothing here caches, so the cost is one
 * transfer of kpad*2 bytes. */
const char *pim_gemv(pim_ctx *c, pim_exec *e, const pim_gemv_w *w,
                     const uint16_t *x, void *xgpr, void *ygpr, uint16_t *y);

/* The same, with the schedule chosen and the launch reported.  pim_gemv() is this
 * with PIM_ACC_SINGLE and no stat.  `st` may be NULL. */
const char *pim_gemv_ex(pim_ctx *c, pim_exec *e, const pim_gemv_w *w,
                        const uint16_t *x, void *xgpr, void *ygpr, uint16_t *y,
                        pim_acc_mode mode, pim_gemv_stat *st);

/* ============================== the reference ==============================
 * The device's arithmetic, on the host, BIT FOR BIT.
 *
 * A BEAT IS BLOCK FLOATING POINT, NOT AN fp32 SUM, and that is not a detail.  The
 * hardware takes the MAX EXPONENT over all 16 lanes of a beat, right-shifts every
 * lane's mantissa to it, and adds those aligned integers exactly.  So the adder
 * tree's wiring cannot affect the answer and every bit that is lost is lost in the
 * ALIGNMENT SHIFT — a bare truncation with a hard kill at 24 places.  That is why a
 * beat of [+A, -A, eps] returns exactly zero on silicon: eps is annihilated during
 * alignment, before any adding happens.  No fp32 model reproduces that, however its
 * additions are ordered.
 *
 * So the reference is a transcription of the RTL (pim_mac_exact.c, eleven rounding
 * sites, each citing its line), not a float loop.  It is the same file the previous
 * runtime used and it agreed with the board.
 *
 * FOUR PRECONDITIONS, all of which pim_gemv() and pim_exec_open() satisfy — they
 * are listed at the top of pim_mac_exact.c and are worth reading before trusting a
 * comparison:
 *   1. both operand buffers zero in the padding lanes  (pim_gemv calloc's both)
 *   2. the accumulator empty on entry                  (pim_prog_verify enforces)
 *   3. T_CCD >= 2                                      (pim_exec_open set_timing)
 *   4. the RNE acc->BF16 converter, not the truncating one  (this bitstream) */
void pim_gemv_golden(const uint16_t *W, uint32_t n, uint32_t k,
                     const uint16_t *x, uint16_t *y);

/* One output's dot product, the way the datapath computes it.  Zero-pads the tail
 * beat itself, so k need not be a multiple of 16. */
uint16_t pim_mac_exact(const uint16_t *w, const uint16_t *x, unsigned k);

#ifdef __cplusplus
}
#endif
#endif /* PIM_GEMV_H */
