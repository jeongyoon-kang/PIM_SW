// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim.h — the runtime for emulator_top.  Allocation and compute.
//
// WHAT THIS IS
//   SW/ holds bring-up tools: one question each, run by hand, printing tables.
//   This is the thing a program links against.  Same hardware, same address map
//   (it includes SW/emu_regs.h so the numbers have one home), different job.
//
// THE ONE KERNEL
//   This device does GEMV.  Everything here exists to place weights so a GEMV can
//   reach them, and to issue as much of one as fits behind a single doorbell.
//
// EVERY RULE BELOW WAS MEASURED ON THE BOARD, 2026-08-10 (HW/r1p0/docs/report.md):
//   - a MAC makes every bank of pu_mask read the SAME (ROW, COL), so the unit of
//     allocation is a DRAM row REPLICATED ACROSS ALL 16 BANKS, not a byte  (§1.4)
//   - RD_MAC returns lane i = bank i, so output n must live in bank n % 16 and a
//     whole result comes back as one contiguous GPR run                    (§1.4)
//   - the accumulator holds fp32 and survives across ISRs, so a dot product longer
//     than one chunk may be chained without an extra rounding             (§1.7)
//   - one WRVEC feeds many MACs (the GB rewinds), so a vector load can be shared
//     across output groups                                                (§1.4)
//   - a beat's sixteen products are reduced by their own adder, starting from
//     zero, and only that result reaches the accumulator.  pim_gemv_golden() is
//     that shape and PIM_EXACT reproduces it bit for bit.
//
// THE TWO SCHEDULES ARE NOT INTERCHANGEABLE
//   PIM_EXACT (group-outer) chains a whole output in the accumulator and rounds
//   once.  PIM_FAST (chunk-outer) is 1.4-1.6x faster but reads the accumulator
//   once per K-chunk, so each output is the host sum of nchunks values that were
//   ALREADY rounded to BF16.  That error is bounded in ABSOLUTE terms — measured
//   at 0.85-1.7% of rms(y) — and NOT in relative terms: on an output whose chunk
//   partials cancel, the relative error is unbounded (14736 ulp was observed).
//   PIM_EXACT is therefore the default and PIM_FAST is an opt-in for callers who
//   know their outputs do not cancel.
//
// THERE IS NO VALIDITY GATE IN THE HARDWARE.  A malformed program returns a wrong
// number or hangs, and nothing can tell you which.  So this library is built to
// make the illegal cases unrepresentable rather than merely detectable:
//   - only pim_emit_*() can produce a 256-bit ISR word
//   - only pim_row_of() can turn a weight handle into an ISR ROW field
//   - pim_prog_verify() walks a whole program for the rules a single ISR cannot
//     show (WRVEC/MAC OPSIZE pairing, latch hygiene, result-word collisions)
//   - the DRAM timing registers are written once at open and never again
//
// EXCLUSIVITY IS THE REAL CONSTRAINT, and nothing here can enforce it.
//   A pim_dev is not thread safe and two processes must not drive the board at
//   once: the IMEM port gives the fetcher priority over host writes, so a host
//   write during someone else's kernel stalls on WREADY until the driver's 10 s
//   timeout, and that one EIO latches the H2C engine until a queue teardown.
//   Worse, a Vivado Hardware Manager attached over JTAG can reprogram the PDI at
//   any moment, which invalidates PCIe enumeration underneath a running host
//   process — every read returns 0xffffffff with no PCIe error to explain it.
//   That happened once (report.md 2.2) and cost an afternoon of misdiagnosis.
//   pim_open() and every launch check for the all-ones signature and say so, but
//   detecting it is all software can do.
//
// ERRORS: every call returns NULL on success or a human-readable reason.  The
// string is owned by the library and stays valid until the next call on that dev.
//////////////////////////////////////////////////////////////////////////////////
#ifndef PIM_H
#define PIM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================== geometry ======================================
// Restated here so a caller need not include emu_regs.h to size its own buffers.
#define PIM_BANKS            16u     // and therefore outputs per RD_MAC word
#define PIM_LANES            16u     // BF16 per 256 b beat
#define PIM_BEATS_PER_ROW    64u     // 2048 B / 32 B
#define PIM_ELEMS_PER_ROW    1024u   // 64 beats x 16 lanes — the K-chunk size
#define PIM_COLS_PER_ROW     64u     // 2048 B page / 32 B column
#define PIM_COLSET_BYTES     512u    // one column in all 16 banks: 32 B x 16
#define PIM_ROBACO_ROW       32768u  // one whole page in all 16 banks — the
                                     // allocator's granule, and exactly what one
                                     // all-bank MAC consumes
// CAPACITY IS NOT A CONSTANT.  It is bank_window x nbank per channel, and
// bank_window comes from platform/*.conf — 256 MiB today, and the ISR's 17-bit ROW
// reaches exactly that.  It used to be PIM_ROWS/PIM_COLS here, which quietly
// assumed one image.  Ask pim_meminfo().

// ============================== device ========================================
typedef struct pim_dev pim_dev;

typedef struct {
    uint64_t launches;      // doorbells rung
    uint64_t isrs;          // ISR words issued
    uint64_t device_us;     // doorbell -> results readable
    uint64_t h2c_bytes, h2c_us;
    uint64_t c2h_bytes, c2h_us;
    uint64_t polls;         // STATUS reads
} pim_stats;

typedef struct {
    // NO PLATFORM FIELD.  The channel count, the bank stride and the bank window
    // are compiled in from platform/<name>.conf — switching platforms means a
    // different bitstream, and rebuilding a library alongside one costs nothing.
    // pim_open() checks that this build matches the currently selected platform.
    const char *bdf;        // NULL -> "0000:01:00.0"
    const char *h2c;        // NULL -> "/dev/qdma01000-MM-0"
    const char *c2h;        // NULL -> "/dev/qdma01000-MM-1"

    // A single program is capped here, and pim_gemv SPLITS across launches to
    // respect it.  The default is the hardware limit: PROG_LEN is 14 bits and IMEM
    // holds 16384 words.  16003 ISRs and a 250 MiB weight upload have both been run
    // [measured 2026-08-10]; nothing suggests a lower ceiling.
    //
    // (An earlier version of this file capped it at 2000 because a long program was
    // in the air when the board stopped answering.  Program length was RULED OUT by
    // re-running the same command 4/4 clean and 16003 ISRs / 250 MiB clean.  The
    // actual cause was never established — most likely concurrent use of the board
    // from elsewhere.  Do not write a cause here that has not been shown.)
    uint32_t max_isrs;      // 0 -> 16383

    // Largest single pwrite/pread.  Also the granularity at which the device is
    // re-checked for liveness during a big upload — which is what turns "it died
    // somewhere in 250 MiB" into "it died at this offset", and that distinction is
    // what identified the real cause the one time this went wrong.
    uint32_t max_xfer;      // 0 -> 4 MiB

    // GPR split: staging for the input vector, and the landing zone for results.
    uint32_t gpr_vec_words; // 0 -> 4096  (up to 64 K-chunks)
} pim_config;

// cfg may be NULL for the defaults above.
const char *pim_open (const pim_config *cfg, pim_dev **out);
void        pim_close(pim_dev *d);
const char *pim_last_error(const pim_dev *d);
void        pim_get_stats (const pim_dev *d, pim_stats *out);
void        pim_reset_stats(pim_dev *d);

// Drain every bank's accumulator and throw the result away.
//
// The latch is cleared ONLY by RD_MAC and by reset, so it is live state that
// crosses doorbells and crosses processes.  A program that dies part way leaves it
// dirty, and the next program's first RD_MAC then returns someone else's partial
// sum with no way to notice.  pim_open() does this once; do it again after any
// failed launch.
const char *pim_scrub(pim_dev *d);

// ============================== memory ========================================
// THE UNIT IS A COLUMN, NOT A ROW.
//   The address encoding is BA-RO-CO and a MAC names (ROW, COL, OPSIZE) — OPSIZE
//   is a COLUMN COUNT.  So what the hardware addresses is a run of columns inside
//   one row, and the smallest thing worth allocating is ONE COLUMN HELD IN ALL 16
//   BANKS: 32 B x 16 = 512 B.  There are PIM_ROWS x 64 of them.
//
//   All-bank operation is what forces the "in all 16 banks" part: an all-bank MAC
//   reads the same offset relative to each bank's base, so anything it touches has
//   to sit at the same (ROW, COL) everywhere.  That is a constraint on ONE MAC,
//   not on an allocation — two tensors that never appear in the same MAC may share
//   a row at different column ranges, and this allocator lets them.
//
//   2048 B is the DRAM PAGE.  Crossing it needs a fresh activate, and the
//   emulator controller does NOT split a MAC at a page boundary — an ISR with
//   COL + OPSIZE > 64 is simply not supported.  So no single MAC may straddle a
//   row, and that is what cuts K at 1024 elements.
typedef struct {
    uint64_t off;       // byte offset in the per-channel RoBaCo space
    uint64_t size;      // bytes, rounded up to a whole RoBaCo row
} pim_buf;

// A REQUEST IS A NUMBER OF BYTES.  Nothing else.
//
// It used to be (ncols, align), and both were wrong.  "Column" is a GEMV word —
// 16 banks x 32 B — so the allocator's own signature spoke the caller's kernel.
// And `align` was passed the SAME value by all three call sites, because page
// alignment is not a caller's choice: it is what the placement arithmetic needs.
// Two of the three then computed a row count and multiplied by 64 just to get back
// to columns.  Three signs of an interface facing the wrong way.
//
// So: bytes in, an offset out, and the allocator rounds and aligns to what the
// hardware makes usable — one RoBaCo row, 16 banks x one 2048 B page = 32 KiB.
// Below that granule nothing can compute: an all-bank MAC reads the same (row, col)
// in every bank, so a fragment of one bank is not an operand.
//
// The waste is the tail of the last row, at most 32 KiB per allocation — for a
// 197-Linear model that is under 6 MiB of 4 GiB.  Packing several tensors into one
// page is possible (they never share a MAC) and is deliberately NOT done: it makes
// every placement carry a starting column, and nothing yet needs the capacity.
//
// THE OFFSET MEANS THE SAME THING IN EVERY CHANNEL.  A supergroup's weights must
// sit at identical (bank, row, col) in all of them — the ISR has one ROW field and
// CH_MASK only decides who executes — so one allocation is replicated, and it costs
// size x nch of real DRAM.
const char *pim_alloc(pim_dev *d, uint64_t nbytes, pim_buf *out);
const char *pim_free (pim_dev *d, pim_buf *b);

// All in bytes, per channel.
void        pim_meminfo(const pim_dev *d, uint64_t *used, uint64_t *freeb,
                        uint64_t *largest_run);

// ============================== weights =======================================
// W is [n][k], row major, BF16.  It is laid out so that
//     output row o  ->  bank (o % 16), group (o / 16)
//     K-chunk c     ->  one run of OPSIZE columns inside one page
// which is the only layout a 16-bank MAC can read and the only one whose result
// comes back as a contiguous GPR run.
//
// COLUMN PACKING.  A chunk is ceil(kpad/16) columns and a page holds 64, so when a
// chunk is short several GROUPS share a page.  That matters enormously for small
// k: at k = 64 a chunk is 4 columns, so 16 groups fit in one page and a
// row-granular layout would waste 93.8% of it.  groups_per_row says what happened.
//
// PADDING.  n is padded up to a multiple of 16 (whole groups) and k up to a
// multiple of 16 (whole beats) with ZEROS.  Zero weights contribute exactly 0.0f
// to a sum in any order, so the padding does not perturb the result — it is not a
// tolerance argument.  k is deliberately NOT padded to a multiple of 1024: the
// final chunk carries a shorter OPSIZE instead, which is 7% less work on a shape
// like k = 896.
// BIAS.  A MAC has no bias input — but it does not need one.  Appending b as an
// extra COLUMN of W and 1.0 as an extra element of x makes W'.[x;1] = W.x + b, and
// the bias then enters the fp32 accumulator BEFORE the single rounding at RD_MAC.
// That is strictly better than adding it to the returned BF16 on the host, which
// rounds twice — and it is what a fused Linear does.
//
// The cost is one more element of k, which is FREE when k is not already a
// multiple of 16 (the padding had room) and otherwise adds one beat — and if k was
// a multiple of 1024 that beat needs a whole extra chunk.  pim_weight_alloc
// reports the outcome in nchunks and last_opsize; look at them before assuming it
// was cheap.
typedef struct {
    pim_buf  buf;
    uint32_t n, k;            // as asked for
    bool     has_bias;        // k+1 columns are stored; column k is b
    uint32_t npad, kpad;      // as stored
    uint32_t ngroups;         // npad / 16
    uint32_t nchunks;         // ceil(kpad / 1024)
    uint16_t full_opsize;     // 64 — columns in every chunk but the last
    uint16_t last_opsize;     // columns in the final chunk, 1..64
    uint32_t groups_per_row;  // how many output groups share a page (1 when k>512)
} pim_weight;

const char *pim_weight_alloc (pim_dev *d, uint32_t n, uint32_t k, pim_weight *out);
const char *pim_weight_alloc_bias(pim_dev *d, uint32_t n, uint32_t k, pim_weight *out);

// W is [n][k].  bias is [n], and must be non-NULL exactly when the handle was
// allocated with pim_weight_alloc_bias.
const char *pim_weight_upload(pim_dev *d, const pim_weight *w, const uint16_t *W,
                              const uint16_t *bias);
const char *pim_weight_free  (pim_dev *d, pim_weight *w);

// Read the weights back through the same aperture and compare against W.
// Nothing in the hardware protects resident weights from a mis-encoded ISR, and
// a corrupted weight is indistinguishable from a bad kernel from the outside.
// Returns NULL if every element matches; otherwise a reason naming the first
// element that does not, with its bank, row and beat.
const char *pim_weight_verify(pim_dev *d, const pim_weight *w, const uint16_t *W,
                              const uint16_t *bias, uint64_t *n_bad);

// ============================== compute =======================================
// PIM_FAST   chunk-outer.  The K-chunk is the outer loop, so the vector is loaded
//            nchunks times for the WHOLE gemv instead of once per output group.
//            [measured] 1.4-1.6x faster; the accumulator is read once per chunk so
//            each output is the host sum of nchunks BF16 partials — at most ONE
//            ulp from PIM_EXACT, and never more, for k up to 8192.
// PIM_EXACT  group-outer.  All of one output's chunks chain into the accumulator
//            and are read once, so the result is BIT-IDENTICAL to
//            pim_gemv_golden().  Slower.  This is what makes it possible to tell
//            "the device is wrong" from "the schedule rounded" — keep it.
typedef enum { PIM_FAST = 0, PIM_EXACT = 1 } pim_mode;

// y[0..n) = W . x[0..k)   — BF16 throughout.  y must hold w->n elements.
const char *pim_gemv(pim_dev *d, const pim_weight *w,
                     const uint16_t *x, uint16_t *y, pim_mode mode);

// Several weights against the SAME x, in as few doorbells as possible.  q/k/v
// share their input, and so do gate/up; running them together removes the
// duplicate vector loads and the extra doorbells.  Every weight must have the
// same k.
const char *pim_gemv_multi(pim_dev *d, const pim_weight *const *ws, unsigned nw,
                           const uint16_t *x, uint16_t *const *ys, pim_mode mode);

// Print the microcode a gemv WOULD issue, without touching the device.  This is
// the whole translation in one place: a weight handle plus a mode becomes a list
// of 256-bit ISR words, and this shows which field of which word each part of the
// allocation ended up in.
const char *pim_gemv_dump(pim_dev *d, const pim_weight *w, pim_mode mode,
                          FILE *out, unsigned max_words);

// ============================== attention =====================================
// Both attention matmuls are GEMVs, and they want OPPOSITE layouts, so the cache
// stores K and V differently:
//
//   scores  s[t] = sum_d q[d]*K[t][d]   reduce over d -> K is POSITION-MAJOR:
//                                       position t lives in bank t%16 and the whole
//                                       of K[t] lies along that row, so appending a
//                                       position is one contiguous write.
//   output  o[d] = sum_t p[t]*V[t][d]   reduce over t -> V is DIMENSION-MAJOR, i.e.
//                                       TRANSPOSED: dimension D = h*head_dim+d
//                                       lives in bank D%16 and positions run along
//                                       the row.
//
// The transpose is what makes p.V expressible at all — a MAC reduces along a row
// inside one bank, so the reduction axis has to be the one stored along the row.
// It also makes p.V the BETTER-shaped of the two: its reduction axis is the
// sequence, so OPSIZE reaches 64, while scores reduce over head_dim and are stuck
// at OPSIZE = head_dim/16.
//
// The cost of the transpose is that one new position owns a single element of
// hkv*head_dim different beats, and without WSTRB a single element cannot be
// written.  So V is BUFFERED on the host and flushed a whole beat at a time;
// whatever has not been flushed is folded into pim_attn_output on the host, in the
// same fp32 accumulator, so the split does not show in the answer.
typedef struct {
    pim_buf   kbuf, vbuf;
    uint32_t  hkv, head_dim, max_pos;
    uint32_t  npos;        // positions appended
    uint32_t  flushed;     // of those, how many of V have reached DRAM
    uint32_t  v_chunks;
    uint32_t  flush_gran;  // buffered positions per flush; a multiple of 16
    uint16_t *pending;     // [flush_gran][hkv*head_dim], the unflushed tail of V
} pim_kv;

// flush_gran 0 -> 64.  Larger means fewer, bigger writes and a longer host tail;
// smaller means the opposite.  It must be a multiple of 16 and divide 1024.
const char *pim_kv_alloc(pim_dev *d, uint32_t hkv, uint32_t head_dim,
                         uint32_t max_pos, uint32_t flush_gran, pim_kv *out);
const char *pim_kv_free (pim_dev *d, pim_kv *c);

// k and v are [hkv][head_dim] for one position.
const char *pim_kv_append(pim_dev *d, pim_kv *c, const uint16_t *k, const uint16_t *v);
const char *pim_kv_flush (pim_dev *d, pim_kv *c);

// ALL query heads of a layer in one call, because that is the granularity the
// hardware wants: per head the compute is ~19 us at S=512 and a launch costs ~50,
// so one call per head spends four fifths of its time on overhead [measured].
// Under GQA several query heads share a kv head; kvhead[] is that mapping.
//   q[h] is [head_dim], s[h] receives npos scores
//   p[h] is [npos],     o[h] receives [head_dim]
const char *pim_attn_scores(pim_dev *d, const pim_kv *c, unsigned nq,
                            const uint32_t *kvhead, const uint16_t *const *q,
                            uint16_t *const *s);
const char *pim_attn_output(pim_dev *d, const pim_kv *c, unsigned nq,
                            const uint32_t *kvhead, const uint16_t *const *p,
                            uint16_t *const *o);

// ============================== binding support ================================
// A ctypes binding has to mirror pim_weight and pim_config by hand.  These let it
// check at import time that its mirror matches the .so it just loaded, instead of
// discovering a stale build through wrong numbers.
size_t pim_weight_sizeof(void);
size_t pim_config_sizeof(void);
size_t pim_stats_sizeof(void);
size_t pim_kv_sizeof(void);

// ============================== reference =====================================
// The device's arithmetic on the host: BF16 in, exact products, accumulation in
// fp32 STRICTLY IN LANE ORDER, one round-to-nearest-even back to BF16.
//
// The order is not a guess and not a convention borrowed from another DUT — it
// was measured on this silicon with vectors built so a tree and a sequential sum
// give answers that are not close (report.md 1.7).  PIM_EXACT reproduces this
// bit for bit; if it ever stops doing so, one of the two is broken.
// bias may be NULL.  When it is not, it is folded in the same way the device does
// it — as one more term of the dot product, inside the fp32 accumulator.
void pim_gemv_golden(const uint16_t *W, uint32_t n, uint32_t k,
                     const uint16_t *x, const uint16_t *bias, uint16_t *y);

uint64_t pim_now_us_pub(void);       // monotonic microseconds, for callers' own timing
uint16_t pim_f32_to_bf16(float f);
float    pim_bf16_to_f32(uint16_t h);
bool     pim_bf16_same(uint16_t a, uint16_t b);   // +0 and -0 are the same number
unsigned pim_bf16_ulps(uint16_t a, uint16_t b);   // distance, for error reporting

#ifdef __cplusplus
}
#endif
#endif // PIM_H
