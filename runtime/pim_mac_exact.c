// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_mac_exact.c — the MAC datapath's arithmetic, transcribed from the RTL.
//
// DERIVED FROM  ref/PIM_HW/src/bank_controller/mac_tree/aim/{bf16_mul,bwms,mau,acc}
// and bank_controller_top.v.  Every rounding site below cites the line it came
// from; if the RTL moves, those citations are how this file gets re-checked.
//
// WHY THIS REPLACES AN fp32 SUM
//   The obvious host model — multiply, sum the 16 products in fp32, accumulate —
//   is not merely imprecise here, it is the WRONG SHAPE.  A beat is BLOCK FLOATING
//   POINT: bwms_top takes the max exponent over all 16 lanes, right-shifts every
//   lane's mantissa to it, and mau_tree then adds those aligned integers EXACTLY.
//   So the adder tree's wiring cannot affect the answer, and all the loss happens
//   in the alignment shift, which is a bare truncation with a hard kill at a
//   24-place shift.  That is why a beat of [+A, -A, eps] returns exactly zero on
//   silicon even though the +-A pair cancels: eps is annihilated during ALIGNMENT,
//   before any adding happens, and no fp32 model reproduces that.
//
// THE ELEVEN PLACES A BIT IS LOST, in order:
//   0  bf16_mul                nothing — 8b x 8b = 16b is exact
//   1  BWMS lane annihilation  shift >= 24 -> the lane becomes exactly 0
//   2  BWMS alignment          bare >> on {man,8'b0}: truncation toward zero
//   -  mau_tree                nothing — exact signed integer adds, 25->29 bits
//   3  acc ingest              tree sum >> K into 24 bits
//   4  acc align sticky        smaller operand's dropped bits OR'ed into its LSB
//   5  acc align saturation    exponent gap > 27 collapses to a sticky
//   6  acc normalize sticky    at lzd_pos == 27
//   7  acc round               RNE, 27b -> 24b
//   8  subnormal encode        TRUNCATING, not RNE
//   9  flush branches          post-round exponent < -22 -> +0
//  10  final 24b -> BF16       RNE on the 7-bit fraction
//
// FOUR PRECONDITIONS THE HARDWARE DOES NOT CHECK.  Each one, if broken, changes
// the answer silently — this model would then be right about a machine that is not
// the one running.
//
//  1. BOTH operand buffers must be ZERO in lanes k..15 of the final beat.  There is
//     no lane mask anywhere on the MAC path (mac_top.sv:152-171 instantiates all 16
//     multipliers unconditionally; bank_controller_top.v:677-678 feeds two whole
//     256-bit words).  A stale lane is not ignored: it joins the exponent max and
//     can annihilate every real lane in its beat, and if its bytes happen to decode
//     as Inf or NaN it poisons the whole result.  Zeroing only ONE side is not
//     enough.  pim_weight_upload memsets its staging block and pim_gemv's vector
//     staging memsets before copying, so both sides are covered here.
//
//  2. The accumulator latch must be EMPTY on entry.  Hardware clears it only on
//     reset and on a completed RD_MAC register handshake (acc_top.sv:161, 575-583).
//     pim_prog refuses to emit EOS while any MAC is undrained, and pim_scrub runs
//     at open and after a failed launch, so this holds by construction.
//
//  3. T_CCD must be >= 2.  acc_top is a 2-stage feedback pipe with no forwarding,
//     so back-to-back beats into one latch land only every OTHER beat
//     (mac_top.sv:26-33).  Not a violation counter — the answer is simply halved.
//     pim_open writes the timing registers once and never again.
//
//  4. THIS BUILD uses the RNE converter, not the truncating one.  There are two
//     acc -> BF16 paths in the tree: bank_controller_top.v:789-804 rounds to
//     nearest even, and mac_tree_top.v:184 (`fp32_word[31:16]`) truncates.  They
//     differ by up to one ulp on about half of all results.  The ch1 FPGA build
//     instantiates bank_controller_top x16 and its filelist
//     (emulator_top/fpga_verify/bd/filelist.f) contains mac_top.sv but NOT
//     mac_tree_top.v or PU_top.v — so the truncating path is not in this
//     bitstream.  If a future build goes through PU_top, the last step below has
//     to change.
//////////////////////////////////////////////////////////////////////////////////
#include <stdint.h>

uint16_t pim_mac_exact(const uint16_t *w, const uint16_t *x, unsigned k)
{
    /* ---- acc_top latch r0_* (acc_top.sv:117-119).  rstn value = all zero,
     *      which by the man==0 rule reads as numerical +0 (acc_top.sv:542-549).
     *      Latch 1 is dead silicon (emulator_controller.v:661,1854 tie
     *      BC_LATCH_LSB to 0), so one latch suffices.                        */
    uint32_t acc_sign     = 0;   /* 1b                                        */
    int64_t  acc_exp      = 0;   /* 10b signed, biased-by-127                 */
    uint32_t acc_man      = 0;   /* 24b, 1.23 form (0.frac when acc_exp == 0) */
    uint32_t acc_is_inf   = 0;
    uint32_t acc_is_nan   = 0;
    uint32_t acc_inf_sign = 0;

    unsigned nbeats = (k + 15u) / 16u;
    unsigned beat, i;

    for (beat = 0; beat < nbeats; beat++) {

        /* =================================================================
         * 16 x bf16_mul  (bf16_mul.sv:48-88, 94-143, 181-184)
         * ================================================================= */
        uint32_t l_sign[16], l_man[16], l_inf[16], l_nan[16];
        int64_t  l_exp[16];

        for (i = 0; i < 16; i++) {
            /* Zero-pad the tail of the vector: a padded lane feeds 0x0000,
             * whose product takes the cond_zero override -> exp 0, man 0. */
            unsigned idx = beat * 16u + i;
            uint16_t a = (idx < k) ? w[idx] : (uint16_t)0;
            uint16_t b = (idx < k) ? x[idx] : (uint16_t)0;

            /* bf16_mul.sv:48-54  straight bit-slice, no pre-processing */
            uint32_t as = ((uint32_t)a >> 15) & 1u;
            uint32_t ae = ((uint32_t)a >>  7) & 0xFFu;
            uint32_t af =  (uint32_t)a        & 0x7Fu;
            uint32_t bs = ((uint32_t)b >> 15) & 1u;
            uint32_t be = ((uint32_t)b >>  7) & 0xFFu;
            uint32_t bfr=  (uint32_t)b        & 0x7Fu;

            /* bf16_mul.sv:57-67 */
            uint32_t a_norm = (ae != 0u)   && (ae != 255u);
            uint32_t a_sub  = (ae == 0u)   && (af != 0u);
            uint32_t a_zero = (ae == 0u)   && (af == 0u);
            uint32_t a_inf  = (ae == 255u) && (af == 0u);
            uint32_t a_nan  = (ae == 255u) && (af != 0u);
            uint32_t b_norm = (be != 0u)   && (be != 255u);
            uint32_t b_sub  = (be == 0u)   && (bfr != 0u);
            uint32_t b_zero = (be == 0u)   && (bfr == 0u);
            uint32_t b_inf  = (be == 255u) && (bfr == 0u);
            uint32_t b_nan  = (be == 255u) && (bfr != 0u);

            /* bf16_mul.sv:70  unconditional XOR, never overridden */
            uint32_t sign = as ^ bs;

            /* bf16_mul.sv:74-78  eff exp 1 for subnormal, hidden bit = is_normal */
            uint32_t ea = a_sub ? 1u : ae;
            uint32_t eb = b_sub ? 1u : be;
            uint32_t ma = (a_norm << 7) | af;    /* Q1.7, 8b */
            uint32_t mb = (b_norm << 7) | bfr;   /* Q1.7, 8b */

            /* bf16_mul.sv:94-143 + 181.  The Wallace tree + 16b CPA is exactly
             * ma*mb: 255*255 = 65025 < 65536, so mod-2^16 is the identity.  No
             * rounding, no guard/sticky anywhere in this module. */
            uint32_t man16 = (ma * mb) & 0xFFFFu;   /* Q2.14 unsigned */

            /* bf16_mul.sv:81-82  single bias removed; range [-125,+381] */
            int64_t exp_raw = (int64_t)ea + (int64_t)eb - 127;

            /* bf16_mul.sv:85-88  priority NaN > Inf > Zero */
            uint32_t c_nan  = a_nan || b_nan || (a_zero && b_inf)
                                             || (a_inf  && b_zero);
            uint32_t c_inf  = !c_nan && (a_inf  || b_inf);
            uint32_t c_zero = !c_nan && !c_inf && (a_zero || b_zero);
            uint32_t ovr    = c_nan || c_inf || c_zero;

            /* bf16_mul.sv:182-184  zero/inf/nan ALL emit man=0, exp=0 --
             * a zero lane is given BIASED exponent 0, not a -inf sentinel. */
            l_sign[i] = sign;
            l_exp[i]  = ovr ? 0 : exp_raw;
            l_man[i]  = ovr ? 0u : man16;
            l_inf[i]  = c_inf;
            l_nan[i]  = c_nan;
        }

        /* =================================================================
         * bwms_exp_compare.sv:32-59  unconditional signed max over ALL 16
         * lanes.  Zero/inf/nan lanes participate with their exp of 0.
         * ================================================================= */
        int64_t max_exp = l_exp[0];
        for (i = 1; i < 16; i++)
            if (l_exp[i] > max_exp) max_exp = l_exp[i];

        /* bwms_top.sv:91-100  flag aggregation */
        uint32_t any_nan = 0, any_pinf = 0, any_ninf = 0;
        for (i = 0; i < 16; i++) {
            if (l_nan[i]) any_nan = 1;
            if (l_inf[i] && !l_sign[i]) any_pinf = 1;
            if (l_inf[i] &&  l_sign[i]) any_ninf = 1;
        }
        uint32_t b_is_nan   = any_nan || (any_pinf && any_ninf);
        uint32_t b_is_inf   = (any_pinf || any_ninf) && !b_is_nan;
        uint32_t b_inf_sign = b_is_nan ? 0u : (any_pinf ? 0u : 1u);

        /* =================================================================
         * bwms_shifter_cell.sv:25-41  align + sign-embed, then
         * mau_tree.sv / mau_pair_add.sv:23  EXACT signed integer tree sum.
         *
         * The tree is exact (|S| <= 16*(2^24-1) = 2^28-16 fits 29b signed),
         * so the pairing order is irrelevant and a flat accumulation in
         * int64_t is bit-identical to the 4-stage tree.
         * ================================================================= */
        int64_t S = 0;
        for (i = 0; i < 16; i++) {
            uint32_t mag24 = (l_man[i] & 0xFFFFu) << 8;      /* line 25: LSB pad */
            int64_t  sh    = max_exp - l_exp[i];             /* line 28          */
            /* line 29 saturate has PRIORITY over the barrel shift (line 36);
             * line 33 is a bare logical right shift of an UNSIGNED vector --
             * pure truncation toward zero, no guard/round/sticky. */
            uint32_t mag   = (sh >= 24) ? 0u : (mag24 >> (uint32_t)(sh & 31));
            /* lines 40-41: zero-extend to 25b THEN negate, so the truncation
             * is on the magnitude (toward zero), not an arithmetic shift. */
            S += l_sign[i] ? -(int64_t)mag : (int64_t)mag;
        }

        /* =================================================================
         * mau_tree.sv:200-214 + mau_lzd28.sv  abs/LZD/shift_amt
         * ================================================================= */
        uint32_t ma_sum   = (uint32_t)(S & 0x1FFFFFFF);            /* 29b 2's-c */
        uint32_t new_sign = (ma_sum >> 28) & 1u;                   /* line 200  */
        uint32_t new_abs  = new_sign ? ((~ma_sum + 1u) & 0x1FFFFFFFu)
                                     : ma_sum;                     /* line 171  */
        uint32_t mag28    = new_abs & 0x0FFFFFFFu;                 /* line 202  */
        uint32_t all_zero = (mag28 == 0u);
        uint32_t K = 0u;
        if (!all_zero) { uint32_t t = mag28; while (t >>= 1) K++; }
        int64_t shift_amt = all_zero ? 0 : ((int64_t)K - 23);      /* lines 213-214 */

        /* =================================================================
         * acc_top.sv:170-183  ingest the beat triple as a (sign,exp,man24)
         * operand.  ROUNDING SITE: the >> drops the low (K-23) magnitude bits
         * with NO guard/round/sticky when K > 23.
         * ================================================================= */
        int64_t  padded52  = (int64_t)new_abs << 23;               /* line 173 */
        uint32_t right_sh  = (uint32_t)(23 + shift_amt) & 0x3Fu;   /* lines 174-175 */
        uint32_t new_man   = (uint32_t)((padded52 >> right_sh) & 0xFFFFFFu);
        int64_t  new_exp_w = max_exp + shift_amt + 1;              /* lines 179-181 */
        int64_t  new_exp   = (int64_t)(((uint32_t)new_exp_w & 0x3FFu) ^ 0x200u)
                             - 0x200;                              /* line 182 [9:0] */
        uint32_t new_is_zero = (new_abs == 0u);                    /* line 183 */

        /* ---- acc_top.sv:197-208  flag resolution ---- */
        uint32_t prev_sign_eff = acc_is_inf ? acc_inf_sign : acc_sign;
        uint32_t new_sign_eff  = b_is_inf   ? b_inf_sign   : new_sign;
        uint32_t flag_nan   = acc_is_nan || b_is_nan ||
                              (acc_is_inf && b_is_inf &&
                               (prev_sign_eff != new_sign_eff));
        uint32_t flag_inf   = (acc_is_inf || b_is_inf) && !flag_nan;
        uint32_t flag_inf_s = flag_nan ? 0u
                            : (acc_is_inf ? prev_sign_eff : new_sign_eff);

        /* ---- acc_top.sv:211-222  zero detect + subnormal-aware prev exp ---- */
        uint32_t prev_is_zero    = (acc_man == 0u) && !acc_is_inf && !acc_is_nan;
        uint32_t new_is_zero_eff = new_is_zero && !b_is_inf && !b_is_nan;
        uint32_t prev_is_subn    = (acc_exp == 0) && (acc_man != 0u)
                                   && !acc_is_inf && !acc_is_nan;
        int64_t  exp_eff         = prev_is_subn ? 1 : acc_exp;

        /* ---- acc_top.sv:227-239  exp diff / bigger-smaller ---- */
        int64_t  exp_diff    = exp_eff - new_exp;
        uint32_t prev_bigger = (exp_diff >= 0);
        int64_t  abs_diff    = prev_bigger ? exp_diff : -exp_diff;
        uint32_t bigger24    = prev_bigger ? acc_man  : new_man;
        uint32_t smaller24   = prev_bigger ? new_man  : acc_man;
        uint32_t bigger_sn   = prev_bigger ? acc_sign : new_sign;
        uint32_t smaller_sn  = prev_bigger ? new_sign : acc_sign;
        int64_t  bigger_exp  = prev_bigger ? exp_eff  : new_exp;

        /* ---- acc_top.sv:242-253  align smaller with GRS + exact sticky ---- */
        uint32_t align_ovf = (abs_diff > 27);
        uint32_t align_sh  = align_ovf ? 27u : (uint32_t)(abs_diff & 0x3F);
        int64_t  ext51     = (int64_t)smaller24 << 27;
        int64_t  sh51      = ext51 >> align_sh;
        uint32_t ali27     = (uint32_t)((sh51 >> 24) & 0x7FFFFFF);
        uint32_t stk_late  = ((sh51 & 0xFFFFFF) != 0);
        uint32_t stk_early = align_ovf ? (smaller24 != 0u) : 0u;
        /* NOTE: sticky is OR'ed into the operand LSB BEFORE the add (line 250),
         * not consumed at the round stage.  This is NOT an IEEE fp32 add. */
        ali27 |= (stk_late || stk_early);
        uint32_t big27 = (bigger24 << 3) & 0x7FFFFFFu;             /* line 253 */

        /* ---- acc_top.sv:256-260  signed add in 29b ---- */
        int64_t sum = (bigger_sn  ? -(int64_t)big27 : (int64_t)big27)
                    + (smaller_sn ? -(int64_t)ali27 : (int64_t)ali27);

        /* ---- acc_top.sv:334-337  abs, then [27:0] slice ---- */
        uint32_t sum_sign = (sum < 0);
        int64_t  sabs     = sum_sign ? -sum : sum;
        uint32_t abs28    = (uint32_t)(sabs & 0x0FFFFFFF);
        uint32_t sum_is_zero = (abs28 == 0u);

        /* ---- acc_top.sv:341-373  inline LZD28 ---- */
        uint32_t lz = 0u;
        if (!sum_is_zero) { uint32_t t = abs28; while (t >>= 1) lz++; }

        /* ---- acc_top.sv:377-387  normalize (exact; bit0 folded to sticky) -- */
        int64_t  ext55  = (int64_t)abs28 << 27;
        uint32_t norm27 = (uint32_t)((ext55 >> (lz + 1u)) & 0x7FFFFFF);
        if (lz == 27u) norm27 |= (abs28 & 1u);
        int64_t  rexp_w = bigger_exp + (int64_t)lz - 26;
        int64_t  res_exp_pre = (int64_t)(((uint32_t)rexp_w & 0x3FFu) ^ 0x200u)
                               - 0x200;                            /* line 387 */

        /* ---- acc_top.sv:389-401  RNE round ---- */
        uint32_t rg   = (norm27 >> 2) & 1u;
        uint32_t rr   = (norm27 >> 1) & 1u;
        uint32_t rs   =  norm27       & 1u;
        uint32_t rlsb = (norm27 >> 3) & 1u;
        uint32_t rup  = rg & (rr | rs | rlsb);
        uint32_t r25  = ((norm27 >> 3) & 0xFFFFFFu) + rup;
        uint32_t rcy  = (r25 >> 24) & 1u;
        uint32_t round_man = rcy ? ((r25 >> 1) & 0xFFFFFFu) : (r25 & 0xFFFFFFu);
        int64_t  rexp2     = rcy ? (res_exp_pre + 1) : res_exp_pre;
        int64_t  round_exp = (int64_t)(((uint32_t)rexp2 & 0x3FFu) ^ 0x200u) - 0x200;

        /* ---- acc_top.sv:408-435  zero bypasses (skip the rounder) ---- */
        uint32_t num_sign, num_man, num_is_zero_pre;
        int64_t  num_exp;
        if (prev_is_zero && new_is_zero_eff) {
            num_sign = 0u; num_exp = 0; num_man = 0u; num_is_zero_pre = 1u;
        } else if (prev_is_zero) {
            num_sign = new_sign; num_exp = new_exp; num_man = new_man;
            num_is_zero_pre = 0u;
        } else if (new_is_zero_eff) {
            /* RAW acc_exp/acc_man here, NOT exp_eff (acc_top.sv:317-318) */
            num_sign = acc_sign; num_exp = acc_exp; num_man = acc_man;
            num_is_zero_pre = 0u;
        } else if (sum_is_zero) {
            num_sign = 0u; num_exp = 0; num_man = 0u; num_is_zero_pre = 1u;
        } else {
            num_sign = sum_sign; num_exp = round_exp; num_man = round_man;
            num_is_zero_pre = 0u;
        }

        /* ---- acc_top.sv:462-531  exp saturation, then NaN/Inf override ---- */
        uint32_t nx_sign, nx_man, nx_is_inf, nx_is_nan, nx_inf_sign;
        int64_t  nx_exp;
        if (flag_nan) {                                            /* line 509 */
            nx_sign = 0u; nx_exp = 0; nx_man = 0u;
            nx_is_inf = 0u; nx_is_nan = 1u; nx_inf_sign = 0u;
        } else if (flag_inf) {                                     /* line 516 */
            nx_sign = flag_inf_s; nx_exp = 255; nx_man = 0u;
            nx_is_inf = 1u; nx_is_nan = 0u; nx_inf_sign = flag_inf_s;
        } else if (num_is_zero_pre) {                              /* line 463 */
            nx_sign = 0u; nx_exp = 0; nx_man = 0u;
            nx_is_inf = 0u; nx_is_nan = 0u; nx_inf_sign = 0u;
        } else if (num_exp > 254) {                                /* line 469 */
            nx_sign = num_sign; nx_exp = 255; nx_man = 0u;
            nx_is_inf = 1u; nx_is_nan = 0u; nx_inf_sign = num_sign;
        } else if (num_exp >= 1) {                                 /* line 475 */
            nx_sign = num_sign; nx_exp = num_exp; nx_man = num_man;
            nx_is_inf = 0u; nx_is_nan = 0u; nx_inf_sign = 0u;
        } else if (num_exp >= -22) {                               /* line 482 */
            /* acc_top.sv:445-447  PLAIN TRUNCATING shift, not RNE */
            uint32_t ssh = (uint32_t)(1 - num_exp) & 0x1Fu;
            nx_sign = num_sign; nx_exp = 0; nx_man = num_man >> ssh;
            nx_is_inf = 0u; nx_is_nan = 0u; nx_inf_sign = 0u;
        } else {                                                   /* line 491 */
            nx_sign = 0u; nx_exp = 0; nx_man = 0u;
            nx_is_inf = 0u; nx_is_nan = 0u; nx_inf_sign = 0u;
        }

        acc_sign     = nx_sign;
        acc_exp      = nx_exp;
        acc_man      = nx_man & 0xFFFFFFu;
        acc_is_inf   = nx_is_inf;
        acc_is_nan   = nx_is_nan;
        acc_inf_sign = nx_inf_sign;
    }

    /* =====================================================================
     * bank_controller_top.v:789-804  acc latch -> BF16, RNE.
     * (bf16_round.sv is NOT on this path -- it is an EWMUL-only tap.)
     * ===================================================================== */
    {
        uint32_t m_sign   = acc_is_inf ? acc_inf_sign : acc_sign;  /* line 135 */
        uint32_t m_iszero = (acc_man == 0u) && !acc_is_inf && !acc_is_nan;

        uint32_t bg     = (acc_man >> 15) & 1u;
        uint32_t blsb   = (acc_man >> 16) & 1u;
        uint32_t bstk   = ((acc_man & 0x7FFFu) != 0u);
        uint32_t brup   = bg & (blsb | bstk);
        uint32_t bsum   = ((acc_man >> 16) & 0xFFu) + brup;        /* 9b */
        uint32_t bcarry = ((acc_man >> 23) & 1u) ? ((bsum >> 8) & 1u)
                                                 : ((bsum >> 7) & 1u);
        uint32_t bfrac  = bcarry ? 0u : (bsum & 0x7Fu);
        int64_t  bexp_w = acc_exp + (int64_t)(bcarry ? 1u : 0u);
        int64_t  bexp   = (int64_t)(((uint32_t)bexp_w & 0x3FFu) ^ 0x200u) - 0x200;

        if (acc_is_nan)   return (uint16_t)0x7FC0u;                       /* qNaN */
        if (acc_is_inf)   return (uint16_t)((m_sign << 15) | (0xFFu << 7));
        if (m_iszero)     return (uint16_t)(m_sign << 15);
        if (bexp > 254)   return (uint16_t)((m_sign << 15) | (0xFFu << 7));
        return (uint16_t)((m_sign << 15)
                          | (((uint32_t)bexp & 0xFFu) << 7)
                          | bfrac);
    }
}