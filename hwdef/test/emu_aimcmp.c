// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// emu_aimcmp — runs a fixed set of PIM workloads on the board, reads the cycle
// counter (RUN_CYC) and the violation statistics of every run, and writes each
// workload as an AiM simulator trace so the two cycle counts can be compared.
//
//     ./emu_aimcmp --out DIR                   every workload, 5 runs each
//     ./emu_aimcmp --out DIR --only gemv_g8    one workload
//     ./emu_aimcmp --list                      the workloads
//
// The timing registers are read and recorded as they are; emu_timing sets them.
//
// DIR receives
//     timing.txt     the ten timing registers, name=value, one per line
//     board.csv      per workload: ISR count, RUN_CYC min/max, violations
//     <name>.trace   the workload as an AiM trace (ramulator2 -t)
// scripts/aim_compare.py turns timing.txt into the simulator's timing, runs the
// simulator on the traces and prints the comparison.
//
// Every workload runs on all channels of this build (one 1-hot RD_MAC per channel,
// on the board and in the trace) with all-bank MACs that take their vector from the
// global buffer.  Before each run a
// two-ISR program (COPY, EOS) leaves the controller in BANK mode with every row
// closed, which is the state the simulator starts in.  The weight rows and the GPR
// hold whatever is already there; only the timing is measured.
//
// Exit: 0 every run finished without a violation, 1 a run failed or violated, 2 usage
//////////////////////////////////////////////////////////////////////////////////
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "emu_regs.h"
#include "pim_platform.h"

#define BDF_DEFAULT  "0000:01:00.0"
#define H2C_DEFAULT  "/dev/qdma01000-MM-0"
#define MAX_ISR      1024u
#define TIMEOUT_US   2000000ull

// Where the workloads point.  Any rows and words inside the device will do.
#define ROW0         100u      // first weight row
#define PREP_ROW     90u       // the row the preparation COPY reads
#define VEC_WORD     0u        // first GPR word of the vectors
#define DST_WORD     2000u     // first GPR word of the results

static volatile uint8_t *g_bar;
static int      g_h2c = -1;
static uint32_t g_all_ch;      // CH_MASK naming every channel of this build

static uint64_t now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

static uint32_t cfr_rd(uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off);
    __sync_synchronize();
    return v;
}

static void cfr_wr(uint32_t off, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(g_bar + EMU_OFF_CFR + off) = v;
    __sync_synchronize();
}

// Writes a program into IMEM over the H2C queue.
static int imem_write(const struct emu_isr *p, unsigned n)
{
    size_t len = (size_t)n * EMU_WORD_BYTES;
    const char *bad = pim_dma_check(EMU_AXI_IMEM, len);

    if (bad) { fprintf(stderr, "IMEM write refused: %s\n", bad); return -1; }
    for (size_t done = 0; done < len; ) {
        ssize_t w = pwrite(g_h2c, (const uint8_t *)p + done, len - done,
                           (off_t)(EMU_AXI_IMEM + done));
        if (w <= 0) {
            fprintf(stderr, "IMEM pwrite: %s\n", w < 0 ? strerror(errno) : "no progress");
            return -1;
        }
        done += (size_t)w;
    }
    return 0;
}

// Waits until the dispatcher has finished the previous program, or has never run
// one.  IMEM and the doorbell are touched only after this.
static bool wait_idle(void)
{
    uint64_t t0 = now_us();

    for (;;) {
        uint32_t st = cfr_rd(CFR_STATUS);
        if (st == 0xFFFFFFFFu) {
            fprintf(stderr, "the CFR reads all-ones: the device is not answering\n");
            return false;
        }
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return true;
        if (now_us() - t0 > TIMEOUT_US) {
            fprintf(stderr, "the dispatcher is still busy (STATUS=0x%08x)\n", st);
            return false;
        }
    }
}

// Runs one program and returns its RUN_CYC.  The doorbell clears RUN_CYC and it
// stops counting when STATUS[31] rises, so it is taken once done is up and two
// reads in a row agree.
static int run_prog(const struct emu_isr *p, unsigned n, uint64_t *cyc)
{
    uint64_t t0;

    if (!wait_idle() || imem_write(p, n) < 0 || !wait_idle()) return -1;
    cfr_wr(CFR_PROG_LEN, n);
    if (cfr_rd(CFR_PROG_LEN) != n) {
        fprintf(stderr, "PROG_LEN read back %u, not %u\n", cfr_rd(CFR_PROG_LEN), n);
        return -1;
    }
    cfr_wr(CFR_CTRL, CFR_CTRL_DOORBELL);

    for (t0 = now_us(); now_us() - t0 < TIMEOUT_US; ) {
        uint32_t lo, hi;
        uint64_t a, b;

        if (!(cfr_rd(CFR_STATUS) & CFR_STATUS_DONE)) continue;
        lo = cfr_rd(CFR_RUN_CYC_LO); hi = cfr_rd(CFR_RUN_CYC_HI); a = CFR_RUN_CYC(lo, hi);
        lo = cfr_rd(CFR_RUN_CYC_LO); hi = cfr_rd(CFR_RUN_CYC_HI); b = CFR_RUN_CYC(lo, hi);
        if (a == b) { *cyc = a; return 0; }
    }
    fprintf(stderr, "the program did not finish in %llu ms (STATUS=0x%08x)\n",
            TIMEOUT_US / 1000ull, cfr_rd(CFR_STATUS));
    return -1;
}

// ---- violations -----------------------------------------------------------------
struct viol { uint32_t act, pre, worst_act, worst_pre; };

// Sums the counts and takes the worst overrun over every bank of every channel.
static void viol_read(struct viol *v)
{
    memset(v, 0, sizeof *v);
    for (unsigned c = 0; c < PIM_NCH; c++)
        for (unsigned bk = 0; bk < EMU_NBANKS; bk++) {
            uint32_t cnt = cfr_rd(VIOL_BANK_BASE(c, bk) + VIOL_CNT);
            uint32_t mx  = cfr_rd(VIOL_BANK_BASE(c, bk) + VIOL_MAX);

            v->act += VIOL_CNT_ACT(cnt);
            v->pre += VIOL_CNT_PRE(cnt);
            if (VIOL_MAX_ACT(mx) > v->worst_act) v->worst_act = VIOL_MAX_ACT(mx);
            if (VIOL_MAX_PRE(mx) > v->worst_pre) v->worst_pre = VIOL_MAX_PRE(mx);
        }
}

// ---- building a workload ----------------------------------------------------------
// A workload under construction: the ISRs for the board, and the same program as
// AiM trace lines written to `tf`.
struct build {
    struct emu_isr isr[MAX_ISR];
    unsigned       n;
    FILE          *tf;
    const char    *err;
};

static void push(struct build *b, const struct emu_isr_spec *s)
{
    const char *e;

    if (b->err) return;
    if (b->n >= MAX_ISR) { b->err = "more ISRs than MAX_ISR"; return; }
    if ((e = emu_isr_build(&b->isr[b->n], s))) { b->err = e; return; }
    b->n++;
}

// WRVEC: L beats from GPR word `word` into the global buffer of each channel in
// `mask`.  AiM: WR_GB opsize GPR channel_mask.
static void b_wrvec(struct build *b, unsigned L, uint32_t word, uint32_t mask)
{
    struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_WRVEC, mask);

    s.opsize = L;
    s.row    = word;
    push(b, &s);
    fprintf(b->tf, "AiM WR_GB %u %u %u\n", L, word, mask);
}

// MAC on all 16 banks of each channel in `mask`: L beats of `row` from beat `col`,
// against the global buffer.  AiM: MAC_ABK opsize channel_mask row.  MAC_ABK starts
// at column 0; the column does not enter the timing.
static void b_mac(struct build *b, unsigned L, uint32_t row, uint32_t col, uint32_t mask)
{
    struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_MAC, mask);

    s.opsize     = L;
    s.row        = row;
    s.col        = col;
    s.pu_mask    = 0xFFFFu;
    s.gb_mc_mask = 0xFFFFu;
    push(b, &s);
    fprintf(b->tf, "AiM MAC_ABK %u %u %u\n", L, mask, row);
}

// RD_MAC, one per channel in `mask`, into GPR word `word` + channel.  Each RD_MAC
// names one landing word, so each reads one channel (CH_MASK 1-hot), and the AiM
// trace carries the same 1-hot RD_MACs in the same order.
static void b_rdmac(struct build *b, uint32_t word, uint32_t mask)
{
    for (unsigned c = 0; c < PIM_NCH; c++) {
        struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_RD_MAC, 1u << c);

        if (!(mask & (1u << c))) continue;
        s.opsize = 0;
        s.row    = word + c;
        push(b, &s);
        fprintf(b->tf, "AiM RD_MAC %u %u\n", word + c, 1u << c);
    }
}

// EOS on each channel in `mask`.  AiM: EOC.
static void b_eos(struct build *b, uint32_t mask)
{
    struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_EOS, mask);

    push(b, &s);
    fprintf(b->tf, "AiM EOC\n");
}

// ---- the workloads --------------------------------------------------------------
static void w_gemv_k1024(struct build *b)
{
    b_wrvec(b, 64, VEC_WORD, g_all_ch);
    b_mac(b, 64, ROW0, 0, g_all_ch);
    b_rdmac(b, DST_WORD, g_all_ch);
    b_eos(b, g_all_ch);
}

static void w_groups(struct build *b, unsigned groups)
{
    b_wrvec(b, 64, VEC_WORD, g_all_ch);
    for (unsigned g = 0; g < groups; g++) {
        b_mac(b, 64, ROW0 + g, 0, g_all_ch);
        b_rdmac(b, DST_WORD + g * PIM_NCH, g_all_ch);
    }
    b_eos(b, g_all_ch);
}
static void w_gemv_g8(struct build *b)  { w_groups(b, 8); }
static void w_gemv_g64(struct build *b) { w_groups(b, 64); }

static void w_gemv_k4096(struct build *b)
{
    for (unsigned c = 0; c < 4; c++) {
        b_wrvec(b, 64, VEC_WORD + c * 64, g_all_ch);
        b_mac(b, 64, ROW0 + c, 0, g_all_ch);
    }
    b_rdmac(b, DST_WORD, g_all_ch);
    b_eos(b, g_all_ch);
}

static void w_qk_h8(struct build *b)
{
    for (unsigned h = 0; h < 8; h++) {
        b_wrvec(b, 4, VEC_WORD + h * 4, g_all_ch);
        b_mac(b, 4, ROW0, h * 4, g_all_ch);
        b_rdmac(b, DST_WORD + h * PIM_NCH, g_all_ch);
    }
    b_eos(b, g_all_ch);
}

static const struct workload {
    const char *name, *what;
    void (*fill)(struct build *);
} WL[] = {
    { "gemv_k1024", "WRVEC(64), MAC(64), RD_MAC per channel: one output group, K 1024",
      w_gemv_k1024 },
    { "gemv_g8",    "WRVEC(64) once, then 8 x { MAC(64) on a new row, RD_MAC per channel }",
      w_gemv_g8 },
    { "gemv_g64",   "as gemv_g8 with 64 output groups", w_gemv_g64 },
    { "gemv_k4096", "4 x { WRVEC(64), MAC(64) on a new row }, then RD_MAC per channel: K 4096",
      w_gemv_k4096 },
    { "qk_h8",      "8 heads of D 64 on one row: { WRVEC(4), MAC(4) at COL 4h, RD_MAC per "
                    "channel }", w_qk_h8 },
};
#define NWL (sizeof WL / sizeof WL[0])

// ---- a program given on the command line ------------------------------------------
// --prog takes items separated by ';':
//     wrvec L [word]     mac L row [col]     rdmac [word]     eos
// Each item may end in @MASK to run on those channels only (default: all of them),
// e.g. "rdmac @1" is the RD_MAC of channel 0 alone.  The items are built with the
// same functions as the workloads above.
static const char *g_spec;

static void w_spec(struct build *b)
{
    char buf[4096], *save = NULL;

    if (strlen(g_spec) >= sizeof buf) { b->err = "--prog is too long"; return; }
    strcpy(buf, g_spec);
    for (char *it = strtok_r(buf, ";", &save); it && !b->err; it = strtok_r(NULL, ";", &save)) {
        char *tok[6], *s2 = NULL;
        unsigned nt = 0, v[4] = { 0, 0, 0, 0 };
        uint32_t mask = g_all_ch;

        for (char *t = strtok_r(it, " \t", &s2); t && nt < 6; t = strtok_r(NULL, " \t", &s2))
            tok[nt++] = t;
        if (!nt) continue;                             // an empty item
        if (tok[nt - 1][0] == '@') {
            mask = (uint32_t)strtoul(tok[nt - 1] + 1, NULL, 0);
            nt--;
            if (!mask || (mask & ~g_all_ch)) { b->err = "--prog: @MASK names no channel of this build"; return; }
        }
        for (unsigned i = 1; i < nt && i <= 4; i++) v[i - 1] = (unsigned)strtoul(tok[i], NULL, 0);

        if      (!strcmp(tok[0], "wrvec") && nt >= 2) b_wrvec(b, v[0], nt >= 3 ? v[1] : VEC_WORD, mask);
        else if (!strcmp(tok[0], "mac") && nt >= 3)   b_mac(b, v[0], v[1], nt >= 4 ? v[2] : 0, mask);
        else if (!strcmp(tok[0], "rdmac"))            b_rdmac(b, nt >= 2 ? v[0] : DST_WORD, mask);
        else if (!strcmp(tok[0], "eos"))              b_eos(b, mask);
        else b->err = "--prog items are 'wrvec L [word]', 'mac L row [col]', "
                      "'rdmac [word]' and 'eos', each optionally ending in @MASK";
    }
}

// COPY one beat of PREP_ROW into the GB, then EOS: BANK mode, every row closed.
static int build_prep(struct emu_isr prep[2])
{
    struct emu_isr_spec s = emu_isr_default_ch(ISR_OP_COPY, g_all_ch);
    const char *e;

    s.opsize   = 1;
    s.row      = PREP_ROW;
    s.col      = 0;
    s.route[0] = GB_PORT;
    if ((e = emu_isr_build(&prep[0], &s))) { fprintf(stderr, "prep COPY: %s\n", e); return -1; }
    s = emu_isr_default_ch(ISR_OP_EOS, g_all_ch);
    if ((e = emu_isr_build(&prep[1], &s))) { fprintf(stderr, "prep EOS: %s\n", e); return -1; }
    return 0;
}

// ---- timing registers -------------------------------------------------------------
static const struct { const char *name; uint32_t off; } TREG[] = {
    { "faw", CFR_T_FAW }, { "rrd", CFR_T_RRD }, { "rcd", CFR_T_RCD },
    { "ccd", CFR_T_CCD }, { "rtp", CFR_T_RTP }, { "rp",  CFR_T_RP  },
    { "wr",  CFR_T_WR  }, { "ras", CFR_T_RAS }, { "mod", CFR_T_MOD },
    { "rpab", CFR_T_RP_AB }, { "gb", CFR_T_GB },
};
#define NTREG (sizeof TREG / sizeof TREG[0])

static FILE *out_open(const char *dir, const char *name)
{
    char path[512];
    FILE *f;

    snprintf(path, sizeof path, "%s/%s", dir, name);
    if (!(f = fopen(path, "w"))) fprintf(stderr, "%s: %s\n", path, strerror(errno));
    return f;
}

static void usage(const char *p)
{
    printf("Usage: %s [--out DIR] [--reps N] [--only NAME] [--list] [--bdf B] [--h2c PATH]\n"
           "       %s --prog SPEC [--name NAME] [--out DIR] [--reps N]\n"
           "\n"
           "Runs PIM workloads on the board and reports RUN_CYC and violations per\n"
           "workload.  With --out, writes timing.txt, board.csv and one AiM trace per\n"
           "workload into DIR.  The timing registers are used as emu_timing left them.\n"
           "\n"
           "  --out DIR    where timing.txt, board.csv and <workload>.trace go\n"
           "  --reps N     runs per workload (default 5)\n"
           "  --only NAME  one workload\n"
           "  --list       print the workloads and exit\n"
           "  --prog SPEC  run this program instead, e.g. \"wrvec 64; mac 64 100; rdmac; eos\".\n"
           "               Items: wrvec L [word], mac L row [col], rdmac [word], eos,\n"
           "               each optionally ending in @MASK (channels; default all).\n"
           "               Its line is appended to DIR/board.csv.\n"
           "  --name NAME  the --prog program's name in board.csv and NAME.trace (default prog)\n"
           "\n"
           "Exit: 0 every run finished without a violation, 1 a run failed or\n"
           "violated, 2 usage\n", p, p);
}

int main(int argc, char **argv)
{
    const char *bdf = BDF_DEFAULT, *h2c = H2C_DEFAULT, *dir = NULL, *only = NULL;
    const char *prog = NULL, *pname = "prog";
    struct workload one = { NULL, "the --prog program", w_spec };
    const struct workload *list = WL;
    unsigned nlist = NWL;
    unsigned reps = 5;
    struct emu_isr prep[2];
    FILE *csv = NULL;
    int rc = 0;

    static struct option lo[] = {
        {"out",1,0,'o'}, {"reps",1,0,'r'}, {"only",1,0,'1'}, {"list",0,0,'l'},
        {"prog",1,0,'p'}, {"name",1,0,'n'},
        {"bdf",1,0,'b'}, {"h2c",1,0,'H'}, {"help",0,0,'h'}, {0,0,0,0}
    };
    for (int o; (o = getopt_long(argc, argv, "h", lo, NULL)) != -1; ) {
        switch (o) {
        case 'o': dir  = optarg; break;
        case 'r': reps = (unsigned)strtoul(optarg, NULL, 0); break;
        case '1': only = optarg; break;
        case 'p': prog = optarg; break;
        case 'n': pname = optarg; break;
        case 'b': bdf  = optarg; break;
        case 'H': h2c  = optarg; break;
        case 'l':
            for (unsigned i = 0; i < NWL; i++) printf("  %-11s %s\n", WL[i].name, WL[i].what);
            return 0;
        case 'h': usage(argv[0]); return 0;
        default:  usage(argv[0]); return 2;
        }
    }
    if (!reps || (prog && only)) { usage(argv[0]); return 2; }
    if (prog) { g_spec = prog; one.name = pname; list = &one; nlist = 1; }
    if (only) {
        bool found = false;
        for (unsigned i = 0; i < NWL; i++) found |= !strcmp(only, WL[i].name);
        if (!found) { fprintf(stderr, "no workload named '%s' (see --list)\n", only); return 2; }
    }

    pim_platform_banner();
    { const char *bad = pim_platform_check(); if (bad) { fprintf(stderr, "%s\n", bad); return 2; } }
    g_all_ch = (1u << PIM_NCH) - 1u;

    {
        char path[256];
        int fd;
        void *m;

        snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource2", bdf);
        if ((fd = open(path, O_RDWR | O_SYNC)) < 0) {
            fprintf(stderr, "open(%s): %s\n", path, strerror(errno)); return 1;
        }
        m = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { fprintf(stderr, "mmap BAR2: %s\n", strerror(errno)); return 1; }
        g_bar = m;
    }
    if ((g_h2c = open(h2c, O_WRONLY)) < 0) {
        fprintf(stderr, "open(%s): %s\n", h2c, strerror(errno)); return 1;
    }
    if (build_prep(prep) < 0) return 1;

    if (dir && mkdir(dir, 0775) < 0 && errno != EEXIST) {
        fprintf(stderr, "mkdir %s: %s\n", dir, strerror(errno)); return 1;
    }

    // ---- the timing this run uses ----
    {
        FILE *tf = dir ? out_open(dir, "timing.txt") : NULL;

        if (dir && !tf) return 1;
        printf("timing  :");
        for (unsigned i = 0; i < NTREG; i++) {
            uint32_t v = cfr_rd(TREG[i].off) & CFR_TIMING_MAX;
            printf(" %s=%u", TREG[i].name, v);
            if (tf) fprintf(tf, "%s=%u\n", TREG[i].name, v);
        }
        printf("\nchannels: %u (CH_MASK 0x%x), %u run(s) per workload\n\n",
               PIM_NCH, g_all_ch, reps);
        if (tf) fclose(tf);
    }

    // A --prog run adds its line to board.csv; a workload run starts it afresh.
    if (dir) {
        char path[512];
        bool fresh;

        snprintf(path, sizeof path, "%s/board.csv", dir);
        fresh = !prog || access(path, F_OK) != 0;
        if (!(csv = fopen(path, prog ? "a" : "w"))) {
            fprintf(stderr, "%s: %s\n", path, strerror(errno)); return 1;
        }
        if (fresh) fprintf(csv, "workload,isrs,run_cyc_min,run_cyc_max,reps,"
                                "act_fill,pre_drain,worst_act,worst_pre\n");
    }

    printf("  %-11s %5s %12s %12s   %9s %9s\n",
           "workload", "ISRs", "RUN_CYC min", "max", "ACT_FILL", "PRE_DRAIN");

    for (unsigned w = 0; w < nlist; w++) {
        static struct build b;
        char   *text = NULL;
        size_t  tlen = 0;
        uint64_t cmin = UINT64_MAX, cmax = 0;
        struct viol tot = { 0, 0, 0, 0 };
        bool failed = false;

        if (only && strcmp(only, list[w].name)) continue;

        memset(&b, 0, sizeof b);
        if (!(b.tf = open_memstream(&text, &tlen))) { perror("open_memstream"); return 1; }
        fprintf(b.tf, "W CFR 0 1\n");          // AiM: MACs take their vector from the GB
        list[w].fill(&b);
        fclose(b.tf);
        if (b.err) {
            fprintf(stderr, "%s: the encoder refused an ISR: %s\n", list[w].name, b.err);
            free(text); return 1;
        }
        if (dir) {
            char name[64];
            FILE *f;

            snprintf(name, sizeof name, "%s.trace", list[w].name);
            if (!(f = out_open(dir, name))) { free(text); return 1; }
            fwrite(text, 1, tlen, f);
            fclose(f);
        }
        free(text);

        for (unsigned r = 0; r < reps && !failed; r++) {
            uint64_t c, prep_cyc;
            struct viol v;

            if (run_prog(prep, 2, &prep_cyc) < 0) { failed = true; break; }
            cfr_wr(VIOL_CTRL, 1u);
            if (run_prog(b.isr, b.n, &c) < 0) { failed = true; break; }
            viol_read(&v);

            if (c < cmin) cmin = c;
            if (c > cmax) cmax = c;
            tot.act += v.act;
            tot.pre += v.pre;
            if (v.worst_act > tot.worst_act) tot.worst_act = v.worst_act;
            if (v.worst_pre > tot.worst_pre) tot.worst_pre = v.worst_pre;
        }
        if (failed) {
            printf("  %-11s %5u   FAILED\n", list[w].name, b.n);
            rc = 1;
            continue;
        }
        printf("  %-11s %5u %12" PRIu64 " %12" PRIu64 "   %9u %9u\n",
               list[w].name, b.n, cmin, cmax, tot.act, tot.pre);
        if (tot.act || tot.pre) rc = 1;
        if (csv) fprintf(csv, "%s,%u,%" PRIu64 ",%" PRIu64 ",%u,%u,%u,%u,%u\n",
                         list[w].name, b.n, cmin, cmax, reps,
                         tot.act, tot.pre, tot.worst_act, tot.worst_pre);
    }
    if (csv) fclose(csv);

    if (rc) printf("\nA run failed or raised a violation; its RUN_CYC is not a timing "
                   "measurement.\n");
    return rc;
}
