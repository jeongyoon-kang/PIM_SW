// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// pim_dev.c — the transport.  The only file that opens fds, mmaps, or moves bytes.
//
// TRANSPORT POLICY, inherited from emu_regs.h and not negotiable here:
//   AXI-Lite (CFR, violation CSR)          MMIO only
//   256 b    (IMEM, GPR, HBM)              QDMA only, 32 B aligned
// A 32 B MMIO store is split into <=16 B TLPs and the 256 b slaves have no WSTRB,
// so the second fragment expands to a full word write and wins.  Measured
// 2026-07-31; it is a property of this host, not a theory.
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include "pim_internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define BDF_DEFAULT  "0000:01:00.0"
#define H2C_DEFAULT  "/dev/qdma01000-MM-0"
#define C2H_DEFAULT  "/dev/qdma01000-MM-1"

uint64_t pim_now_us_impl(void) { return pim_now_us(); }

uint64_t pim_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)ts.tv_nsec / 1000ull;
}

const char *pim_err(pim_dev *d, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(d->err, PIM_ERRLEN, fmt, ap);
    va_end(ap);
    return d->err;
}

size_t pim_weight_sizeof(void) { return sizeof(pim_weight); }
size_t pim_config_sizeof(void) { return sizeof(pim_config); }
size_t pim_stats_sizeof (void) { return sizeof(pim_stats);  }
size_t pim_kv_sizeof    (void) { return sizeof(pim_kv);     }

const char *pim_last_error(const pim_dev *d) { return d->err[0] ? d->err : NULL; }
void pim_get_stats(const pim_dev *d, pim_stats *out) { *out = d->st; }
void pim_reset_stats(pim_dev *d) { memset(&d->st, 0, sizeof d->st); }

// ---- AXI-Lite over MMIO ------------------------------------------------------
uint32_t pim_cfr_rd(pim_dev *d, uint32_t off)
{
    __sync_synchronize();
    uint32_t v = *(volatile uint32_t *)(d->bar + EMU_OFF_CFR + off);
    __sync_synchronize();
    return v;
}
void pim_cfr_wr(pim_dev *d, uint32_t off, uint32_t v)
{
    __sync_synchronize();
    *(volatile uint32_t *)(d->bar + EMU_OFF_CFR + off) = v;
    __sync_synchronize();
}

// ---- the 256 b windows over QDMA ---------------------------------------------
// Split at max_xfer, and re-read STATUS between fragments.  The read costs one
// MMIO cycle and turns "the board died somewhere in a 125 MiB upload" into "the
// board died at offset X" — which is the difference between a diagnosis and a
// shrug.  It is also how a wedge stops the upload instead of feeding it 30 more
// doomed transfers.
const char *pim_axi_write(pim_dev *d, uint64_t axi, const void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) return pim_err(d, "refused H2C to 0x%011llx (%zu B): %s",
                            (unsigned long long)axi, len, bad);
    const uint8_t *p = buf;
    uint64_t t0 = pim_now_us();
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > d->max_xfer ? d->max_xfer : len - done;
        ssize_t n = pwrite(d->h2c, p + done, want, (off_t)(axi + done));
        if (n <= 0)
            return pim_err(d, "H2C failed at 0x%011llx + 0x%zx (%zu B of %zu): %s.  "
                              "An EIO latches the engine; every later transfer will "
                              "fail until the queues are torn down and set up again.",
                           (unsigned long long)axi, done, want, len,
                           n < 0 ? strerror(errno) : "no progress");
        done += (size_t)n;
        if (done < len && pim_cfr_rd(d, CFR_STATUS) == 0xFFFFFFFFu)
            return pim_err(d, "the control plane went to all-ones during an H2C "
                              "transfer, at 0x%011llx + 0x%zx of %zu B.  The device "
                              "stopped answering here.",
                           (unsigned long long)axi, done, len);
    }
    d->st.h2c_bytes += len;
    d->st.h2c_us    += pim_now_us() - t0;
    return NULL;
}

const char *pim_axi_read(pim_dev *d, uint64_t axi, void *buf, size_t len)
{
    const char *bad = pim_dma_check(axi, len);
    if (bad) return pim_err(d, "refused C2H from 0x%011llx (%zu B): %s",
                            (unsigned long long)axi, len, bad);
    uint8_t *p = buf;
    uint64_t t0 = pim_now_us();
    for (size_t done = 0; done < len; ) {
        size_t want = len - done > d->max_xfer ? d->max_xfer : len - done;
        ssize_t n = pread(d->c2h, p + done, want, (off_t)(axi + done));
        if (n <= 0)
            return pim_err(d, "C2H failed at 0x%011llx + 0x%zx (%zu B of %zu): %s",
                           (unsigned long long)axi, done, want, len,
                           n < 0 ? strerror(errno) : "no progress");
        done += (size_t)n;
    }
    d->st.c2h_bytes += len;
    d->st.c2h_us    += pim_now_us() - t0;
    return NULL;
}

const char *pim_gpr_write(pim_dev *d, uint32_t word, const void *src, uint32_t n)
{
    if ((uint64_t)word + n > EMU_GPR_WORDS)
        return pim_err(d, "GPR write [%u, %u) runs past the end of the GPR (%u words). "
                          "The CFR begins one beat past it and its offset 0 is the "
                          "doorbell.", word, word + n, EMU_GPR_WORDS);
    return pim_axi_write(d, EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, src,
                         (size_t)n * EMU_WORD_BYTES);
}
const char *pim_gpr_read(pim_dev *d, uint32_t word, void *dst, uint32_t n)
{
    if ((uint64_t)word + n > EMU_GPR_WORDS)
        return pim_err(d, "GPR read [%u, %u) runs past the end of the GPR (%u words)",
                       word, word + n, EMU_GPR_WORDS);
    return pim_axi_read(d, EMU_AXI_GPR + (uint64_t)word * EMU_WORD_BYTES, dst,
                        (size_t)n * EMU_WORD_BYTES);
}

// The dispatcher must be parked before the host touches anything it owns.  The
// IMEM command port gives the FETCHER priority, so a host write during a live
// kernel holds WREADY low until the driver's 10 s timeout — and that EIO latches
// the H2C engine.  emu_gpr_wrap arbitrates the same way.
const char *pim_require_idle(pim_dev *d, const char *what, uint32_t timeout_ms)
{
    uint64_t t = pim_now_us();
    for (;;) {
        uint32_t st = pim_cfr_rd(d, CFR_STATUS);
        if (st == 0xFFFFFFFFu)
            return pim_err(d, "cannot %s: the control plane reads all-ones.  Either "
                              "the BAR mapping is stale or the device stopped "
                              "answering.", what);
        if ((st & CFR_STATUS_DONE) || CFR_STATUS_STATE(st) == 0) return NULL;
        if (pim_now_us() - t > (uint64_t)timeout_ms * 1000ull)
            return pim_err(d, "cannot %s: the dispatcher is still busy after %u ms "
                              "(STATUS=0x%08x, state=%u).  A previous kernel has not "
                              "finished.", what, timeout_ms, st, CFR_STATUS_STATE(st));
    }
}

const char *pim_stage(pim_dev *d, size_t bytes, uint8_t **out)
{
    if (bytes > d->stage_bytes) {
        uint8_t *p = realloc(d->stage, bytes);
        if (!p) return pim_err(d, "out of memory staging %zu B", bytes);
        d->stage = p;
        d->stage_bytes = bytes;
    }
    *out = d->stage;
    return NULL;
}

// ---- open / close ------------------------------------------------------------
const char *pim_open(const pim_config *cfg, pim_dev **out)
{
    static char boot_err[PIM_ERRLEN];       // before a dev exists there is nowhere else
    pim_config c = { 0 };
    if (cfg) c = *cfg;
    if (!c.bdf) c.bdf = BDF_DEFAULT;
    if (!c.h2c) c.h2c = H2C_DEFAULT;
    if (!c.c2h) c.c2h = C2H_DEFAULT;
    if (!c.max_isrs) c.max_isrs = CFR_PROG_LEN_MAX;
    if (!c.max_xfer) c.max_xfer = 4u << 20;
    if (!c.gpr_vec_words) c.gpr_vec_words = 4096;

    if (c.max_isrs > CFR_PROG_LEN_MAX) {
        snprintf(boot_err, sizeof boot_err,
                 "max_isrs %u exceeds PROG_LEN's 14 bits (%u)", c.max_isrs,
                 CFR_PROG_LEN_MAX);
        return boot_err;
    }
    if (c.gpr_vec_words >= EMU_GPR_WORDS) {
        snprintf(boot_err, sizeof boot_err, "gpr_vec_words %u leaves no room for results",
                 c.gpr_vec_words);
        return boot_err;
    }

    pim_dev *d = calloc(1, sizeof *d);
    if (!d) { snprintf(boot_err, sizeof boot_err, "out of memory"); return boot_err; }

    // The topology is compiled in (hwdef/pim_config.h, generated from the selected
    // platform conf).  What can still be wrong is that this library was built for a
    // different platform than the one now selected — choosing one and forgetting to
    // rebuild — so that is what is checked.
    {
        const char *e = pim_platform_check();
        if (e) {
            snprintf(boot_err, sizeof boot_err, "%s", e);
            free(d);
            return boot_err;
        }
    }

    d->bar_fd = d->h2c = d->c2h = -1;
    d->max_isrs = c.max_isrs;
    d->max_xfer = c.max_xfer;

    char path[256];
    snprintf(path, sizeof path, "/sys/bus/pci/devices/%s/resource2", c.bdf);
    d->bar_fd = open(path, O_RDWR | O_SYNC);
    if (d->bar_fd < 0) {
        const char *e = pim_err(d, "open(%s): %s", path, strerror(errno));
        snprintf(boot_err, sizeof boot_err, "%s", e); free(d); return boot_err;
    }
    void *m = mmap(NULL, EMU_BAR2_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, d->bar_fd, 0);
    if (m == MAP_FAILED) {
        const char *e = pim_err(d, "mmap BAR2: %s", strerror(errno));
        snprintf(boot_err, sizeof boot_err, "%s", e);
        close(d->bar_fd); free(d); return boot_err;
    }
    d->bar = m;
    d->h2c = open(c.h2c, O_WRONLY);
    d->c2h = open(c.c2h, O_RDONLY);
    if (d->h2c < 0 || d->c2h < 0) {
        const char *e = pim_err(d, "cannot open the MM queues (%s / %s): %s.  Both are "
                                   "required: IMEM, the GPR and HBM are DMA-only.  "
                                   "sudo ./qdma_queues.sh setup", c.h2c, c.c2h,
                                strerror(errno));
        snprintf(boot_err, sizeof boot_err, "%s", e);
        pim_close(d); return boot_err;
    }

    // ---- the control plane must answer before anything else happens ----------
    // Every register resets to zero and undecoded BAR space reads all-ones, so a
    // write/readback of a DISTINCT value is the only check that means anything.
    pim_cfr_wr(d, CFR_T_RAS, 0x5A);
    if (pim_cfr_rd(d, CFR_T_RAS) != 0x5A) {
        uint32_t got = pim_cfr_rd(d, CFR_T_RAS);
        const char *e = pim_err(d, "the CFR did not read back (wrote 0x5A, read 0x%08x). "
                                   "%s", got,
                                got == 0xFFFFFFFFu
                                  ? "All-ones means the PCIe mapping is stale or the "
                                    "device is not answering; remove and rescan, or "
                                    "reprogram."
                                  : "Run SW/emu_sanity.");
        snprintf(boot_err, sizeof boot_err, "%s", e);
        pim_close(d); return boot_err;
    }

    // ---- DRAM timing: written once, here, and never again --------------------
    // T_CCD MUST NOT drop below 2.  Two beats into the same latch back to back
    // accumulate every OTHER beat, silently (mac_top.sv:26-33).  Nothing in the
    // hardware refuses it, so the runtime owning these registers IS the guard.
    const struct emu_timing t = EMU_TIMING_SIM;
    const struct { uint32_t off; uint8_t v; const char *n; } TM[] = {
        { CFR_T_FAW, t.faw, "T_FAW" }, { CFR_T_RRD, t.rrd, "T_RRD" },
        { CFR_T_RCD, t.rcd, "T_RCD" }, { CFR_T_CCD, t.ccd, "T_CCD" },
        { CFR_T_RTP, t.rtp, "T_RTP" }, { CFR_T_RP,  t.rp,  "T_RP"  },
        { CFR_T_WR,  t.wr,  "T_WR"  }, { CFR_T_RAS, t.ras, "T_RAS" },
    };
    for (unsigned i = 0; i < 8; i++) pim_cfr_wr(d, TM[i].off, TM[i].v);
    for (unsigned i = 0; i < 8; i++)
        if (pim_cfr_rd(d, TM[i].off) != TM[i].v) {
            const char *e = pim_err(d, "%s did not read back (wrote %u, read %u)",
                                    TM[i].n, TM[i].v, pim_cfr_rd(d, TM[i].off));
            snprintf(boot_err, sizeof boot_err, "%s", e);
            pim_close(d); return boot_err;
        }

    // ---- allocator, buffers --------------------------------------------------
    d->holes = calloc(1, sizeof *d->holes);
    d->prog  = calloc(d->max_isrs, sizeof *d->prog);
    d->gpr_vec_base  = 0;
    d->gpr_vec_words = c.gpr_vec_words;
    d->gpr_res_base  = c.gpr_vec_words;
    d->gpr_res_words = EMU_GPR_WORDS - c.gpr_vec_words;
    d->res = calloc((size_t)d->max_isrs, PIM_LANES * sizeof(uint16_t));
    d->acc = calloc((size_t)d->max_isrs, PIM_LANES * sizeof(float));
    if (!d->holes || !d->prog || !d->res || !d->acc) {
        snprintf(boot_err, sizeof boot_err, "out of memory");
        pim_close(d); return boot_err;
    }
    // The whole per-channel RoBaCo space starts free.  Its size comes from the
    // platform (bank_window x nbank), not from a constant here — a constant would
    // silently assume one image.
    d->holes->off  = 0;
    d->holes->size = (uint64_t)PIM_BANK_WINDOW * PIM_NBANK;
    d->holes->next = NULL;

    // ---- the accumulator is live state that crosses processes ----------------
    const char *e = pim_scrub(d);
    if (e) { snprintf(boot_err, sizeof boot_err, "%s", e); pim_close(d); return boot_err; }

    *out = d;
    return NULL;
}

void pim_close(pim_dev *d)
{
    if (!d) return;
    for (struct pim_hole *h = d->holes; h; ) { struct pim_hole *n = h->next; free(h); h = n; }
    free(d->prog); free(d->res); free(d->acc); free(d->stage);
    if (d->h2c >= 0) close(d->h2c);
    if (d->c2h >= 0) close(d->c2h);
    if (d->bar) munmap((void *)d->bar, EMU_BAR2_SIZE);
    if (d->bar_fd >= 0) close(d->bar_fd);
    free(d);
}
