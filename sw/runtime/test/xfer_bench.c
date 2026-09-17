// SPDX-License-Identifier: MIT
//////////////////////////////////////////////////////////////////////////////////
// xfer_bench.c — what is the ~13 us a small pwrite costs?
//
// eager_board measured that a pwrite to a QDMA MM queue costs about 13 us almost
// regardless of payload: 256 bytes and 2048 bytes come out within 2 us of each
// other.  That number decides real design questions — whether a KV cache may append
// one row per token, whether a scattered V writeback needs batching — so it is worth
// knowing WHAT it is, because the answer changes what to do about it:
//
//   LATENCY, and the queue can hold several in flight
//        -> more threads on the same queue overlaps them.  Cheap.
//   LATENCY, but the queue serialises
//        -> more QUEUES are needed; threads alone buy nothing.  qmax is 4096 here.
//   CPU TIME in the driver
//        -> threads help only up to core count and the ceiling is low.
//
// So: the same total bytes, moved as N small transfers, spread over 1, 2, 4 and 8
// threads sharing ONE queue.  If the per-transfer cost falls with thread count the
// queue pipelines and threading is the answer; if it is flat the queue serialises
// and only a second queue can help.
//
// A big single transfer is measured first as the ceiling — that is what the
// transport does when nothing is in the way.
//
// SAFE TO RUN CONCURRENTLY DESPITE pim.h's WARNING.  The unsafe parts of a context
// are the error string and the pool; pwrite itself takes an explicit offset and
// shares no file position, and every thread here writes a disjoint range that was
// allocated up front.
//
//     ./xfer_bench [--bytes N] [--reps N]
//////////////////////////////////////////////////////////////////////////////////
#define _GNU_SOURCE

#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "pim/pim.h"
#include "pim/pim_addr.h"

#define MAXTHREADS 8

static uint64_t now_us(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000ull + (uint64_t)t.tv_nsec / 1000ull;
}

struct job {
    pim_ctx  *c;
    char     *dst;          /* PIM pointer, this thread's slice */
    void     *src;
    size_t    chunk;        /* bytes per transfer */
    uint32_t  n;            /* transfers this thread does */
    uint64_t  stride;       /* distance between this thread's transfers */
    int       fd;           /* -1 = go through libpim's single queue          */
    uint64_t  axi;          /* card address of dst, when fd >= 0              */
    uint32_t  err;          /* short or failed pwrites                        */
};

static void *worker(void *arg)
{
    struct job *j = arg;

    for (uint32_t i = 0; i < j->n; i++) {
        if (j->fd >= 0) {
            // OWN QUEUE.  Straight pwrite, so nothing this thread does touches
            // anything another thread has.  The AXI address IS the file offset.
            ssize_t k = pwrite(j->fd, j->src, j->chunk,
                               (off_t)(j->axi + (uint64_t)i * j->stride));
            if (k != (ssize_t)j->chunk) j->err++;
        } else
            pim_memcpy_ctx(j->c, j->dst + (size_t)i * j->stride, j->src, j->chunk,
                           PIM_TO_DEV, 0);
    }
    return NULL;
}

// Open one queue node per thread.  Returns how many it got; the caller falls back
// to the shared queue when there are not enough.
static int open_queues(int *fd, int want, const char *qdev, int first_idx)
{
    char path[128];
    int  n = 0;

    for (int i = 0; i < want; i++) {
        snprintf(path, sizeof path, "/dev/%s-MM-%d", qdev, first_idx + i);
        fd[i] = open(path, O_WRONLY);
        if (fd[i] < 0) break;
        n++;
    }
    return n;
}

int main(int argc, char **argv)
{
    pim_ctx *c = NULL;
    const char *bad;
    size_t   chunk = 256;
    uint32_t reps  = 1024;
    const char *qdev = "qdma01000";
    int      first_idx = 2;        /* 0 and 1 are libpim's pair */
    int      qfd[MAXTHREADS], nq = 0;
    uint64_t base_axi = 0;
    void    *dev;
    char    *host;
    size_t   span;
    uint64_t stride = 4096;          /* one GPR page apart, so no two overlap */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bytes") && i + 1 < argc) chunk = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--qdev") && i + 1 < argc) qdev = argv[++i];
        else if (!strcmp(argv[i], "--first-q") && i + 1 < argc) first_idx = atoi(argv[++i]);
    }

    printf("small-transfer cost on one QDMA queue\n");
    if ((bad = pim_open(NULL, &c))) { printf("SKIPPED: %s\n", bad); return 0; }

    span = (size_t)reps * stride;
    dev  = pim_alloc_ctx(c, span, PIM_MEM_DRAM);
    host = malloc(chunk);
    if (!dev || !host) { printf("  alloc: %s\n", pim_last_error_ctx(c)); return 1; }
    memset(host, 0x5A, chunk);
    printf("  %u transfers of %zu B, %zu KiB of PIM reserved\n\n",
           reps, chunk, span >> 10);

    // ---- the ceiling: one transfer of the same total ------------------------
    {
        size_t total = (size_t)reps * chunk;
        void  *big   = pim_alloc_ctx(c, total, PIM_MEM_DRAM);
        char  *hb    = malloc(total);
        uint64_t t0;

        if (big && hb) {
            memset(hb, 0x5A, total);
            t0 = now_us();
            pim_memcpy_ctx(c, big, hb, total, PIM_TO_DEV, 0);
            uint64_t us = now_us() - t0;
            printf("  one transfer of %zu KiB            %6llu us   %5.0f MB/s\n",
                   total >> 10, (unsigned long long)us,
                   us ? (double)total / us : 0.0);
            pim_free_ctx(c, big);
        }
        free(hb);
    }

    // ---- how many private queues are there? --------------------------------
    nq = open_queues(qfd, MAXTHREADS, qdev, first_idx);
    if (nq) {
        pim_loc loc; size_t run;
        if (pim_resolve_ctx(c, dev, chunk, &loc, &run)) nq = 0;
        else base_axi = loc.axi;
    }
    printf("\n  private queues at /dev/%s-MM-%d.. : %d\n", qdev, first_idx, nq);
    if (!nq)
        printf("     (none — create them with dma-ctl to compare against the shared one)\n");

    // ---- the same bytes, split, across N threads ----------------------------
    for (int mode = 0; mode < 2; mode++) {
      if (mode == 1 && nq < 2) break;
      printf("\n  %zu B transfers, %s:\n", chunk,
             mode ? "ONE QUEUE PER THREAD" : "one shared queue");
      printf("  %-8s %-10s %-12s %s\n", "threads", "total us", "us/transfer", "MB/s");
      for (int nt = 1; nt <= MAXTHREADS; nt *= 2) {
        if (mode == 1 && nt > nq) break;
        pthread_t th[MAXTHREADS];
        struct job jb[MAXTHREADS];
        uint32_t per = reps / (uint32_t)nt;
        uint64_t t0, us;

        if (!per) break;
        t0 = now_us();
        for (int t = 0; t < nt; t++) {
            jb[t] = (struct job){
                .c = c, .src = host, .chunk = chunk, .n = per, .stride = stride,
                .dst = (char *)dev + (size_t)t * per * stride,
                .fd  = mode ? qfd[t] : -1,
                .axi = base_axi + (uint64_t)t * per * stride,
            };
            pthread_create(&th[t], NULL, worker, &jb[t]);
        }
        for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
        us = now_us() - t0;
        for (int t = 0; t < nt; t++)
            if (jb[t].err) printf("     thread %d: %u failed transfer(s)\n", t, jb[t].err);

        printf("  %-8d %-10llu %-12.2f %.0f\n", nt, (unsigned long long)us,
               us ? (double)us / (per * (uint32_t)nt) : 0.0,
               us ? (double)per * nt * chunk / us : 0.0);
      }
    }

    printf("\n  read it: if the two tables match, one queue already pipelines and\n"
           "  per-thread queues buy nothing.  If the per-thread one keeps scaling\n"
           "  where the shared one flattens, the ring is the limit and each thread\n"
           "  wants its own.\n");

    for (int i = 0; i < nq; i++) close(qfd[i]);
    pim_free_ctx(c, dev);
    free(host);
    pim_close(c);
    return 0;
}
