/* dataset.h — HARDWARE AWARE stream-only loader */
#ifndef ND_DATASET_H
#define ND_DATASET_H

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

/* Binary layout:
 * magic "NDSET001" (8)
 * version u32, n_samples u32, max_src u32, max_tgt u32
 * then n_samples × u64 file offsets to records
 * each record: u32 src_len, u32 tgt_len, src_ids[src_len] i32, tgt_ids[tgt_len] i32, labels[tgt_len] i32
 * labels = next-token targets for decoder (teacher forcing)
 */

typedef struct nd_dataset {
    FILE    *fp;
    uint32_t n_samples;
    uint32_t max_src, max_tgt;
    uint64_t *offsets;   /* n_samples only — NOT full data */
    int     *indices;
    int      cursor;
} nd_dataset;

typedef struct nd_batch {
    int B, Tsrc, Ttgt;
    int *src_ids;   /* B * max_src  (caller owns / reused) */
    int *tgt_ids;
    int *labels;
    int *src_lens;
    int *tgt_lens;
} nd_batch;

nd_dataset *nd_dataset_open(const char *path);
void        nd_dataset_close(nd_dataset *d);
void        nd_dataset_shuffle(nd_dataset *d, unsigned seed);
/* Fill batch buffers (preallocated by caller). Returns batch size or 0 at epoch end. */
int         nd_dataset_next_batch(nd_dataset *d, nd_batch *b, int batch_req);
void        nd_dataset_rewind(nd_dataset *d);

/* Write helper for tests/synth — still written sample-by-sample, not one giant load at read */
int nd_dataset_write(const char *path, int n, int max_src, int max_tgt,
                     const int *src_ids, const int *src_lens,
                     const int *tgt_ids, const int *tgt_lens,
                     const int *labels);

#endif
