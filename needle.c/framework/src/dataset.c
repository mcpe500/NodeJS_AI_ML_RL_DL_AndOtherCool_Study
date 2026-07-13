#include "dataset.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>

#define MAGIC "NDSET001"

nd_dataset *nd_dataset_open(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { nd_log("dataset open fail %s", path); return NULL; }
    char mag[8];
    if (fread(mag, 1, 8, fp) != 8 || memcmp(mag, MAGIC, 8) != 0) {
        nd_log("bad magic"); fclose(fp); return NULL;
    }
    uint32_t ver, n, max_src, max_tgt;
    if (fread(&ver, 4, 1, fp) != 1 || fread(&n, 4, 1, fp) != 1 ||
        fread(&max_src, 4, 1, fp) != 1 || fread(&max_tgt, 4, 1, fp) != 1) {
        fclose(fp); return NULL;
    }
    nd_dataset *d = (nd_dataset *)nd_xcalloc(1, sizeof(nd_dataset));
    d->fp = fp;
    d->n_samples = n;
    d->max_src = max_src;
    d->max_tgt = max_tgt;
    /* HARDWARE AWARE: only offsets + indices, never full corpus */
    d->offsets = (uint64_t *)nd_xmalloc((size_t)n * sizeof(uint64_t));
    d->indices = (int *)nd_xmalloc((size_t)n * sizeof(int));
    if (fread(d->offsets, sizeof(uint64_t), n, fp) != n) {
        nd_dataset_close(d); return NULL;
    }
    for (uint32_t i = 0; i < n; ++i) d->indices[i] = (int)i;
    d->cursor = 0;
    return d;
}

void nd_dataset_close(nd_dataset *d) {
    if (!d) return;
    if (d->fp) fclose(d->fp);
    free(d->offsets);
    free(d->indices);
    free(d);
}

void nd_dataset_rewind(nd_dataset *d) { if (d) d->cursor = 0; }

void nd_dataset_shuffle(nd_dataset *d, unsigned seed) {
    nd_seed(seed);
    for (int i = (int)d->n_samples - 1; i > 0; --i) {
        int j = (int)(nd_rand_uniform() * (i + 1));
        if (j < 0) j = 0; if (j > i) j = i;
        int tmp = d->indices[i]; d->indices[i] = d->indices[j]; d->indices[j] = tmp;
    }
    d->cursor = 0;
}

int nd_dataset_next_batch(nd_dataset *d, nd_batch *b, int batch_req) {
    if (!d || d->cursor >= (int)d->n_samples) return 0;
    int B = batch_req;
    if (d->cursor + B > (int)d->n_samples) B = (int)d->n_samples - d->cursor;
    int max_src = (int)d->max_src, max_tgt = (int)d->max_tgt;
    memset(b->src_ids, 0, (size_t)B * max_src * sizeof(int));
    memset(b->tgt_ids, 0, (size_t)B * max_tgt * sizeof(int));
    memset(b->labels, 0, (size_t)B * max_tgt * sizeof(int));
    for (int i = 0; i < B; ++i) {
        int idx = d->indices[d->cursor + i];
        if (fseek(d->fp, (long)d->offsets[idx], SEEK_SET) != 0) return 0;
        uint32_t sl, tl;
        if (fread(&sl, 4, 1, d->fp) != 1 || fread(&tl, 4, 1, d->fp) != 1) return 0;
        if (sl > (uint32_t)max_src) sl = (uint32_t)max_src;
        if (tl > (uint32_t)max_tgt) tl = (uint32_t)max_tgt;
        b->src_lens[i] = (int)sl;
        b->tgt_lens[i] = (int)tl;
        if (sl && fread(b->src_ids + i * max_src, 4, sl, d->fp) != sl) return 0;
        if (tl && fread(b->tgt_ids + i * max_tgt, 4, tl, d->fp) != tl) return 0;
        if (tl && fread(b->labels + i * max_tgt, 4, tl, d->fp) != tl) return 0;
    }
    d->cursor += B;
    b->B = B; b->Tsrc = max_src; b->Ttgt = max_tgt;
    return B;
}

int nd_dataset_write(const char *path, int n, int max_src, int max_tgt,
                     const int *src_ids, const int *src_lens,
                     const int *tgt_ids, const int *tgt_lens,
                     const int *labels) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite(MAGIC, 1, 8, fp);
    uint32_t ver = 1, nn = (uint32_t)n, ms = (uint32_t)max_src, mt = (uint32_t)max_tgt;
    fwrite(&ver, 4, 1, fp); fwrite(&nn, 4, 1, fp); fwrite(&ms, 4, 1, fp); fwrite(&mt, 4, 1, fp);
    long off_table = ftell(fp);
    uint64_t *offsets = (uint64_t *)calloc((size_t)n, sizeof(uint64_t));
    fwrite(offsets, sizeof(uint64_t), (size_t)n, fp); /* placeholder */
    for (int i = 0; i < n; ++i) {
        offsets[i] = (uint64_t)ftell(fp);
        uint32_t sl = (uint32_t)src_lens[i], tl = (uint32_t)tgt_lens[i];
        fwrite(&sl, 4, 1, fp); fwrite(&tl, 4, 1, fp);
        fwrite(src_ids + i * max_src, 4, sl, fp);
        fwrite(tgt_ids + i * max_tgt, 4, tl, fp);
        fwrite(labels + i * max_tgt, 4, tl, fp);
    }
    fseek(fp, off_table, SEEK_SET);
    fwrite(offsets, sizeof(uint64_t), (size_t)n, fp);
    free(offsets);
    fclose(fp);
    return 0;
}
