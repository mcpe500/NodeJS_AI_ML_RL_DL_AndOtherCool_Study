/* Eval harness — load ckpt, stream val, print headline metrics */
#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void batch_to_tensors(nd_batch *b, int max_src, int max_tgt,
                             nd_tensor **src, nd_tensor **tgt, nd_tensor **lab) {
    int B = b->B;
    int ssh[2] = {B, max_src}, tsh[2] = {B, max_tgt}, lsh[1] = {B * max_tgt};
    *src = nd_zeros(ssh, 2, false);
    *tgt = nd_zeros(tsh, 2, false);
    *lab = nd_zeros(lsh, 1, false);
    for (int i = 0; i < B * max_src; ++i) (*src)->data[i] = (float)b->src_ids[i];
    for (int i = 0; i < B * max_tgt; ++i) {
        (*tgt)->data[i] = (float)b->tgt_ids[i];
        (*lab)->data[i] = (float)b->labels[i];
    }
}

/* argv: model{A|B|C} data_path ckpt_path [vocab_path] [merges_path] */
int main(int argc, char **argv) {
    const char *which = argc > 1 ? argv[1] : "A";
    const char *data = argc > 2 ? argv[2] : "data/smoke.bin";
    const char *ckpt = argc > 3 ? argv[3] : "ckpts/model.nd";
    const char *vocab = argc > 4 ? argv[4] : NULL;
    const char *merges = argc > 5 ? argv[5] : NULL;
    nd_needle_config cfg;
    int batch = 4;
    if (which[0] == 'B' || which[0] == 'b') { cfg = nd_cfg_pilot(); batch = 1; }
    else if (which[0] == 'C' || which[0] == 'c') { cfg = nd_cfg_full(); batch = 1; }
    else { cfg = nd_cfg_sanity(); batch = 4; }
    cfg.vocab = 64;
    if (vocab && merges) {
        nd_bpe bpe;
        if (nd_bpe_load(&bpe, vocab, merges) == 0) {
            cfg.vocab = bpe.vocab_size;
            printf("eval: vocab=%d from BPE\n", cfg.vocab);
        }
    }
    nd_dataset *ds = nd_dataset_open(data);
    if (!ds) { fprintf(stderr, "data open fail\n"); return 1; }
    nd_module *model = nd_needle_create(&cfg);
    if (nd_checkpoint_load(ckpt, model) != 0)
        fprintf(stderr, "warn: ckpt load fail %s — eval random weights\n", ckpt);
    int max_src = (int)ds->max_src, max_tgt = (int)ds->max_tgt;
    nd_batch b;
    b.src_ids = calloc((size_t)batch * max_src, sizeof(int));
    b.tgt_ids = calloc((size_t)batch * max_tgt, sizeof(int));
    b.labels = calloc((size_t)batch * max_tgt, sizeof(int));
    b.src_lens = calloc((size_t)batch, sizeof(int));
    b.tgt_lens = calloc((size_t)batch, sizeof(int));
    long ok = 0, tot = 0, full = 0, full_ok = 0;
    nd_dataset_rewind(ds);
    for (;;) {
        int got = nd_dataset_next_batch(ds, &b, batch);
        if (got <= 0) break;
        nd_tensor *src, *tgt, *lab;
        batch_to_tensors(&b, max_src, max_tgt, &src, &tgt, &lab);
        nd_tensor *logits = model->forward2(model, src, tgt);
        int V = logits->shape[2];
        for (int i = 0; i < got; ++i) {
            int tl = b.tgt_lens[i], row_ok = 1;
            for (int t = 0; t < tl; ++t) {
                float *row = logits->data + (i * max_tgt + t) * V;
                int pred = 0; float best = row[0];
                for (int v = 1; v < V; ++v) if (row[v] > best) { best = row[v]; pred = v; }
                if (pred == b.labels[i * max_tgt + t]) ok++; else row_ok = 0;
                tot++;
            }
            full++; if (row_ok) full_ok++;
        }
        nd_tensor_free(logits);
        nd_tensor_free(src); nd_tensor_free(tgt); nd_tensor_free(lab);
    }
    float token_em = tot ? (float)ok / (float)tot : 0;
    float full_em = full ? (float)full_ok / (float)full : 0;
    printf("model=%s\n", which);
    printf("token_EM=%.4f\n", token_em);
    printf("val_full_call_EM=%.4f\n", full_em);
    printf("peak_rss_mb=%.1f\n", nd_peak_rss_mb());
    free(b.src_ids); free(b.tgt_ids); free(b.labels);
    free(b.src_lens); free(b.tgt_lens);
    nd_module_free(model);
    nd_dataset_close(ds);
    return 0;
}
