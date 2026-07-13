/* Shared train entry for models A/B/C — stream, free graph, full f32, full epochs */
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

static float train_epochs(nd_module *model, nd_optim *opt, nd_dataset *ds,
                          int epochs, int batch, int accum, const char *tag) {
    int max_src = (int)ds->max_src, max_tgt = (int)ds->max_tgt;
    nd_batch b;
    b.src_ids = calloc((size_t)batch * max_src, sizeof(int));
    b.tgt_ids = calloc((size_t)batch * max_tgt, sizeof(int));
    b.labels = calloc((size_t)batch * max_tgt, sizeof(int));
    b.src_lens = calloc((size_t)batch, sizeof(int));
    b.tgt_lens = calloc((size_t)batch, sizeof(int));
    float last = 0, first = -1.f;
    for (int ep = 0; ep < epochs; ++ep) {
        nd_dataset_shuffle(ds, (unsigned)(7 + ep * 13));
        nd_dataset_rewind(ds);
        int steps = 0, micro = 0;
        float ep_loss = 0;
        opt->zero_grad(opt);
        for (;;) {
            int got = nd_dataset_next_batch(ds, &b, batch);
            if (got <= 0) break;
            nd_tensor *src, *tgt, *lab;
            batch_to_tensors(&b, max_src, max_tgt, &src, &tgt, &lab);
            nd_tensor *logits = model->forward2(model, src, tgt);
            nd_tensor *loss = nd_cross_entropy(logits, lab);
            float lv = loss->data[0];
            ep_loss += lv; steps++;
            if (first < 0) first = lv;
            last = lv;
            nd_backward(loss);
            micro++;
            if (micro >= accum) {
                opt->step(opt);
                opt->zero_grad(opt);
                micro = 0;
            }
            nd_tensor_free(loss);
            nd_tensor_free(logits);
            nd_tensor_free(src); nd_tensor_free(tgt); nd_tensor_free(lab);
            float peak = nd_peak_rss_mb();
            if (peak > 900.f) {
                printf("OOM peak_rss_mb=%.1f cur=%.1f\n", peak, nd_current_rss_mb());
                free(b.src_ids); free(b.tgt_ids); free(b.labels);
                free(b.src_lens); free(b.tgt_lens);
                return -1.f;
            }
        }
        if (micro > 0) { opt->step(opt); opt->zero_grad(opt); }
        printf("%s epoch=%d/%d loss=%.4f steps=%d peak_rss_mb=%.1f cur_rss_mb=%.1f\n",
               tag, ep + 1, epochs, steps ? ep_loss / steps : 0, steps,
               nd_peak_rss_mb(), nd_current_rss_mb());
        fflush(stdout);
    }
    free(b.src_ids); free(b.tgt_ids); free(b.labels);
    free(b.src_lens); free(b.tgt_lens);
    printf("%s first_loss=%.4f last_loss=%.4f\n", tag, first, last);
    return last;
}

static float eval_token_em(nd_module *model, nd_dataset *ds, int batch) {
    int max_src = (int)ds->max_src, max_tgt = (int)ds->max_tgt;
    nd_batch b;
    b.src_ids = calloc((size_t)batch * max_src, sizeof(int));
    b.tgt_ids = calloc((size_t)batch * max_tgt, sizeof(int));
    b.labels = calloc((size_t)batch * max_tgt, sizeof(int));
    b.src_lens = calloc((size_t)batch, sizeof(int));
    b.tgt_lens = calloc((size_t)batch, sizeof(int));
    nd_dataset_rewind(ds);
    long ok = 0, tot = 0, full = 0, full_ok = 0;
    for (;;) {
        int got = nd_dataset_next_batch(ds, &b, batch);
        if (got <= 0) break;
        nd_tensor *src, *tgt, *lab;
        batch_to_tensors(&b, max_src, max_tgt, &src, &tgt, &lab);
        nd_tensor *logits = model->forward2(model, src, tgt);
        int V = logits->shape[2];
        for (int i = 0; i < got; ++i) {
            int tl = b.tgt_lens[i];
            int row_ok = 1;
            for (int t = 0; t < tl; ++t) {
                float *row = logits->data + (i * max_tgt + t) * V;
                int pred = 0; float best = row[0];
                for (int v = 1; v < V; ++v) if (row[v] > best) { best = row[v]; pred = v; }
                int y = b.labels[i * max_tgt + t];
                if (pred == y) ok++; else row_ok = 0;
                tot++;
            }
            full++;
            if (row_ok) full_ok++;
        }
        nd_tensor_free(logits);
        nd_tensor_free(src); nd_tensor_free(tgt); nd_tensor_free(lab);
    }
    free(b.src_ids); free(b.tgt_ids); free(b.labels);
    free(b.src_lens); free(b.tgt_lens);
    float em = tot ? (float)ok / (float)tot : 0;
    float full_em = full ? (float)full_ok / (float)full : 0;
    printf("token_EM=%.4f val_full_call_EM=%.4f peak_rss_mb=%.1f\n",
           em, full_em, nd_peak_rss_mb());
    return full_em;
}

static void log_results(const char *path, const char *commit, float em, float rss,
                        const char *status, const char *desc) {
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    fprintf(fp, "%s\t%.4f\t%.1f\t%s\t%s\n", commit, em, rss, status, desc);
    fclose(fp);
}

/* argv: model{A|B|C} mode{smoke|train|fc} data_path [ckpt_path] [results.tsv] [vocab_path] [merges_path] */
int main(int argc, char **argv) {
    const char *which = argc > 1 ? argv[1] : "A";
    const char *mode = argc > 2 ? argv[2] : "smoke";
    const char *data = argc > 3 ? argv[3] : "data/smoke.bin";
    const char *ckpt = argc > 4 ? argv[4] : "ckpts/model.nd";
    const char *results = argc > 5 ? argv[5] : "results.tsv";
    const char *vocab_path = argc > 6 ? argv[6] : NULL;
    const char *merges_path = argc > 7 ? argv[7] : NULL;

    nd_seed(42);
    nd_needle_config cfg;
    int epochs = 50, batch = 4, accum = 1;
    float lr = 1e-3f;
    int is_fc = strcmp(mode, "fc") == 0;
    if (which[0] == 'A' || which[0] == 'a') {
        cfg = nd_cfg_sanity();
        cfg.vocab = 64;
        cfg.max_len = 128; /* FC bins use max_src=128 */
        if (is_fc) { epochs = 20; batch = 4; accum = 1; lr = 1e-3f; }
        else { epochs = strcmp(mode, "train") == 0 ? 20 : 50; batch = 4; accum = 1; lr = 1e-3f; }
    } else if (which[0] == 'B' || which[0] == 'b') {
        cfg = nd_cfg_pilot();
        cfg.vocab = 64; /* fixture vocab; real BPE later */
        epochs = strcmp(mode, "train") == 0 ? 10 : 5;
        batch = 1; accum = 1; lr = 5e-4f;
    } else {
        cfg = nd_cfg_full();
        cfg.vocab = 64;
        epochs = strcmp(mode, "train") == 0 ? 3 : 2;
        batch = 1; accum = 4; lr = 3e-4f; /* grad accum for C */
    }

    /* FC mode: load BPE, override vocab + max_len from data bin */
    if (is_fc && vocab_path && merges_path) {
        nd_bpe bpe;
        if (nd_bpe_load(&bpe, vocab_path, merges_path) != 0) {
            fprintf(stderr, "BPE load fail %s\n", vocab_path);
            return 1;
        }
        cfg.vocab = bpe.vocab_size;
        printf("fc: vocab=%d from %s\n", bpe.vocab_size, vocab_path);
        fflush(stdout);
    }

    nd_dataset *ds = nd_dataset_open(data);
    if (!ds) { fprintf(stderr, "open fail %s\n", data); return 1; }
    nd_module *model = nd_needle_create(&cfg);
    if (!model) { nd_dataset_close(ds); return 1; }
    long np = nd_module_num_params(model);
    printf("model=%s mode=%s n_params=%ld vocab=%d d=%d B=%d accum=%d epochs=%d\n",
           which, mode, np, cfg.vocab, cfg.d_model, batch, accum, epochs);
    fflush(stdout);
    nd_tensor **params; int n;
    nd_module_parameters(model, &params, &n);
    nd_optim *opt = nd_adamw(params, n, lr, 0.9f, 0.999f, 0.01f, 1e-8f);
    float last = train_epochs(model, opt, ds, epochs, batch, accum, which);
    float em = 0;
    const char *status = "keep";
    if (last < 0) status = "crash";
    else em = eval_token_em(model, ds, batch);
    float rss = nd_peak_rss_mb();
    if (rss > 900.f) status = "crash";
    nd_checkpoint_save(ckpt, model);
    char desc[128];
    snprintf(desc, sizeof desc, "model %s %s full epochs f32 stream", which, mode);
    log_results(results, "local", em, rss, status, desc);
    printf("peak_rss_mb=%.1f status=%s\n", rss, status);
    nd_optim_free(opt); free(params);
    nd_module_free(model);
    nd_dataset_close(ds);
    return (last < 0 || rss > 900.f) ? 1 : 0;
}
