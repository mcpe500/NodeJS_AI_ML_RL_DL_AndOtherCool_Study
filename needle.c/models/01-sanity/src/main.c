/* Model A 01-sanity — full float32, stream data, free graph every step */
#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
                          int epochs, int batch, const char *tag) {
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
        int steps = 0;
        float ep_loss = 0;
        for (;;) {
            int got = nd_dataset_next_batch(ds, &b, batch);
            if (got <= 0) break;
            nd_tensor *src, *tgt, *lab;
            batch_to_tensors(&b, max_src, max_tgt, &src, &tgt, &lab);
            opt->zero_grad(opt);
            nd_tensor *logits = model->forward2(model, src, tgt);
            nd_tensor *loss = nd_cross_entropy(logits, lab);
            float lv = loss->data[0];
            ep_loss += lv; steps++;
            if (first < 0) first = lv;
            last = lv;
            nd_backward(loss);
            opt->step(opt);
            /* free loss (drops CE retain) then logits (walks graph) then leaves */
            nd_tensor_free(loss);
            nd_tensor_free(logits);
            nd_tensor_free(src); nd_tensor_free(tgt); nd_tensor_free(lab);
            float cur = nd_current_rss_mb();
            float peak = nd_peak_rss_mb();
            if (peak > 900.f) {
                printf("OOM peak_rss_mb=%.1f cur=%.1f\n", peak, cur);
                free(b.src_ids); free(b.tgt_ids); free(b.labels);
                free(b.src_lens); free(b.tgt_lens);
                return -1.f;
            }
        }
        float rss = nd_peak_rss_mb();
        printf("%s epoch=%d/%d loss=%.4f steps=%d peak_rss_mb=%.1f cur_rss_mb=%.1f\n",
               tag, ep + 1, epochs, steps ? ep_loss / steps : 0, steps, rss, nd_current_rss_mb());
    }
    free(b.src_ids); free(b.tgt_ids); free(b.labels);
    free(b.src_lens); free(b.tgt_lens);
    printf("%s first_loss=%.4f last_loss=%.4f\n", tag, first, last);
    return last;
}

/* greedy argmax EM on labels for full-call proxy (token EM) */
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

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "smoke";
    const char *data = argc > 2 ? argv[2] : "../../data/smoke.bin";
    int epochs_smoke = 50, epochs_train = 20;
    int batch = 4;
    nd_seed(42);
    nd_needle_config cfg = nd_cfg_sanity();
    cfg.vocab = 64; /* smoke fixture vocab */
    if (strcmp(mode, "train") == 0) {
        /* small train may use same fixture path or larger bin */
    }
    nd_dataset *ds = nd_dataset_open(data);
    if (!ds) {
        fprintf(stderr, "open dataset fail: %s\n", data);
        return 1;
    }
    nd_module *model = nd_needle_create(&cfg);
    if (!model) { nd_dataset_close(ds); return 1; }
    long np = nd_module_num_params(model);
    printf("mode=%s n_params=%ld vocab=%d d=%d\n", mode, np, cfg.vocab, cfg.d_model);
    nd_tensor **params; int n;
    nd_module_parameters(model, &params, &n);
    nd_optim *opt = nd_adamw(params, n, 1e-3f, 0.9f, 0.999f, 0.01f, 1e-8f);
    int epochs = strcmp(mode, "train") == 0 ? epochs_train : epochs_smoke;
    float last = train_epochs(model, opt, ds, epochs, batch, mode);
    float em = 0;
    const char *status = "keep";
    if (last < 0) status = "crash";
    else em = eval_token_em(model, ds, batch);
    float rss = nd_peak_rss_mb();
    if (rss > 900.f) status = "crash";
    nd_checkpoint_save("ckpts/sanity.nd", model);
    log_results("results.tsv", "local", em, rss, status,
                strcmp(mode, "train") == 0 ? "A small train full epochs" : "A smoke full epochs");
    printf("peak_rss_mb=%.1f status=%s\n", rss, status);
    nd_optim_free(opt); free(params);
    nd_module_free(model);
    nd_dataset_close(ds);
    return (last < 0 || rss > 900.f) ? 1 : 0;
}
