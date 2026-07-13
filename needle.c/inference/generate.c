/* Greedy decode demo — no KV-cache yet (stretch). */
#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *ckpt = argc > 1 ? argv[1] : "models/01-sanity/ckpts/sanity.nd";
    nd_seed(1);
    nd_needle_config cfg = nd_cfg_sanity();
    cfg.vocab = 64;
    nd_module *model = nd_needle_create(&cfg);
    if (nd_checkpoint_load(ckpt, model) != 0)
        fprintf(stderr, "warn: no ckpt, random weights\n");
    int B = 1, S = 4, T = 4;
    int ssh[2] = {B, S}, tsh[2] = {B, T};
    nd_tensor *src = nd_zeros(ssh, 2, false);
    nd_tensor *tgt = nd_zeros(tsh, 2, false);
    for (int i = 0; i < S; ++i) src->data[i] = (float)((i + 1) % 64);
    tgt->data[0] = 1.f; /* BOS-ish */
    /* teacher-force fill then one greedy step on last position */
    nd_tensor *logits = model->forward2(model, src, tgt);
    int V = logits->shape[2];
    int pred[8];
    for (int t = 0; t < T; ++t) {
        float *row = logits->data + t * V;
        int p = 0; float best = row[0];
        for (int v = 1; v < V; ++v) if (row[v] > best) { best = row[v]; p = v; }
        pred[t] = p;
    }
    printf("greedy_ids:");
    for (int t = 0; t < T; ++t) printf(" %d", pred[t]);
    printf("\npeak_rss_mb=%.1f\n", nd_peak_rss_mb());
    /* toy JSON parseable shell */
    printf("{\"tool\":\"demo\",\"args\":{\"ids\":[%d,%d,%d,%d]}}\n",
           pred[0], pred[1], pred[2], pred[3]);
    nd_tensor_free(logits); nd_tensor_free(src); nd_tensor_free(tgt);
    nd_module_free(model);
    return 0;
}
