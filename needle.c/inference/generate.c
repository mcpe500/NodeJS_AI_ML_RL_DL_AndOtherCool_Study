/* Live greedy generate: query+tools → tool-call JSON.
 * Loads BPE + ckpt, encodes src, greedy-decodes (recompute per step — no KV),
 * detokenizes. Prints JSON string to stdout.
 *
 * Usage:
 *   ./generate --ckpt <path> --vocab <path> --merges <path> \
 *              --query "..." --tools '[...]'
 */
#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SRC 128
#define MAX_TGT 64

static int find_arg(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], key) == 0) return i + 1;
    return -1;
}

int main(int argc, char **argv) {
    int a_ckpt = find_arg(argc, argv, "--ckpt");
    int a_vocab = find_arg(argc, argv, "--vocab");
    int a_merges = find_arg(argc, argv, "--merges");
    int a_query = find_arg(argc, argv, "--query");
    int a_tools = find_arg(argc, argv, "--tools");

    if (a_ckpt < 0 || a_vocab < 0 || a_merges < 0 || a_query < 0) {
        fprintf(stderr,
            "usage: generate --ckpt m.nd --vocab v.json --merges m.txt --query q [--tools json]\n");
        return 2;
    }
    const char *ckpt = argv[a_ckpt];
    const char *vocab = argv[a_vocab];
    const char *merges = argv[a_merges];
    const char *query = argv[a_query];
    const char *tools = a_tools > 0 ? argv[a_tools] : "[]";

    nd_bpe bpe;
    if (nd_bpe_load(&bpe, vocab, merges) != 0) {
        fprintf(stderr, "bpe load fail\n");
        return 2;
    }

    /* src text template (matches tools/jsonl_to_bin.py) */
    char src_text[4096];
    snprintf(src_text, sizeof src_text, "Query: %s\nTools: %s", query, tools);

    int src_ids[MAX_SRC];
    int ns = nd_bpe_encode(&bpe, src_text, src_ids, MAX_SRC, 1);
    if (ns <= 0) { fprintf(stderr, "encode fail\n"); return 2; }

    /* pad src to MAX_SRC */
    int src_pad[MAX_SRC];
    for (int i = 0; i < MAX_SRC; ++i)
        src_pad[i] = i < ns ? src_ids[i] : bpe.pad_id;

    nd_seed(1);
    nd_needle_config cfg = nd_cfg_sanity();
    cfg.vocab = bpe.vocab_size;
    cfg.max_len = MAX_SRC > MAX_TGT ? MAX_SRC : MAX_TGT;
    nd_module *model = nd_needle_create(&cfg);
    if (nd_checkpoint_load(ckpt, model) != 0) {
        fprintf(stderr, "warn: no ckpt at %s — random weights\n", ckpt);
    }

    /* greedy decode: re-feed growing tgt each step. */
    int tgt_ids[MAX_TGT];
    int nt = 1;
    tgt_ids[0] = bpe.bos_id;
    char out_buf[2048];
    int stopped = 0;

    for (int step = 0; step < MAX_TGT - 1 && nt < MAX_TGT; ++step) {
        int tgt_pad[MAX_TGT];
        for (int i = 0; i < MAX_TGT; ++i)
            tgt_pad[i] = i < nt ? tgt_ids[i] : bpe.pad_id;

        int ssh[2] = {1, MAX_SRC}, tsh[2] = {1, MAX_TGT};
        nd_tensor *src = nd_zeros(ssh, 2, false);
        nd_tensor *tgt = nd_zeros(tsh, 2, false);
        for (int i = 0; i < MAX_SRC; ++i) src->data[i] = (float)src_pad[i];
        for (int i = 0; i < MAX_TGT; ++i) tgt->data[i] = (float)tgt_pad[i];

        nd_tensor *logits = model->forward2(model, src, tgt);
        int V = logits->shape[2];
        int last = nt - 1;
        float *row = logits->data + last * V;
        int best = 0; float bv = row[0];
        for (int v = 1; v < V; ++v) if (row[v] > bv) { bv = row[v]; best = v; }
        nd_tensor_free(logits);
        nd_tensor_free(src); nd_tensor_free(tgt);

        if (best == bpe.eos_id) { stopped = 1; break; }
        tgt_ids[nt++] = best;
    }

    int dn = nd_bpe_decode(&bpe, tgt_ids + 1, nt - 1, out_buf, sizeof out_buf);
    (void)dn;
    printf("%s\n", out_buf);
    fprintf(stderr, "generate: src_len=%d tgt_len=%d stopped=%d peak_rss_mb=%.1f\n",
            ns, nt, stopped, nd_peak_rss_mb());
    nd_module_free(model);
    return 0;
}
