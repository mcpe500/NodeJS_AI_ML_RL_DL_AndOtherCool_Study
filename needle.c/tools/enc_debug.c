#include "needle.h"
#include <stdio.h>
#include <string.h>
int main(void) {
    nd_bpe b; memset(&b, 0, sizeof b);
    int rc = nd_bpe_load(&b, "tokenizer/vocab.json", "tokenizer/merges.txt");
    printf("load=%d vocab=%d merges=%d specials=%d/%d/%d/%d\n",
           rc, b.vocab_size, b.n_merges, b.pad_id, b.bos_id, b.eos_id, b.unk_id);
    int ids[256];
    const char *s = "Query: What's the weather in San Francisco?\nTools: [{\"name\":\"get_weather\",\"parameters\":{\"location\":\"string\"}}]";
    int n = nd_bpe_encode(&b, s, ids, 256, 1);
    printf("encode n=%d ids:", n);
    for (int i = 0; i < n; i++) printf(" %d", ids[i]);
    printf("\n");
    char buf[1024];
    int dn = nd_bpe_decode(&b, ids + 1, n > 2 ? n - 2 : 0, buf, sizeof buf);
    printf("dec[%d]=%s\n", dn, buf);
    n = nd_bpe_encode(&b, s, ids, 256, 0);
    printf("nospecial n=%d\n", n);
    /* checkpoint load */
    nd_needle_config cfg = nd_cfg_sanity();
    cfg.vocab = b.vocab_size;
    nd_module *m = nd_needle_create(&cfg);
    rc = nd_checkpoint_load("models/01-sanity/ckpts/sanity_fc.nd", m);
    printf("ckpt_load=%d n_params=%ld\n", rc, nd_module_num_params(m));
    nd_module_free(m);
    return 0;
}
