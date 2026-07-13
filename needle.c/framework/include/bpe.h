/* bpe.h — minimal BPE load / encode / decode for needle.c */
#ifndef ND_BPE_H
#define ND_BPE_H

#include <stddef.h>

#define ND_BPE_MAX_TOKEN 64
#define ND_BPE_MAX_VOCAB 8192
#define ND_BPE_MAX_MERGES 8192

typedef struct nd_bpe {
    int   vocab_size;
    char  tokens[ND_BPE_MAX_VOCAB][ND_BPE_MAX_TOKEN]; /* id → token string */
    int   n_merges;
    char  merge_a[ND_BPE_MAX_MERGES][ND_BPE_MAX_TOKEN];
    char  merge_b[ND_BPE_MAX_MERGES][ND_BPE_MAX_TOKEN];
    int   pad_id, bos_id, eos_id, unk_id;
} nd_bpe;

/* Load tokenizer/vocab.json + tokenizer/merges.txt (simple line formats).
 * vocab.json lines: "token"\t id   OR  JSON object {"token": id, ...}
 * merges.txt: one "a b" pair per line (rank = line order).
 * Returns 0 on success. */
int  nd_bpe_load(nd_bpe *b, const char *vocab_path, const char *merges_path);
void nd_bpe_free(nd_bpe *b); /* no-op for stack alloc; clears */

/* Encode UTF-8-ish bytes as char-level then apply merges. Returns n tokens
 * written (≤ max_out). Adds bos at start if add_specials; eos at end if room. */
int  nd_bpe_encode(const nd_bpe *b, const char *text, int *out_ids, int max_out,
                   int add_specials);

/* Decode ids → null-terminated string into buf (size buf_n). Returns length. */
int  nd_bpe_decode(const nd_bpe *b, const int *ids, int n, char *buf, int buf_n);

int  nd_bpe_token_to_id(const nd_bpe *b, const char *tok);

#endif
