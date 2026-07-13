#include "bpe.h"
#include "util.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

void nd_bpe_free(nd_bpe *b) {
    if (!b) return;
    memset(b, 0, sizeof *b);
}

int nd_bpe_token_to_id(const nd_bpe *b, const char *tok) {
    if (!b || !tok) return -1;
    for (int i = 0; i < b->vocab_size; ++i)
        if (strcmp(b->tokens[i], tok) == 0) return i;
    return b->unk_id >= 0 ? b->unk_id : -1;
}

/* vocab format: one "token\tid" per line (token may be escaped \n \t \\) */
static int unescape_tok(const char *in, char *out, size_t outn) {
    size_t j = 0;
    for (size_t i = 0; in[i] && j + 1 < outn; ++i) {
        if (in[i] == '\\' && in[i + 1]) {
            char c = in[++i];
            if (c == 'n') out[j++] = '\n';
            else if (c == 't') out[j++] = '\t';
            else if (c == 'r') out[j++] = '\r';
            else if (c == 's') out[j++] = ' '; /* space as \s so lines stay clean */
            else out[j++] = c;
        } else out[j++] = in[i];
    }
    out[j] = 0;
    return (int)j;
}

int nd_bpe_load(nd_bpe *b, const char *vocab_path, const char *merges_path) {
    if (!b) return -1;
    memset(b, 0, sizeof *b);
    b->pad_id = b->bos_id = b->eos_id = b->unk_id = -1;

    FILE *fv = fopen(vocab_path, "r");
    if (!fv) { nd_log("bpe: open vocab fail %s", vocab_path); return -1; }
    char line[512];
    while (fgets(line, sizeof line, fv)) {
        /* strip newline */
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L || line[0] == '#') continue;
        /* find last tab */
        char *tab = strrchr(line, '\t');
        if (!tab) continue;
        *tab = 0;
        int id = atoi(tab + 1);
        if (id < 0 || id >= ND_BPE_MAX_VOCAB) continue;
        char tok[ND_BPE_MAX_TOKEN];
        unescape_tok(line, tok, sizeof tok);
        if (id >= b->vocab_size) b->vocab_size = id + 1;
        strncpy(b->tokens[id], tok, ND_BPE_MAX_TOKEN - 1);
        b->tokens[id][ND_BPE_MAX_TOKEN - 1] = 0;
        if (strcmp(tok, "<pad>") == 0) b->pad_id = id;
        else if (strcmp(tok, "<bos>") == 0) b->bos_id = id;
        else if (strcmp(tok, "<eos>") == 0) b->eos_id = id;
        else if (strcmp(tok, "<unk>") == 0) b->unk_id = id;
    }
    fclose(fv);

    FILE *fm = fopen(merges_path, "r");
    if (!fm) { nd_log("bpe: open merges fail %s", merges_path); return -1; }
    while (fgets(line, sizeof line, fm) && b->n_merges < ND_BPE_MAX_MERGES) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L || line[0] == '#') continue;
        char a[ND_BPE_MAX_TOKEN], bb[ND_BPE_MAX_TOKEN];
        if (sscanf(line, "%63s %63s", a, bb) != 2) continue;
        char ua[ND_BPE_MAX_TOKEN], ub[ND_BPE_MAX_TOKEN];
        unescape_tok(a, ua, sizeof ua);
        unescape_tok(bb, ub, sizeof ub);
        strncpy(b->merge_a[b->n_merges], ua, ND_BPE_MAX_TOKEN - 1);
        strncpy(b->merge_b[b->n_merges], ub, ND_BPE_MAX_TOKEN - 1);
        b->n_merges++;
    }
    fclose(fm);

    if (b->vocab_size < 4) return -1;
    if (b->pad_id < 0) b->pad_id = 0;
    if (b->bos_id < 0) b->bos_id = 1;
    if (b->eos_id < 0) b->eos_id = 2;
    if (b->unk_id < 0) b->unk_id = 3;
    return 0;
}

/* apply BPE merges in-place on list of token strings */
static void apply_merges(const nd_bpe *b, char toks[][ND_BPE_MAX_TOKEN], int *n) {
    for (int m = 0; m < b->n_merges; ++m) {
        int i = 0;
        while (i + 1 < *n) {
            if (strcmp(toks[i], b->merge_a[m]) == 0 &&
                strcmp(toks[i + 1], b->merge_b[m]) == 0) {
                char merged[ND_BPE_MAX_TOKEN];
                snprintf(merged, sizeof merged, "%s%s", toks[i], toks[i + 1]);
                strncpy(toks[i], merged, ND_BPE_MAX_TOKEN - 1);
                toks[i][ND_BPE_MAX_TOKEN - 1] = 0;
                for (int j = i + 1; j + 1 < *n; ++j)
                    memcpy(toks[j], toks[j + 1], ND_BPE_MAX_TOKEN);
                (*n)--;
                /* stay at i to chain merges of same pair */
            } else {
                ++i;
            }
        }
    }
}

int nd_bpe_encode(const nd_bpe *b, const char *text, int *out_ids, int max_out,
                  int add_specials) {
    if (!b || !text || !out_ids || max_out <= 0) return 0;
    int n_out = 0;
    if (add_specials && n_out < max_out) out_ids[n_out++] = b->bos_id;

    /* split into single-byte tokens (ASCII-first) */
    char toks[512][ND_BPE_MAX_TOKEN];
    int nt = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p && nt < 512; ++p) {
        toks[nt][0] = (char)*p;
        toks[nt][1] = 0;
        nt++;
    }
    apply_merges(b, toks, &nt);

    for (int i = 0; i < nt && n_out < max_out - (add_specials ? 1 : 0); ++i) {
        int id = nd_bpe_token_to_id(b, toks[i]);
        if (id < 0) id = b->unk_id;
        out_ids[n_out++] = id;
    }
    if (add_specials && n_out < max_out) out_ids[n_out++] = b->eos_id;
    return n_out;
}

int nd_bpe_decode(const nd_bpe *b, const int *ids, int n, char *buf, int buf_n) {
    if (!b || !ids || !buf || buf_n <= 0) return 0;
    int pos = 0;
    buf[0] = 0;
    for (int i = 0; i < n; ++i) {
        int id = ids[i];
        if (id < 0 || id >= b->vocab_size) continue;
        if (id == b->pad_id || id == b->bos_id || id == b->eos_id) continue;
        const char *t = b->tokens[id];
        int tl = (int)strlen(t);
        if (pos + tl >= buf_n) break;
        memcpy(buf + pos, t, (size_t)tl);
        pos += tl;
    }
    buf[pos] = 0;
    return pos;
}
