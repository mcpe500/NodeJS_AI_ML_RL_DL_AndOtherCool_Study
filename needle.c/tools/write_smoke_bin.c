/* write tiny streamable smoke dataset for Model A */
#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "data/smoke.bin";
    int n = argc > 2 ? atoi(argv[2]) : 64;
    int max_src = 8, max_tgt = 8, vocab = 64;
    int *src = calloc((size_t)n * max_src, sizeof(int));
    int *tgt = calloc((size_t)n * max_tgt, sizeof(int));
    int *lab = calloc((size_t)n * max_tgt, sizeof(int));
    int *sl = calloc((size_t)n, sizeof(int));
    int *tl = calloc((size_t)n, sizeof(int));
    if (!src || !tgt || !lab || !sl || !tl) return 1;
    for (int i = 0; i < n; ++i) {
        sl[i] = 4; tl[i] = 4;
        for (int j = 0; j < 4; ++j) {
            int a = (i + j + 1) % vocab;
            src[i * max_src + j] = a;
            tgt[i * max_tgt + j] = (a + 1) % vocab;
            lab[i * max_tgt + j] = (a + 2) % vocab;
        }
    }
    int rc = nd_dataset_write(path, n, max_src, max_tgt, src, sl, tgt, tl, lab);
    free(src); free(tgt); free(lab); free(sl); free(tl);
    if (rc != 0) { fprintf(stderr, "write fail %s\n", path); return 1; }
    printf("wrote %s n=%d\n", path, n);
    return 0;
}
