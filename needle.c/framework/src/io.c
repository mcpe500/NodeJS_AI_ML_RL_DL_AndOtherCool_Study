#include "io.h"
#include "util.h"
#include <stdio.h>
#include <string.h>

int nd_checkpoint_save(const char *path, nd_module *m) {
    nd_tensor **params; int n;
    nd_module_parameters(m, &params, &n);
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    fwrite("NDCKP001", 1, 8, fp);
    fwrite(&n, sizeof(int), 1, fp);
    for (int i = 0; i < n; ++i) {
        nd_tensor *t = params[i];
        fwrite(&t->ndim, sizeof(int), 1, fp);
        fwrite(t->shape, sizeof(int), (size_t)t->ndim, fp);
        fwrite(t->data, sizeof(float), (size_t)t->size, fp);
    }
    free(params);
    fclose(fp);
    return 0;
}

int nd_checkpoint_load(const char *path, nd_module *m) {
    nd_tensor **params; int n;
    nd_module_parameters(m, &params, &n);
    FILE *fp = fopen(path, "rb");
    if (!fp) { free(params); return -1; }
    char mag[8];
    if (fread(mag, 1, 8, fp) != 8) { fclose(fp); free(params); return -1; }
    int nn;
    if (fread(&nn, sizeof(int), 1, fp) != 1 || nn != n) { fclose(fp); free(params); return -1; }
    for (int i = 0; i < n; ++i) {
        int ndim; fread(&ndim, sizeof(int), 1, fp);
        int shape[8]; fread(shape, sizeof(int), (size_t)ndim, fp);
        int sz = 1; for (int k = 0; k < ndim; ++k) sz *= shape[k];
        if (sz != params[i]->size) { fclose(fp); free(params); return -1; }
        fread(params[i]->data, sizeof(float), (size_t)sz, fp);
    }
    free(params);
    fclose(fp);
    return 0;
}
