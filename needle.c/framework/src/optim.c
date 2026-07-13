#include "optim.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

typedef struct {
    float *m, *v;
    int t;
    float b1, b2, wd, eps;
    int n_elems; /* total floats across params — we store per-param separately */
    float **pm, **pv;
} AdamWState;

static void adamw_zero(nd_optim *o) {
    for (int i = 0; i < o->n_params; ++i) nd_tensor_zero_grad(o->params[i]);
}

static void adamw_step(nd_optim *o) {
    AdamWState *st = (AdamWState *)o->state;
    st->t++;
    float b1t = 1.f - powf(st->b1, (float)st->t);
    float b2t = 1.f - powf(st->b2, (float)st->t);
    for (int p = 0; p < o->n_params; ++p) {
        nd_tensor *t = o->params[p];
        float *m = st->pm[p], *v = st->pv[p];
        for (int i = 0; i < t->size; ++i) {
            float g = t->grad[i];
            if (st->wd != 0.f) t->data[i] -= o->lr * st->wd * t->data[i];
            m[i] = st->b1 * m[i] + (1.f - st->b1) * g;
            v[i] = st->b2 * v[i] + (1.f - st->b2) * g * g;
            float mh = m[i] / b1t;
            float vh = v[i] / b2t;
            t->data[i] -= o->lr * mh / (sqrtf(vh) + st->eps);
        }
    }
}

static void adamw_free(nd_optim *o) {
    AdamWState *st = (AdamWState *)o->state;
    for (int p = 0; p < o->n_params; ++p) { free(st->pm[p]); free(st->pv[p]); }
    free(st->pm); free(st->pv); free(st);
}

nd_optim *nd_adamw(nd_tensor **params, int n, float lr, float b1, float b2, float wd, float eps) {
    nd_optim *o = (nd_optim *)nd_xcalloc(1, sizeof(nd_optim));
    o->params = params; o->n_params = n; o->lr = lr;
    o->step = adamw_step; o->zero_grad = adamw_zero; o->free_state = adamw_free;
    AdamWState *st = (AdamWState *)nd_xcalloc(1, sizeof(AdamWState));
    st->b1 = b1; st->b2 = b2; st->wd = wd; st->eps = eps;
    st->pm = (float **)nd_xmalloc((size_t)n * sizeof(float *));
    st->pv = (float **)nd_xmalloc((size_t)n * sizeof(float *));
    for (int p = 0; p < n; ++p) {
        st->pm[p] = (float *)nd_xcalloc((size_t)params[p]->size, sizeof(float));
        st->pv[p] = (float *)nd_xcalloc((size_t)params[p]->size, sizeof(float));
    }
    o->state = st;
    return o;
}

typedef struct { float momentum, wd; float **v; } SGDState;

static void sgd_zero(nd_optim *o) {
    for (int i = 0; i < o->n_params; ++i) nd_tensor_zero_grad(o->params[i]);
}
static void sgd_step(nd_optim *o) {
    SGDState *st = (SGDState *)o->state;
    for (int p = 0; p < o->n_params; ++p) {
        nd_tensor *t = o->params[p];
        for (int i = 0; i < t->size; ++i) {
            float g = t->grad[i] + st->wd * t->data[i];
            st->v[p][i] = st->momentum * st->v[p][i] + g;
            t->data[i] -= o->lr * st->v[p][i];
        }
    }
}
static void sgd_free(nd_optim *o) {
    SGDState *st = (SGDState *)o->state;
    for (int p = 0; p < o->n_params; ++p) free(st->v[p]);
    free(st->v); free(st);
}
nd_optim *nd_sgd(nd_tensor **params, int n, float lr, float momentum, float wd) {
    nd_optim *o = (nd_optim *)nd_xcalloc(1, sizeof(nd_optim));
    o->params = params; o->n_params = n; o->lr = lr;
    o->step = sgd_step; o->zero_grad = sgd_zero; o->free_state = sgd_free;
    SGDState *st = (SGDState *)nd_xcalloc(1, sizeof(SGDState));
    st->momentum = momentum; st->wd = wd;
    st->v = (float **)nd_xmalloc((size_t)n * sizeof(float *));
    for (int p = 0; p < n; ++p) st->v[p] = (float *)nd_xcalloc((size_t)params[p]->size, sizeof(float));
    o->state = st;
    return o;
}

void nd_optim_free(nd_optim *o) {
    if (!o) return;
    if (o->free_state) o->free_state(o);
    free(o);
}
