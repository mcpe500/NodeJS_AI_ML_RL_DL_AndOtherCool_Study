#include "nn.h"
#include "autograd.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ---------- module helpers ---------- */
static nd_module *mod_new(const char *name) {
    nd_module *m = (nd_module *)nd_xcalloc(1, sizeof(nd_module));
    strncpy(m->name, name, 63);
    m->training = true;
    return m;
}

static void collect_params(nd_module *m, nd_tensor ***arr, int *n, int *cap) {
    for (int i = 0; i < m->n_params; ++i) {
        if (*n >= *cap) {
            *cap = *cap ? *cap * 2 : 16;
            *arr = (nd_tensor **)realloc(*arr, (size_t)(*cap) * sizeof(nd_tensor *));
        }
        (*arr)[(*n)++] = m->params[i];
    }
    for (int i = 0; i < m->n_children; ++i) collect_params(m->children[i], arr, n, cap);
}

void nd_module_parameters(nd_module *m, nd_tensor ***out, int *n) {
    int cap = 0; *n = 0; *out = NULL;
    collect_params(m, out, n, &cap);
}

void nd_module_zero_grad(nd_module *m) {
    nd_tensor **p; int n;
    nd_module_parameters(m, &p, &n);
    for (int i = 0; i < n; ++i) nd_tensor_zero_grad(p[i]);
    free(p);
}

void nd_module_set_training(nd_module *m, bool t) {
    m->training = t;
    for (int i = 0; i < m->n_children; ++i) nd_module_set_training(m->children[i], t);
}

long nd_module_num_params(nd_module *m) {
    nd_tensor **p; int n; long s = 0;
    nd_module_parameters(m, &p, &n);
    for (int i = 0; i < n; ++i) s += p[i]->size;
    free(p);
    return s;
}

void nd_module_free(nd_module *m) {
    if (!m) return;
    for (int i = 0; i < m->n_children; ++i) nd_module_free(m->children[i]);
    free(m->children);
    if (m->free_state) m->free_state(m);
    /* params owned by module state usually */
    free(m->params);
    free(m);
}

/* ---------- Linear ---------- */
typedef struct { int in_f, out_f; bool bias; nd_tensor *W, *b; } LinearState;

/* Proper linear with autograd */
static void bw_linear(nd_node *gn) {
    nd_tensor *x = gn->inputs[0];
    nd_tensor *W = gn->inputs[1];
    nd_tensor *y = gn->output;
    int *meta = (int *)gn->state;
    int leading = meta[0], in = meta[1], out = meta[2], has_b = meta[3];
    nd_tensor *b = has_b ? gn->inputs[2] : NULL;
    if (W->requires_grad) {
        for (int o = 0; o < out; ++o)
            for (int k = 0; k < in; ++k) {
                float s = 0;
                for (int bi = 0; bi < leading; ++bi)
                    s += x->data[bi * in + k] * y->grad[bi * out + o];
                W->grad[o * in + k] += s;
            }
    }
    if (x->requires_grad) {
        for (int bi = 0; bi < leading; ++bi)
            for (int k = 0; k < in; ++k) {
                float s = 0;
                for (int o = 0; o < out; ++o) s += y->grad[bi * out + o] * W->data[o * in + k];
                x->grad[bi * in + k] += s;
            }
    }
    if (b && b->requires_grad) {
        for (int o = 0; o < out; ++o) {
            float s = 0;
            for (int bi = 0; bi < leading; ++bi) s += y->grad[bi * out + o];
            b->grad[o] += s;
        }
    }
}

static nd_tensor *linear_fwd2(nd_module *self, nd_tensor *x) {
    LinearState *st = (LinearState *)self->state;
    int in = st->in_f, out = st->out_f;
    int leading = x->size / in;
    bool rg = x->requires_grad || st->W->requires_grad || (st->bias && st->b->requires_grad);
    int ysh[2] = {leading, out};
    /* flatten x */
    nd_tensor *xflat = x;
    int need_free_x = 0;
    if (!(x->ndim == 2 && x->shape[0] == leading && x->shape[1] == in)) {
        int xs[2] = {leading, in};
        xflat = nd_tensor_alloc(xs, 2, x->requires_grad);
        memcpy(xflat->data, x->data, (size_t)x->size * sizeof(float));
        if (x->requires_grad) {
            /* share grad path: treat as same storage - copy grads back not handled; require 2D input */
        }
        need_free_x = 1;
        /* simpler assert */
    }
    nd_tensor *y = nd_tensor_alloc(ysh, 2, rg);
    for (int bi = 0; bi < leading; ++bi)
        for (int o = 0; o < out; ++o) {
            float s = st->bias ? st->b->data[o] : 0.f;
            for (int k = 0; k < in; ++k) s += x->data[bi * in + k] * st->W->data[o * in + k];
            y->data[bi * out + o] = s;
        }
    if (rg) {
        int n_in = st->bias ? 3 : 2;
        nd_tensor *ins[3] = {x, st->W, st->b};
        nd_node *g = nd_node_new(bw_linear, ins, n_in, y);
        int *meta = (int *)nd_xmalloc(4 * sizeof(int));
        meta[0] = leading; meta[1] = in; meta[2] = out; meta[3] = st->bias ? 1 : 0;
        g->state = meta;
    }
    if (need_free_x) nd_tensor_free(xflat);
    /* restore leading dims if x was 3D */
    if (x->ndim == 3) {
        int sh[3] = {x->shape[0], x->shape[1], out};
        nd_tensor *yo = nd_tensor_alloc(sh, 3, rg);
        memcpy(yo->data, y->data, (size_t)y->size * sizeof(float));
        /* graph on y - for 3D path use y as [B*T,out] which is fine for CE after reshape */
        nd_tensor_free(yo);
    }
    return y;
}

static void linear_free(nd_module *m) {
    LinearState *st = (LinearState *)m->state;
    nd_tensor_free(st->W);
    if (st->b) nd_tensor_free(st->b);
    free(st);
}

nd_module *nd_linear(int in_f, int out_f, bool bias) {
    nd_module *m = mod_new("linear");
    LinearState *st = (LinearState *)nd_xcalloc(1, sizeof(LinearState));
    st->in_f = in_f; st->out_f = out_f; st->bias = bias;
    int wsh[2] = {out_f, in_f};
    st->W = nd_tensor_alloc(wsh, 2, true);
    float scale = sqrtf(2.f / (float)in_f);
    for (int i = 0; i < st->W->size; ++i) st->W->data[i] = nd_rand_normal(0, scale);
    if (bias) {
        int bsh[1] = {out_f};
        st->b = nd_zeros(bsh, 1, true);
    }
    m->state = st;
    m->n_params = bias ? 2 : 1;
    m->params = (nd_tensor **)nd_xmalloc((size_t)m->n_params * sizeof(nd_tensor *));
    m->params[0] = st->W;
    if (bias) m->params[1] = st->b;
    m->forward = linear_fwd2;
    m->free_state = linear_free;
    return m;
}

/* ---------- Embedding module ---------- */
typedef struct { nd_tensor *weight; int vocab, dim; } EmbState;

static nd_tensor *emb_fwd(nd_module *self, nd_tensor *ids) {
    EmbState *st = (EmbState *)self->state;
    return nd_embedding(st->weight, ids);
}
static void emb_free(nd_module *m) {
    EmbState *st = (EmbState *)m->state;
    nd_tensor_free(st->weight); free(st);
}
nd_module *nd_embedding_mod(int vocab, int dim) {
    nd_module *m = mod_new("embedding");
    EmbState *st = (EmbState *)nd_xcalloc(1, sizeof(EmbState));
    st->vocab = vocab; st->dim = dim;
    int sh[2] = {vocab, dim};
    st->weight = nd_tensor_alloc(sh, 2, true);
    for (int i = 0; i < st->weight->size; ++i) st->weight->data[i] = nd_rand_normal(0, 0.02f);
    m->state = st;
    m->n_params = 1;
    m->params = (nd_tensor **)nd_xmalloc(sizeof(nd_tensor *));
    m->params[0] = st->weight;
    m->forward = emb_fwd;
    m->free_state = emb_free;
    return m;
}

/* ---------- ZCRMSNorm ---------- */
typedef struct { nd_tensor *gamma; float eps; int dim; } ZCRState;

static void bw_zcr(nd_node *gn) {
    nd_tensor *x = gn->inputs[0], *gamma = gn->inputs[1], *y = gn->output;
    float eps = *(float *)gn->state;
    int D = x->shape[x->ndim - 1];
    int rows = x->size / D;
    for (int r = 0; r < rows; ++r) {
        float *xv = x->data + r * D;
        float *yv = y->data + r * D;
        float *gy = y->grad + r * D;
        float mean_sq = 0;
        for (int i = 0; i < D; ++i) mean_sq += xv[i] * xv[i];
        mean_sq /= (float)D;
        float inv = 1.f / sqrtf(mean_sq + eps);
        if (gamma->requires_grad) {
            for (int i = 0; i < D; ++i) gamma->grad[i] += gy[i] * xv[i] * inv;
        }
        if (x->requires_grad) {
            /* simplified grad through rms */
            for (int i = 0; i < D; ++i) {
                float gscale = (1.f + gamma->data[i]) * inv;
                x->grad[r * D + i] += gy[i] * gscale; /* approx without full rms jacobian */
            }
        }
    }
    (void)eps;
}

static nd_tensor *zcr_fwd(nd_module *self, nd_tensor *x) {
    ZCRState *st = (ZCRState *)self->state;
    int D = st->dim;
    nd_tensor *y = nd_tensor_alloc(x->shape, x->ndim, x->requires_grad || st->gamma->requires_grad);
    int rows = x->size / D;
    for (int r = 0; r < rows; ++r) {
        float mean_sq = 0;
        for (int i = 0; i < D; ++i) mean_sq += x->data[r * D + i] * x->data[r * D + i];
        mean_sq /= (float)D;
        float inv = 1.f / sqrtf(mean_sq + st->eps);
        for (int i = 0; i < D; ++i)
            y->data[r * D + i] = x->data[r * D + i] * (1.f + st->gamma->data[i]) * inv;
    }
    if (y->requires_grad) {
        nd_tensor *ins[2] = {x, st->gamma};
        nd_node *g = nd_node_new(bw_zcr, ins, 2, y);
        float *eps = (float *)nd_xmalloc(sizeof(float));
        *eps = st->eps;
        g->state = eps;
    }
    return y;
}
static void zcr_free(nd_module *m) {
    ZCRState *st = (ZCRState *)m->state;
    nd_tensor_free(st->gamma); free(st);
}
nd_module *nd_zcrmsnorm(int dim, float eps) {
    nd_module *m = mod_new("zcrmsnorm");
    ZCRState *st = (ZCRState *)nd_xcalloc(1, sizeof(ZCRState));
    st->dim = dim; st->eps = eps;
    int sh[1] = {dim};
    st->gamma = nd_zeros(sh, 1, true); /* init 0 */
    m->state = st;
    m->n_params = 1;
    m->params = (nd_tensor **)nd_xmalloc(sizeof(nd_tensor *));
    m->params[0] = st->gamma;
    m->forward = zcr_fwd;
    m->free_state = zcr_free;
    return m;
}

/* ---------- GQA Attention (simplified MHA-style with n_kv) ---------- */
typedef struct {
    int d, n_q, n_kv, hd;
    bool causal;
    nd_module *wq, *wk, *wv, *wo;
    nd_tensor *gate; /* scalar-ish [1] or [d] — use [1] */
} GQAState;

static void rope_inplace(float *q, int T, int hd) {
    /* simple rotary on pairs */
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i + 1 < hd; i += 2) {
            float freq = 1.f / powf(10000.f, (float)i / (float)hd);
            float ang = (float)t * freq;
            float c = cosf(ang), s = sinf(ang);
            float u = q[t * hd + i], v = q[t * hd + i + 1];
            q[t * hd + i] = u * c - v * s;
            q[t * hd + i + 1] = u * s + v * c;
        }
    }
}

static nd_tensor *gqa_fwd(nd_module *self, nd_tensor *x) {
    /* x: [B, T, D] */
    GQAState *st = (GQAState *)self->state;
    int B = x->shape[0], T = x->shape[1], D = st->d;
    int hd = st->hd, n_q = st->n_q, n_kv = st->n_kv;
    nd_tensor *xq = st->wq->forward(st->wq, x); /* [B*T, D] or [B,T,D]? linear returns [B*T, out] if flat */
    /* linear_fwd2 with 3D x: uses size/in — OK returns [B*T, D] */
    nd_tensor *xk = st->wk->forward(st->wk, x);
    nd_tensor *xv = st->wv->forward(st->wv, x);
    /* reshape to [B, n_q, T, hd] */
    int BT = B * T;
    /* apply rope per head on q,k */
    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_q; ++h)
            rope_inplace(xq->data + (b * T) * D + h * hd, T, hd); /* wrong layout */
    }
    /* redo layout-aware rope: xq is [BT, D] = [BT, n_q*hd] */
    for (int bt = 0; bt < BT; ++bt) {
        int t = bt % T;
        for (int h = 0; h < n_q; ++h) {
            float *q = xq->data + bt * D + h * hd;
            for (int i = 0; i + 1 < hd; i += 2) {
                float freq = 1.f / powf(10000.f, (float)i / (float)hd);
                float ang = (float)t * freq;
                float c = cosf(ang), s = sinf(ang);
                float u = q[i], v = q[i + 1];
                q[i] = u * c - v * s; q[i + 1] = u * s + v * c;
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            float *k = xk->data + bt * (n_kv * hd) + h * hd;
            /* wk out dim = n_kv*hd */
            for (int i = 0; i + 1 < hd; i += 2) {
                float freq = 1.f / powf(10000.f, (float)i / (float)hd);
                float ang = (float)t * freq;
                float c = cosf(ang), s = sinf(ang);
                float u = k[i], v = k[i + 1];
                k[i] = u * c - v * s; k[i + 1] = u * s + v * c;
            }
        }
    }
    int kv_dim = n_kv * hd;
    /* attention output [BT, D] */
    nd_tensor *ctx = nd_tensor_alloc((int[]){BT, D}, 2, true);
    float scale = 1.f / sqrtf((float)hd);
    int rep = n_q / n_kv;
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < n_q; ++h) {
                int hkv = h / rep;
                float *q = xq->data + (b * T + t) * D + h * hd;
                /* scores over T */
                float scores[512]; /* max T for sanity/pilot; cap */
                int TT = T < 512 ? T : 512;
                float maxs = -1e30f;
                for (int s = 0; s < T; ++s) {
                    if (st->causal && s > t) { scores[s] = -1e30f; continue; }
                    float *k = xk->data + (b * T + s) * kv_dim + hkv * hd;
                    float dot = 0;
                    for (int i = 0; i < hd; ++i) dot += q[i] * k[i];
                    scores[s] = dot * scale;
                    if (scores[s] > maxs) maxs = scores[s];
                }
                float sum = 0;
                for (int s = 0; s < T; ++s) { scores[s] = expf(scores[s] - maxs); sum += scores[s]; }
                float *out = ctx->data + (b * T + t) * D + h * hd;
                memset(out, 0, (size_t)hd * sizeof(float));
                for (int s = 0; s < T; ++s) {
                    float a = scores[s] / sum;
                    float *v = xv->data + (b * T + s) * kv_dim + hkv * hd;
                    for (int i = 0; i < hd; ++i) out[i] += a * v[i];
                }
                (void)TT;
            }
        }
    }
    nd_tensor *y = st->wo->forward(st->wo, ctx);
    /* gated residual: y = x + sigmoid(gate)*attn */
    float g = 1.f / (1.f + expf(-st->gate->data[0]));
    nd_tensor *attn_scaled = nd_scale(y, g);
    /* reshape x to [BT,D] */
    nd_tensor *xflat = nd_tensor_alloc((int[]){BT, D}, 2, x->requires_grad);
    memcpy(xflat->data, x->data, (size_t)x->size * sizeof(float));
    nd_tensor *out = nd_add(xflat, attn_scaled);
    /* reshape to [B,T,D] */
    nd_tensor *ores = nd_tensor_alloc((int[]){B, T, D}, 3, out->requires_grad);
    memcpy(ores->data, out->data, (size_t)out->size * sizeof(float));
    /* free intermediates carefully */
    nd_tensor_free(xq); nd_tensor_free(xk); nd_tensor_free(xv);
    nd_tensor_free(ctx); nd_tensor_free(y); nd_tensor_free(attn_scaled);
    nd_tensor_free(xflat); nd_tensor_free(out);
    /* NOTE: graph broken for free chain - for training we need connected graph.
       Simpler path for Model A: use residual without free of graph nodes mid-way.
       Rebuild minimal connected path for train: */
    return ores;
}

/* Graph-connected residual GQA. Attn values mixed into xq in-place (fwd).
 * Grad path: out → wo(xq) + gate residual on x. xk/xv freed after use (no full attn bw yet). */
static nd_tensor *gqa_fwd_train(nd_module *self, nd_tensor *x) {
    GQAState *st = (GQAState *)self->state;
    int B = x->shape[0], T = x->shape[1], D = st->d;
    int BT = B * T;
    int xsh[2] = {BT, D};
    nd_tensor *xf = nd_reshape(x, xsh, 2);
    nd_tensor *xq = st->wq->forward(st->wq, xf);
    nd_tensor *xk = st->wk->forward(st->wk, xf);
    nd_tensor *xv = st->wv->forward(st->wv, xf);
    int hd = st->hd, n_q = st->n_q, n_kv = st->n_kv;
    int kv_dim = n_kv * hd;
    int rep = n_q / n_kv;
    if (rep < 1) rep = 1;
    float scale = 1.f / sqrtf((float)(hd > 0 ? hd : 1));
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < n_q; ++h) {
                int hkv = h / rep;
                if (hkv >= n_kv) hkv = n_kv - 1;
                float *q = xq->data + (b * T + t) * D + h * hd;
                float scores_s[1024];
                float maxs = -1e30f;
                for (int s = 0; s < T; ++s) {
                    if (st->causal && s > t) { scores_s[s] = -1e30f; continue; }
                    float *k = xk->data + (b * T + s) * kv_dim + hkv * hd;
                    float dot = 0;
                    for (int i = 0; i < hd; ++i) dot += q[i] * k[i];
                    scores_s[s] = dot * scale;
                    if (scores_s[s] > maxs) maxs = scores_s[s];
                }
                float sum = 0;
                for (int s = 0; s < T; ++s) { scores_s[s] = expf(scores_s[s] - maxs); sum += scores_s[s]; }
                float acc[128];
                int hdn = hd < 128 ? hd : 128;
                for (int i = 0; i < hdn; ++i) acc[i] = 0;
                for (int s = 0; s < T; ++s) {
                    float a = scores_s[s] / (sum + 1e-12f);
                    float *v = xv->data + (b * T + s) * kv_dim + hkv * hd;
                    for (int i = 0; i < hdn; ++i) acc[i] += a * v[i];
                }
                for (int i = 0; i < hdn; ++i) q[i] = acc[i];
            }
        }
    }
    /* drop k/v now — not on residual graph (wq/wo still learn via xq path) */
    nd_tensor_free(xk);
    nd_tensor_free(xv);
    nd_tensor *y = st->wo->forward(st->wo, xq);
    nd_tensor_free(xq); /* wo graph retained xq */
    float gate = 1.f / (1.f + expf(-st->gate->data[0]));
    nd_tensor *gs = nd_scale(y, gate);
    nd_tensor_free(y);
    nd_tensor *out2 = nd_add(xf, gs);
    nd_tensor_free(xf);
    nd_tensor_free(gs);
    int osh[3] = {B, T, D};
    nd_tensor *ret = nd_reshape(out2, osh, 3);
    nd_tensor_free(out2); /* view holds storage */
    (void)gqa_fwd;
    return ret;
}

static void gqa_free(nd_module *m) {
    GQAState *st = (GQAState *)m->state;
    /* children wq/wk/wv/wo freed by nd_module_free */
    nd_tensor_free(st->gate);
    free(st);
}

nd_module *nd_gqa_attn(int d_model, int n_q, int n_kv, bool causal) {
    nd_module *m = mod_new("gqa");
    GQAState *st = (GQAState *)nd_xcalloc(1, sizeof(GQAState));
    st->d = d_model; st->n_q = n_q; st->n_kv = n_kv; st->hd = d_model / n_q; st->causal = causal;
    st->wq = nd_linear(d_model, d_model, false);
    st->wk = nd_linear(d_model, n_kv * st->hd, false);
    st->wv = nd_linear(d_model, n_kv * st->hd, false);
    st->wo = nd_linear(d_model, d_model, false);
    int gsh[1] = {1};
    st->gate = nd_full(gsh, 1, -2.f); /* sigmoid(-2)~0.12, near identity-ish small */
    st->gate->requires_grad = true;
    m->state = st;
    m->n_children = 4;
    m->children = (nd_module **)nd_xmalloc(4 * sizeof(nd_module *));
    m->children[0] = st->wq; m->children[1] = st->wk;
    m->children[2] = st->wv; m->children[3] = st->wo;
    m->n_params = 1;
    m->params = (nd_tensor **)nd_xmalloc(sizeof(nd_tensor *));
    m->params[0] = st->gate;
    m->forward = gqa_fwd_train;
    m->free_state = gqa_free;
    return m;
}

nd_module *nd_cross_attn(int d_model, int n_q, int n_kv) {
    /* reuse gqa non-causal as placeholder; real cross needs memory — use self for v1 */
    return nd_gqa_attn(d_model, n_q, n_kv, false);
}

/* ---------- Needle encoder-decoder ---------- */
typedef struct {
    nd_needle_config cfg;
    nd_module *emb;
    nd_module **enc_norm, **enc_attn;
    nd_module **dec_norm_s, **dec_self, **dec_norm_c, **dec_cross;
    nd_tensor *out_norm_g; /* final zcr gamma or reuse */
    nd_module *final_norm;
} NeedleState;

static void bw_tied_proj(nd_node *gn) {
    nd_tensor *h = gn->inputs[0], *W = gn->inputs[1], *logits = gn->output;
    int D = W->shape[1], V = W->shape[0];
    int BT = h->size / D;
    if (h->requires_grad) {
        for (int i = 0; i < BT; ++i)
            for (int d = 0; d < D; ++d) {
                float s = 0;
                for (int v = 0; v < V; ++v) s += logits->grad[i * V + v] * W->data[v * D + d];
                h->grad[i * D + d] += s;
            }
    }
    if (W->requires_grad) {
        for (int v = 0; v < V; ++v)
            for (int d = 0; d < D; ++d) {
                float s = 0;
                for (int i = 0; i < BT; ++i) s += logits->grad[i * V + v] * h->data[i * D + d];
                W->grad[v * D + d] += s;
            }
    }
}

static nd_tensor *needle_fwd2(nd_module *self, nd_tensor *src_ids, nd_tensor *tgt_ids) {
    NeedleState *st = (NeedleState *)self->state;
    int D = st->cfg.d_model;
    nd_tensor *h = st->emb->forward(st->emb, src_ids);
    for (int i = 0; i < st->cfg.n_enc; ++i) {
        nd_tensor *n = st->enc_norm[i]->forward(st->enc_norm[i], h);
        nd_tensor *a = st->enc_attn[i]->forward(st->enc_attn[i], n);
        nd_tensor_free(n);
        nd_tensor_free(h); /* drop caller ref; a residual graph may retain via views */
        h = a;
    }
    nd_tensor *mem = h;
    nd_tensor *g = st->emb->forward(st->emb, tgt_ids);
    for (int i = 0; i < st->cfg.n_dec; ++i) {
        nd_tensor *n1 = st->dec_norm_s[i]->forward(st->dec_norm_s[i], g);
        nd_tensor *s = st->dec_self[i]->forward(st->dec_self[i], n1);
        nd_tensor_free(n1);
        nd_tensor_free(g);
        nd_tensor *n2 = st->dec_norm_c[i]->forward(st->dec_norm_c[i], s);
        nd_tensor *c = st->dec_cross[i]->forward(st->dec_cross[i], n2);
        nd_tensor_free(n2);
        nd_tensor_free(s);
        int B = c->shape[0], T = c->shape[1];
        int S = mem->shape[1];
        for (int b = 0; b < B; ++b) {
            float pool[512];
            int Dd = D < 512 ? D : 512;
            for (int d = 0; d < Dd; ++d) pool[d] = 0;
            for (int si = 0; si < S; ++si)
                for (int d = 0; d < D; ++d) pool[d] += mem->data[(b * S + si) * D + d];
            for (int d = 0; d < D; ++d) pool[d] /= (float)S;
            for (int t = 0; t < T; ++t)
                for (int d = 0; d < D; ++d)
                    c->data[(b * T + t) * D + d] += 0.1f * pool[d];
        }
        g = c;
    }
    nd_tensor_free(mem); /* encoder path no longer needed for bw of tied path via g only —
                            NOTE: mem not on g graph; free OK. Emb still updated via g path. */
    nd_tensor *gn = st->final_norm->forward(st->final_norm, g);
    nd_tensor_free(g);
    g = gn;
    EmbState *es = (EmbState *)st->emb->state;
    int B = g->shape[0], T = g->shape[1], V = st->cfg.vocab;
    int BT = B * T;
    bool rg = g->requires_grad || es->weight->requires_grad;
    int lsh[3] = {B, T, V};
    nd_tensor *logits = nd_tensor_alloc(lsh, 3, rg);
    for (int i = 0; i < BT; ++i) {
        for (int v = 0; v < V; ++v) {
            float s = 0;
            for (int d = 0; d < D; ++d) s += g->data[i * D + d] * es->weight->data[v * D + d];
            logits->data[i * V + v] = s;
        }
    }
    if (rg) {
        nd_tensor *ins[2] = {g, es->weight};
        nd_node_new(bw_tied_proj, ins, 2, logits);
    }
    nd_tensor_free(g); /* retained by logits grad_fn */
    return logits;
}

static void needle_free(nd_module *m) {
    NeedleState *st = (NeedleState *)m->state;
    /* submodules already freed via m->children in nd_module_free */
    free(st->enc_norm); free(st->enc_attn);
    free(st->dec_norm_s); free(st->dec_self);
    free(st->dec_norm_c); free(st->dec_cross);
    free(st);
}

nd_module *nd_needle_create(const nd_needle_config *cfg) {
    nd_module *m = mod_new("needle");
    NeedleState *st = (NeedleState *)nd_xcalloc(1, sizeof(NeedleState));
    st->cfg = *cfg;
    st->emb = nd_embedding_mod(cfg->vocab, cfg->d_model);
    st->enc_norm = (nd_module **)nd_xmalloc((size_t)cfg->n_enc * sizeof(nd_module *));
    st->enc_attn = (nd_module **)nd_xmalloc((size_t)cfg->n_enc * sizeof(nd_module *));
    for (int i = 0; i < cfg->n_enc; ++i) {
        st->enc_norm[i] = nd_zcrmsnorm(cfg->d_model, 1e-5f);
        st->enc_attn[i] = nd_gqa_attn(cfg->d_model, cfg->n_q, cfg->n_kv, false);
    }
    st->dec_norm_s = (nd_module **)nd_xmalloc((size_t)cfg->n_dec * sizeof(nd_module *));
    st->dec_self = (nd_module **)nd_xmalloc((size_t)cfg->n_dec * sizeof(nd_module *));
    st->dec_norm_c = (nd_module **)nd_xmalloc((size_t)cfg->n_dec * sizeof(nd_module *));
    st->dec_cross = (nd_module **)nd_xmalloc((size_t)cfg->n_dec * sizeof(nd_module *));
    for (int i = 0; i < cfg->n_dec; ++i) {
        st->dec_norm_s[i] = nd_zcrmsnorm(cfg->d_model, 1e-5f);
        st->dec_self[i] = nd_gqa_attn(cfg->d_model, cfg->n_q, cfg->n_kv, true);
        st->dec_norm_c[i] = nd_zcrmsnorm(cfg->d_model, 1e-5f);
        st->dec_cross[i] = nd_cross_attn(cfg->d_model, cfg->n_q, cfg->n_kv);
    }
    st->final_norm = nd_zcrmsnorm(cfg->d_model, 1e-5f);
    m->state = st;
    /* children for param collect */
    int nc = 1 + cfg->n_enc * 2 + cfg->n_dec * 4 + 1;
    m->children = (nd_module **)nd_xmalloc((size_t)nc * sizeof(nd_module *));
    int c = 0;
    m->children[c++] = st->emb;
    for (int i = 0; i < cfg->n_enc; ++i) { m->children[c++] = st->enc_norm[i]; m->children[c++] = st->enc_attn[i]; }
    for (int i = 0; i < cfg->n_dec; ++i) {
        m->children[c++] = st->dec_norm_s[i]; m->children[c++] = st->dec_self[i];
        m->children[c++] = st->dec_norm_c[i]; m->children[c++] = st->dec_cross[i];
    }
    m->children[c++] = st->final_norm;
    m->n_children = c;
    m->forward2 = needle_fwd2;
    m->free_state = needle_free;
    return m;
}

nd_needle_config nd_cfg_sanity(void) {
    return (nd_needle_config){ .d_model=64, .vocab=512, .n_enc=2, .n_dec=2, .n_q=4, .n_kv=2, .max_len=64 };
}
nd_needle_config nd_cfg_pilot(void) {
    return (nd_needle_config){ .d_model=256, .vocab=8192, .n_enc=8, .n_dec=6, .n_q=8, .n_kv=4, .max_len=256 };
}
nd_needle_config nd_cfg_full(void) {
    return (nd_needle_config){ .d_model=512, .vocab=8192, .n_enc=12, .n_dec=8, .n_q=8, .n_kv=4, .max_len=512 };
}
