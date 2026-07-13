/* nn.h — modules */
#ifndef ND_NN_H
#define ND_NN_H

#include "tensor.h"

typedef struct nd_module nd_module;

struct nd_module {
    nd_tensor **params;
    int         n_params;
    nd_module **children;
    int         n_children;
    bool        training;
    char        name[64];
    void       *state;
    nd_tensor *(*forward)(nd_module *self, nd_tensor *x);
    /* multi-arg forward for enc-dec: src_ids, tgt_ids */
    nd_tensor *(*forward2)(nd_module *self, nd_tensor *a, nd_tensor *b);
    void       (*free_state)(nd_module *self);
};

nd_module *nd_linear(int in_f, int out_f, bool bias);
nd_module *nd_embedding_mod(int vocab, int dim);
nd_module *nd_zcrmsnorm(int dim, float eps);
/* GQA self-attn; causal=1 for decoder self */
nd_module *nd_gqa_attn(int d_model, int n_q, int n_kv, bool causal);
nd_module *nd_cross_attn(int d_model, int n_q, int n_kv);
/* full needle model */
typedef struct {
    int d_model, vocab, n_enc, n_dec, n_q, n_kv, max_len;
} nd_needle_config;

nd_module *nd_needle_create(const nd_needle_config *cfg);
nd_needle_config nd_cfg_sanity(void);
nd_needle_config nd_cfg_pilot(void);
nd_needle_config nd_cfg_full(void);

void nd_module_parameters(nd_module *m, nd_tensor ***out, int *n);
void nd_module_zero_grad(nd_module *m);
void nd_module_set_training(nd_module *m, bool t);
void nd_module_free(nd_module *m);
long nd_module_num_params(nd_module *m);

#endif
