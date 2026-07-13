/* optim.h */
#ifndef ND_OPTIM_H
#define ND_OPTIM_H

#include "tensor.h"

typedef struct nd_optim {
    nd_tensor **params;
    int         n_params;
    float       lr;
    void       *state;
    void      (*step)(struct nd_optim *o);
    void      (*zero_grad)(struct nd_optim *o);
    void      (*free_state)(struct nd_optim *o);
} nd_optim;

nd_optim *nd_adamw(nd_tensor **params, int n, float lr, float b1, float b2, float wd, float eps);
nd_optim *nd_sgd(nd_tensor **params, int n, float lr, float momentum, float wd);
void      nd_optim_free(nd_optim *o);

#endif
