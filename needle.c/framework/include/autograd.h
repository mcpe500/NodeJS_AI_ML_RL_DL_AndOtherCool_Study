/* autograd.h */
#ifndef ND_AUTOGRAD_H
#define ND_AUTOGRAD_H

#include "tensor.h"

struct nd_node {
    void       (*backward)(struct nd_node *self);
    nd_tensor **inputs;
    int         n_inputs;
    nd_tensor  *output;
    struct nd_node **prev;
    int         n_prev;
    bool        visited;
    void       *state;
};

void nd_autograd_backward(nd_tensor *loss);

/* helpers for ops */
nd_node *nd_node_new(void (*bw)(nd_node *), nd_tensor **ins, int n, nd_tensor *out);

#endif
