#include "autograd.h"
#include "util.h"
#include <stdlib.h>
#include <string.h>

nd_node *nd_node_new(void (*bw)(nd_node *), nd_tensor **ins, int n, nd_tensor *out) {
    nd_node *g = (nd_node *)nd_xcalloc(1, sizeof(nd_node));
    if (!g) return NULL;
    g->backward = bw;
    g->n_inputs = n;
    g->inputs = (nd_tensor **)nd_xmalloc((size_t)n * sizeof(nd_tensor *));
    g->prev = (nd_node **)nd_xcalloc((size_t)n, sizeof(nd_node *));
    g->n_prev = 0;
    for (int i = 0; i < n; ++i) {
        g->inputs[i] = ins[i];
        nd_tensor_retain(ins[i]);
        if (ins[i]->grad_fn) g->prev[g->n_prev++] = ins[i]->grad_fn;
    }
    g->output = out;
    out->grad_fn = g;
    return g;
}

typedef struct {
    nd_node **data;
    int n, cap;
} Stack;

static void push(Stack *s, nd_node *g) {
    if (s->n == s->cap) {
        s->cap = s->cap ? s->cap * 2 : 16;
        s->data = (nd_node **)realloc(s->data, (size_t)s->cap * sizeof(nd_node *));
    }
    s->data[s->n++] = g;
}

static void topo(nd_node *g, Stack *o) {
    if (!g || g->visited) return;
    g->visited = true;
    for (int i = 0; i < g->n_prev; ++i) topo(g->prev[i], o);
    push(o, g);
}

void nd_autograd_backward(nd_tensor *loss) {
    if (!loss) return;
    for (int i = 0; i < loss->size; ++i) loss->grad[i] = 1.f;
    Stack order = {0};
    topo(loss->grad_fn, &order);
    for (int i = order.n - 1; i >= 0; --i) {
        nd_node *g = order.data[i];
        if (g->backward) g->backward(g);
        g->visited = false;
    }
    free(order.data);
}

void nd_backward(nd_tensor *loss) {
    nd_autograd_backward(loss);
}
