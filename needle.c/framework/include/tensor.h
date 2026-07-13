/* tensor.h — nd_tensor + ops */
#ifndef ND_TENSOR_H
#define ND_TENSOR_H

#include <stdbool.h>
#include <stdint.h>

#define ND_MAX_DIMS 8

typedef struct nd_node nd_node;

typedef struct nd_tensor {
    float     *data;
    float     *grad;
    int        shape[ND_MAX_DIMS];
    int        strides[ND_MAX_DIMS];
    int        ndim;
    int        size;
    bool       requires_grad;
    bool       owns_data;
    struct nd_tensor *storage;
    nd_node   *grad_fn;
    int        refcount;
} nd_tensor;

nd_tensor *nd_tensor_alloc(const int *shape, int ndim, bool requires_grad);
nd_tensor *nd_zeros(const int *shape, int ndim, bool requires_grad);
nd_tensor *nd_ones(const int *shape, int ndim, bool requires_grad);
nd_tensor *nd_randn(const int *shape, int ndim, float mean, float std);
nd_tensor *nd_from_arr(const int *shape, int ndim, const float *data);
nd_tensor *nd_full(const int *shape, int ndim, float value);

void       nd_tensor_retain(nd_tensor *t);
void       nd_tensor_free(nd_tensor *t);
void       nd_tensor_zero_grad(nd_tensor *t);
nd_tensor *nd_tensor_detach(nd_tensor *t);

nd_tensor *nd_view(nd_tensor *t, const int *shape, int ndim);
nd_tensor *nd_reshape(nd_tensor *t, const int *shape, int ndim);
nd_tensor *nd_transpose(nd_tensor *t, int d0, int d1);

nd_tensor *nd_add(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_sub(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_mul(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_div(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_neg(nd_tensor *a);
nd_tensor *nd_scale(nd_tensor *a, float s);

nd_tensor *nd_relu(nd_tensor *a);
nd_tensor *nd_sigmoid(nd_tensor *a);
nd_tensor *nd_tanh_op(nd_tensor *a);
nd_tensor *nd_softmax(nd_tensor *a, int dim);

nd_tensor *nd_matmul(nd_tensor *a, nd_tensor *b);
nd_tensor *nd_sum(nd_tensor *a);
nd_tensor *nd_mean(nd_tensor *a);
nd_tensor *nd_sum_dim(nd_tensor *a, int dim, bool keepdim);
nd_tensor *nd_mean_dim(nd_tensor *a, int dim, bool keepdim);

/* embedding: weight [V,D], ids as float storage of integer ids shape [B,T] */
nd_tensor *nd_embedding(nd_tensor *weight, nd_tensor *ids);
nd_tensor *nd_cross_entropy(nd_tensor *logits, nd_tensor *targets); /* logits [B,T,V] or [N,V], targets float ids */

/* internals used by autograd */
void nd_compute_strides(const int *shape, int ndim, int *strides);
int  nd_same_shape(const nd_tensor *a, const nd_tensor *b);
void nd_backward(nd_tensor *loss);

#endif
