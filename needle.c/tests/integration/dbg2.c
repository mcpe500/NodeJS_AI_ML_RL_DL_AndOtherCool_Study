#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
    nd_seed(7);
    nd_needle_config cfg = nd_cfg_sanity();
    cfg.vocab = 64;
    printf("1\n"); fflush(stdout);
    nd_module *model = nd_needle_create(&cfg);
    printf("2 np=%ld\n", nd_module_num_params(model)); fflush(stdout);
    int B=1,S=4,T=4;
    int ssh[2]={B,S}, tsh[2]={B,T}, lsh[1]={B*T};
    nd_tensor *src = nd_zeros(ssh,2,false);
    nd_tensor *tgt = nd_zeros(tsh,2,false);
    nd_tensor *lab = nd_zeros(lsh,1,false);
    for(int i=0;i<B*S;i++) src->data[i]=1;
    for(int i=0;i<B*T;i++){ tgt->data[i]=2; lab->data[i]=3; }
    nd_tensor **params; int n;
    nd_module_parameters(model,&params,&n);
    nd_optim *opt = nd_adamw(params,n,1e-3f,0.9f,0.999f,0.01f,1e-8f);
    for(int step=0; step<5; step++){
      printf("step %d zero\n", step); fflush(stdout);
      opt->zero_grad(opt);
      printf("fwd\n"); fflush(stdout);
      nd_tensor *logits = model->forward2(model,src,tgt);
      int N=B*T,V=cfg.vocab; int fsh[2]={N,V};
      nd_tensor *flat = nd_tensor_alloc(fsh,2,true);
      memcpy(flat->data,logits->data,(size_t)N*V*sizeof(float));
      nd_tensor *loss = nd_cross_entropy(flat, lab);
      printf("loss %f bw\n", loss->data[0]); fflush(stdout);
      nd_backward(loss);
      opt->step(opt);
      printf("free loss\n"); fflush(stdout);
      nd_tensor_free(loss);
      printf("free logits\n"); fflush(stdout);
      nd_tensor_free(logits);
    }
    printf("done free model\n"); fflush(stdout);
    nd_optim_free(opt);
    free(params);
    nd_module_free(model);
    nd_tensor_free(src); nd_tensor_free(tgt); nd_tensor_free(lab);
    printf("OK\n");
    return 0;
}
