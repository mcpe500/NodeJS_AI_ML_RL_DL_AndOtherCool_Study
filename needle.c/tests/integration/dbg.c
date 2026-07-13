#include "needle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(){
  nd_seed(7);
  nd_needle_config cfg = nd_cfg_sanity();
  cfg.vocab = 64;
  nd_module *m = nd_needle_create(&cfg);
  int B=1,S=4,T=4;
  nd_tensor *src = nd_zeros((int[]){B,S},2,false);
  nd_tensor *tgt = nd_zeros((int[]){B,T},2,false);
  nd_tensor *lab = nd_zeros((int[]){B,T},2,false);
  for(int i=0;i<B*S;i++) src->data[i]=1;
  for(int i=0;i<B*T;i++) { tgt->data[i]=2; lab->data[i]=3; }
  nd_tensor **params; int n;
  nd_module_parameters(m,&params,&n);
  printf("nparams tensors=%d\n", n);
  nd_optim *opt = nd_adamw(params,n,1e-3f,0.9f,0.999f,0.01f,1e-8f);
  printf("fwd\n"); fflush(stdout);
  opt->zero_grad(opt);
  nd_tensor *logits = m->forward2(m,src,tgt);
  printf("logits ok\n"); fflush(stdout);
  int N=B*T,V=cfg.vocab;
  nd_tensor *flat = nd_tensor_alloc((int[]){N,V},2,true);
  memcpy(flat->data,logits->data,(size_t)N*V*sizeof(float));
  printf("ce\n"); fflush(stdout);
  nd_tensor *loss = nd_cross_entropy(flat, lab);
  printf("loss=%f\n", loss->data[0]); fflush(stdout);
  printf("backward\n"); fflush(stdout);
  nd_backward(loss);
  printf("step\n"); fflush(stdout);
  opt->step(opt);
  printf("free\n"); fflush(stdout);
  nd_tensor_free(loss);
  nd_tensor_free(flat);
  nd_tensor_free(logits);
  printf("ok rss=%.1f\n", nd_peak_rss_mb());
  return 0;
}
