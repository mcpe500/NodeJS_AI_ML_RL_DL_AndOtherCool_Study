/* io.h — checkpoint */
#ifndef ND_IO_H
#define ND_IO_H

#include "nn.h"

int nd_checkpoint_save(const char *path, nd_module *m);
int nd_checkpoint_load(const char *path, nd_module *m);

#endif
