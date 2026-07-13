/* util.h — log, assert, seed, RSS, clock */
#ifndef ND_UTIL_H
#define ND_UTIL_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define nd_assert(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "nd_assert %s:%d: %s\n", __FILE__, __LINE__, (msg)); abort(); } \
} while (0)

void     nd_log(const char *fmt, ...);
void     nd_seed(uint64_t s);
float    nd_rand_uniform(void);
float    nd_rand_normal(float mean, float std);
double   nd_now_sec(void);
/* Peak / current RSS in MB from /proc/self/status; 0 if unavailable */
float    nd_peak_rss_mb(void);
float    nd_current_rss_mb(void);
void    *nd_xmalloc(size_t n);
void    *nd_xcalloc(size_t n, size_t sz);

#endif
