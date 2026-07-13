#define _POSIX_C_SOURCE 200809L
#include "util.h"
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include <time.h>

static uint64_t g_rng = 0x9ac4f1e2ULL;

void nd_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

void nd_seed(uint64_t s) {
    g_rng = s ? s : 0x9ac4f1e2ULL;
}

static uint64_t xorshift(void) {
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}

float nd_rand_uniform(void) {
    return (float)(xorshift() >> 11) / (float)(1ULL << 53);
}

float nd_rand_normal(float mean, float std) {
    float u1 = nd_rand_uniform();
    float u2 = nd_rand_uniform();
    if (u1 < 1e-12f) u1 = 1e-12f;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    return mean + std * z;
}

double nd_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static float nd_rss_field_mb(const char *key) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0.f;
    char line[256];
    long kb = 0;
    size_t klen = strlen(key);
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, klen) == 0) {
            sscanf(line + klen, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb / 1024.f;
}

float nd_peak_rss_mb(void) { return nd_rss_field_mb("VmHWM:"); }
float nd_current_rss_mb(void) { return nd_rss_field_mb("VmRSS:"); }

void *nd_xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p && n) nd_log("OOM malloc %zu", n);
    return p;
}

void *nd_xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p && n && sz) nd_log("OOM calloc %zu*%zu", n, sz);
    return p;
}
