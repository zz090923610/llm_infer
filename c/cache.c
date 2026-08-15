#include "cache.h"
#include "util.h"

static size_t cache_n(const KVCache *c) {
    return (size_t)c->n_layer * (size_t)c->batch * (size_t)c->n_head_kv * (size_t)c->max_seq *
           (size_t)c->head_dim;
}

KVCache *kvcache_create(const LlamaHParams *hp, int batch, int max_seq) {
    KVCache *c = xcalloc(1, sizeof(KVCache));
    c->n_layer = hp->n_layer;
    c->batch = batch;
    c->n_head_kv = hp->n_head_kv;
    c->head_dim = hp->head_dim;
    c->max_seq = max_seq;
    size_t n = cache_n(c);
    c->k = xcalloc(n, sizeof(float));
    c->v = xcalloc(n, sizeof(float));
    c->n_seq = xcalloc((size_t)batch, sizeof(int));
    return c;
}

void kvcache_free(KVCache *c) {
    if (!c) return;
    free(c->k);
    free(c->v);
    free(c->n_seq);
    free(c);
}

void kvcache_reset(KVCache *c) {
    memset(c->n_seq, 0, (size_t)c->batch * sizeof(int));
}

KVCache *kvcache_clone_empty(const KVCache *c) {
    KVCache *n = xcalloc(1, sizeof(KVCache));
    *n = *c;
    size_t elems = cache_n(c);
    n->k = xcalloc(elems, sizeof(float));
    n->v = xcalloc(elems, sizeof(float));
    n->n_seq = xcalloc((size_t)c->batch, sizeof(int));
    return n;
}

size_t kvcache_layer_stride(const KVCache *c) {
    return (size_t)c->batch * (size_t)c->n_head_kv * (size_t)c->max_seq * (size_t)c->head_dim;
}

size_t kvcache_batch_stride(const KVCache *c) {
    return (size_t)c->n_head_kv * (size_t)c->max_seq * (size_t)c->head_dim;
}

size_t kvcache_head_stride(const KVCache *c) {
    return (size_t)c->max_seq * (size_t)c->head_dim;
}
