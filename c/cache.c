#include "cache.h"
#include "util.h"

static size_t cache_n(const KVCache *c) {
    return (size_t)c->n_layer * (size_t)c->batch * (size_t)c->n_head_kv * (size_t)c->max_seq *
           (size_t)c->head_dim;
}

static size_t conv_n(const KVCache *c) {
    if (c->conv_dim <= 0 || c->d_conv <= 1) return 0;
    return (size_t)c->n_layer * (size_t)c->batch * (size_t)c->conv_dim * (size_t)(c->d_conv - 1);
}

static size_t ssm_n(const KVCache *c) {
    if (c->n_v_heads <= 0 || c->d_state <= 0) return 0;
    return (size_t)c->n_layer * (size_t)c->batch * (size_t)c->n_v_heads * (size_t)c->d_state *
           (size_t)c->d_state;
}

KVCache *kvcache_create(const LlamaHParams *hp, int batch, int max_seq) {
    KVCache *c = xcalloc(1, sizeof(KVCache));
    int n_layer = hp->n_layer_fwd > 0 ? hp->n_layer_fwd : hp->n_layer;
    c->n_layer = n_layer;
    c->batch = batch;
    c->n_head_kv = hp->n_head_kv;
    c->head_dim = hp->head_dim;
    c->max_seq = max_seq;
    if (hp->arch == LLM_ARCH_QWEN35 && hp->ssm_d_conv > 0) {
        int key_dim = hp->ssm_d_state * hp->ssm_n_group;
        int value_dim = hp->ssm_d_inner;
        c->conv_dim = 2 * key_dim + value_dim;
        c->d_conv = hp->ssm_d_conv;
        c->n_v_heads = hp->ssm_dt_rank;
        c->d_state = hp->ssm_d_state;
    }
    size_t n = cache_n(c);
    c->k = xcalloc(n, sizeof(float));
    c->v = xcalloc(n, sizeof(float));
    size_t nc = conv_n(c);
    size_t ns = ssm_n(c);
    if (nc) c->conv = xcalloc(nc, sizeof(float));
    if (ns) c->ssm = xcalloc(ns, sizeof(float));
    c->n_seq = xcalloc((size_t)batch, sizeof(int));
    return c;
}

void kvcache_free(KVCache *c) {
    if (!c) return;
    free(c->k);
    free(c->v);
    free(c->conv);
    free(c->ssm);
    free(c->n_seq);
    free(c);
}

void kvcache_reset(KVCache *c) {
    memset(c->n_seq, 0, (size_t)c->batch * sizeof(int));
    size_t nc = conv_n(c);
    size_t ns = ssm_n(c);
    if (c->conv && nc) memset(c->conv, 0, nc * sizeof(float));
    if (c->ssm && ns) memset(c->ssm, 0, ns * sizeof(float));
}

KVCache *kvcache_clone_empty(const KVCache *c) {
    KVCache *n = xcalloc(1, sizeof(KVCache));
    *n = *c;
    size_t elems = cache_n(c);
    n->k = xcalloc(elems, sizeof(float));
    n->v = xcalloc(elems, sizeof(float));
    size_t nc = conv_n(c);
    size_t ns = ssm_n(c);
    n->conv = nc ? xcalloc(nc, sizeof(float)) : NULL;
    n->ssm = ns ? xcalloc(ns, sizeof(float)) : NULL;
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

size_t kvcache_conv_layer_stride(const KVCache *c) {
    if (c->conv_dim <= 0 || c->d_conv <= 1) return 0;
    return (size_t)c->batch * (size_t)c->conv_dim * (size_t)(c->d_conv - 1);
}

size_t kvcache_ssm_layer_stride(const KVCache *c) {
    if (c->n_v_heads <= 0 || c->d_state <= 0) return 0;
    return (size_t)c->batch * (size_t)c->n_v_heads * (size_t)c->d_state * (size_t)c->d_state;
}
