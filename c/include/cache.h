#ifndef LLM_CACHE_H
#define LLM_CACHE_H

#include "gguf.h"

typedef struct {
    float *k; /* (n_layer, batch, n_kv_head, max_seq, head_dim) */
    float *v;
    float *conv; /* (n_layer, batch, conv_dim, d_conv-1) or NULL */
    float *ssm;  /* (n_layer, batch, n_v_heads, d_state, d_state) or NULL */
    int *n_seq; /* (batch,) */
    int max_seq;
    int n_layer;
    int batch;
    int n_head_kv;
    int head_dim;
    int conv_dim;
    int d_conv;
    int n_v_heads;
    int d_state;
} KVCache;

KVCache *kvcache_create(const LlamaHParams *hp, int batch, int max_seq);
void kvcache_free(KVCache *c);
void kvcache_reset(KVCache *c);
KVCache *kvcache_clone_empty(const KVCache *c);

size_t kvcache_layer_stride(const KVCache *c);
size_t kvcache_batch_stride(const KVCache *c);
size_t kvcache_head_stride(const KVCache *c);
size_t kvcache_conv_layer_stride(const KVCache *c);
size_t kvcache_ssm_layer_stride(const KVCache *c);

#endif
