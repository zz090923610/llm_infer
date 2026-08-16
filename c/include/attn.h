#ifndef LLM_ATTN_H
#define LLM_ATTN_H

#include <stddef.h>

/* src: (B, S, n_head * d)  ->  dst: (B, n_head, S, d) */
void attn_pack_heads(const float *src, float *dst, int B, int S, int n_head, int d);

/* src: (B, n_head, S, d)  ->  dst: (B, S, n_head * d) */
void attn_merge_heads(const float *src, float *dst, int B, int S, int n_head, int d);

/* Write new K/V tokens into one cache layer.
   k, v: (B, n_head_kv, S, d). cache_k/v point at the start of this layer. */
void attn_cache_store(float *cache_k, float *cache_v, const float *k, const float *v,
                      size_t batch_stride, size_t head_stride, const int *valid_len,
                      const int *starts, int B, int S, int n_head_kv, int d);

/* GQA attention over a KV-cache layer.
   q, y: (B, n_head, S, d)
   attn: scratch (B, n_head, S, max_k)
   cache_k/v: start of this layer, layout (batch, n_kv_head, max_seq, head_dim) */
void attn_gqa(float *y, float *attn, const float *q, const float *cache_k, const float *cache_v,
              size_t batch_stride, size_t head_stride, const int *positions, const int *valid_len,
              const int *key_len, int B, int n_head, int S, int n_head_kv, int head_dim, int max_k,
              float scale);

#endif
