#include "attn.h"
#include "tensor.h"
#include <math.h>
#include <string.h>

static size_t head_index(int b, int h, int s, int n_head, int S, int d) {
    return ((((size_t)b * (size_t)n_head + (size_t)h) * (size_t)S + (size_t)s) * (size_t)d);
}

void attn_pack_heads(const float *src, float *dst, int B, int S, int n_head, int d) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            const float *row = src + (size_t)(b * S + s) * (size_t)n_head * (size_t)d;
            for (int h = 0; h < n_head; h++) {
                memcpy(dst + head_index(b, h, s, n_head, S, d), row + h * d, (size_t)d * sizeof(float));
            }
        }
    }
}

void attn_merge_heads(const float *src, float *dst, int B, int S, int n_head, int d) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            float *row = dst + (size_t)(b * S + s) * (size_t)n_head * (size_t)d;
            for (int h = 0; h < n_head; h++) {
                memcpy(row + h * d, src + head_index(b, h, s, n_head, S, d), (size_t)d * sizeof(float));
            }
        }
    }
}

void attn_cache_store(float *cache_k, float *cache_v, const float *k, const float *v,
                      size_t batch_stride, size_t head_stride, const int *valid_len,
                      const int *starts, int B, int S, int n_head_kv, int d) {
    for (int b = 0; b < B; b++) {
        int n = valid_len[b];
        int s0 = starts[b];
        for (int h = 0; h < n_head_kv; h++) {
            for (int t = 0; t < n; t++) {
                float *dst_k = cache_k + (size_t)b * batch_stride + (size_t)h * head_stride +
                               (size_t)(s0 + t) * (size_t)d;
                float *dst_v = cache_v + (size_t)b * batch_stride + (size_t)h * head_stride +
                               (size_t)(s0 + t) * (size_t)d;
                memcpy(dst_k, k + head_index(b, h, t, n_head_kv, S, d), (size_t)d * sizeof(float));
                memcpy(dst_v, v + head_index(b, h, t, n_head_kv, S, d), (size_t)d * sizeof(float));
            }
        }
    }
}

void attn_gqa(float *y, float *attn, const float *q, const float *cache_k, const float *cache_v,
              size_t batch_stride, size_t head_stride, const int *positions, const int *valid_len,
              const int *key_len, int B, int n_head, int S, int n_head_kv, int head_dim, int max_k,
              float scale) {
    int n_rep = n_head / n_head_kv;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / n_rep;
            for (int s = 0; s < S; s++) {
                float *row = attn + ((((size_t)b * n_head + (size_t)h) * (size_t)S + (size_t)s) *
                                     (size_t)max_k);
                int qpos = positions[b * S + s];
                int q_valid = s < valid_len[b];
                const float *qrow = q + head_index(b, h, s, n_head, S, head_dim);
                for (int kp = 0; kp < max_k; kp++) {
                    if (!q_valid || kp > qpos || kp >= key_len[b]) {
                        row[kp] = -INFINITY;
                        continue;
                    }
                    const float *krow = cache_k + (size_t)b * batch_stride + (size_t)kv_h * head_stride +
                                        (size_t)kp * (size_t)head_dim;
                    float acc = 0.0f;
                    for (int i = 0; i < head_dim; i++) acc += qrow[i] * krow[i];
                    row[kp] = acc * scale;
                }
                softmax_inplace(row, max_k);
                float *yrow = y + head_index(b, h, s, n_head, S, head_dim);
                memset(yrow, 0, (size_t)head_dim * sizeof(float));
                for (int kp = 0; kp < max_k; kp++) {
                    float p = row[kp];
                    if (p == 0.0f) continue;
                    const float *vrow = cache_v + (size_t)b * batch_stride + (size_t)kv_h * head_stride +
                                        (size_t)kp * (size_t)head_dim;
                    for (int i = 0; i < head_dim; i++) yrow[i] += p * vrow[i];
                }
            }
        }
    }
}
