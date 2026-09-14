#include "attn.h"
#include "backend.h"
#include "pool.h"
#include "simd.h"
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

static void mix_v(float *yrow, const float *row, const float *cache_v, size_t kv_off, int max_k,
                  int head_dim) {
    memset(yrow, 0, (size_t)head_dim * sizeof(float));
    for (int kp = 0; kp < max_k; kp++) {
        float p = row[kp];
        if (p == 0.0f) continue;
        const float *vrow = cache_v + kv_off + (size_t)kp * (size_t)head_dim;
        llm_axpy_f32(yrow, p, vrow, head_dim);
    }
}

typedef struct {
    float *y;
    float *attn;
    const float *q;
    const float *cache_k;
    const float *cache_v;
    size_t batch_stride;
    size_t head_stride;
    const int *positions;
    const int *valid_len;
    const int *key_len;
    int B;
    int n_head;
    int S;
    int n_head_kv;
    int head_dim;
    int max_k;
    float scale;
} AttnJob;

static void attn_head_range(const AttnJob *j, int w0, int w1) {
    int n_rep = j->n_head / j->n_head_kv;
    int n_head = j->n_head;
    int S = j->S;
    int d = j->head_dim;
    int max_k = j->max_k;
    for (int w = w0; w < w1; w++) {
        int b = w / n_head;
        int h = w % n_head;
        int kv_h = h / n_rep;
        size_t kv_off = (size_t)b * j->batch_stride + (size_t)kv_h * j->head_stride;
        for (int s = 0; s < S; s++) {
            float *row = j->attn + ((((size_t)b * n_head + (size_t)h) * (size_t)S + (size_t)s) *
                                    (size_t)max_k);
            int qpos = j->positions[b * S + s];
            int q_valid = s < j->valid_len[b];
            const float *qrow = j->q + head_index(b, h, s, n_head, S, d);
            int klen = j->key_len[b];
            for (int kp = 0; kp < max_k; kp++) {
                if (!q_valid || kp > qpos || kp >= klen) {
                    row[kp] = -INFINITY;
                    continue;
                }
                const float *krow = j->cache_k + kv_off + (size_t)kp * (size_t)d;
                row[kp] = llm_dot_f32(qrow, krow, d) * j->scale;
            }
            softmax_inplace(row, max_k);
            float *yrow = j->y + head_index(b, h, s, n_head, S, d);
            mix_v(yrow, row, j->cache_v, kv_off, max_k, d);
        }
    }
}

static void attn_job(int tid, int n_threads, void *ctx) {
    AttnJob *j = (AttnJob *)ctx;
    int n_work = j->B * j->n_head;
    int n = n_threads;
    if (n_work < n) n = n_work;
    if (n < 1) n = 1;
    if (tid >= n) return;
    int w0 = n_work * tid / n;
    int w1 = n_work * (tid + 1) / n;
    attn_head_range(j, w0, w1);
}

void attn_gqa(float *y, float *attn, const float *q, const float *cache_k, const float *cache_v,
              size_t batch_stride, size_t head_stride, const int *positions, const int *valid_len,
              const int *key_len, int B, int n_head, int S, int n_head_kv, int head_dim, int max_k,
              float scale) {
    if (B <= 0 || n_head <= 0 || S <= 0) return;
    AttnJob job = {y,      attn,     q,        cache_k, cache_v, batch_stride, head_stride,
                   positions, valid_len, key_len, B,      n_head,  S,           n_head_kv,
                   head_dim,  max_k,     scale};
    int nth = S > 1 ? llm_backend_n_prefill_threads() : llm_backend_n_decode_threads();
    llm_pool_run(attn_job, &job, nth);
}
