#include "model.h"
#include "rope.h"
#include "tensor.h"
#include "util.h"
#include <math.h>

static const float *must_weight(LoadedModel *loaded, const char *name) {
    const WeightTensor *w = loaded_find_weight(loaded, name);
    if (!w) die("missing weight %s", name);
    return w->data;
}

static void rope_tables(LlamaModel *m, int need) {
    const LlamaHParams *hp = &m->hparams;
    if (m->rope_cos && need <= m->rope_len) return;
    int n = need > hp->n_ctx ? need : hp->n_ctx;
    free(m->rope_cos);
    free(m->rope_sin);
    m->rope_cos = xmalloc((size_t)n * (size_t)hp->head_dim * sizeof(float));
    m->rope_sin = xmalloc((size_t)n * (size_t)hp->head_dim * sizeof(float));
    build_rope_cache(m->rope_cos, m->rope_sin, n, hp->head_dim, hp->rope_theta);
    m->rope_len = n;
}

static void ensure_scratch(LlamaModel *m, int B, int S, int max_k) {
    const LlamaHParams *hp = &m->hparams;
    if (B <= m->scratch_B && S <= m->scratch_S && max_k <= m->scratch_K && m->x) return;
    int b = B > m->scratch_B ? B : m->scratch_B;
    int s = S > m->scratch_S ? S : m->scratch_S;
    int k = max_k > m->scratch_K ? max_k : m->scratch_K;
    if (b < 1) b = 1;
    if (s < 1) s = 1;
    if (k < 1) k = 1;
    m->scratch_B = b;
    m->scratch_S = s;
    m->scratch_K = k;
    size_t BS = (size_t)b * (size_t)s;
    size_t D = (size_t)hp->n_embd;
    size_t H = (size_t)hp->n_head;
    size_t KV = (size_t)hp->n_head_kv;
    size_t d = (size_t)hp->head_dim;
    size_t F = (size_t)hp->n_ff;
    free(m->x);
    free(m->h);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->y);
    free(m->ffn_gate);
    free(m->ffn_up);
    free(m->logits);
    free(m->positions);
    free(m->key_len);
    free(m->starts);
    m->x = xmalloc(BS * D * sizeof(float));
    m->h = xmalloc(BS * D * sizeof(float));
    m->q = xmalloc((size_t)b * H * (size_t)s * d * sizeof(float));
    m->k = xmalloc((size_t)b * KV * (size_t)s * d * sizeof(float));
    m->v = xmalloc((size_t)b * KV * (size_t)s * d * sizeof(float));
    m->attn = xmalloc((size_t)b * H * (size_t)s * (size_t)k * sizeof(float));
    m->y = xmalloc((size_t)b * H * (size_t)s * d * sizeof(float));
    m->ffn_gate = xmalloc(BS * F * sizeof(float));
    m->ffn_up = xmalloc(BS * F * sizeof(float));
    m->logits = xmalloc((size_t)b * (size_t)hp->n_vocab * sizeof(float));
    m->positions = xmalloc((size_t)b * (size_t)s * sizeof(int));
    m->key_len = xmalloc((size_t)b * sizeof(int));
    m->starts = xmalloc((size_t)b * sizeof(int));
}

LlamaModel *llama_model_init(LoadedModel *loaded) {
    LlamaModel *m = xcalloc(1, sizeof(LlamaModel));
    m->hparams = loaded->hparams;
    m->hparams.chat_template = NULL;
    m->hparams.tokenizer_pre = NULL;
    m->tok_embd = must_weight(loaded, "token_embd.weight");
    m->output_norm = must_weight(loaded, "output_norm.weight");
    m->layers = xcalloc((size_t)m->hparams.n_layer, sizeof(LayerWeights));
    char name[128];
    for (int i = 0; i < m->hparams.n_layer; i++) {
        LayerWeights *L = &m->layers[i];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        L->attn_norm = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
        L->wq = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
        L->wk = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
        L->wv = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
        L->wo = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        L->ffn_norm = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        L->gate = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        L->up = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        L->down = must_weight(loaded, name);
    }
    return m;
}

LlamaModel *llama_model_from_file(const char *path, int progress) {
    LoadedModel *loaded = load_model(path, 1, progress);
    LlamaModel *m = llama_model_init(loaded);
    m->owned = loaded;
    return m;
}

void llama_model_free(LlamaModel *m) {
    if (!m) return;
    if (m->owned) loaded_model_free(m->owned);
    free(m->layers);
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->x);
    free(m->h);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->y);
    free(m->ffn_gate);
    free(m->ffn_up);
    free(m->logits);
    free(m->positions);
    free(m->key_len);
    free(m->starts);
    free(m);
}

KVCache *llama_model_new_cache(LlamaModel *m, int batch, int max_seq) {
    if (max_seq <= 0) max_seq = m->hparams.n_ctx;
    return kvcache_create(&m->hparams, batch, max_seq);
}

static size_t q_index(int b, int h, int s, int B, int H, int S, int d) {
    (void)B;
    return ((((size_t)b * (size_t)H + (size_t)h) * (size_t)S + (size_t)s) * (size_t)d);
}

float *llama_model_forward(LlamaModel *m, const int *tokens, int B, int S, KVCache *cache,
                           const int *valid_len, float *logits_out) {
    const LlamaHParams *hp = &m->hparams;
    int D = hp->n_embd, H = hp->n_head, KV = hp->n_head_kv, d = hp->head_dim, F = hp->n_ff;
    int n_rep = H / KV;
    float scale = 1.0f / sqrtf((float)d);

    int *vl_local = NULL;
    if (!valid_len) {
        vl_local = xmalloc((size_t)B * sizeof(int));
        for (int b = 0; b < B; b++) vl_local[b] = S;
        valid_len = vl_local;
    }

    int max_end = 0;
    for (int b = 0; b < B; b++) {
        int end = cache->n_seq[b] + valid_len[b];
        if (end > max_end) max_end = end;
    }
    if (max_end > cache->max_seq) die("cache overflow: %d > %d", max_end, cache->max_seq);

    rope_tables(m, max_end);
    ensure_scratch(m, B, S, max_end > 0 ? max_end : 1);

    for (int b = 0; b < B; b++) m->starts[b] = cache->n_seq[b];
    memset(m->positions, 0, (size_t)B * (size_t)S * sizeof(int));
    for (int b = 0; b < B; b++) {
        int n = valid_len[b];
        for (int t = 0; t < n; t++) m->positions[b * S + t] = m->starts[b] + t;
        m->key_len[b] = m->starts[b] + n;
    }

    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            int tid = tokens[b * S + s];
            if (tid < 0 || tid >= hp->n_vocab) die("token id %d out of range", tid);
            memcpy(m->x + ((size_t)(b * S + s) * (size_t)D), m->tok_embd + (size_t)tid * (size_t)D,
                   (size_t)D * sizeof(float));
        }
    }

    size_t layer_stride = kvcache_layer_stride(cache);
    size_t batch_stride = kvcache_batch_stride(cache);
    size_t head_stride = kvcache_head_stride(cache);

    for (int li = 0; li < hp->n_layer; li++) {
        LayerWeights *L = &m->layers[li];
        rmsnorm_rows(m->x, L->attn_norm, m->h, B * S, D, hp->rms_eps);

        /* q,k,v projections then pack as (B, heads, S, d) */
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                const float *xin = m->h + (size_t)(b * S + s) * (size_t)D;
                float *qtmp = m->y + (size_t)(b * S + s) * (size_t)(H * d); /* reuse y as q_lin */
                linear(L->wq, xin, qtmp, H * d, D);
                float *ktmp = m->ffn_gate + (size_t)(b * S + s) * (size_t)(KV * d);
                linear(L->wk, xin, ktmp, KV * d, D);
                float *vtmp = m->ffn_up + (size_t)(b * S + s) * (size_t)(KV * d);
                linear(L->wv, xin, vtmp, KV * d, D);
            }
        }
        /* ffn_gate/up used as k_lin/v_lin; y as q_lin. Scatter into q,k,v. */
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                const float *qlin = m->y + (size_t)(b * S + s) * (size_t)(H * d);
                for (int h = 0; h < H; h++) {
                    memcpy(m->q + q_index(b, h, s, B, H, S, d), qlin + h * d, (size_t)d * sizeof(float));
                }
                const float *klin = m->ffn_gate + (size_t)(b * S + s) * (size_t)(KV * d);
                const float *vlin = m->ffn_up + (size_t)(b * S + s) * (size_t)(KV * d);
                for (int h = 0; h < KV; h++) {
                    memcpy(m->k + q_index(b, h, s, B, KV, S, d), klin + h * d, (size_t)d * sizeof(float));
                    memcpy(m->v + q_index(b, h, s, B, KV, S, d), vlin + h * d, (size_t)d * sizeof(float));
                }
            }
        }

        apply_rope(m->q, m->rope_cos, m->rope_sin, m->positions, B, H, S, d);
        apply_rope(m->k, m->rope_cos, m->rope_sin, m->positions, B, KV, S, d);

        for (int b = 0; b < B; b++) {
            int n = valid_len[b];
            int s0 = m->starts[b];
            for (int h = 0; h < KV; h++) {
                for (int t = 0; t < n; t++) {
                    float *dst_k = cache->k + (size_t)li * layer_stride + (size_t)b * batch_stride +
                                   (size_t)h * head_stride + (size_t)(s0 + t) * (size_t)d;
                    float *dst_v = cache->v + (size_t)li * layer_stride + (size_t)b * batch_stride +
                                   (size_t)h * head_stride + (size_t)(s0 + t) * (size_t)d;
                    memcpy(dst_k, m->k + q_index(b, h, t, B, KV, S, d), (size_t)d * sizeof(float));
                    memcpy(dst_v, m->v + q_index(b, h, t, B, KV, S, d), (size_t)d * sizeof(float));
                }
            }
        }

        int max_k = 0;
        for (int b = 0; b < B; b++)
            if (m->key_len[b] > max_k) max_k = m->key_len[b];

        /* attention */
        for (int b = 0; b < B; b++) {
            for (int h = 0; h < H; h++) {
                int kv_h = h / n_rep;
                for (int s = 0; s < S; s++) {
                    float *row = m->attn + ((((size_t)b * H + (size_t)h) * (size_t)S + (size_t)s) * (size_t)max_k);
                    int qpos = m->positions[b * S + s];
                    int q_valid = s < valid_len[b];
                    const float *qrow = m->q + q_index(b, h, s, B, H, S, d);
                    for (int kp = 0; kp < max_k; kp++) {
                        if (!q_valid || kp > qpos || kp >= m->key_len[b]) {
                            row[kp] = -INFINITY;
                            continue;
                        }
                        const float *krow = cache->k + (size_t)li * layer_stride + (size_t)b * batch_stride +
                                            (size_t)kv_h * head_stride + (size_t)kp * (size_t)d;
                        float acc = 0.0f;
                        for (int i = 0; i < d; i++) acc += qrow[i] * krow[i];
                        row[kp] = acc * scale;
                    }
                    softmax_inplace(row, max_k);
                    float *yrow = m->y + q_index(b, h, s, B, H, S, d);
                    memset(yrow, 0, (size_t)d * sizeof(float));
                    for (int kp = 0; kp < max_k; kp++) {
                        float p = row[kp];
                        if (p == 0.0f) continue;
                        const float *vrow = cache->v + (size_t)li * layer_stride + (size_t)b * batch_stride +
                                            (size_t)kv_h * head_stride + (size_t)kp * (size_t)d;
                        for (int i = 0; i < d; i++) yrow[i] += p * vrow[i];
                    }
                }
            }
        }

        /* merge heads -> (B,S,D) in h, then x += y @ wo.T */
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                float *merged = m->h + (size_t)(b * S + s) * (size_t)D;
                for (int h = 0; h < H; h++) {
                    memcpy(merged + h * d, m->y + q_index(b, h, s, B, H, S, d), (size_t)d * sizeof(float));
                }
                float *proj = m->ffn_gate + (size_t)(b * S + s) * (size_t)D;
                linear(L->wo, merged, proj, D, D);
                float *xr = m->x + (size_t)(b * S + s) * (size_t)D;
                vec_add(xr, proj, xr, D);
            }
        }

        rmsnorm_rows(m->x, L->ffn_norm, m->h, B * S, D, hp->rms_eps);
        for (int b = 0; b < B; b++) {
            for (int s = 0; s < S; s++) {
                const float *hin = m->h + (size_t)(b * S + s) * (size_t)D;
                float *g = m->ffn_gate + (size_t)(b * S + s) * (size_t)F;
                float *u = m->ffn_up + (size_t)(b * S + s) * (size_t)F;
                linear(L->gate, hin, g, F, D);
                silu(g, g, F);
                linear(L->up, hin, u, F, D);
                vec_mul(g, u, g, F);
                float *down = m->y + (size_t)(b * S + s) * (size_t)D;
                linear(L->down, g, down, D, F);
                float *xr = m->x + (size_t)(b * S + s) * (size_t)D;
                vec_add(xr, down, xr, D);
            }
        }
    }

    rmsnorm_rows(m->x, m->output_norm, m->h, B * S, D, hp->rms_eps);
    float *logits = logits_out ? logits_out : m->logits;
    for (int b = 0; b < B; b++) {
        int last = valid_len[b] - 1;
        if (last < 0) last = 0;
        if (last > S - 1) last = S - 1;
        const float *hlast = m->h + (size_t)(b * S + last) * (size_t)D;
        linear(m->tok_embd, hlast, logits + (size_t)b * (size_t)hp->n_vocab, hp->n_vocab, D);
        cache->n_seq[b] = m->key_len[b];
    }
    free(vl_local);
    return logits;
}
