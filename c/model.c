#include "model.h"
#include "attn.h"
#include "backend.h"
#include "gdn.h"
#include "rope.h"
#include "tensor.h"
#include "util.h"
#include <math.h>
#include <string.h>

static const float *must_weight(LoadedModel *loaded, const char *name) {
    const WeightTensor *w = loaded_find_weight(loaded, name);
    if (!w) die("missing weight %s", name);
    return w->data;
}

static const float *opt_weight(LoadedModel *loaded, const char *name) {
    const WeightTensor *w = loaded_find_weight(loaded, name);
    return w ? w->data : NULL;
}

static void intern_w(const float *p, size_t bytes) {
    if (p) llm_backend_intern_weight(p, bytes);
}

static void intern_f16(const float *p, size_t bytes) {
    if (p) llm_backend_intern_weight_f16(p, bytes);
}

static void add_bias_rows(float *y, const float *bias, int n_tok, int n) {
    if (!bias) return;
    for (int t = 0; t < n_tok; t++) {
        float *row = y + (size_t)t * (size_t)n;
        vec_add(row, bias, row, n);
    }
}

static void split_q_gate(const float *src, float *q, float *gate, int B, int S, int H, int d) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            const float *row = src + (size_t)(b * S + s) * (size_t)H * 2 * (size_t)d;
            float *qrow = q + (size_t)(b * S + s) * (size_t)H * (size_t)d;
            float *grow = gate + (size_t)(b * S + s) * (size_t)H * (size_t)d;
            for (int h = 0; h < H; h++) {
                memcpy(qrow + h * d, row + (size_t)h * 2 * d, (size_t)d * sizeof(float));
                memcpy(grow + h * d, row + (size_t)h * 2 * d + d, (size_t)d * sizeof(float));
            }
        }
    }
}

static void sigmoid_mul(float *y, const float *gate, int n) {
    for (int i = 0; i < n; i++) y[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

static size_t gdn_scratch_floats(const LlamaHParams *hp) {
    if (hp->arch != LLM_ARCH_QWEN35) return 0;
    int key_dim = hp->ssm_d_state * hp->ssm_n_group;
    int value_dim = hp->ssm_d_inner;
    int conv_dim = 2 * key_dim + value_dim;
    int n_v = hp->ssm_dt_rank;
    return (size_t)(2 * conv_dim + 5 * value_dim + 2 * n_v + hp->ssm_d_state + 2 * key_dim + 16);
}

static void rope_tables(LlamaModel *m, int need) {
    const LlamaHParams *hp = &m->hparams;
    int n_rot = hp->n_rot > 0 ? hp->n_rot : hp->head_dim;
    int tab_dim = hp->rope_neox ? n_rot : hp->head_dim;
    if (m->rope_cos && need <= m->rope_len && m->rope_n_rot == tab_dim) return;
    int n = need > hp->n_ctx ? need : hp->n_ctx;
    if (n > LLM_DEFAULT_MAX_SEQ) n = need > LLM_DEFAULT_MAX_SEQ ? need : LLM_DEFAULT_MAX_SEQ;
    free(m->rope_cos);
    free(m->rope_sin);
    m->rope_cos = xmalloc((size_t)n * (size_t)tab_dim * sizeof(float));
    m->rope_sin = xmalloc((size_t)n * (size_t)tab_dim * sizeof(float));
    if (hp->rope_neox) build_rope_cache_n(m->rope_cos, m->rope_sin, n, n_rot, hp->rope_theta);
    else build_rope_cache(m->rope_cos, m->rope_sin, n, hp->head_dim, hp->rope_theta);
    intern_w(m->rope_cos, (size_t)n * (size_t)tab_dim * sizeof(float));
    intern_w(m->rope_sin, (size_t)n * (size_t)tab_dim * sizeof(float));
    m->rope_len = n;
    m->rope_n_rot = tab_dim;
}

static void apply_rope_any(LlamaModel *m, float *x, int B, int n_head, int S, int d) {
    const LlamaHParams *hp = &m->hparams;
    if (hp->rope_neox)
        apply_rope_neox(x, m->rope_cos, m->rope_sin, m->positions, B, n_head, S, d, hp->n_rot);
    else
        apply_rope(x, m->rope_cos, m->rope_sin, m->positions, B, n_head, S, d);
}

static int imax(int a, int b) { return a > b ? a : b; }

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
    int q_out = (int)(H * d);
    if (hp->arch == LLM_ARCH_QWEN35) q_out = (int)(H * 2 * d);
    int gdn_mix = 0;
    if (hp->arch == LLM_ARCH_QWEN35) {
        int key_dim = hp->ssm_d_state * hp->ssm_n_group;
        gdn_mix = 2 * key_dim + hp->ssm_d_inner;
    }
    int lin = imax((int)F, q_out);
    lin = imax(lin, (int)(KV * d));
    lin = imax(lin, gdn_mix);
    lin = imax(lin, (int)(H * d));
    free(m->x);
    free(m->h);
    free(m->q);
    free(m->k);
    free(m->v);
    free(m->attn);
    free(m->y);
    free(m->ffn_gate);
    free(m->ffn_up);
    free(m->gate_buf);
    free(m->gdn_scratch);
    free(m->logits);
    free(m->positions);
    free(m->key_len);
    free(m->starts);
    free(m->valid_buf);
    free(m->tok_ids);
    size_t h_w = D > H * d ? D : H * d;
    m->x = xmalloc(BS * D * sizeof(float));
    m->h = xmalloc(BS * h_w * sizeof(float));
    m->q = xmalloc((size_t)b * H * (size_t)s * d * sizeof(float));
    m->k = xmalloc((size_t)b * KV * (size_t)s * d * sizeof(float));
    m->v = xmalloc((size_t)b * KV * (size_t)s * d * sizeof(float));
    m->attn = xmalloc((size_t)b * H * (size_t)s * (size_t)k * sizeof(float));
    m->y = xmalloc(BS * (size_t)lin * sizeof(float));
    m->ffn_gate = xmalloc(BS * (size_t)lin * sizeof(float));
    m->ffn_up = xmalloc(BS * (size_t)lin * sizeof(float));
    m->gate_buf = xmalloc(BS * H * d * sizeof(float));
    m->gdn_scratch_n = gdn_scratch_floats(hp);
    m->gdn_scratch = m->gdn_scratch_n ? xmalloc(m->gdn_scratch_n * sizeof(float)) : NULL;
    m->logits = xmalloc((size_t)b * (size_t)hp->n_vocab * sizeof(float));
    m->positions = xmalloc((size_t)b * (size_t)s * sizeof(int));
    m->key_len = xmalloc((size_t)b * sizeof(int));
    m->starts = xmalloc((size_t)b * sizeof(int));
    m->valid_buf = xmalloc((size_t)b * sizeof(int));
    m->tok_ids = xmalloc((size_t)b * (size_t)s * sizeof(int));
    llm_backend_intern_rw(m->x, BS * D * sizeof(float));
    llm_backend_intern_rw(m->h, BS * h_w * sizeof(float));
    llm_backend_intern_rw(m->q, (size_t)b * H * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->k, (size_t)b * KV * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->v, (size_t)b * KV * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->y, BS * (size_t)lin * sizeof(float));
    llm_backend_intern_rw(m->ffn_gate, BS * (size_t)lin * sizeof(float));
    llm_backend_intern_rw(m->ffn_up, BS * (size_t)lin * sizeof(float));
    llm_backend_intern_rw(m->gate_buf, BS * H * d * sizeof(float));
    llm_backend_intern_rw(m->logits, (size_t)b * (size_t)hp->n_vocab * sizeof(float));
    llm_backend_intern_rw(m->positions, (size_t)b * (size_t)s * sizeof(int));
    llm_backend_intern_rw(m->key_len, (size_t)b * sizeof(int));
    llm_backend_intern_rw(m->starts, (size_t)b * sizeof(int));
    llm_backend_intern_rw(m->valid_buf, (size_t)b * sizeof(int));
    llm_backend_intern_rw(m->tok_ids, (size_t)b * (size_t)s * sizeof(int));
}

LlamaModel *llama_model_init(LoadedModel *loaded) {
    LlamaModel *m = xcalloc(1, sizeof(LlamaModel));
    m->hparams = loaded->hparams;
    m->hparams.chat_template = NULL;
    m->hparams.tokenizer_pre = NULL;
    m->hparams.model_name = NULL;
    m->tok_embd = must_weight(loaded, "token_embd.weight");
    m->output_norm = must_weight(loaded, "output_norm.weight");
    m->output = opt_weight(loaded, "output.weight");
    if (!m->output) m->output = m->tok_embd;
    int n_layer = m->hparams.n_layer_fwd > 0 ? m->hparams.n_layer_fwd : m->hparams.n_layer;
    m->layers = xcalloc((size_t)n_layer, sizeof(LayerWeights));
    char name[128];
    const LlamaHParams *hp = &m->hparams;
    int D = hp->n_embd, H = hp->n_head, KV = hp->n_head_kv, d = hp->head_dim, F = hp->n_ff;
    for (int i = 0; i < n_layer; i++) {
        LayerWeights *L = &m->layers[i];
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", i);
        L->attn_norm = must_weight(loaded, name);
        L->is_gdn = layer_is_gdn(hp, i);
        if (L->is_gdn) {
            snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", i);
            L->wqkv = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", i);
            L->attn_gate = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_a", i);
            L->ssm_a = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", i);
            L->ssm_alpha = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", i);
            L->ssm_beta = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_conv1d.weight", i);
            L->ssm_conv1d = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", i);
            L->ssm_dt = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", i);
            L->ssm_norm = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", i);
            L->ssm_out = must_weight(loaded, name);
        } else {
            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", i);
            L->wq = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", i);
            L->wk = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", i);
            L->wv = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", i);
            L->wo = must_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_q.bias", i);
            L->bq = opt_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k.bias", i);
            L->bk = opt_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_v.bias", i);
            L->bv = opt_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", i);
            L->q_norm = opt_weight(loaded, name);
            snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", i);
            L->k_norm = opt_weight(loaded, name);
            L->wq_out = (hp->arch == LLM_ARCH_QWEN35) ? H * 2 * d : H * d;
            L->wo_in = H * d;
        }
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", i);
        L->ffn_norm = opt_weight(loaded, name);
        if (!L->ffn_norm) {
            snprintf(name, sizeof(name), "blk.%d.post_attention_norm.weight", i);
            L->ffn_norm = must_weight(loaded, name);
        }
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", i);
        L->gate = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        L->up = must_weight(loaded, name);
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", i);
        L->down = must_weight(loaded, name);
    }
    intern_f16(m->tok_embd, (size_t)hp->n_vocab * (size_t)D * sizeof(float));
    if (m->output != m->tok_embd)
        intern_f16(m->output, (size_t)hp->n_vocab * (size_t)D * sizeof(float));
    intern_w(m->output_norm, (size_t)D * sizeof(float));
    for (int i = 0; i < n_layer; i++) {
        LayerWeights *L = &m->layers[i];
        intern_w(L->attn_norm, (size_t)D * sizeof(float));
        intern_w(L->ffn_norm, (size_t)D * sizeof(float));
        intern_f16(L->gate, (size_t)F * (size_t)D * sizeof(float));
        intern_f16(L->up, (size_t)F * (size_t)D * sizeof(float));
        intern_f16(L->down, (size_t)D * (size_t)F * sizeof(float));
        if (L->is_gdn) {
            int key_dim = hp->ssm_d_state * hp->ssm_n_group;
            int value_dim = hp->ssm_d_inner;
            int conv_dim = 2 * key_dim + value_dim;
            intern_f16(L->wqkv, (size_t)conv_dim * (size_t)D * sizeof(float));
            intern_f16(L->attn_gate, (size_t)value_dim * (size_t)D * sizeof(float));
            intern_w(L->ssm_a, (size_t)hp->ssm_dt_rank * sizeof(float));
            intern_w(L->ssm_dt, (size_t)hp->ssm_dt_rank * sizeof(float));
            intern_f16(L->ssm_alpha, (size_t)hp->ssm_dt_rank * (size_t)D * sizeof(float));
            intern_f16(L->ssm_beta, (size_t)hp->ssm_dt_rank * (size_t)D * sizeof(float));
            intern_w(L->ssm_conv1d, (size_t)conv_dim * (size_t)hp->ssm_d_conv * sizeof(float));
            intern_w(L->ssm_norm, (size_t)hp->ssm_d_state * sizeof(float));
            intern_f16(L->ssm_out, (size_t)D * (size_t)value_dim * sizeof(float));
        } else {
            intern_f16(L->wq, (size_t)L->wq_out * (size_t)D * sizeof(float));
            intern_f16(L->wk, (size_t)KV * (size_t)d * (size_t)D * sizeof(float));
            intern_f16(L->wv, (size_t)KV * (size_t)d * (size_t)D * sizeof(float));
            intern_f16(L->wo, (size_t)D * (size_t)L->wo_in * sizeof(float));
            intern_w(L->bq, (size_t)L->wq_out * sizeof(float));
            intern_w(L->bk, (size_t)KV * (size_t)d * sizeof(float));
            intern_w(L->bv, (size_t)KV * (size_t)d * sizeof(float));
            intern_w(L->q_norm, (size_t)d * sizeof(float));
            intern_w(L->k_norm, (size_t)d * sizeof(float));
        }
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
    free(m->gate_buf);
    free(m->gdn_scratch);
    free(m->logits);
    free(m->positions);
    free(m->key_len);
    free(m->starts);
    free(m->valid_buf);
    free(m->tok_ids);
    free(m);
}

KVCache *llama_model_new_cache(LlamaModel *m, int batch, int max_seq) {
    if (max_seq <= 0) {
        max_seq = m->hparams.n_ctx;
        if (max_seq > LLM_DEFAULT_MAX_SEQ) max_seq = LLM_DEFAULT_MAX_SEQ;
    }
    KVCache *c = kvcache_create(&m->hparams, batch, max_seq);
    size_t layer = kvcache_layer_stride(c) * sizeof(float);
    for (int i = 0; i < c->n_layer; i++) {
        llm_backend_intern_device(c->k + (size_t)i * kvcache_layer_stride(c), layer);
        llm_backend_intern_device(c->v + (size_t)i * kvcache_layer_stride(c), layer);
    }
    return c;
}

static void full_attn_layer(LlamaModel *m, LayerWeights *L, int B, int S, int D, int H, int KV,
                            int d, float scale, float *layer_k, float *layer_v, size_t batch_stride,
                            size_t head_stride) {
    int BS = B * S;
    int max_k = 0;
    for (int b = 0; b < B; b++)
        if (m->key_len[b] > max_k) max_k = m->key_len[b];
    float eps = m->hparams.rms_eps;

    linear_rows(L->wq, m->h, m->y, BS, L->wq_out, D);
    add_bias_rows(m->y, L->bq, BS, L->wq_out);
    linear_rows(L->wk, m->h, m->ffn_gate, BS, KV * d, D);
    add_bias_rows(m->ffn_gate, L->bk, BS, KV * d);
    linear_rows(L->wv, m->h, m->ffn_up, BS, KV * d, D);
    add_bias_rows(m->ffn_up, L->bv, BS, KV * d);

    int gated = L->wq_out == H * 2 * d;
    float *q_bs = m->y;
    float *g_bs = NULL;
    if (gated) {
        split_q_gate(m->y, m->h, m->gate_buf, B, S, H, d);
        q_bs = m->h;
        g_bs = m->gate_buf;
    }

    if (S == 1) {
        if (L->q_norm) rmsnorm_rows(q_bs, L->q_norm, q_bs, B * H, d, eps);
        if (L->k_norm) rmsnorm_rows(m->ffn_gate, L->k_norm, m->ffn_gate, B * KV, d, eps);
        apply_rope_any(m, q_bs, B, H, S, d);
        apply_rope_any(m, m->ffn_gate, B, KV, S, d);
        attn_cache_store(layer_k, layer_v, m->ffn_gate, m->ffn_up, batch_stride, head_stride,
                         m->valid_buf, m->starts, B, S, KV, d);
        attn_gqa(m->y, m->attn, q_bs, layer_k, layer_v, batch_stride, head_stride, m->positions,
                 m->valid_buf, m->key_len, B, H, S, KV, d, max_k, scale);
        if (gated) sigmoid_mul(m->y, g_bs, B * H * d);
        linear_rows_add(L->wo, m->y, m->x, BS, D, L->wo_in);
        return;
    }

    attn_pack_heads(q_bs, m->q, B, S, H, d);
    if (gated) {
        attn_pack_heads(g_bs, m->y, B, S, H, d);
        memcpy(m->gate_buf, m->y, (size_t)B * (size_t)H * (size_t)S * (size_t)d * sizeof(float));
    }
    attn_pack_heads(m->ffn_gate, m->k, B, S, KV, d);
    attn_pack_heads(m->ffn_up, m->v, B, S, KV, d);
    if (L->q_norm) rmsnorm_rows(m->q, L->q_norm, m->q, B * H * S, d, eps);
    if (L->k_norm) rmsnorm_rows(m->k, L->k_norm, m->k, B * KV * S, d, eps);
    apply_rope_any(m, m->q, B, H, S, d);
    apply_rope_any(m, m->k, B, KV, S, d);
    attn_cache_store(layer_k, layer_v, m->k, m->v, batch_stride, head_stride, m->valid_buf, m->starts,
                     B, S, KV, d);
    attn_gqa(m->y, m->attn, m->q, layer_k, layer_v, batch_stride, head_stride, m->positions,
             m->valid_buf, m->key_len, B, H, S, KV, d, max_k, scale);
    if (gated) sigmoid_mul(m->y, m->gate_buf, B * H * S * d);
    attn_merge_heads(m->y, m->h, B, S, H, d);
    linear_rows_add(L->wo, m->h, m->x, BS, D, L->wo_in);
}

float *llama_model_forward(LlamaModel *m, const int *tokens, int B, int S, KVCache *cache,
                           const int *valid_len, float *logits_out) {
    const LlamaHParams *hp = &m->hparams;
    int D = hp->n_embd, H = hp->n_head, KV = hp->n_head_kv, d = hp->head_dim, F = hp->n_ff;
    float scale = 1.0f / sqrtf((float)d);
    int BS = B * S;
    int n_layer = hp->n_layer_fwd > 0 ? hp->n_layer_fwd : hp->n_layer;

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
    memcpy(m->valid_buf, valid_len, (size_t)B * sizeof(int));
    memset(m->positions, 0, (size_t)B * (size_t)S * sizeof(int));
    for (int b = 0; b < B; b++) {
        int n = m->valid_buf[b];
        for (int t = 0; t < n; t++) m->positions[b * S + t] = m->starts[b] + t;
        m->key_len[b] = m->starts[b] + n;
    }
    llm_backend_host_write(m->positions);
    llm_backend_host_write(m->starts);
    llm_backend_host_write(m->key_len);
    llm_backend_host_write(m->valid_buf);

    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            int tid = tokens[b * S + s];
            if (tid < 0 || tid >= hp->n_vocab) die("token id %d out of range", tid);
            m->tok_ids[b * S + s] = tid;
        }
    }
    llm_backend_host_write(m->tok_ids);
    embed_gather(m->tok_embd, m->tok_ids, m->x, BS, D);

    size_t layer_stride = kvcache_layer_stride(cache);
    size_t batch_stride = kvcache_batch_stride(cache);
    size_t head_stride = kvcache_head_stride(cache);
    size_t conv_stride = kvcache_conv_layer_stride(cache);
    size_t ssm_stride = kvcache_ssm_layer_stride(cache);

    for (int li = 0; li < n_layer; li++) {
        LayerWeights *L = &m->layers[li];
        rmsnorm_rows(m->x, L->attn_norm, m->h, BS, D, hp->rms_eps);

        if (L->is_gdn) {
            float *conv = cache->conv ? cache->conv + (size_t)li * conv_stride : NULL;
            float *ssm = cache->ssm ? cache->ssm + (size_t)li * ssm_stride : NULL;
            gdn_layer_forward(L, hp, m->h, m->y, B, S, m->valid_buf, conv, ssm, m->gdn_scratch);
            vec_add(m->x, m->y, m->x, BS * D);
        } else {
            float *layer_k = cache->k + (size_t)li * layer_stride;
            float *layer_v = cache->v + (size_t)li * layer_stride;
            full_attn_layer(m, L, B, S, D, H, KV, d, scale, layer_k, layer_v, batch_stride,
                            head_stride);
        }

        rmsnorm_rows(m->x, L->ffn_norm, m->h, BS, D, hp->rms_eps);
        linear_rows(L->gate, m->h, m->ffn_gate, BS, F, D);
        linear_rows(L->up, m->h, m->ffn_up, BS, F, D);
        silu_mul(m->ffn_gate, m->ffn_up, m->ffn_gate, BS * F);
        linear_rows_add(L->down, m->ffn_gate, m->x, BS, D, F);
    }

    rmsnorm_rows(m->x, m->output_norm, m->h, BS, D, hp->rms_eps);
    float *logits = logits_out ? logits_out : m->logits;
    for (int b = 0; b < B; b++) {
        int last = valid_len[b] - 1;
        if (last < 0) last = 0;
        if (last > S - 1) last = S - 1;
        const float *hlast = m->h + (size_t)(b * S + last) * (size_t)D;
        linear_rows(m->output, hlast, logits + (size_t)b * (size_t)hp->n_vocab, 1, hp->n_vocab, D);
        cache->n_seq[b] = m->key_len[b];
    }
    free(vl_local);
    return logits;
}
