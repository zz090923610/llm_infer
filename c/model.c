#include "model.h"
#include "attn.h"
#include "backend.h"
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
    llm_backend_intern_weight(m->rope_cos, (size_t)n * (size_t)hp->head_dim * sizeof(float));
    llm_backend_intern_weight(m->rope_sin, (size_t)n * (size_t)hp->head_dim * sizeof(float));
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
    free(m->valid_buf);
    free(m->tok_ids);
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
    m->valid_buf = xmalloc((size_t)b * sizeof(int));
    m->tok_ids = xmalloc((size_t)b * (size_t)s * sizeof(int));
    llm_backend_intern_rw(m->x, BS * D * sizeof(float));
    llm_backend_intern_rw(m->h, BS * D * sizeof(float));
    llm_backend_intern_rw(m->q, (size_t)b * H * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->k, (size_t)b * KV * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->v, (size_t)b * KV * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->y, (size_t)b * H * (size_t)s * d * sizeof(float));
    llm_backend_intern_rw(m->ffn_gate, BS * F * sizeof(float));
    llm_backend_intern_rw(m->ffn_up, BS * F * sizeof(float));
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
    /* Upload weights before the first timed forward (no-op on CPU backends). */
    const LlamaHParams *hp = &m->hparams;
    int D = hp->n_embd, H = hp->n_head, KV = hp->n_head_kv, d = hp->head_dim, F = hp->n_ff;
    llm_backend_intern_weight_f16(m->tok_embd, (size_t)hp->n_vocab * (size_t)D * sizeof(float));
    llm_backend_intern_weight(m->output_norm, (size_t)D * sizeof(float));
    for (int i = 0; i < hp->n_layer; i++) {
        LayerWeights *L = &m->layers[i];
        llm_backend_intern_weight(L->attn_norm, (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->wq, (size_t)H * (size_t)d * (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->wk, (size_t)KV * (size_t)d * (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->wv, (size_t)KV * (size_t)d * (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->wo, (size_t)D * (size_t)H * (size_t)d * sizeof(float));
        llm_backend_intern_weight(L->ffn_norm, (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->gate, (size_t)F * (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->up, (size_t)F * (size_t)D * sizeof(float));
        llm_backend_intern_weight_f16(L->down, (size_t)D * (size_t)F * sizeof(float));
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
    free(m->valid_buf);
    free(m->tok_ids);
    free(m);
}

KVCache *llama_model_new_cache(LlamaModel *m, int batch, int max_seq) {
    if (max_seq <= 0) max_seq = m->hparams.n_ctx;
    KVCache *c = kvcache_create(&m->hparams, batch, max_seq);
    size_t layer = kvcache_layer_stride(c) * sizeof(float);
    for (int i = 0; i < c->n_layer; i++) {
        llm_backend_intern_device(c->k + (size_t)i * kvcache_layer_stride(c), layer);
        llm_backend_intern_device(c->v + (size_t)i * kvcache_layer_stride(c), layer);
    }
    return c;
}

float *llama_model_forward(LlamaModel *m, const int *tokens, int B, int S, KVCache *cache,
                           const int *valid_len, float *logits_out) {
    const LlamaHParams *hp = &m->hparams;
    int D = hp->n_embd, H = hp->n_head, KV = hp->n_head_kv, d = hp->head_dim, F = hp->n_ff;
    float scale = 1.0f / sqrtf((float)d);
    int BS = B * S;

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

    for (int li = 0; li < hp->n_layer; li++) {
        LayerWeights *L = &m->layers[li];
        rmsnorm_rows(m->x, L->attn_norm, m->h, BS, D, hp->rms_eps);

        /* q,k,v projections; reuse y / ffn_gate / ffn_up as q_lin / k_lin / v_lin */
        linear_rows(L->wq, m->h, m->y, BS, H * d, D);
        linear_rows(L->wk, m->h, m->ffn_gate, BS, KV * d, D);
        linear_rows(L->wv, m->h, m->ffn_up, BS, KV * d, D);

        float *layer_k = cache->k + (size_t)li * layer_stride;
        float *layer_v = cache->v + (size_t)li * layer_stride;
        int max_k = 0;
        for (int b = 0; b < B; b++)
            if (m->key_len[b] > max_k) max_k = m->key_len[b];

        /* S==1: (B,S,H,d) matches packed (B,H,S,d), so skip pack/merge. */
        if (S == 1) {
            apply_rope(m->y, m->rope_cos, m->rope_sin, m->positions, B, H, S, d);
            apply_rope(m->ffn_gate, m->rope_cos, m->rope_sin, m->positions, B, KV, S, d);
            attn_cache_store(layer_k, layer_v, m->ffn_gate, m->ffn_up, batch_stride, head_stride,
                             m->valid_buf, m->starts, B, S, KV, d);
            attn_gqa(m->y, m->attn, m->y, layer_k, layer_v, batch_stride, head_stride, m->positions,
                     m->valid_buf, m->key_len, B, H, S, KV, d, max_k, scale);
            linear_rows_add(L->wo, m->y, m->x, BS, D, D);
        } else {
            attn_pack_heads(m->y, m->q, B, S, H, d);
            attn_pack_heads(m->ffn_gate, m->k, B, S, KV, d);
            attn_pack_heads(m->ffn_up, m->v, B, S, KV, d);
            apply_rope(m->q, m->rope_cos, m->rope_sin, m->positions, B, H, S, d);
            apply_rope(m->k, m->rope_cos, m->rope_sin, m->positions, B, KV, S, d);
            attn_cache_store(layer_k, layer_v, m->k, m->v, batch_stride, head_stride, m->valid_buf,
                             m->starts, B, S, KV, d);
            attn_gqa(m->y, m->attn, m->q, layer_k, layer_v, batch_stride, head_stride, m->positions,
                     m->valid_buf, m->key_len, B, H, S, KV, d, max_k, scale);
            attn_merge_heads(m->y, m->h, B, S, H, d);
            linear_rows_add(L->wo, m->h, m->x, BS, D, D);
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
        linear_rows(m->tok_embd, hlast, logits + (size_t)b * (size_t)hp->n_vocab, 1, hp->n_vocab, D);
        cache->n_seq[b] = m->key_len[b];
    }
    free(vl_local);
    return logits;
}
