#include "tensor.h"
#include "backend.h"
#include "attn.h"
#include "rope.h"
#include "weight.h"
#include "quant.h"
#include "gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>

static int fails;
static uint32_t rng_state = 1u;

static void expect_eq(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fails++;
    }
}

static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

static void fill_rand(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = frand();
}

static void linear_ref(const float *W, const float *x, float *y, int n_out, int n_in) {
    for (int i = 0; i < n_out; i++) {
        const float *w = W + (size_t)i * (size_t)n_in;
        double acc = 0.0;
        for (int j = 0; j < n_in; j++) acc += (double)w[j] * (double)x[j];
        y[i] = (float)acc;
    }
}

static void check_close(const float *got, const float *ref, int n, int n_in, const char *tag) {
    float max_abs = 0.0f;
    float tol = 1e-4f * (float)n_in;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > max_abs) max_abs = e;
        if (e > tol) {
            fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", tag, i, got[i], ref[i], e,
                    tol);
            fails++;
            return;
        }
    }
    (void)max_abs;
}

static void test_linear_shape(int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_in);
    linear(W, x, y, n_out, n_in);
    linear_ref(W, x, ref, n_out, n_in);
    char tag[64];
    snprintf(tag, sizeof(tag), "linear n_out=%d n_in=%d", n_out, n_in);
    check_close(y, ref, n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void test_linear_rows_shape(int n_tok, int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear_rows");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_tok * n_in);
    linear_rows(W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        linear_ref(W, x + (size_t)t * (size_t)n_in, ref + (size_t)t * (size_t)n_out, n_out, n_in);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_rows n_tok=%d n_out=%d n_in=%d", n_tok, n_out, n_in);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void test_linear_rows_add_shape(int n_tok, int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear_rows_add");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_tok * n_in);
    fill_rand(y, n_tok * n_out);
    memcpy(ref, y, (size_t)n_tok * (size_t)n_out * sizeof(float));
    llm_backend_host_write(y);
    linear_rows_add(W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        float *tmp = malloc((size_t)n_out * sizeof(float));
        linear_ref(W, x + (size_t)t * (size_t)n_in, tmp, n_out, n_in);
        for (int i = 0; i < n_out; i++)
            ref[(size_t)t * (size_t)n_out + i] += tmp[i];
        free(tmp);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_rows_add n_tok=%d n_out=%d n_in=%d", n_tok, n_out, n_in);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void check_close_f16(const float *got, const float *ref, int n, int n_in, const char *tag) {
    float max_abs = 0.0f;
    float tol = 5e-3f * (float)n_in + 0.05f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > max_abs) max_abs = e;
        if (e > tol) {
            fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", tag, i, got[i], ref[i], e,
                    tol);
            fails++;
            return;
        }
    }
    (void)max_abs;
}

static void test_linear_q8(int n_tok, int n_out, int n_in) {
    expect_eq(n_in % QK8_0 == 0, "q8 n_in aligned");
    int n = n_out * n_in;
    unsigned char *pack = malloc((size_t)(n / QK8_0) * BLOCK_Q8_0);
    float *Wf = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(pack && Wf && x && y && ref, "alloc linear q8");
    if (!pack || !Wf || !x || !y || !ref) {
        free(pack);
        free(Wf);
        free(x);
        free(y);
        free(ref);
        return;
    }
    uint16_t hs = 0x3c00; /* fp16 1.0 */
    for (int b = 0; b < n / QK8_0; b++) {
        unsigned char *blk = pack + (size_t)b * BLOCK_Q8_0;
        memcpy(blk, &hs, 2);
        memset(blk + 2, 2, QK8_0); /* qs=2 → w=2 */
    }
    expect_eq(dequantize_q8_0(pack, n, Wf) == 0, "q8 pack dequant");
    fill_rand(x, n_tok * n_in);
    WeightTensor W;
    memset(&W, 0, sizeof(W));
    W.name = "q8W";
    W.data = pack;
    W.ggml_type = GGML_Q8_0;
    W.n_elements = n;
    W.ndim = 2;
    W.shape[0] = n_out;
    W.shape[1] = n_in;
    llm_backend_intern_weight_q8(&W, pack, n);
    linear_rows_wt(&W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        linear_ref(Wf, x + (size_t)t * (size_t)n_in, ref + (size_t)t * (size_t)n_out, n_out, n_in);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_wt q8 n_tok=%d n_out=%d n_in=%d", n_tok, n_out, n_in);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(pack);
    free(Wf);
    free(x);
    free(y);
    free(ref);
}

static void test_embed_q8(void) {
    const int n_vocab = 8, d = 32;
    unsigned char pack[8 * BLOCK_Q8_0];
    float table[8 * 32];
    uint16_t hs = 0x3c00;
    for (int r = 0; r < n_vocab; r++) {
        unsigned char *blk = pack + (size_t)r * BLOCK_Q8_0;
        memcpy(blk, &hs, 2);
        memset(blk + 2, (int8_t)(r + 1), QK8_0);
    }
    expect_eq(dequantize_q8_0(pack, n_vocab * d, table) == 0, "embed pack dequant");
    WeightTensor W;
    memset(&W, 0, sizeof(W));
    W.name = "emb";
    W.data = pack;
    W.ggml_type = GGML_Q8_0;
    W.n_elements = n_vocab * d;
    int ids[3] = {0, 3, 7};
    float out[3 * 32], ref[3 * 32];
    embed_gather_wt(&W, ids, out, 3, d);
    llm_backend_sync();
    llm_backend_intern_weight(table, (size_t)n_vocab * (size_t)d * sizeof(float));
    embed_gather(table, ids, ref, 3, d);
    llm_backend_sync();
    check_close(out, ref, 3 * d, d, "embed_gather_wt q8");
}

static void fill_q4k_block(unsigned char *blk) {
    memset(blk, 0, BLOCK_Q4_K);
    uint16_t d = 0x3c00, dmin = 0x3800; /* 1.0, 0.5 */
    memcpy(blk, &d, 2);
    memcpy(blk + 2, &dmin, 2);
    blk[4] = blk[5] = blk[6] = blk[7] = 1;
    memset(blk + 16, 0x22, 128);
}

static void fill_q6k_block(unsigned char *blk) {
    memset(blk, 0, BLOCK_Q6_K);
    memset(blk + 128 + 64, 1, 16);
    uint16_t d = 0x3c00;
    memcpy(blk + 128 + 64 + 16, &d, 2);
}

static void fill_iq4_block(unsigned char *blk) {
    memset(blk, 0, BLOCK_IQ4_XS);
    uint16_t d = 0x3c00;
    memcpy(blk, &d, 2);
}

static void test_linear_kquant(int ggml_type, int n_tok, int n_out) {
    int n_in = QK_K;
    int n = n_out * n_in;
    int blksz = ggml_type_size(ggml_type);
    unsigned char *pack = malloc((size_t)(n / QK_K) * (size_t)blksz);
    float *Wf = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(pack && Wf && x && y && ref, "alloc linear kquant");
    if (!pack || !Wf || !x || !y || !ref) {
        free(pack);
        free(Wf);
        free(x);
        free(y);
        free(ref);
        return;
    }
    int nb = n / QK_K;
    for (int b = 0; b < nb; b++) {
        unsigned char *blk = pack + (size_t)b * (size_t)blksz;
        if (ggml_type == GGML_Q4_K) fill_q4k_block(blk);
        else if (ggml_type == GGML_Q6_K) fill_q6k_block(blk);
        else fill_iq4_block(blk);
    }
    expect_eq(dequantize_row(ggml_type, pack, n, Wf) == 0, "kquant dequant");
    fill_rand(x, n_tok * n_in);
    WeightTensor W;
    memset(&W, 0, sizeof(W));
    W.name = "kW";
    W.data = pack;
    W.ggml_type = ggml_type;
    W.n_elements = n;
    W.ndim = 2;
    W.shape[0] = n_out;
    W.shape[1] = n_in;
    llm_backend_intern_weight_quant(&W, pack, n, ggml_type);
    linear_rows_wt(&W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        linear_ref(Wf, x + (size_t)t * (size_t)n_in, ref + (size_t)t * (size_t)n_out, n_out, n_in);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_wt type=%d n_tok=%d n_out=%d", ggml_type, n_tok, n_out);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(pack);
    free(Wf);
    free(x);
    free(y);
    free(ref);
}

static void test_embed_kquant(void) {
    const int n_vocab = 4, d = QK_K;
    unsigned char pack[4 * BLOCK_Q4_K];
    float table[4 * QK_K];
    for (int r = 0; r < n_vocab; r++) fill_q4k_block(pack + (size_t)r * BLOCK_Q4_K);
    expect_eq(dequantize_q4_k(pack, n_vocab * d, table) == 0, "embed q4_k dequant");
    WeightTensor W;
    memset(&W, 0, sizeof(W));
    W.name = "emb4";
    W.data = pack;
    W.ggml_type = GGML_Q4_K;
    W.n_elements = n_vocab * d;
    int ids[3] = {0, 2, 3};
    float out[3 * QK_K], ref[3 * QK_K];
    embed_gather_wt(&W, ids, out, 3, d);
    llm_backend_sync();
    llm_backend_intern_weight(table, (size_t)n_vocab * (size_t)d * sizeof(float));
    embed_gather(table, ids, ref, 3, d);
    llm_backend_sync();
    check_close(out, ref, 3 * d, d, "embed_gather_wt q4_k");
}

static void test_linear_f16_texel(int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear f16");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_in);
    llm_backend_intern_weight_f16(W, (size_t)n_out * (size_t)n_in * sizeof(float));
    linear_rows(W, x, y, 1, n_out, n_in);
    llm_backend_sync();
    linear_ref(W, x, ref, n_out, n_in);
    char tag[64];
    snprintf(tag, sizeof(tag), "linear f16 texel n_out=%d n_in=%d", n_out, n_in);
    check_close_f16(y, ref, n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void check_close_abs(const float *got, const float *ref, int n, float tol, const char *tag) {
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > tol) {
            fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", tag, i, got[i], ref[i], e,
                    tol);
            fails++;
            return;
        }
    }
}

static void apply_rope_ref(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                           int B, int n_head, int S, int head_dim) {
    int pairs = head_dim / 2;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int s = 0; s < S; s++) {
                int pos = positions[b * S + s];
                float *row = x + ((((size_t)b * n_head + (size_t)h) * S + (size_t)s) * (size_t)head_dim);
                const float *c = cos_tab + (size_t)pos * (size_t)head_dim;
                const float *si = sin_tab + (size_t)pos * (size_t)head_dim;
                for (int i = 0; i < pairs; i++) {
                    float x0 = row[2 * i];
                    float x1 = row[2 * i + 1];
                    float cv = c[2 * i];
                    float sv = si[2 * i];
                    row[2 * i] = x0 * cv - x1 * sv;
                    row[2 * i + 1] = x0 * sv + x1 * cv;
                }
            }
        }
    }
}

static void apply_rope_neox_ref(float *x, const float *cos_tab, const float *sin_tab,
                                const int *positions, int B, int n_head, int S, int head_dim,
                                int n_rot) {
    int half = n_rot / 2;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int s = 0; s < S; s++) {
                int pos = positions[b * S + s];
                float *row = x + ((((size_t)b * n_head + (size_t)h) * S + (size_t)s) * (size_t)head_dim);
                const float *c = cos_tab + (size_t)pos * (size_t)n_rot;
                const float *si = sin_tab + (size_t)pos * (size_t)n_rot;
                for (int i = 0; i < half; i++) {
                    float x0 = row[i];
                    float x1 = row[i + half];
                    float cv = c[2 * i];
                    float sv = si[2 * i];
                    row[i] = x0 * cv - x1 * sv;
                    row[i + half] = x0 * sv + x1 * cv;
                }
            }
        }
    }
}

static void test_apply_rope(void) {
    const int B = 1, H = 3, S = 4, d = 64, seq = 8;
    int n = B * H * S * d;
    float *x = malloc((size_t)n * sizeof(float));
    float *ref = malloc((size_t)n * sizeof(float));
    float *cos_tab = malloc((size_t)seq * d * sizeof(float));
    float *sin_tab = malloc((size_t)seq * d * sizeof(float));
    int pos[4] = {0, 1, 2, 7};
    expect_eq(x && ref && cos_tab && sin_tab, "alloc rope");
    if (!x || !ref || !cos_tab || !sin_tab) return;
    fill_rand(x, n);
    memcpy(ref, x, (size_t)n * sizeof(float));
    build_rope_cache(cos_tab, sin_tab, seq, d, 10000.0f);
    apply_rope(x, cos_tab, sin_tab, pos, B, H, S, d);
    llm_backend_sync();
    apply_rope_ref(ref, cos_tab, sin_tab, pos, B, H, S, d);
    check_close_abs(x, ref, n, 1e-5f, "apply_rope");
    free(x);
    free(ref);
    free(cos_tab);
    free(sin_tab);
}

static void test_apply_rope_neox(void) {
    const int B = 1, H = 2, S = 3, d = 256, n_rot = 64, seq = 8;
    int n = B * H * S * d;
    float *x = malloc((size_t)n * sizeof(float));
    float *ref = malloc((size_t)n * sizeof(float));
    float *cos_tab = malloc((size_t)seq * n_rot * sizeof(float));
    float *sin_tab = malloc((size_t)seq * n_rot * sizeof(float));
    int pos[3] = {0, 3, 5};
    expect_eq(x && ref && cos_tab && sin_tab, "alloc rope neox");
    if (!x || !ref || !cos_tab || !sin_tab) return;
    fill_rand(x, n);
    memcpy(ref, x, (size_t)n * sizeof(float));
    build_rope_cache_n(cos_tab, sin_tab, seq, n_rot, 1000000.0f);
    apply_rope_neox(x, cos_tab, sin_tab, pos, B, H, S, d, n_rot);
    llm_backend_sync();
    apply_rope_neox_ref(ref, cos_tab, sin_tab, pos, B, H, S, d, n_rot);
    check_close_abs(x, ref, n, 1e-5f, "apply_rope_neox");
    {
        const int d2 = 128;
        int n2 = B * H * S * d2;
        float *x2 = malloc((size_t)n2 * sizeof(float));
        float *r2 = malloc((size_t)n2 * sizeof(float));
        float *c2 = malloc((size_t)seq * d2 * sizeof(float));
        float *s2 = malloc((size_t)seq * d2 * sizeof(float));
        expect_eq(x2 && r2 && c2 && s2, "alloc rope neox full");
        if (x2 && r2 && c2 && s2) {
            fill_rand(x2, n2);
            memcpy(r2, x2, (size_t)n2 * sizeof(float));
            build_rope_cache_n(c2, s2, seq, d2, 10000.0f);
            apply_rope_neox(x2, c2, s2, pos, B, H, S, d2, d2);
            llm_backend_sync();
            apply_rope_neox_ref(r2, c2, s2, pos, B, H, S, d2, d2);
            check_close_abs(x2, r2, n2, 1e-5f, "apply_rope_neox_full");
        }
        free(x2);
        free(r2);
        free(c2);
        free(s2);
    }
    free(x);
    free(ref);
    free(cos_tab);
    free(sin_tab);
}

static void softmax_ref(float *x, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        float v = isfinite(x[i]) ? x[i] : -1e9f;
        x[i] = v;
        if (v > m) m = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void attn_gqa_ref(float *y, float *attn, const float *q, const float *cache_k,
                         const float *cache_v, size_t batch_stride, size_t head_stride,
                         const int *positions, const int *valid_len, const int *key_len, int B,
                         int n_head, int S, int n_head_kv, int head_dim, int max_k, float scale) {
    int n_rep = n_head / n_head_kv;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            int kv_h = h / n_rep;
            for (int s = 0; s < S; s++) {
                float *row = attn + ((((size_t)b * n_head + (size_t)h) * (size_t)S + (size_t)s) *
                                     (size_t)max_k);
                int qpos = positions[b * S + s];
                int q_valid = s < valid_len[b];
                const float *qrow = q + ((((size_t)b * n_head + (size_t)h) * (size_t)S + (size_t)s) *
                                         (size_t)head_dim);
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
                softmax_ref(row, max_k);
                float *yrow = y + ((((size_t)b * n_head + (size_t)h) * (size_t)S + (size_t)s) *
                                   (size_t)head_dim);
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

static void test_attn_gqa(void) {
    const int B = 1, H = 4, KV = 2, S = 2, d = 128, max_k = 8;
    size_t head_stride = (size_t)max_k * (size_t)d;
    size_t batch_stride = (size_t)KV * head_stride;
    int nq = B * H * S * d;
    int nk = B * KV * max_k * d;
    float *q = malloc((size_t)nq * sizeof(float));
    float *ck = malloc((size_t)nk * sizeof(float));
    float *cv = malloc((size_t)nk * sizeof(float));
    float *y = malloc((size_t)nq * sizeof(float));
    float *yref = malloc((size_t)nq * sizeof(float));
    float *attn = malloc((size_t)B * H * S * max_k * sizeof(float));
    float *areff = malloc((size_t)B * H * S * max_k * sizeof(float));
    int pos[2] = {5, 6};
    int valid[1] = {2};
    int klen[1] = {7};
    expect_eq(q && ck && cv && y && yref && attn && areff, "alloc attn");
    if (!q || !ck || !cv || !y || !yref || !attn || !areff) return;
    fill_rand(q, nq);
    fill_rand(ck, nk);
    fill_rand(cv, nk);
    memset(y, 0, (size_t)nq * sizeof(float));
    memset(yref, 0, (size_t)nq * sizeof(float));
    float scale = 1.0f / sqrtf((float)d);
    attn_gqa(y, attn, q, ck, cv, batch_stride, head_stride, pos, valid, klen, B, H, S, KV, d, max_k,
             scale);
    llm_backend_sync();
    attn_gqa_ref(yref, areff, q, ck, cv, batch_stride, head_stride, pos, valid, klen, B, H, S, KV, d,
                 max_k, scale);
    check_close_abs(y, yref, nq, 2e-4f, "attn_gqa");
    free(q);
    free(ck);
    free(cv);
    free(y);
    free(yref);
    free(attn);
    free(areff);
}

static void test_host_read_preserves_cpu_write(void) {
    const int n = 64;
    float *a = malloc((size_t)n * sizeof(float));
    float *b = malloc((size_t)n * sizeof(float));
    float *y = malloc((size_t)n * sizeof(float));
    expect_eq(a && b && y, "alloc host_rw");
    if (!a || !b || !y) return;
    fill_rand(a, n);
    fill_rand(b, n);
    vec_add(a, b, y, n);
    llm_backend_sync();
    for (int i = 0; i < n; i++) y[i] = (float)i;
    llm_backend_host_write(y);
    llm_backend_host_read(y);
    for (int i = 0; i < n; i++) {
        if (y[i] != (float)i) {
            fprintf(stderr, "FAIL: host_read clobbered CPU write y[%d]=%g\n", i, y[i]);
            fails++;
            break;
        }
    }
    vec_add(a, b, y, n);
    llm_backend_host_read(y);
    float *ref = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) ref[i] = a[i] + b[i];
    check_close_abs(y, ref, n, 1e-5f, "host_read after gpu write");
    free(a);
    free(b);
    free(y);
    free(ref);
}

int main(void) {
    llm_backend_set_threads(4);
    const int n_ins[] = {960, 2560, 7, 64};
    for (int k = 0; k < 4; k++) {
        test_linear_shape(16, n_ins[k]);
        test_linear_rows_shape(1, 16, n_ins[k]);
        test_linear_rows_shape(5, 16, n_ins[k]);
    }
    test_linear_shape(960, 960);
    test_linear_rows_shape(1, 960, 960);
    test_linear_rows_shape(4, 2560, 960);
    test_linear_rows_shape(32, 960, 960);
    test_linear_rows_add_shape(1, 16, 960);
    test_linear_rows_add_shape(4, 16, 960);
    test_linear_rows_add_shape(5, 960, 960);
    test_linear_q8(1, 16, 64);
    test_linear_q8(1, 16, 960);
    test_linear_q8(4, 16, 64);
    test_linear_q8(1, 960, 960);
    test_linear_q8(5, 960, 960);
    test_embed_q8();
    test_linear_kquant(GGML_Q4_K, 1, 8);
    test_linear_kquant(GGML_Q4_K, 4, 8);
    test_linear_kquant(GGML_Q6_K, 1, 4);
    test_linear_kquant(GGML_IQ4_XS, 1, 4);
    test_embed_kquant();
    test_linear_f16_texel(960, 960);
    test_linear_f16_texel(16, 960);
    test_apply_rope();
    test_apply_rope_neox();
    test_attn_gqa();
    test_host_read_preserves_cpu_write();
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("test_linear ok\n");
    return 0;
}
