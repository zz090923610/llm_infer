#include "gdn.h"
#include "tensor.h"
#include "weight.h"
#include "util.h"
#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include "simd.h"
#endif

int layer_is_gdn(const LlamaHParams *hp, int il) {
    if (hp->arch != LLM_ARCH_QWEN35) return 0;
    int iv = hp->full_attention_interval > 0 ? hp->full_attention_interval : 4;
    return ((il + 1) % iv) != 0;
}

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static float softplus_f(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void l2norm(float *x, int n, float eps) {
#if defined(__AVX2__) && defined(__FMA__)
    float ss = llm_dot_f32(x, x, n);
    float inv = 1.0f / sqrtf(ss + eps);
    llm_scale_f32(x, inv, n);
#else
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss + eps);
    for (int i = 0; i < n; i++) x[i] *= inv;
#endif
}

static void gdn_delta_rule(float *S, const float *qv, const float *kv, const float *vv, float *ov,
                           float *delta, float gv, float beta, float scale, int d_state) {
    int ss = d_state * d_state;
#if defined(__AVX2__) && defined(__FMA__)
    llm_scale_f32(S, gv, ss);
    for (int j = 0; j < d_state; j++) {
        float sum = llm_dot_f32(S + (size_t)j * (size_t)d_state, kv, d_state);
        delta[j] = (vv[j] - sum) * beta;
    }
    for (int j = 0; j < d_state; j++) {
        llm_axpy_f32(S + (size_t)j * (size_t)d_state, delta[j], kv, d_state);
    }
    for (int j = 0; j < d_state; j++) {
        ov[j] = llm_dot_f32(S + (size_t)j * (size_t)d_state, qv, d_state) * scale;
    }
#else
    for (int i = 0; i < ss; i++) S[i] *= gv;
    for (int j = 0; j < d_state; j++) {
        float sum = 0.0f;
        const float *row = S + (size_t)j * (size_t)d_state;
        for (int i = 0; i < d_state; i++) sum += row[i] * kv[i];
        delta[j] = (vv[j] - sum) * beta;
    }
    for (int j = 0; j < d_state; j++) {
        float *row = S + (size_t)j * (size_t)d_state;
        float dj = delta[j];
        for (int i = 0; i < d_state; i++) row[i] += dj * kv[i];
    }
    for (int j = 0; j < d_state; j++) {
        float sum = 0.0f;
        const float *row = S + (size_t)j * (size_t)d_state;
        for (int i = 0; i < d_state; i++) sum += row[i] * qv[i];
        ov[j] = sum * scale;
    }
#endif
}

static void silu_mul_inplace(float *ov, const float *zv, int n) {
    for (int i = 0; i < n; i++) {
        float silu = zv[i] / (1.0f + expf(-zv[i]));
        ov[i] *= silu;
    }
}

void gdn_layer_forward(const LayerWeights *L, const LlamaHParams *hp, const float *x, float *y,
                       int B, int S, const int *valid_len, float *conv_state, float *ssm_state,
                       float *scratch) {
    const int D = hp->n_embd;
    const int d_conv = hp->ssm_d_conv;
    const int d_state = hp->ssm_d_state;
    const int n_k_heads = hp->ssm_n_group;
    const int n_v_heads = hp->ssm_dt_rank;
    const int key_dim = d_state * n_k_heads;
    const int value_dim = hp->ssm_d_inner;
    const int conv_dim = 2 * key_dim + value_dim;
    const int hist = d_conv > 1 ? d_conv - 1 : 0;
    const float scale = 1.0f / sqrtf((float)d_state);
    const float eps = hp->rms_eps;

    float *qkv = scratch;
    float *z = qkv + conv_dim;
    float *beta = z + value_dim;
    float *g = beta + n_v_heads;
    float *mix = g + n_v_heads;
    float *q = mix + conv_dim;
    float *k = q + n_v_heads * d_state;
    float *v = k + n_v_heads * d_state;
    float *delta = v + value_dim;
    float *out = delta + d_state;
    float *q_raw = out + value_dim;
    float *k_raw = q_raw + key_dim;

    memset(y, 0, (size_t)B * (size_t)S * (size_t)D * sizeof(float));

    for (int b = 0; b < B; b++) {
        int n = valid_len ? valid_len[b] : S;
        float *conv_b = conv_state + (size_t)b * (size_t)conv_dim * (size_t)hist;
        float *ssm_b = ssm_state + (size_t)b * (size_t)n_v_heads * (size_t)d_state * (size_t)d_state;
        for (int t = 0; t < n; t++) {
            const float *xt = x + (size_t)(b * S + t) * (size_t)D;
            linear_wt(L->wqkv, xt, qkv, conv_dim, D);
            linear_wt(L->attn_gate, xt, z, value_dim, D);
            linear_wt(L->ssm_beta, xt, beta, n_v_heads, D);
            linear_wt(L->ssm_alpha, xt, g, n_v_heads, D);
            for (int i = 0; i < n_v_heads; i++) {
                beta[i] = sigmoid_f(beta[i]);
                g[i] = softplus_f(g[i] + L->ssm_dt[i]) * L->ssm_a[i];
            }

            for (int c = 0; c < conv_dim; c++) {
                float acc = 0.0f;
                const float *w = L->ssm_conv1d + (size_t)c * (size_t)d_conv;
                float *st = conv_b + (size_t)c * (size_t)hist;
                for (int k = 0; k < hist; k++) acc += w[k] * st[k];
                acc += w[hist] * qkv[c];
                mix[c] = acc / (1.0f + expf(-acc));
                if (hist > 0) {
                    memmove(st, st + 1, (size_t)(hist - 1) * sizeof(float));
                    st[hist - 1] = qkv[c];
                }
            }

            memcpy(q_raw, mix, (size_t)key_dim * sizeof(float));
            memcpy(k_raw, mix + key_dim, (size_t)key_dim * sizeof(float));
            memcpy(v, mix + 2 * key_dim, (size_t)value_dim * sizeof(float));
            for (int h = 0; h < n_k_heads; h++) {
                l2norm(q_raw + h * d_state, d_state, eps);
                l2norm(k_raw + h * d_state, d_state, eps);
            }
            /* llama.cpp qwen35 CPU GDN broadcasts with h % n_k (tile), not
               repeat_interleave. MiniCPM has n_k == n_v so both agree. */
            for (int vh = 0; vh < n_v_heads; vh++) {
                int kh = n_k_heads > 0 ? vh % n_k_heads : 0;
                memcpy(q + vh * d_state, q_raw + kh * d_state, (size_t)d_state * sizeof(float));
                memcpy(k + vh * d_state, k_raw + kh * d_state, (size_t)d_state * sizeof(float));
            }

            for (int vh = 0; vh < n_v_heads; vh++) {
                float *S = ssm_b + (size_t)vh * (size_t)d_state * (size_t)d_state;
                const float *qv = q + vh * d_state;
                const float *kv = k + vh * d_state;
                const float *vv = v + vh * d_state;
                float *ov = out + vh * d_state;
                gdn_delta_rule(S, qv, kv, vv, ov, delta, expf(g[vh]), beta[vh], scale, d_state);
                rmsnorm(ov, L->ssm_norm, ov, d_state, eps);
                silu_mul_inplace(ov, z + vh * d_state, d_state);
            }
            linear_wt(L->ssm_out, out, y + (size_t)(b * S + t) * (size_t)D, D, value_dim);
        }
    }
}
