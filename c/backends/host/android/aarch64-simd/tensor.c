#include "tensor.h"
#include "pool.h"
#include "backend.h"
#include <arm_neon.h>
#include <math.h>
#include <float.h>
#include <string.h>

static float hsum128(float32x4_t v) {
    return vaddvq_f32(v);
}

static float dot_neon(const float *w, const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + j), vld1q_f32(x + j));
        a1 = vfmaq_f32(a1, vld1q_f32(w + j + 4), vld1q_f32(x + j + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(w + j + 8), vld1q_f32(x + j + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(w + j + 12), vld1q_f32(x + j + 12));
    }
    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    for (; j + 4 <= n; j += 4) {
        a0 = vfmaq_f32(a0, vld1q_f32(w + j), vld1q_f32(x + j));
    }
    float sum = hsum128(a0);
    for (; j < n; j++) sum += w[j] * x[j];
    return sum;
}

static void linear_range(const float *W, const float *x, float *y, int n_out, int n_in, int i0,
                         int i1, int acc) {
    (void)n_out;
    int i = i0;
    for (; i + 4 <= i1; i += 4) {
        const float *w0 = W + (size_t)(i + 0) * (size_t)n_in;
        const float *w1 = W + (size_t)(i + 1) * (size_t)n_in;
        const float *w2 = W + (size_t)(i + 2) * (size_t)n_in;
        const float *w3 = W + (size_t)(i + 3) * (size_t)n_in;
        if (i + 4 < i1) {
            __builtin_prefetch(W + (size_t)(i + 4) * (size_t)n_in, 0, 3);
        }
        float32x4_t a00 = vdupq_n_f32(0.0f), a01 = vdupq_n_f32(0.0f);
        float32x4_t a10 = vdupq_n_f32(0.0f), a11 = vdupq_n_f32(0.0f);
        float32x4_t a20 = vdupq_n_f32(0.0f), a21 = vdupq_n_f32(0.0f);
        float32x4_t a30 = vdupq_n_f32(0.0f), a31 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 8 <= n_in; j += 8) {
            float32x4_t x0 = vld1q_f32(x + j);
            float32x4_t x1 = vld1q_f32(x + j + 4);
            a00 = vfmaq_f32(a00, vld1q_f32(w0 + j), x0);
            a01 = vfmaq_f32(a01, vld1q_f32(w0 + j + 4), x1);
            a10 = vfmaq_f32(a10, vld1q_f32(w1 + j), x0);
            a11 = vfmaq_f32(a11, vld1q_f32(w1 + j + 4), x1);
            a20 = vfmaq_f32(a20, vld1q_f32(w2 + j), x0);
            a21 = vfmaq_f32(a21, vld1q_f32(w2 + j + 4), x1);
            a30 = vfmaq_f32(a30, vld1q_f32(w3 + j), x0);
            a31 = vfmaq_f32(a31, vld1q_f32(w3 + j + 4), x1);
        }
        float s0 = hsum128(vaddq_f32(a00, a01));
        float s1 = hsum128(vaddq_f32(a10, a11));
        float s2 = hsum128(vaddq_f32(a20, a21));
        float s3 = hsum128(vaddq_f32(a30, a31));
        for (; j < n_in; j++) {
            float xv = x[j];
            s0 += w0[j] * xv;
            s1 += w1[j] * xv;
            s2 += w2[j] * xv;
            s3 += w3[j] * xv;
        }
        y[i + 0] = acc ? y[i + 0] + s0 : s0;
        y[i + 1] = acc ? y[i + 1] + s1 : s1;
        y[i + 2] = acc ? y[i + 2] + s2 : s2;
        y[i + 3] = acc ? y[i + 3] + s3 : s3;
    }
    for (; i < i1; i++) {
        if (i + 1 < i1) {
            __builtin_prefetch(W + (size_t)(i + 1) * (size_t)n_in, 0, 3);
        }
        float s = dot_neon(W + (size_t)i * (size_t)n_in, x, n_in);
        y[i] = acc ? y[i] + s : s;
    }
}

static void linear_rows_range(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in,
                              int i0, int i1, int acc) {
    int t = 0;
    for (; t + 4 <= n_tok; t += 4) {
        const float *x0 = x + (size_t)(t + 0) * (size_t)n_in;
        const float *x1 = x + (size_t)(t + 1) * (size_t)n_in;
        const float *x2 = x + (size_t)(t + 2) * (size_t)n_in;
        const float *x3 = x + (size_t)(t + 3) * (size_t)n_in;
        float *y0 = y + (size_t)(t + 0) * (size_t)n_out;
        float *y1 = y + (size_t)(t + 1) * (size_t)n_out;
        float *y2 = y + (size_t)(t + 2) * (size_t)n_out;
        float *y3 = y + (size_t)(t + 3) * (size_t)n_out;
        for (int i = i0; i < i1; i++) {
            const float *w = W + (size_t)i * (size_t)n_in;
            float32x4_t a0 = vdupq_n_f32(0.0f);
            float32x4_t a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f);
            float32x4_t a3 = vdupq_n_f32(0.0f);
            int j = 0;
            for (; j + 16 <= n_in; j += 16) {
                float32x4_t vw0 = vld1q_f32(w + j);
                float32x4_t vw1 = vld1q_f32(w + j + 4);
                float32x4_t vw2 = vld1q_f32(w + j + 8);
                float32x4_t vw3 = vld1q_f32(w + j + 12);
                a0 = vfmaq_f32(a0, vw0, vld1q_f32(x0 + j));
                a1 = vfmaq_f32(a1, vw0, vld1q_f32(x1 + j));
                a2 = vfmaq_f32(a2, vw0, vld1q_f32(x2 + j));
                a3 = vfmaq_f32(a3, vw0, vld1q_f32(x3 + j));
                a0 = vfmaq_f32(a0, vw1, vld1q_f32(x0 + j + 4));
                a1 = vfmaq_f32(a1, vw1, vld1q_f32(x1 + j + 4));
                a2 = vfmaq_f32(a2, vw1, vld1q_f32(x2 + j + 4));
                a3 = vfmaq_f32(a3, vw1, vld1q_f32(x3 + j + 4));
                a0 = vfmaq_f32(a0, vw2, vld1q_f32(x0 + j + 8));
                a1 = vfmaq_f32(a1, vw2, vld1q_f32(x1 + j + 8));
                a2 = vfmaq_f32(a2, vw2, vld1q_f32(x2 + j + 8));
                a3 = vfmaq_f32(a3, vw2, vld1q_f32(x3 + j + 8));
                a0 = vfmaq_f32(a0, vw3, vld1q_f32(x0 + j + 12));
                a1 = vfmaq_f32(a1, vw3, vld1q_f32(x1 + j + 12));
                a2 = vfmaq_f32(a2, vw3, vld1q_f32(x2 + j + 12));
                a3 = vfmaq_f32(a3, vw3, vld1q_f32(x3 + j + 12));
            }
            for (; j + 8 <= n_in; j += 8) {
                float32x4_t vw0 = vld1q_f32(w + j);
                float32x4_t vw1 = vld1q_f32(w + j + 4);
                a0 = vfmaq_f32(a0, vw0, vld1q_f32(x0 + j));
                a1 = vfmaq_f32(a1, vw0, vld1q_f32(x1 + j));
                a2 = vfmaq_f32(a2, vw0, vld1q_f32(x2 + j));
                a3 = vfmaq_f32(a3, vw0, vld1q_f32(x3 + j));
                a0 = vfmaq_f32(a0, vw1, vld1q_f32(x0 + j + 4));
                a1 = vfmaq_f32(a1, vw1, vld1q_f32(x1 + j + 4));
                a2 = vfmaq_f32(a2, vw1, vld1q_f32(x2 + j + 4));
                a3 = vfmaq_f32(a3, vw1, vld1q_f32(x3 + j + 4));
            }
            float s0 = hsum128(a0);
            float s1 = hsum128(a1);
            float s2 = hsum128(a2);
            float s3 = hsum128(a3);
            for (; j < n_in; j++) {
                float wv = w[j];
                s0 += wv * x0[j];
                s1 += wv * x1[j];
                s2 += wv * x2[j];
                s3 += wv * x3[j];
            }
            y0[i] = acc ? y0[i] + s0 : s0;
            y1[i] = acc ? y1[i] + s1 : s1;
            y2[i] = acc ? y2[i] + s2 : s2;
            y3[i] = acc ? y3[i] + s3 : s3;
        }
    }
    for (; t < n_tok; t++) {
        linear_range(W, x + (size_t)t * (size_t)n_in, y + (size_t)t * (size_t)n_out, n_out, n_in, i0,
                     i1, acc);
    }
}

typedef struct {
    const float *W;
    const float *x;
    float *y;
    int n_tok;
    int n_out;
    int n_in;
    int rows;
    int acc;
} LinearJob;

static void row_slice(int tid, int n_threads, int n_out, int *i0, int *i1) {
    int n = n_threads;
    if (n_out < n) n = n_out;
    if (n < 1) n = 1;
    if (tid >= n) {
        *i0 = 0;
        *i1 = 0;
        return;
    }
    *i0 = n_out * tid / n;
    *i1 = n_out * (tid + 1) / n;
}

static void linear_job(int tid, int n_threads, void *ctx) {
    LinearJob *j = (LinearJob *)ctx;
    int i0, i1;
    row_slice(tid, n_threads, j->n_out, &i0, &i1);
    if (i0 >= i1) return;
    if (j->rows) {
        linear_rows_range(j->W, j->x, j->y, j->n_tok, j->n_out, j->n_in, i0, i1, j->acc);
    } else {
        linear_range(j->W, j->x, j->y, j->n_out, j->n_in, i0, i1, j->acc);
    }
}

void linear(const float *W, const float *x, float *y, int n_out, int n_in) {
    LinearJob job = {W, x, y, 1, n_out, n_in, 0, 0};
    llm_pool_run(linear_job, &job, llm_backend_n_decode_threads());
}

void linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    if (n_tok <= 0) return;
    if (n_tok == 1) {
        linear(W, x, y, n_out, n_in);
        return;
    }
    LinearJob job = {W, x, y, n_tok, n_out, n_in, 1, 0};
    llm_pool_run(linear_job, &job, llm_backend_n_prefill_threads());
}

void linear_rows_add(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    if (n_tok <= 0) return;
    LinearJob job = {W, x, y, n_tok, n_out, n_in, n_tok > 1, 1};
    llm_pool_run(linear_job, &job, n_tok > 1 ? llm_backend_n_prefill_threads()
                                             : llm_backend_n_decode_threads());
}

void rmsnorm(const float *x, const float *weight, float *y, int n, float eps) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t v2 = vld1q_f32(x + i + 8);
        float32x4_t v3 = vld1q_f32(x + i + 12);
        acc = vfmaq_f32(acc, v0, v0);
        acc = vfmaq_f32(acc, v1, v1);
        acc = vfmaq_f32(acc, v2, v2);
        acc = vfmaq_f32(acc, v3, v3);
    }
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        acc = vfmaq_f32(acc, v, v);
    }
    float ms = hsum128(acc);
    for (; i < n; i++) ms += x[i] * x[i];
    ms /= (float)n;
    float inv = 1.0f / sqrtf(ms + eps);
    float32x4_t vinv = vdupq_n_f32(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(x0, vinv), w0));
        vst1q_f32(y + i + 4, vmulq_f32(vmulq_f32(x1, vinv), w1));
    }
    for (; i < n; i++) y[i] = x[i] * inv * weight[i];
}

void rmsnorm_rows(const float *x, const float *weight, float *y, int n_tok, int n, float eps) {
    for (int t = 0; t < n_tok; t++) {
        rmsnorm(x + (size_t)t * (size_t)n, weight, y + (size_t)t * (size_t)n, n, eps);
    }
}

void silu(const float *x, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

void silu_mul(const float *x, const float *g, float *y, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        y[i] = (v / (1.0f + expf(-v))) * g[i];
    }
}

void softmax_inplace(float *x, int n) {
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

void softmax_rows(float *x, int n_rows, int n) {
    for (int r = 0; r < n_rows; r++) softmax_inplace(x + (size_t)r * (size_t)n, n);
}

void vec_add(const float *a, const float *b, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(y + i + 4, vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    for (; i < n; i++) y[i] = a[i] + b[i];
}

void vec_mul(const float *a, const float *b, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        vst1q_f32(y + i + 4, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    for (; i < n; i++) y[i] = a[i] * b[i];
}

void embed_gather(const float *table, const int *ids, float *out, int n_rows, int d) {
    for (int t = 0; t < n_rows; t++)
        memcpy(out + (size_t)t * (size_t)d, table + (size_t)ids[t] * (size_t)d,
               (size_t)d * sizeof(float));
}

int argmax_f64(const double *x, int n) {
    int best = 0;
    double m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) {
            m = x[i];
            best = i;
        }
    }
    return best;
}

int argmax_f32(const float *x, int n) {
    int best = 0;
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) {
            m = x[i];
            best = i;
        }
    }
    return best;
}
