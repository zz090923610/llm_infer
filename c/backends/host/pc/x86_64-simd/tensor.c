#include "tensor.h"
#include "pool.h"
#include "backend.h"
#include "simd.h"
#include <immintrin.h>
#include <math.h>
#include <float.h>
#include <string.h>

static float hsum256(__m256 v) { return llm_hsum256(v); }

static float dot_range(const float *w, const float *x, int n_in) {
    return llm_dot_f32(w, x, n_in);
}

static void linear_range(const float *W, const float *x, float *y, int n_out, int n_in, int i0,
                         int i1, int add) {
    (void)n_out;
    int i = i0;
    for (; i + 4 <= i1; i += 4) {
        const float *w0 = W + (size_t)(i + 0) * (size_t)n_in;
        const float *w1 = W + (size_t)(i + 1) * (size_t)n_in;
        const float *w2 = W + (size_t)(i + 2) * (size_t)n_in;
        const float *w3 = W + (size_t)(i + 3) * (size_t)n_in;
        if (i + 4 < i1) {
            _mm_prefetch((const char *)(W + (size_t)(i + 4) * (size_t)n_in), _MM_HINT_T0);
        }
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 16 <= n_in; j += 16) {
            __m256 x0 = _mm256_loadu_ps(x + j);
            __m256 x1 = _mm256_loadu_ps(x + j + 8);
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), x0, a00);
            a01 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j + 8), x1, a01);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), x0, a10);
            a11 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j + 8), x1, a11);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), x0, a20);
            a21 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j + 8), x1, a21);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), x0, a30);
            a31 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j + 8), x1, a31);
        }
        for (; j + 8 <= n_in; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            a00 = _mm256_fmadd_ps(_mm256_loadu_ps(w0 + j), xv, a00);
            a10 = _mm256_fmadd_ps(_mm256_loadu_ps(w1 + j), xv, a10);
            a20 = _mm256_fmadd_ps(_mm256_loadu_ps(w2 + j), xv, a20);
            a30 = _mm256_fmadd_ps(_mm256_loadu_ps(w3 + j), xv, a30);
        }
        float s0 = hsum256(_mm256_add_ps(a00, a01));
        float s1 = hsum256(_mm256_add_ps(a10, a11));
        float s2 = hsum256(_mm256_add_ps(a20, a21));
        float s3 = hsum256(_mm256_add_ps(a30, a31));
        for (; j < n_in; j++) {
            float xv = x[j];
            s0 += w0[j] * xv;
            s1 += w1[j] * xv;
            s2 += w2[j] * xv;
            s3 += w3[j] * xv;
        }
        y[i + 0] = add ? y[i + 0] + s0 : s0;
        y[i + 1] = add ? y[i + 1] + s1 : s1;
        y[i + 2] = add ? y[i + 2] + s2 : s2;
        y[i + 3] = add ? y[i + 3] + s3 : s3;
    }
    for (; i < i1; i++) {
        if (i + 1 < i1) {
            _mm_prefetch((const char *)(W + (size_t)(i + 1) * (size_t)n_in), _MM_HINT_T0);
        }
        float sum = dot_range(W + (size_t)i * (size_t)n_in, x, n_in);
        y[i] = add ? y[i] + sum : sum;
    }
}

static void linear_rows_range(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in,
                              int i0, int i1, int add) {
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
            __m256 a0 = _mm256_setzero_ps();
            __m256 a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps();
            __m256 a3 = _mm256_setzero_ps();
            int j = 0;
            for (; j + 8 <= n_in; j += 8) {
                __m256 vw = _mm256_loadu_ps(w + j);
                a0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x0 + j), a0);
                a1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x1 + j), a1);
                a2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x2 + j), a2);
                a3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x3 + j), a3);
            }
            float s0 = hsum256(a0);
            float s1 = hsum256(a1);
            float s2 = hsum256(a2);
            float s3 = hsum256(a3);
            for (; j < n_in; j++) {
                float wv = w[j];
                s0 += wv * x0[j];
                s1 += wv * x1[j];
                s2 += wv * x2[j];
                s3 += wv * x3[j];
            }
            y0[i] = add ? y0[i] + s0 : s0;
            y1[i] = add ? y1[i] + s1 : s1;
            y2[i] = add ? y2[i] + s2 : s2;
            y3[i] = add ? y3[i] + s3 : s3;
        }
    }
    for (; t < n_tok; t++) {
        linear_range(W, x + (size_t)t * (size_t)n_in, y + (size_t)t * (size_t)n_out, n_out, n_in, i0,
                     i1, add);
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
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ms = hsum256(acc);
    for (; i < n; i++) ms += x[i] * x[i];
    ms /= (float)n;
    float inv = 1.0f / sqrtf(ms + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
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
        _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    for (; i < n; i++) y[i] = a[i] + b[i];
}

void vec_mul(const float *a, const float *b, float *y, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
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
