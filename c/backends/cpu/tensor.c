#include "tensor.h"
#include <math.h>
#include <float.h>
#include <string.h>

void linear(const float *W, const float *x, float *y, int n_out, int n_in) {
    for (int i = 0; i < n_out; i++) {
        const float *w = W + (size_t)i * (size_t)n_in;
        float acc = 0.0f;
        for (int j = 0; j < n_in; j++) acc += w[j] * x[j];
        y[i] = acc;
    }
}

void linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    for (int t = 0; t < n_tok; t++) {
        linear(W, x + (size_t)t * (size_t)n_in, y + (size_t)t * (size_t)n_out, n_out, n_in);
    }
}

void linear_rows_add(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    for (int t = 0; t < n_tok; t++) {
        const float *xt = x + (size_t)t * (size_t)n_in;
        float *yt = y + (size_t)t * (size_t)n_out;
        for (int i = 0; i < n_out; i++) {
            const float *w = W + (size_t)i * (size_t)n_in;
            float acc = 0.0f;
            for (int j = 0; j < n_in; j++) acc += w[j] * xt[j];
            yt[i] += acc;
        }
    }
}

void rmsnorm(const float *x, const float *weight, float *y, int n, float eps) {
    float ms = 0.0f;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms /= (float)n;
    float inv = 1.0f / sqrtf(ms + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * weight[i];
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
    for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
}

void vec_mul(const float *a, const float *b, float *y, int n) {
    for (int i = 0; i < n; i++) y[i] = a[i] * b[i];
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
