#include "rope.h"
#include "simd.h"
#include "util.h"
#include <math.h>

void build_rope_cache(float *cos_tab, float *sin_tab, int seq_len, int head_dim, float theta) {
    int half = head_dim / 2;
    float *inv = xmalloc((size_t)half * sizeof(float));
    for (int i = 0; i < half; i++) {
        inv[i] = (float)pow((double)theta, -(double)i / (double)half);
    }
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < half; i++) {
            float freq = (float)t * inv[i];
            float c = cosf(freq);
            float s = sinf(freq);
            cos_tab[(size_t)t * head_dim + 2 * i] = c;
            cos_tab[(size_t)t * head_dim + 2 * i + 1] = c;
            sin_tab[(size_t)t * head_dim + 2 * i] = s;
            sin_tab[(size_t)t * head_dim + 2 * i + 1] = s;
        }
    }
    free(inv);
}

void apply_rope(float *x, const float *cos_tab, const float *sin_tab, const int *positions, int B,
                int n_head, int S, int head_dim) {
    /* After swap-adjacent: [-x1, x0, -x3, x2, ...] so y = x*c + xrot*s. set_ps is high→low. */
    const __m256 neg_even = _mm256_set_ps(0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int s = 0; s < S; s++) {
                int pos = positions[b * S + s];
                float *row = x + ((((size_t)b * n_head + (size_t)h) * S + (size_t)s) * (size_t)head_dim);
                const float *c = cos_tab + (size_t)pos * (size_t)head_dim;
                const float *si = sin_tab + (size_t)pos * (size_t)head_dim;
                int i = 0;
                for (; i + 8 <= head_dim; i += 8) {
                    __m256 vx = _mm256_loadu_ps(row + i);
                    __m256 vc = _mm256_loadu_ps(c + i);
                    __m256 vs = _mm256_loadu_ps(si + i);
                    __m256 xrot = _mm256_xor_ps(_mm256_permute_ps(vx, 0xB1), neg_even);
                    _mm256_storeu_ps(row + i, _mm256_fmadd_ps(xrot, vs, _mm256_mul_ps(vx, vc)));
                }
                int pairs = head_dim / 2;
                int p = i / 2;
                for (; p < pairs; p++) {
                    float x0 = row[2 * p];
                    float x1 = row[2 * p + 1];
                    float cv = c[2 * p];
                    float sv = si[2 * p];
                    row[2 * p] = x0 * cv - x1 * sv;
                    row[2 * p + 1] = x0 * sv + x1 * cv;
                }
            }
        }
    }
}
