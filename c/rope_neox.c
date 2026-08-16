#include "rope.h"
#include "backend.h"
#include "util.h"
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include "simd.h"
#endif

void build_rope_cache_n(float *cos_tab, float *sin_tab, int seq_len, int n_rot, float theta) {
    int half = n_rot / 2;
    if (half <= 0) return;
    float *inv = xmalloc((size_t)half * sizeof(float));
    for (int i = 0; i < half; i++) {
        inv[i] = (float)pow((double)theta, -(double)i / (double)half);
    }
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < half; i++) {
            float freq = (float)t * inv[i];
            float c = cosf(freq);
            float s = sinf(freq);
            cos_tab[(size_t)t * n_rot + 2 * i] = c;
            cos_tab[(size_t)t * n_rot + 2 * i + 1] = c;
            sin_tab[(size_t)t * n_rot + 2 * i] = s;
            sin_tab[(size_t)t * n_rot + 2 * i + 1] = s;
        }
    }
    free(inv);
}

void apply_rope_neox(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                     int B, int n_head, int S, int head_dim, int n_rot) {
    if (n_rot <= 0) n_rot = head_dim;
    if (n_rot > head_dim) n_rot = head_dim;
    int half = n_rot / 2;
    llm_backend_host_read(x);
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int s = 0; s < S; s++) {
                int pos = positions[b * S + s];
                float *row = x + ((((size_t)b * n_head + (size_t)h) * S + (size_t)s) * (size_t)head_dim);
                const float *c = cos_tab + (size_t)pos * (size_t)n_rot;
                const float *si = sin_tab + (size_t)pos * (size_t)n_rot;
                int i = 0;
#if defined(__AVX2__) && defined(__FMA__)
                for (; i + 8 <= half; i += 8) {
                    __m256 x0 = _mm256_loadu_ps(row + i);
                    __m256 x1 = _mm256_loadu_ps(row + i + half);
                    __m256 vc = llm_pack_even(_mm256_loadu_ps(c + 2 * i), _mm256_loadu_ps(c + 2 * i + 8));
                    __m256 vs = llm_pack_even(_mm256_loadu_ps(si + 2 * i), _mm256_loadu_ps(si + 2 * i + 8));
                    __m256 y0 = _mm256_fnmadd_ps(x1, vs, _mm256_mul_ps(x0, vc));
                    __m256 y1 = _mm256_fmadd_ps(x0, vs, _mm256_mul_ps(x1, vc));
                    _mm256_storeu_ps(row + i, y0);
                    _mm256_storeu_ps(row + i + half, y1);
                }
#endif
                for (; i < half; i++) {
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
    llm_backend_host_write(x);
}
