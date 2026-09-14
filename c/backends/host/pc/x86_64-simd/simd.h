#ifndef LLM_X86_SIMD_H
#define LLM_X86_SIMD_H

#include <immintrin.h>

static inline float llm_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

/* Pack even lanes from 16 interleaved floats (two YMM): [a0,a0,a1,a1,...] -> [a0,a1,...]. */
static inline __m256 llm_pack_even(__m256 a, __m256 b) {
    const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    __m256 ae = _mm256_permutevar8x32_ps(a, idx);
    __m256 be = _mm256_permutevar8x32_ps(b, idx);
    return _mm256_permute2f128_ps(ae, be, 0x20);
}

static inline float llm_dot_f32(const float *a, const float *b, int n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    for (; i + 8 <= n; i += 8) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), a0);
    }
    float s = llm_hsum256(a0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline void llm_axpy_f32(float *y, float a, const float *x, int n) {
    __m256 va = _mm256_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
        _mm256_storeu_ps(y + i + 8,
                         _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), _mm256_loadu_ps(y + i + 8)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += a * x[i];
}

static inline void llm_scale_f32(float *x, float s, int n) {
    __m256 vs = _mm256_set1_ps(s);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
        _mm256_storeu_ps(x + i + 8, _mm256_mul_ps(_mm256_loadu_ps(x + i + 8), vs));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
    }
    for (; i < n; i++) x[i] *= s;
}

#endif
