#include "weight.h"
#include "backend.h"
#include "quant.h"
#include "tensor.h"
#include "util.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef LLM_HAS_POOL
#include "pool.h"
#else
typedef void (*llm_pool_fn)(int tid, int n_threads, void *ctx);
static void llm_pool_run(llm_pool_fn fn, void *ctx, int n) {
    (void)n;
    fn(0, 1, ctx);
}
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

const float *weight_f32(const WeightTensor *w) {
    if (!w) return NULL;
    if (w->ggml_type != GGML_F32) die("weight %s is quantized (type %d), expected f32", w->name,
                                      w->ggml_type);
    return (const float *)w->data;
}

#if defined(__AVX2__) && defined(__FMA__)
static float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, sh);
    sh = _mm_movehl_ps(sh, s);
    s = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

static float dot_f32(const float *a, const float *b, int n) {
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
    float s = hsum256(a0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_dot_q8_0(const void *vx, const float *y, int n) {
    const unsigned char *p = (const unsigned char *)vx;
    int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        uint16_t hs;
        memcpy(&hs, p, 2);
        __m256 vd = _mm256_set1_ps(fp16_to_fp32(hs));
        const int8_t *q = (const int8_t *)(p + 2);
        __m128i q16 = _mm_loadu_si128((const __m128i *)q);
        __m128i q16b = _mm_loadu_si128((const __m128i *)(q + 16));
        __m256 q0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16));
        __m256 q1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16, 8)));
        __m256 q2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16b));
        __m256 q3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16b, 8)));
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, q0), _mm256_loadu_ps(y + 0), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, q1), _mm256_loadu_ps(y + 8), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, q2), _mm256_loadu_ps(y + 16), acc);
        acc = _mm256_fmadd_ps(_mm256_mul_ps(vd, q3), _mm256_loadu_ps(y + 24), acc);
        p += BLOCK_Q8_0;
        y += QK8_0;
    }
    return hsum256(acc);
}

static void dots4_f32(const float *w, const float *x0, const float *x1, const float *x2,
                      const float *x3, float *s0, float *s1, float *s2, float *s3, int n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 vw = _mm256_loadu_ps(w + j);
        a0 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x0 + j), a0);
        a1 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x1 + j), a1);
        a2 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x2 + j), a2);
        a3 = _mm256_fmadd_ps(vw, _mm256_loadu_ps(x3 + j), a3);
    }
    float t0 = hsum256(a0), t1 = hsum256(a1), t2 = hsum256(a2), t3 = hsum256(a3);
    for (; j < n; j++) {
        float wv = w[j];
        t0 += wv * x0[j];
        t1 += wv * x1[j];
        t2 += wv * x2[j];
        t3 += wv * x3[j];
    }
    *s0 = t0;
    *s1 = t1;
    *s2 = t2;
    *s3 = t3;
}

static void dots4_q8_0(const void *vx, const float *x0, const float *x1, const float *x2,
                       const float *x3, float *s0, float *s1, float *s2, float *s3, int n) {
    const unsigned char *p = (const unsigned char *)vx;
    int nb = n / QK8_0;
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    for (int b = 0; b < nb; b++) {
        uint16_t hs;
        memcpy(&hs, p, 2);
        __m256 vd = _mm256_set1_ps(fp16_to_fp32(hs));
        const int8_t *q = (const int8_t *)(p + 2);
        __m128i q16 = _mm_loadu_si128((const __m128i *)q);
        __m128i q16b = _mm_loadu_si128((const __m128i *)(q + 16));
        __m256 w0 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16)));
        __m256 w1 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16, 8))));
        __m256 w2 = _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16b)));
        __m256 w3 = _mm256_mul_ps(
            vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16b, 8))));
        const float *p0 = x0 + b * QK8_0;
        const float *p1 = x1 + b * QK8_0;
        const float *p2 = x2 + b * QK8_0;
        const float *p3 = x3 + b * QK8_0;
        a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p0 + 0), a0);
        a1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p1 + 0), a1);
        a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p2 + 0), a2);
        a3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(p3 + 0), a3);
        a0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p0 + 8), a0);
        a1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p1 + 8), a1);
        a2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p2 + 8), a2);
        a3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(p3 + 8), a3);
        a0 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p0 + 16), a0);
        a1 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p1 + 16), a1);
        a2 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p2 + 16), a2);
        a3 = _mm256_fmadd_ps(w2, _mm256_loadu_ps(p3 + 16), a3);
        a0 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p0 + 24), a0);
        a1 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p1 + 24), a1);
        a2 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p2 + 24), a2);
        a3 = _mm256_fmadd_ps(w3, _mm256_loadu_ps(p3 + 24), a3);
        p += BLOCK_Q8_0;
    }
    *s0 = hsum256(a0);
    *s1 = hsum256(a1);
    *s2 = hsum256(a2);
    *s3 = hsum256(a3);
}
#elif defined(__aarch64__)
static float hsum128(float32x4_t v) {
    return vaddvq_f32(v);
}

static float dot_f32(const float *a, const float *b, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    a0 = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    for (; i + 4 <= n; i += 4) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float s = hsum128(a0);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float32x4_t q8_scale_dup(const unsigned char *blk) {
    float16_t h;
    memcpy(&h, blk, 2);
    return vcvt_f32_f16(vdup_n_f16(h));
}

static inline float32x4_t q8_w4(float32x4_t vd, int16x8_t s, int hi) {
    int32x4_t q32 = vmovl_s16(hi ? vget_high_s16(s) : vget_low_s16(s));
    return vmulq_f32(vd, vcvtq_f32_s32(q32));
}

static float vec_dot_q8_0(const void *vx, const float *y, int n) {
    const unsigned char *p = (const unsigned char *)vx;
    int nb = n / QK8_0;
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int i = 0; i < nb; i++) {
        float32x4_t vd = q8_scale_dup(p);
        const int8_t *q = (const int8_t *)(p + 2);
        int8x16_t q0 = vld1q_s8(q);
        int8x16_t q1 = vld1q_s8(q + 16);
        int16x8_t s0 = vmovl_s8(vget_low_s8(q0));
        int16x8_t s1 = vmovl_s8(vget_high_s8(q0));
        int16x8_t s2 = vmovl_s8(vget_low_s8(q1));
        int16x8_t s3 = vmovl_s8(vget_high_s8(q1));
        acc = vfmaq_f32(acc, q8_w4(vd, s0, 0), vld1q_f32(y + 0));
        acc = vfmaq_f32(acc, q8_w4(vd, s0, 1), vld1q_f32(y + 4));
        acc = vfmaq_f32(acc, q8_w4(vd, s1, 0), vld1q_f32(y + 8));
        acc = vfmaq_f32(acc, q8_w4(vd, s1, 1), vld1q_f32(y + 12));
        acc = vfmaq_f32(acc, q8_w4(vd, s2, 0), vld1q_f32(y + 16));
        acc = vfmaq_f32(acc, q8_w4(vd, s2, 1), vld1q_f32(y + 20));
        acc = vfmaq_f32(acc, q8_w4(vd, s3, 0), vld1q_f32(y + 24));
        acc = vfmaq_f32(acc, q8_w4(vd, s3, 1), vld1q_f32(y + 28));
        p += BLOCK_Q8_0;
        y += QK8_0;
    }
    return hsum128(acc);
}

/* Prefill: convert each Q8_0 block once in registers and FMA 4 activation rows.
   Avoids writing a f32 weight row and streaming it back from L1/DRAM. */
static void dots4_q8_0(const void *vx, const float *x0, const float *x1, const float *x2,
                       const float *x3, float *s0, float *s1, float *s2, float *s3, int n) {
    const unsigned char *p = (const unsigned char *)vx;
    int nb = n / QK8_0;
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    for (int b = 0; b < nb; b++) {
        if (b + 1 < nb) __builtin_prefetch(p + BLOCK_Q8_0, 0, 3);
        float32x4_t vd = q8_scale_dup(p);
        const int8_t *q = (const int8_t *)(p + 2);
        int8x16_t q0 = vld1q_s8(q);
        int8x16_t q1 = vld1q_s8(q + 16);
        int16x8_t t0 = vmovl_s8(vget_low_s8(q0));
        int16x8_t t1 = vmovl_s8(vget_high_s8(q0));
        int16x8_t t2 = vmovl_s8(vget_low_s8(q1));
        int16x8_t t3 = vmovl_s8(vget_high_s8(q1));
        const float *p0 = x0 + b * QK8_0;
        const float *p1 = x1 + b * QK8_0;
        const float *p2 = x2 + b * QK8_0;
        const float *p3 = x3 + b * QK8_0;
        float32x4_t w;
        w = q8_w4(vd, t0, 0);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 0));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 0));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 0));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 0));
        w = q8_w4(vd, t0, 1);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 4));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 4));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 4));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 4));
        w = q8_w4(vd, t1, 0);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 8));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 8));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 8));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 8));
        w = q8_w4(vd, t1, 1);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 12));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 12));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 12));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 12));
        w = q8_w4(vd, t2, 0);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 16));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 16));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 16));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 16));
        w = q8_w4(vd, t2, 1);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 20));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 20));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 20));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 20));
        w = q8_w4(vd, t3, 0);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 24));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 24));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 24));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 24));
        w = q8_w4(vd, t3, 1);
        a0 = vfmaq_f32(a0, w, vld1q_f32(p0 + 28));
        a1 = vfmaq_f32(a1, w, vld1q_f32(p1 + 28));
        a2 = vfmaq_f32(a2, w, vld1q_f32(p2 + 28));
        a3 = vfmaq_f32(a3, w, vld1q_f32(p3 + 28));
        p += BLOCK_Q8_0;
    }
    *s0 = hsum128(a0);
    *s1 = hsum128(a1);
    *s2 = hsum128(a2);
    *s3 = hsum128(a3);
}

static void dots4_f32(const float *w, const float *x0, const float *x1, const float *x2,
                      const float *x3, float *s0, float *s1, float *s2, float *s3, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    int j = 0;
    for (; j + 16 <= n; j += 16) {
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
    for (; j + 8 <= n; j += 8) {
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
    float t0 = hsum128(a0), t1 = hsum128(a1), t2 = hsum128(a2), t3 = hsum128(a3);
    for (; j < n; j++) {
        float wv = w[j];
        t0 += wv * x0[j];
        t1 += wv * x1[j];
        t2 += wv * x2[j];
        t3 += wv * x3[j];
    }
    *s0 = t0;
    *s1 = t1;
    *s2 = t2;
    *s3 = t3;
}
#else
static float dot_f32(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static float vec_dot_q8_0(const void *vx, const float *y, int n) {
    const unsigned char *p = (const unsigned char *)vx;
    int nb = n / QK8_0;
    float sum = 0.0f;
    for (int i = 0; i < nb; i++) {
        uint16_t hs;
        memcpy(&hs, p, 2);
        float d = fp16_to_fp32(hs);
        const int8_t *q = (const int8_t *)(p + 2);
        float acc = 0.0f;
        for (int j = 0; j < QK8_0; j++) acc += (float)q[j] * y[j];
        sum += d * acc;
        p += BLOCK_Q8_0;
        y += QK8_0;
    }
    return sum;
}

static void dots4_f32(const float *w, const float *x0, const float *x1, const float *x2,
                      const float *x3, float *s0, float *s1, float *s2, float *s3, int n) {
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (int j = 0; j < n; j++) {
        float wv = w[j];
        t0 += wv * x0[j];
        t1 += wv * x1[j];
        t2 += wv * x2[j];
        t3 += wv * x3[j];
    }
    *s0 = t0;
    *s1 = t1;
    *s2 = t2;
    *s3 = t3;
}
#endif

typedef struct {
    const WeightTensor *W;
    const float *x;
    float *y;
    int n_tok;
    int n_out;
    int n_in;
    int acc;
} QLinearJob;

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

static void apply_acc(float *dst, float s, int acc) {
    *dst = acc ? *dst + s : s;
}

static void qlinear_job(int tid, int n_threads, void *ctx) {
    QLinearJob *j = (QLinearJob *)ctx;
    int i0, i1;
    row_slice(tid, n_threads, j->n_out, &i0, &i1);
    if (i0 >= i1) return;

    const WeightTensor *W = j->W;
    const float *x = j->x;
    float *y = j->y;
    int n_tok = j->n_tok, n_out = j->n_out, n_in = j->n_in, acc = j->acc;
    int q8 = W->ggml_type == GGML_Q8_0 && n_in % QK8_0 == 0;
    int q8_decode = q8 && n_tok == 1;
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__aarch64__)
    int q8_fused = q8 && n_tok != 1;
#else
    int q8_fused = 0;
#endif

    float *wrow = NULL;
    if (!q8_decode && !q8_fused) {
        wrow = xmalloc((size_t)n_in * sizeof(float));
    }

    for (int i = i0; i < i1; i++) {
        const void *rp = ggml_row_data(W->data, W->ggml_type, i, n_in);
        if (i + 1 < i1) {
            __builtin_prefetch(ggml_row_data(W->data, W->ggml_type, i + 1, n_in), 0, 3);
        }
        if (q8_decode) {
            float s = vec_dot_q8_0(rp, x, n_in);
            apply_acc(y + i, s, acc);
#if (defined(__AVX2__) && defined(__FMA__)) || defined(__aarch64__)
        } else if (q8_fused) {
            int t = 0;
            for (; t + 4 <= n_tok; t += 4) {
                const float *x0 = x + (size_t)(t + 0) * (size_t)n_in;
                const float *x1 = x + (size_t)(t + 1) * (size_t)n_in;
                const float *x2 = x + (size_t)(t + 2) * (size_t)n_in;
                const float *x3 = x + (size_t)(t + 3) * (size_t)n_in;
                float s0, s1, s2, s3;
                dots4_q8_0(rp, x0, x1, x2, x3, &s0, &s1, &s2, &s3, n_in);
                apply_acc(y + (size_t)(t + 0) * (size_t)n_out + i, s0, acc);
                apply_acc(y + (size_t)(t + 1) * (size_t)n_out + i, s1, acc);
                apply_acc(y + (size_t)(t + 2) * (size_t)n_out + i, s2, acc);
                apply_acc(y + (size_t)(t + 3) * (size_t)n_out + i, s3, acc);
            }
            for (; t < n_tok; t++) {
                float s = vec_dot_q8_0(rp, x + (size_t)t * (size_t)n_in, n_in);
                apply_acc(y + (size_t)t * (size_t)n_out + i, s, acc);
            }
#endif
        } else {
            if (dequantize_row(W->ggml_type, rp, n_in, wrow) != 0)
                die("dequant row failed for %s", W->name);

            int t = 0;
            for (; t + 4 <= n_tok; t += 4) {
                const float *x0 = x + (size_t)(t + 0) * (size_t)n_in;
                const float *x1 = x + (size_t)(t + 1) * (size_t)n_in;
                const float *x2 = x + (size_t)(t + 2) * (size_t)n_in;
                const float *x3 = x + (size_t)(t + 3) * (size_t)n_in;
                float s0, s1, s2, s3;
                dots4_f32(wrow, x0, x1, x2, x3, &s0, &s1, &s2, &s3, n_in);
                apply_acc(y + (size_t)(t + 0) * (size_t)n_out + i, s0, acc);
                apply_acc(y + (size_t)(t + 1) * (size_t)n_out + i, s1, acc);
                apply_acc(y + (size_t)(t + 2) * (size_t)n_out + i, s2, acc);
                apply_acc(y + (size_t)(t + 3) * (size_t)n_out + i, s3, acc);
            }
            for (; t < n_tok; t++) {
                float s = dot_f32(wrow, x + (size_t)t * (size_t)n_in, n_in);
                apply_acc(y + (size_t)t * (size_t)n_out + i, s, acc);
            }
        }
        if (i == i0 || (i + 1) % 64 == 0 || i + 1 == i1) {
            LLM_STEP("qlinear %s %d/%d\n",
                     (W->name && W->name[0]) ? W->name : "?", i + 1, n_out);
        }
    }
    free(wrow);
}

static void qlinear(const WeightTensor *W, const float *x, float *y, int n_tok, int n_out, int n_in,
                    int acc) {
    if (n_tok <= 0 || n_out <= 0 || n_in <= 0) return;
    llm_backend_host_read(x);
    if (acc) llm_backend_host_read(y);
    LLM_STEP("qlinear %s tok=%d out=%d in=%d acc=%d\n",
             (W->name && W->name[0]) ? W->name : "?", n_tok, n_out, n_in, acc);
    QLinearJob job = {W, x, y, n_tok, n_out, n_in, acc};
    int nth = n_tok > 1 ? llm_backend_n_prefill_threads() : llm_backend_n_decode_threads();
    llm_pool_run(qlinear_job, &job, nth);
    llm_backend_host_write(y);
}

static int use_device_quant(const WeightTensor *W) {
    return W && llm_backend_quant_linear(W->ggml_type);
}

void linear_wt(const WeightTensor *W, const float *x, float *y, int n_out, int n_in) {
    linear_rows_wt(W, x, y, 1, n_out, n_in);
}

void linear_rows_wt(const WeightTensor *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    if (!W || n_tok <= 0) return;
    if (W->ggml_type == GGML_F32) {
        linear_rows((const float *)W->data, x, y, n_tok, n_out, n_in);
        return;
    }
    if (use_device_quant(W)) {
        linear_rows((const float *)W, x, y, n_tok, n_out, n_in);
        return;
    }
    qlinear(W, x, y, n_tok, n_out, n_in, 0);
}

void linear_rows_add_wt(const WeightTensor *W, const float *x, float *y, int n_tok, int n_out,
                        int n_in) {
    if (!W || n_tok <= 0) return;
    if (W->ggml_type == GGML_F32) {
        linear_rows_add((const float *)W->data, x, y, n_tok, n_out, n_in);
        return;
    }
    if (use_device_quant(W)) {
        linear_rows_add((const float *)W, x, y, n_tok, n_out, n_in);
        return;
    }
    qlinear(W, x, y, n_tok, n_out, n_in, 1);
}

void embed_gather_wt(const WeightTensor *table, const int *ids, float *out, int n_rows, int d) {
    if (!table || n_rows <= 0 || d <= 0) return;
    if (table->ggml_type == GGML_F32) {
        embed_gather((const float *)table->data, ids, out, n_rows, d);
        return;
    }
    for (int t = 0; t < n_rows; t++) {
        const void *rp = ggml_row_data(table->data, table->ggml_type, ids[t], d);
        if (dequantize_row(table->ggml_type, rp, d, out + (size_t)t * (size_t)d) != 0)
            die("embed dequant failed for %s id=%d", table->name, ids[t]);
    }
    llm_backend_host_write(out);
}
