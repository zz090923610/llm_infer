#include "pim_func/mac_simd.h"
#include "pim_func/fp16.h"

#include <cstdlib>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#if defined(__F16C__)
#define PIM_HAS_F16C 1
#endif
#endif

namespace pim_func {

bool fast_mac_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *s = std::getenv("PIM_GEMV_SCALAR_STEPS");
        cached = (s && s[0] && s[0] != '0') ? 0 : 1;
    }
    return cached != 0;
}

#if defined(__AVX2__) && defined(__FMA__)

/* Pairwise reduce; avoids _mm_hadd_ps (high latency). */
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehdup_ps(s));
    s = _mm_add_ss(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(s);
}

static inline __m256 wdot_partial(const Burst &w, __m256 x0, __m256 x1) {
#if PIM_HAS_F16C
    __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w.data()));
    __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(w.data() + 8));
    __m256 a0 = _mm256_cvtph_ps(h0);
    __m256 a1 = _mm256_cvtph_ps(h1);
#else
    alignas(32) float wf[kLanes];
    for (int i = 0; i < kLanes; i++) wf[i] = f16_to_f32(w[static_cast<size_t>(i)]);
    __m256 a0 = _mm256_loadu_ps(wf);
    __m256 a1 = _mm256_loadu_ps(wf + 8);
#endif
    return _mm256_fmadd_ps(a0, x0, _mm256_mul_ps(a1, x1));
}

void burst_f16_to_f32(const Burst &b, float *out_f32) {
#if PIM_HAS_F16C
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b.data()));
    __m256 lo = _mm256_cvtph_ps(h);
    __m128i h2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(b.data() + 8));
    __m256 hi = _mm256_cvtph_ps(h2);
    _mm256_storeu_ps(out_f32, lo);
    _mm256_storeu_ps(out_f32 + 8, hi);
#else
    for (int i = 0; i < kLanes; i++) out_f32[i] = f16_to_f32(b[static_cast<size_t>(i)]);
#endif
}

float burst_dot_f16_xf32(const Burst &w, const float *x_f32) {
    __m256 x0 = _mm256_loadu_ps(x_f32);
    __m256 x1 = _mm256_loadu_ps(x_f32 + 8);
    return hsum256(wdot_partial(w, x0, x1));
}

void mac_banks16(const Burst *base, size_t bank_stride, const float *x_f32, float *acc) {
    __m256 x0 = _mm256_loadu_ps(x_f32);
    __m256 x1 = _mm256_loadu_ps(x_f32 + 8);
    /* Contiguous banks: unroll 4 so FMAs overlap; prefetch next column's slab. */
    if (bank_stride == 1) {
        _mm_prefetch(reinterpret_cast<const char *>(base + kBanks), _MM_HINT_T0);
        for (int bank = 0; bank < kBanks; bank += 4) {
            __m256 p0 = wdot_partial(base[bank + 0], x0, x1);
            __m256 p1 = wdot_partial(base[bank + 1], x0, x1);
            __m256 p2 = wdot_partial(base[bank + 2], x0, x1);
            __m256 p3 = wdot_partial(base[bank + 3], x0, x1);
            acc[bank + 0] += hsum256(p0);
            acc[bank + 1] += hsum256(p1);
            acc[bank + 2] += hsum256(p2);
            acc[bank + 3] += hsum256(p3);
        }
        return;
    }
    for (int bank = 0; bank < kBanks; bank++) {
        acc[bank] += hsum256(wdot_partial(base[static_cast<size_t>(bank) * bank_stride], x0, x1));
    }
}

float burst_dot_f16(const Burst &w, const Burst &x) {
    alignas(32) float xf[kLanes];
    burst_f16_to_f32(x, xf);
    return burst_dot_f16_xf32(w, xf);
}

#else

void burst_f16_to_f32(const Burst &b, float *out_f32) {
    for (int i = 0; i < kLanes; i++) out_f32[i] = f16_to_f32(b[static_cast<size_t>(i)]);
}

float burst_dot_f16_xf32(const Burst &w, const float *x_f32) {
    float sum = 0.0f;
    for (int i = 0; i < kLanes; i++) {
        sum += f16_to_f32(w[static_cast<size_t>(i)]) * x_f32[i];
    }
    return sum;
}

void mac_banks16(const Burst *base, size_t bank_stride, const float *x_f32, float *acc) {
    for (int bank = 0; bank < kBanks; bank++) {
        acc[bank] += burst_dot_f16_xf32(base[static_cast<size_t>(bank) * bank_stride], x_f32);
    }
}

float burst_dot_f16(const Burst &w, const Burst &x) {
    float sum = 0.0f;
    for (int i = 0; i < kLanes; i++) {
        sum += f16_to_f32(w[static_cast<size_t>(i)]) * f16_to_f32(x[static_cast<size_t>(i)]);
    }
    return sum;
}

#endif

} // namespace pim_func
