#include "quant.h"
#include "gguf.h"
#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp = (uint32_t)(h >> 10) & 0x1fu;
    uint32_t man = (uint32_t)h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 14;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ffu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (man << 13);
    } else {
        f = (sign << 31) | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

int ggml_blck_size(int ggml_type) {
    switch (ggml_type) {
    case GGML_F32:
        return 1;
    case GGML_Q8_0:
        return QK8_0;
    case GGML_Q4_K:
    case GGML_Q6_K:
    case GGML_IQ4_XS:
        return QK_K;
    default:
        return 0;
    }
}

int ggml_type_size(int ggml_type) {
    switch (ggml_type) {
    case GGML_F32:
        return (int)sizeof(float);
    case GGML_Q8_0:
        return BLOCK_Q8_0;
    case GGML_Q4_K:
        return BLOCK_Q4_K;
    case GGML_Q6_K:
        return BLOCK_Q6_K;
    case GGML_IQ4_XS:
        return BLOCK_IQ4_XS;
    default:
        return 0;
    }
}

size_t ggml_nbytes(int ggml_type, int n_elements) {
    int qk = ggml_blck_size(ggml_type);
    int ts = ggml_type_size(ggml_type);
    if (qk <= 0 || ts <= 0 || n_elements < 0 || n_elements % qk != 0) return 0;
    return (size_t)(n_elements / qk) * (size_t)ts;
}

const void *ggml_row_data(const void *data, int ggml_type, int row, int n_in) {
    return (const unsigned char *)data + (size_t)row * ggml_nbytes(ggml_type, n_in);
}

int dequantize_row(int ggml_type, const void *row, int n_in, float *out) {
    switch (ggml_type) {
    case GGML_F32:
        return dequantize_f32(row, n_in, out);
    case GGML_Q8_0:
        return dequantize_q8_0(row, n_in, out);
    case GGML_Q4_K:
        return dequantize_q4_k(row, n_in, out);
    case GGML_Q6_K:
        return dequantize_q6_k(row, n_in, out);
    case GGML_IQ4_XS:
        return dequantize_iq4_xs(row, n_in, out);
    default:
        return -1;
    }
}

int dequantize_q8_0(const void *data, int n_elements, float *out) {
    if (n_elements % QK8_0 != 0) return -1;
    int n_blocks = n_elements / QK8_0;
    const unsigned char *raw = (const unsigned char *)data;
#if defined(__aarch64__)
    for (int b = 0; b < n_blocks; b++) {
        const unsigned char *blk = raw + (size_t)b * BLOCK_Q8_0;
        float16_t h;
        memcpy(&h, blk, 2);
        float32x4_t vs = vcvt_f32_f16(vdup_n_f16(h));
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *dst = out + (size_t)b * QK8_0;
        int8x16_t q0 = vld1q_s8(qs);
        int8x16_t q1 = vld1q_s8(qs + 16);
        int16x8_t s0 = vmovl_s8(vget_low_s8(q0));
        int16x8_t s1 = vmovl_s8(vget_high_s8(q0));
        int16x8_t s2 = vmovl_s8(vget_low_s8(q1));
        int16x8_t s3 = vmovl_s8(vget_high_s8(q1));
        vst1q_f32(dst + 0, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s0)))));
        vst1q_f32(dst + 4, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s0)))));
        vst1q_f32(dst + 8, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s1)))));
        vst1q_f32(dst + 12, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s1)))));
        vst1q_f32(dst + 16, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s2)))));
        vst1q_f32(dst + 20, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s2)))));
        vst1q_f32(dst + 24, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_low_s16(s3)))));
        vst1q_f32(dst + 28, vmulq_f32(vs, vcvtq_f32_s32(vmovl_s16(vget_high_s16(s3)))));
    }
#else
    for (int b = 0; b < n_blocks; b++) {
        const unsigned char *blk = raw + (size_t)b * BLOCK_Q8_0;
        uint16_t hs;
        memcpy(&hs, blk, 2);
        float scale = fp16_to_fp32(hs);
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *dst = out + (size_t)b * QK8_0;
        for (int i = 0; i < QK8_0; i++) dst[i] = (float)qs[i] * scale;
    }
#endif
    return 0;
}

int dequantize_f32(const void *data, int n_elements, float *out) {
    memcpy(out, data, (size_t)n_elements * sizeof(float));
    return 0;
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

int dequantize_q4_k(const void *data, int n_elements, float *out) {
    if (n_elements % QK_K != 0) return -1;
    int nb = n_elements / QK_K;
    const unsigned char *raw = (const unsigned char *)data;
    float *y = out;
    for (int i = 0; i < nb; i++) {
        const unsigned char *blk = raw + (size_t)i * BLOCK_Q4_K;
        uint16_t hd, hm;
        memcpy(&hd, blk, 2);
        memcpy(&hm, blk + 2, 2);
        const float d = fp16_to_fp32(hd);
        const float minv = fp16_to_fp32(hm);
        const uint8_t *scales = blk + 4;
        const uint8_t *q = blk + 4 + K_SCALE_SIZE;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * (float)sc;
            const float m1 = minv * (float)m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * (float)sc;
            const float m2 = minv * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
    return 0;
}

int dequantize_q6_k(const void *data, int n_elements, float *out) {
    if (n_elements % QK_K != 0) return -1;
    int nb = n_elements / QK_K;
    const unsigned char *raw = (const unsigned char *)data;
    float *y = out;
    for (int i = 0; i < nb; i++) {
        const unsigned char *blk = raw + (size_t)i * BLOCK_Q6_K;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + QK_K / 2;
        const int8_t *sc = (const int8_t *)(blk + QK_K / 2 + QK_K / 4);
        uint16_t hd;
        memcpy(&hd, blk + QK_K / 2 + QK_K / 4 + QK_K / 16, 2);
        const float d = fp16_to_fp32(hd);
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return 0;
}

static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

int dequantize_iq4_xs(const void *data, int n_elements, float *out) {
    if (n_elements % QK_K != 0) return -1;
    int nb = n_elements / QK_K;
    const unsigned char *raw = (const unsigned char *)data;
    float *y = out;
    for (int i = 0; i < nb; i++) {
        const unsigned char *blk = raw + (size_t)i * BLOCK_IQ4_XS;
        uint16_t hd, scales_h;
        memcpy(&hd, blk, 2);
        memcpy(&scales_h, blk + 2, 2);
        const float d = fp16_to_fp32(hd);
        const uint8_t *scales_l = blk + 4;
        const uint8_t *qs = blk + 8;
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
            const float dl = d * (float)(ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * (float)kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * (float)kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
    return 0;
}
