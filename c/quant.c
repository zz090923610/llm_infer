#include "quant.h"
#include <string.h>

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

int dequantize_q8_0(const void *data, int n_elements, float *out) {
    if (n_elements % QK8_0 != 0) return -1;
    int n_blocks = n_elements / QK8_0;
    const unsigned char *raw = (const unsigned char *)data;
    for (int b = 0; b < n_blocks; b++) {
        const unsigned char *blk = raw + (size_t)b * BLOCK_Q8_0;
        uint16_t hs;
        memcpy(&hs, blk, 2);
        float scale = fp16_to_fp32(hs);
        const int8_t *qs = (const int8_t *)(blk + 2);
        float *dst = out + (size_t)b * QK8_0;
        for (int i = 0; i < QK8_0; i++) dst[i] = (float)qs[i] * scale;
    }
    return 0;
}

int dequantize_f32(const void *data, int n_elements, float *out) {
    memcpy(out, data, (size_t)n_elements * sizeof(float));
    return 0;
}
