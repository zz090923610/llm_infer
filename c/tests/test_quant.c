#include "quant.h"
#include "gguf.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int fails;

static void expect_eq(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fails++;
    }
}

static uint16_t fp32_to_fp16_known(float x) {
    if (x == 0.5f) return 0x3800;
    if (x == 1.0f) return 0x3c00;
    if (x == 2.0f) return 0x4000;
    return 0;
}

static void test_known_block(void) {
    unsigned char block[BLOCK_Q8_0];
    uint16_t hs = fp32_to_fp16_known(0.5f);
    memcpy(block, &hs, 2);
    for (int i = 0; i < QK8_0; i++) block[2 + i] = (unsigned char)(int8_t)(i - 16);
    float y[QK8_0];
    expect_eq(dequantize_q8_0(block, QK8_0, y) == 0, "dequant known block");
    for (int i = 0; i < QK8_0; i++) {
        float exp = (float)(i - 16) * 0.5f;
        if (y[i] != exp) {
            fprintf(stderr, "FAIL: y[%d]=%g expected %g\n", i, y[i], exp);
            fails++;
        }
    }
}

static void test_two_blocks(void) {
    unsigned char raw[BLOCK_Q8_0 * 2];
    uint16_t d0 = fp32_to_fp16_known(1.0f);
    uint16_t d1 = fp32_to_fp16_known(2.0f);
    memcpy(raw, &d0, 2);
    memset(raw + 2, 1, QK8_0);
    memcpy(raw + BLOCK_Q8_0, &d1, 2);
    memset(raw + BLOCK_Q8_0 + 2, 3, QK8_0);
    float y[QK8_0 * 2];
    expect_eq(dequantize_q8_0(raw, QK8_0 * 2, y) == 0, "dequant two blocks");
    expect_eq(dequantize_row(GGML_Q8_0, raw, QK8_0 * 2, y) == 0, "dequant_row two blocks");
    for (int i = 0; i < QK8_0; i++) {
        if (y[i] != 1.0f) {
            fprintf(stderr, "FAIL: block0 y[%d]=%g\n", i, y[i]);
            fails++;
        }
        if (y[QK8_0 + i] != 6.0f) {
            fprintf(stderr, "FAIL: block1 y[%d]=%g\n", i, y[QK8_0 + i]);
            fails++;
        }
    }
}

static void test_rejects_ragged(void) {
    unsigned char buf[10] = {0};
    float y[10];
    expect_eq(dequantize_q8_0(buf, 10, y) != 0, "reject ragged length");
    expect_eq(dequantize_q4_k(buf, 10, y) != 0, "reject ragged q4_k");
    expect_eq(dequantize_q6_k(buf, 10, y) != 0, "reject ragged q6_k");
    expect_eq(dequantize_iq4_xs(buf, 10, y) != 0, "reject ragged iq4_xs");
}

static void test_q4_k_known(void) {
    unsigned char blk[BLOCK_Q4_K];
    memset(blk, 0, sizeof(blk));
    uint16_t d = fp32_to_fp16_known(1.0f);
    uint16_t dmin = fp32_to_fp16_known(0.5f);
    memcpy(blk, &d, 2);
    memcpy(blk + 2, &dmin, 2);
    /* first 4 scale groups: sc=1, min=0 */
    blk[4] = 1;
    blk[5] = 1;
    blk[6] = 1;
    blk[7] = 1;
    memset(blk + 16, 0x22, 128); /* nibbles = 2 */
    float y[QK_K];
    expect_eq(dequantize_q4_k(blk, QK_K, y) == 0, "q4_k dequant");
    /* first 64 elems use is=0,1 with sc=1, min=0 → 1*1*2 - 0 = 2 */
    for (int i = 0; i < 64; i++) {
        if (fabsf(y[i] - 2.0f) > 1e-5f) {
            fprintf(stderr, "FAIL: q4_k y[%d]=%g expected 2\n", i, y[i]);
            fails++;
            break;
        }
    }
}

static void test_q6_k_known(void) {
    unsigned char blk[BLOCK_Q6_K];
    memset(blk, 0, sizeof(blk));
    /* scales all 1 */
    memset(blk + 128 + 64, 1, 16);
    uint16_t d = fp32_to_fp16_known(1.0f);
    memcpy(blk + 128 + 64 + 16, &d, 2);
    float y[QK_K];
    expect_eq(dequantize_q6_k(blk, QK_K, y) == 0, "q6_k dequant");
    /* ql=0,qh=0 → q = -32, scale=1, d=1 → -32 */
    for (int i = 0; i < 16; i++) {
        if (fabsf(y[i] + 32.0f) > 1e-5f) {
            fprintf(stderr, "FAIL: q6_k y[%d]=%g expected -32\n", i, y[i]);
            fails++;
            break;
        }
    }
}

static void test_iq4_xs_known(void) {
    unsigned char blk[BLOCK_IQ4_XS];
    memset(blk, 0, sizeof(blk));
    uint16_t d = fp32_to_fp16_known(1.0f);
    memcpy(blk, &d, 2);
    float y[QK_K];
    expect_eq(dequantize_iq4_xs(blk, QK_K, y) == 0, "iq4_xs dequant");
    /* ls=0 → dl=-32; qs=0 → kvalues[0]=-127 → 4064 */
    if (fabsf(y[0] - 4064.0f) > 1e-3f) {
        fprintf(stderr, "FAIL: iq4_xs y[0]=%g expected 4064\n", y[0]);
        fails++;
    }
}

int main(void) {
    test_known_block();
    test_two_blocks();
    test_rejects_ragged();
    test_q4_k_known();
    test_q6_k_known();
    test_iq4_xs_known();
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("test_quant ok\n");
    return 0;
}
