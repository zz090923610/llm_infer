#include "quant.h"
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
}

int main(void) {
    test_known_block();
    test_two_blocks();
    test_rejects_ragged();
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("test_quant ok\n");
    return 0;
}
