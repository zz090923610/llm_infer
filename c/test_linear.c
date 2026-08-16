#include "tensor.h"
#include "backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int fails;
static uint32_t rng_state = 1u;

static void expect_eq(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        fails++;
    }
}

static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

static void fill_rand(float *a, int n) {
    for (int i = 0; i < n; i++) a[i] = frand();
}

static void linear_ref(const float *W, const float *x, float *y, int n_out, int n_in) {
    for (int i = 0; i < n_out; i++) {
        const float *w = W + (size_t)i * (size_t)n_in;
        double acc = 0.0;
        for (int j = 0; j < n_in; j++) acc += (double)w[j] * (double)x[j];
        y[i] = (float)acc;
    }
}

static void check_close(const float *got, const float *ref, int n, int n_in, const char *tag) {
    float max_abs = 0.0f;
    float tol = 1e-4f * (float)n_in;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > max_abs) max_abs = e;
        if (e > tol) {
            fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", tag, i, got[i], ref[i], e,
                    tol);
            fails++;
            return;
        }
    }
    (void)max_abs;
}

static void test_linear_shape(int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_in);
    linear(W, x, y, n_out, n_in);
    linear_ref(W, x, ref, n_out, n_in);
    char tag[64];
    snprintf(tag, sizeof(tag), "linear n_out=%d n_in=%d", n_out, n_in);
    check_close(y, ref, n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void test_linear_rows_shape(int n_tok, int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear_rows");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_tok * n_in);
    linear_rows(W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        linear_ref(W, x + (size_t)t * (size_t)n_in, ref + (size_t)t * (size_t)n_out, n_out, n_in);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_rows n_tok=%d n_out=%d n_in=%d", n_tok, n_out, n_in);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void test_linear_rows_add_shape(int n_tok, int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_tok * (size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_tok * (size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear_rows_add");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_tok * n_in);
    fill_rand(y, n_tok * n_out);
    memcpy(ref, y, (size_t)n_tok * (size_t)n_out * sizeof(float));
    llm_backend_host_write(y);
    linear_rows_add(W, x, y, n_tok, n_out, n_in);
    llm_backend_sync();
    for (int t = 0; t < n_tok; t++) {
        float *tmp = malloc((size_t)n_out * sizeof(float));
        linear_ref(W, x + (size_t)t * (size_t)n_in, tmp, n_out, n_in);
        for (int i = 0; i < n_out; i++)
            ref[(size_t)t * (size_t)n_out + i] += tmp[i];
        free(tmp);
    }
    char tag[80];
    snprintf(tag, sizeof(tag), "linear_rows_add n_tok=%d n_out=%d n_in=%d", n_tok, n_out, n_in);
    check_close(y, ref, n_tok * n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

static void check_close_f16(const float *got, const float *ref, int n, int n_in, const char *tag) {
    float max_abs = 0.0f;
    float tol = 5e-3f * (float)n_in + 0.05f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(got[i] - ref[i]);
        if (e > max_abs) max_abs = e;
        if (e > tol) {
            fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", tag, i, got[i], ref[i], e,
                    tol);
            fails++;
            return;
        }
    }
    (void)max_abs;
}

static void test_linear_f16_texel(int n_out, int n_in) {
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    float *ref = malloc((size_t)n_out * sizeof(float));
    expect_eq(W && x && y && ref, "alloc linear f16");
    if (!W || !x || !y || !ref) {
        free(W);
        free(x);
        free(y);
        free(ref);
        return;
    }
    fill_rand(W, n_out * n_in);
    fill_rand(x, n_in);
    llm_backend_intern_weight_f16(W, (size_t)n_out * (size_t)n_in * sizeof(float));
    linear_rows(W, x, y, 1, n_out, n_in);
    llm_backend_sync();
    linear_ref(W, x, ref, n_out, n_in);
    char tag[64];
    snprintf(tag, sizeof(tag), "linear f16 texel n_out=%d n_in=%d", n_out, n_in);
    check_close_f16(y, ref, n_out, n_in, tag);
    free(W);
    free(x);
    free(y);
    free(ref);
}

int main(void) {
    llm_backend_set_threads(4);
    const int n_ins[] = {960, 2560, 7, 64};
    for (int k = 0; k < 4; k++) {
        test_linear_shape(16, n_ins[k]);
        test_linear_rows_shape(1, 16, n_ins[k]);
        test_linear_rows_shape(5, 16, n_ins[k]);
    }
    test_linear_shape(960, 960);
    test_linear_rows_shape(1, 960, 960);
    test_linear_rows_shape(4, 2560, 960);
    test_linear_rows_shape(32, 960, 960);
    test_linear_rows_add_shape(1, 16, 960);
    test_linear_rows_add_shape(4, 16, 960);
    test_linear_rows_add_shape(5, 960, 960);
    test_linear_f16_texel(960, 960);
    test_linear_f16_texel(16, 960);
    if (fails) {
        fprintf(stderr, "%d failure(s)\n", fails);
        return 1;
    }
    printf("test_linear ok\n");
    return 0;
}
