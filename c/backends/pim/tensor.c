#include "tensor.h"
#include "backend.h"
#include "pim_func/c_api.h"

#include <stdlib.h>
#include <string.h>

void cpu_linear(const float *W, const float *x, float *y, int n_out, int n_in);
void cpu_linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in);
void cpu_linear_rows_add(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in);
int pim_backend_gemv(const void *key, const float *x, float *y, int n_out, int n_in, int acc);

enum { PIM_OP_OUTPUT = 1, PIM_OP_FFN = 2, PIM_OP_ATTN = 4 };

static int pim_ops_mask(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    const char *s = getenv("LLM_PIM_OPS");
    if (!s || !s[0] || strcmp(s, "all") == 0) {
        cached = PIM_OP_OUTPUT | PIM_OP_FFN | PIM_OP_ATTN;
        return cached;
    }
    int m = 0;
    if (strstr(s, "output")) m |= PIM_OP_OUTPUT;
    if (strstr(s, "ffn")) m |= PIM_OP_FFN;
    if (strstr(s, "attn")) m |= PIM_OP_ATTN;
    cached = m;
    return cached;
}

static int class_from_name(const char *n) {
    if (!n || !n[0]) return 0;
    if (strstr(n, "attn")) return PIM_OP_ATTN;
    if (strstr(n, "ffn") || strstr(n, "w1") || strstr(n, "w2") || strstr(n, "w3")) return PIM_OP_FFN;
    if (strstr(n, "output") || strstr(n, "lm_head") || strstr(n, "token_embd")) return PIM_OP_OUTPUT;
    if (strstr(n, "wq") || strstr(n, "wk") || strstr(n, "wv") || strstr(n, "wo") ||
        strstr(n, "q_proj") || strstr(n, "k_proj") || strstr(n, "v_proj") || strstr(n, "o_proj"))
        return PIM_OP_ATTN;
    if (strstr(n, "gate_proj") || strstr(n, "up_proj") || strstr(n, "down_proj")) return PIM_OP_FFN;
    return -1; /* named but unknown: keep on CPU (unit-test dummy weights) */
}

static int class_from_shape(int n_out, int n_in) {
    if (n_out >= 4096) return PIM_OP_OUTPUT;
    if (n_out >= n_in * 2 || n_in >= n_out * 2) return PIM_OP_FFN;
    return PIM_OP_ATTN;
}

static int should_offload(const void *key, int n_out, int n_in) {
    if (!pim_interned_f32(key, NULL)) return 0;
    int cls = class_from_name(pim_interned_name(key));
    if (cls < 0) return 0;
    if (cls == 0) cls = class_from_shape(n_out, n_in);
    return (pim_ops_mask() & cls) != 0;
}

static const float *cpu_W(const float *W) {
    const float *f = pim_interned_f32(W, NULL);
    return f ? f : W;
}

void linear(const float *W, const float *x, float *y, int n_out, int n_in) {
    linear_rows(W, x, y, 1, n_out, n_in);
}

static void ensure_gemv_threads(void) {
    static int once;
    if (once) return;
    int nd = llm_backend_n_decode_threads();
    int gemv_n = nd;
    if (gemv_n > PIM_MAX_GEMV_THREADS) gemv_n = PIM_MAX_GEMV_THREADS;
    pim_set_threads(gemv_n);
    once = 1;
}

static void linear_rows_ex(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in,
                           int acc) {
    if (n_tok <= 0 || n_out <= 0 || n_in <= 0) return;
    if (n_tok == 1 && should_offload(W, n_out, n_in)) {
        ensure_gemv_threads();
        if (pim_backend_gemv(W, x, y, n_out, n_in, acc) == 0) return;
    }
    const float *Wf = cpu_W(W);
    if (acc) cpu_linear_rows_add(Wf, x, y, n_tok, n_out, n_in);
    else cpu_linear_rows(Wf, x, y, n_tok, n_out, n_in);
}

void linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    linear_rows_ex(W, x, y, n_tok, n_out, n_in, 0);
}

void linear_rows_add(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    linear_rows_ex(W, x, y, n_tok, n_out, n_in, 1);
}
