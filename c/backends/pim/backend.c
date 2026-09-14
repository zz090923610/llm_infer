#include "backend.h"
#include "gguf.h"
#include "pim_func/c_api.h"
#include "pim_func/hybrid.h"
#include "quant.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>
#ifndef PIM_USE_X86_CPU_OPS
#include <unistd.h>
#endif

const char *llm_backend_name(void) {
    return LLM_BACKEND_NAME;
}

#ifndef PIM_USE_X86_CPU_OPS
static int req_prefill; /* 0 = unset */
static int req_decode;
static int inited_threads;

static int clamp_n(int n) {
    if (n < 1) n = 1;
    if (n > 32) n = 32;
    return n;
}

static int env_n(const char *name) {
    const char *e = getenv(name);
    if (e && e[0]) {
        int n = atoi(e);
        if (n > 0) return clamp_n(n);
    }
    return 0;
}

static int default_nthreads(void) {
    int n = env_n("LLM_NTHREADS");
    if (n > 0) return n;
    n = env_n("PIM_NTHREADS");
    if (n > 0) return n;
    long p = sysconf(_SC_NPROCESSORS_ONLN);
    if (p < 1) p = 1;
    return clamp_n((int)p);
}

static void ensure_threads(void) {
    if (inited_threads) return;
    int np = req_prefill > 0 ? req_prefill : env_n("LLM_PREFILL_THREADS");
    int nd = req_decode > 0 ? req_decode : env_n("LLM_DECODE_THREADS");
    int def = default_nthreads();
    if (np <= 0) np = def;
    if (nd <= 0) nd = def;
    int n = np > nd ? np : nd;
    if (n > PIM_MAX_GEMV_THREADS) n = PIM_MAX_GEMV_THREADS;
    pim_set_threads(n);
    inited_threads = 1;
}

void llm_backend_set_threads(int n) {
    if (inited_threads) return;
    int v = n > 0 ? clamp_n(n) : 0;
    req_prefill = v;
    req_decode = v;
}

void llm_backend_set_prefill_threads(int n) {
    if (inited_threads) return;
    req_prefill = n > 0 ? clamp_n(n) : 0;
}

void llm_backend_set_decode_threads(int n) {
    if (inited_threads) return;
    req_decode = n > 0 ? clamp_n(n) : 0;
}

int llm_backend_n_threads(void) {
    ensure_threads();
    return pim_n_threads();
}

int llm_backend_n_prefill_threads(void) {
    ensure_threads();
    return pim_n_threads();
}

int llm_backend_n_decode_threads(void) {
    ensure_threads();
    return pim_n_threads();
}
#endif /* !PIM_USE_X86_CPU_OPS */

static void ensure_hybrid(void) {
    if (pim_issue_mode() != PIM_ISSUE_SHM) return;
    if (!pim_hybrid_active() && pim_hybrid_attach_client() != 0)
        die("pim hybrid attach failed (start gem5/B first)");
    /* Do not redirect xmalloc: tokenizer/hashmap free() libc pointers.
     * Interned banks go through pim_hybrid_alloc in ensure_dense; GEMV x/y
     * are copied into the arena in pim_hybrid_submit. */
}

void llm_backend_sync(void) {}

void llm_backend_intern_weight(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}

void llm_backend_intern_weight_f16(const void *p, size_t bytes) {
    ensure_hybrid();
    if (!p || bytes < sizeof(float) || bytes % sizeof(float) != 0) return;
    int n = (int)(bytes / sizeof(float));
    if (pim_intern_f32(p, (const float *)p, n, NULL) != 0)
        die("pim intern f16 failed n=%d", n);
}

static float *dequant_owned(const void *blob, int n_elements, int ggml_type) {
    float *f32 = (float *)xmalloc((size_t)n_elements * sizeof(float));
    int rc = -1;
    if (ggml_type == GGML_Q8_0) rc = dequantize_q8_0(blob, n_elements, f32);
    else if (ggml_type == GGML_Q4_K) rc = dequantize_q4_k(blob, n_elements, f32);
    else if (ggml_type == GGML_Q6_K) rc = dequantize_q6_k(blob, n_elements, f32);
    else if (ggml_type == GGML_IQ4_XS) rc = dequantize_iq4_xs(blob, n_elements, f32);
    else if (ggml_type == GGML_F32) rc = dequantize_f32(blob, n_elements, f32);
    if (rc != 0) {
        free(f32);
        return NULL;
    }
    return f32;
}

void llm_backend_intern_weight_q8(const void *host_key, const void *q8_blob, int n_elements) {
    llm_backend_intern_weight_quant(host_key, q8_blob, n_elements, GGML_Q8_0);
}

void llm_backend_intern_weight_quant(const void *host_key, const void *blob, int n_elements,
                                     int ggml_type) {
    ensure_hybrid();
    if (!host_key || !blob || n_elements <= 0) return;
    float *f32 = dequant_owned(blob, n_elements, ggml_type);
    if (!f32) die("pim dequant failed type=%d n=%d", ggml_type, n_elements);
    const WeightTensor *W = (const WeightTensor *)host_key;
    const char *name = (W && W->name) ? W->name : NULL;
    if (pim_intern_f32_owned(host_key, f32, n_elements, name) != 0) {
        free(f32);
        die("pim intern quant failed n=%d", n_elements);
    }
}

int llm_backend_q8_linear(void) {
    return 1;
}

int llm_backend_quant_linear(int ggml_type) {
    return ggml_type == GGML_Q8_0 || ggml_type == GGML_Q4_K || ggml_type == GGML_Q6_K ||
           ggml_type == GGML_IQ4_XS;
}

void llm_backend_host_write(void *p) {
    pim_invalidate(p);
}

void llm_backend_host_read(const void *p) {
    (void)p;
}

void llm_backend_intern_rw(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}

void llm_backend_intern_device(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}
