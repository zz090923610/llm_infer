#ifndef LLM_BACKEND_H
#define LLM_BACKEND_H

#include <stddef.h>

/* Compile-time compute backend. Select with cmake -DLLM_BACKEND=<name>.
 *
 * Each backend under c/backends/{host,pim}/ implements:
 *   tensor.h  — linear, RMSNorm, SiLU, softmax, vec ops
 *   rope.h    — RoPE tables + apply
 *   attn.h    — GQA attention, head pack/merge, KV cache store
 *
 * Host PC:      plain-cpu (portable scalar C),
 *               x86_64-simd (AVX2/FMA GEMV/GEMM, GQA, RoPE, GDN + pthread pool).
 * Host Android: aarch64-simd (NEON FMA + pthread pool; OnePlus Ace 5 / SD 8 Gen 3),
 *               gpu (Vulkan compute; Adreno 750 on Ace 5).
 * PIM:          pim (decode GEMV via pim_func).
 * Later:   npu — same symbols, different impl.
 */

#ifndef LLM_BACKEND_NAME
#define LLM_BACKEND_NAME "plain-cpu"
#endif

const char *llm_backend_name(void);

/* Thread counts for the x86_64-simd / aarch64-simd pool. plain-cpu ignores these
   and always reports 1.
   n <= 0 means default. Call before the first linear.
   set_threads(n) sets both prefill and decode.
   Default: LLM_NTHREADS, else nproc (Android aarch64-simd: 2).
   Prefill/decode can also be set via LLM_PREFILL_THREADS / LLM_DECODE_THREADS. */
void llm_backend_set_threads(int n);
void llm_backend_set_prefill_threads(int n);
void llm_backend_set_decode_threads(int n);
int llm_backend_n_threads(void); /* max(prefill, decode) — pool size */
int llm_backend_n_prefill_threads(void);
int llm_backend_n_decode_threads(void);

/* gpu: submit in-flight compute and copy RW results to host. Other backends: no-op. */
void llm_backend_sync(void);

/* gpu: copy a weight tensor into a device buffer at load. Other backends: no-op.
   intern_weight is f32; intern_weight_f16 packs host f32 as device f16. */
void llm_backend_intern_weight(const void *p, size_t bytes);
void llm_backend_intern_weight_f16(const void *p, size_t bytes);
/* gpu: intern GGUF Q8_0 bytes keyed by host_key (WeightTensor* or f32*).
   n_elements must be % 32 == 0. Other backends: no-op. */
void llm_backend_intern_weight_q8(const void *host_key, const void *q8_blob, int n_elements);
/* gpu: intern a packed 2D weight if this backend has a fused kernel for ggml_type.
   Q4_K / Q6_K / IQ4_XS stay host-side and use on-demand dequant in weight.c. */
void llm_backend_intern_weight_quant(const void *host_key, const void *blob, int n_elements,
                                     int ggml_type);
/* gpu: 1 if linear() can consume interned Q8 keyed by a WeightTensor*. */
int llm_backend_q8_linear(void);
/* gpu: 1 if linear() can consume interned packed weights of this ggml type. */
int llm_backend_quant_linear(int ggml_type);

/* gpu: CPU just wrote host memory (re-upload on next use). Other backends: no-op. */
void llm_backend_host_write(void *p);

/* gpu: wait and copy this buffer back to host. Other backends: no-op. */
void llm_backend_host_read(const void *p);

/* gpu: reserve a host-mapped RW/device buffer. Other backends: no-op. */
void llm_backend_intern_rw(const void *p, size_t bytes);
void llm_backend_intern_device(const void *p, size_t bytes);

#endif
