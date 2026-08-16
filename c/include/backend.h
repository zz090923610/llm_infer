#ifndef LLM_BACKEND_H
#define LLM_BACKEND_H

#include <stddef.h>

/* Compile-time compute backend. Select with cmake -DLLM_BACKEND=<name>.
 *
 * Each backend under c/backends/<name>/ implements the kernels in:
 *   tensor.h  — linear, RMSNorm, SiLU, softmax, vec ops
 *   rope.h    — RoPE tables + apply
 *   attn.h    — GQA attention, head pack/merge, KV cache store
 *
 * Current: cpu (portable scalar C),
 *          x86_64 (AVX2/FMA GEMV/GEMM, GQA, RoPE, GDN + pthread pool),
 *          aarch64 (NEON FMA GEMV/GEMM + pthread pool; OnePlus Ace 5 / SD 8 Gen 3),
 *          gpu (Vulkan compute; Adreno 750 on Ace 5).
 * Later:   npu — same symbols, different impl.
 */

#ifndef LLM_BACKEND_NAME
#define LLM_BACKEND_NAME "cpu"
#endif

const char *llm_backend_name(void);

/* Thread counts for the x86_64 / aarch64 pool. cpu ignores these and always reports 1.
   n <= 0 means default. Call before the first linear.
   set_threads(n) sets both prefill and decode.
   Default: LLM_NTHREADS, else nproc (Android aarch64: 2).
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
/* gpu: intern GGUF Q8_0 bytes keyed by host f32 pointer. n_elements must be % 32 == 0. */
void llm_backend_intern_weight_q8(const void *host_key, const void *q8_blob, int n_elements);

/* gpu: CPU just wrote host memory (re-upload on next use). Other backends: no-op. */
void llm_backend_host_write(void *p);

/* gpu: wait and copy this buffer back to host. Other backends: no-op. */
void llm_backend_host_read(const void *p);

/* gpu: reserve a host-mapped RW/device buffer. Other backends: no-op. */
void llm_backend_intern_rw(const void *p, size_t bytes);
void llm_backend_intern_device(const void *p, size_t bytes);

#endif
