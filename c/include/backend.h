#ifndef LLM_BACKEND_H
#define LLM_BACKEND_H

/* Compile-time compute backend. Select with cmake -DLLM_BACKEND=<name>.
 *
 * Each backend under c/backends/<name>/ implements the kernels in:
 *   tensor.h  — linear, RMSNorm, SiLU, softmax, vec ops
 *   rope.h    — RoPE tables + apply
 *   attn.h    — GQA attention, head pack/merge, KV cache store
 *
 * Current: cpu (portable scalar C),
 *          x86_64 (AVX2/FMA GEMV/GEMM + pthread pool),
 *          aarch64 (NEON FMA GEMV/GEMM + pthread pool; OnePlus Ace 5 / SD 8 Gen 3).
 * Later:   npu, gpu — same symbols, different impl.
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

#endif
