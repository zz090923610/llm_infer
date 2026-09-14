#ifndef LLM_X86_POOL_H
#define LLM_X86_POOL_H

/* Internal fork-join pool. Workers never call back into llm_pool_run. */

typedef void (*llm_pool_fn)(int tid, int n_threads, void *ctx);

/* Run fn on threads 0..n-1. n is clamped to the pool size. */
void llm_pool_run(llm_pool_fn fn, void *ctx, int n);

#endif
