#pragma once

namespace pim_func {

using PoolFn = void (*)(int tid, int n_threads, void *ctx);

/* Cap matches llm_infer x86/aarch64 pools. */
constexpr int kMaxPoolThreads = 32;

/* Requested worker count (0 = unset). Must be called before the first pool_run
   if changing from the default; ignored after the pool has started. */
void pool_set_threads(int n);

/* Resolved worker count: set value, else PIM_NTHREADS, else 1. */
int pool_n_threads();

/* Fork-join: run fn on threads 0..n-1 (n clamped to pool size). */
void pool_run(PoolFn fn, void *ctx, int n);

void pool_shutdown();

} // namespace pim_func
