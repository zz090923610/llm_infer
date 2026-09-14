#include "pim_func/pool.h"

#include <cstdint>
#include <cstdlib>
#include <pthread.h>

namespace pim_func {
namespace {

pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv_worker[kMaxPoolThreads];
pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;

int req_n = 0; /* 0 = unset */
int nth = 1;
int inited = 0;
int stop = 0;
int generation = 0;
int remaining = 0;
int job_n = 0;
PoolFn job_fn = nullptr;
void *job_ctx = nullptr;
pthread_t workers[kMaxPoolThreads];

int clamp_n(int n) {
    if (n < 1) n = 1;
    if (n > kMaxPoolThreads) n = kMaxPoolThreads;
    return n;
}

int env_n() {
    const char *e = std::getenv("PIM_NTHREADS");
    if (e && e[0]) {
        int n = std::atoi(e);
        if (n > 0) return clamp_n(n);
    }
    return 0;
}

int resolve_n() {
    if (req_n > 0) return clamp_n(req_n);
    int e = env_n();
    if (e > 0) return e;
    return 1;
}

void *worker_main(void *arg) {
    int tid = static_cast<int>(reinterpret_cast<intptr_t>(arg));
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (generation == seen && !stop) pthread_cond_wait(&cv_worker[tid], &mu);
        if (stop) {
            pthread_mutex_unlock(&mu);
            return nullptr;
        }
        seen = generation;
        PoolFn fn = job_fn;
        void *ctx = job_ctx;
        int n = job_n;
        pthread_mutex_unlock(&mu);

        if (tid < n) fn(tid, n, ctx);

        pthread_mutex_lock(&mu);
        if (tid < n && --remaining == 0) pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&mu);
    }
}

void ensure_init() {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    nth = resolve_n();
    stop = 0;
    generation = 0;
    inited = 1;
    for (int t = 1; t < nth; t++) pthread_cond_init(&cv_worker[t], nullptr);
    pthread_mutex_unlock(&mu);

    for (int t = 1; t < nth; t++) {
        if (pthread_create(&workers[t - 1], nullptr, worker_main,
                           reinterpret_cast<void *>(static_cast<intptr_t>(t))) != 0) {
            /* Fall back to single-threaded if spawn fails. */
            pthread_mutex_lock(&mu);
            nth = 1;
            pthread_mutex_unlock(&mu);
            return;
        }
    }
}

} // namespace

void pool_set_threads(int n) {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    req_n = n > 0 ? clamp_n(n) : 0;
    pthread_mutex_unlock(&mu);
}

int pool_n_threads() {
    ensure_init();
    return nth;
}

void pool_run(PoolFn fn, void *ctx, int n) {
    ensure_init();
    if (n < 1) n = 1;
    if (n > nth) n = nth;
    if (n == 1) {
        fn(0, 1, ctx);
        return;
    }
    pthread_mutex_lock(&mu);
    job_fn = fn;
    job_ctx = ctx;
    job_n = n;
    remaining = n - 1;
    generation++;
    for (int t = 1; t < n; t++) pthread_cond_signal(&cv_worker[t]);
    pthread_mutex_unlock(&mu);

    fn(0, n, ctx);

    pthread_mutex_lock(&mu);
    while (remaining > 0) pthread_cond_wait(&cv_done, &mu);
    pthread_mutex_unlock(&mu);
}

void pool_shutdown() {
    pthread_mutex_lock(&mu);
    if (!inited || stop) {
        pthread_mutex_unlock(&mu);
        return;
    }
    stop = 1;
    int n = nth;
    for (int t = 1; t < n; t++) pthread_cond_signal(&cv_worker[t]);
    pthread_mutex_unlock(&mu);
    for (int t = 1; t < n; t++) pthread_join(workers[t - 1], nullptr);
    pthread_mutex_lock(&mu);
    inited = 0;
    stop = 0;
    nth = 1;
    pthread_mutex_unlock(&mu);
}

} // namespace pim_func
