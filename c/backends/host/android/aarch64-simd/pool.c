#include "pool.h"
#include "backend.h"
#include "util.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#define LLM_MAX_THREADS 32

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cv_worker[LLM_MAX_THREADS];
static pthread_cond_t cv_done = PTHREAD_COND_INITIALIZER;

static int req_prefill; /* 0 = unset */
static int req_decode;
static int n_prefill;
static int n_decode;
static int nth; /* worker count = max(prefill, decode) */
static int inited;
static int stop;
static int generation;
static int remaining;
static int job_n;
static llm_pool_fn job_fn;
static void *job_ctx;
static pthread_t workers[LLM_MAX_THREADS];

static int clamp_n(int n) {
    if (n < 1) n = 1;
    if (n > LLM_MAX_THREADS) n = LLM_MAX_THREADS;
    return n;
}

static int env_n(const char *name) {
    const char *e = getenv(name);
    if (e && e[0]) {
        int n = atoi(e);
        if (n > 0) return n;
    }
    return 0;
}

static int default_nthreads(void) {
    int n = env_n("LLM_NTHREADS");
    if (n > 0) return n;
#ifdef __ANDROID__
    /* Decode GEMV is DRAM-bound. On Ace 5, 2 threads beat 6+: less
       memory contention, and less fork-join on the small (D=960) linears. */
    return 2;
#else
    long p = sysconf(_SC_NPROCESSORS_ONLN);
    if (p < 1) p = 1;
    return (int)p;
#endif
}

static void pool_shutdown(void) {
    pthread_mutex_lock(&mu);
    if (!inited || stop) {
        pthread_mutex_unlock(&mu);
        return;
    }
    stop = 1;
    int n = nth;
    for (int t = 1; t < n; t++) pthread_cond_signal(&cv_worker[t]);
    pthread_mutex_unlock(&mu);
    for (int t = 1; t < n; t++) pthread_join(workers[t - 1], NULL);
}

static void *worker_main(void *arg) {
    int tid = (int)(intptr_t)arg;
    int seen = 0;
    for (;;) {
        pthread_mutex_lock(&mu);
        while (generation == seen && !stop) pthread_cond_wait(&cv_worker[tid], &mu);
        if (stop) {
            pthread_mutex_unlock(&mu);
            return NULL;
        }
        seen = generation;
        llm_pool_fn fn = job_fn;
        void *ctx = job_ctx;
        int n = job_n;
        pthread_mutex_unlock(&mu);

        if (tid < n) fn(tid, n, ctx);

        pthread_mutex_lock(&mu);
        if (tid < n && --remaining == 0) pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&mu);
    }
}

static void ensure_init(void) {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    int def = default_nthreads();
    int np = req_prefill > 0 ? req_prefill : env_n("LLM_PREFILL_THREADS");
    int nd = req_decode > 0 ? req_decode : env_n("LLM_DECODE_THREADS");
    if (np <= 0) np = def;
    if (nd <= 0) nd = def;
    n_prefill = clamp_n(np);
    n_decode = clamp_n(nd);
    nth = n_prefill > n_decode ? n_prefill : n_decode;
    stop = 0;
    generation = 0;
    inited = 1;
    int n_workers = nth;
    for (int t = 1; t < n_workers; t++) pthread_cond_init(&cv_worker[t], NULL);
    pthread_mutex_unlock(&mu);

    for (int t = 1; t < n_workers; t++) {
        if (pthread_create(&workers[t - 1], NULL, worker_main, (void *)(intptr_t)t) != 0) {
            die("pthread_create failed");
        }
    }
    atexit(pool_shutdown);
}

void llm_backend_set_threads(int n) {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    int v = n > 0 ? clamp_n(n) : 0;
    req_prefill = v;
    req_decode = v;
    pthread_mutex_unlock(&mu);
}

void llm_backend_set_prefill_threads(int n) {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    req_prefill = n > 0 ? clamp_n(n) : 0;
    pthread_mutex_unlock(&mu);
}

void llm_backend_set_decode_threads(int n) {
    pthread_mutex_lock(&mu);
    if (inited) {
        pthread_mutex_unlock(&mu);
        return;
    }
    req_decode = n > 0 ? clamp_n(n) : 0;
    pthread_mutex_unlock(&mu);
}

int llm_backend_n_threads(void) {
    ensure_init();
    return nth;
}

int llm_backend_n_prefill_threads(void) {
    ensure_init();
    return n_prefill;
}

int llm_backend_n_decode_threads(void) {
    ensure_init();
    return n_decode;
}

void llm_pool_run(llm_pool_fn fn, void *ctx, int n) {
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
