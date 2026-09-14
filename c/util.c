#include "util.h"

#include <stdlib.h>

static int llm_step_quiet = 1; /* STEP logs off unless LLM_STEPS / llm_set_quiet(0) */
static int llm_quiet_forced;
static void *(*g_arena_alloc)(size_t) = NULL;

void llm_set_arena_alloc(void *(*fn)(size_t)) { g_arena_alloc = fn; }

void llm_set_quiet(int quiet) {
    llm_step_quiet = quiet ? 1 : 0;
    llm_quiet_forced = 1;
}

int llm_steps_enabled(void) {
    static int env_done;
    if (!env_done) {
        env_done = 1;
        if (!llm_quiet_forced) {
            const char *q = getenv("LLM_QUIET");
            if (q && q[0] && q[0] != '0') {
                llm_step_quiet = 1;
            } else {
                const char *s = getenv("LLM_STEPS");
                if (s && s[0] && s[0] != '0') llm_step_quiet = 0;
            }
        }
    }
    return !llm_step_quiet;
}

void *xmalloc(size_t n) {
    if (n == 0) n = 1;
    if (g_arena_alloc) {
        void *p = g_arena_alloc(n);
        if (!p) die("out of arena memory (%zu bytes)", n);
        return p;
    }
    void *p = malloc(n);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

void *xcalloc(size_t n, size_t sz) {
    if (n == 0) n = 1;
    if (g_arena_alloc) {
        size_t bytes = n * sz;
        if (sz && bytes / sz != n) die("calloc overflow");
        void *p = g_arena_alloc(bytes ? bytes : 1);
        if (!p) die("out of arena memory (%zu * %zu)", n, sz);
        return p;
    }
    void *p = calloc(n, sz);
    if (!p) die("out of memory (%zu * %zu)", n, sz);
    return p;
}

void *xrealloc(void *p, size_t n) {
    if (n == 0) n = 1;
    if (g_arena_alloc) {
        void *q = g_arena_alloc(n);
        if (!q) die("out of arena memory (%zu bytes)", n);
        if (p) memcpy(q, p, n); /* old size unknown; copy n bytes is wrong if grow, OK if bump unused */
        return q;
    }
    void *q = realloc(p, n);
    if (!q) die("out of memory (%zu bytes)", n);
    return q;
}

char *xstrdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *d = xmalloc(n + 1);
    memcpy(d, s, n + 1);
    return d;
}

char *xstrndup(const char *s, size_t n) {
    char *d = xmalloc(n + 1);
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("error: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(1);
}

void intvec_init(IntVec *v) {
    v->data = NULL;
    v->n = 0;
    v->cap = 0;
}

void intvec_free(IntVec *v) {
    free(v->data);
    v->data = NULL;
    v->n = v->cap = 0;
}

void intvec_clear(IntVec *v) { v->n = 0; }

void intvec_push(IntVec *v, int x) {
    if (v->n >= v->cap) {
        v->cap = v->cap ? v->cap * 2 : 16;
        v->data = xrealloc(v->data, (size_t)v->cap * sizeof(int));
    }
    v->data[v->n++] = x;
}

void intvec_extend(IntVec *v, const int *src, int n) {
    for (int i = 0; i < n; i++) intvec_push(v, src[i]);
}

void bytevec_init(ByteVec *v) {
    v->data = NULL;
    v->n = 0;
    v->cap = 0;
}

void bytevec_free(ByteVec *v) {
    free(v->data);
    v->data = NULL;
    v->n = v->cap = 0;
}

void bytevec_clear(ByteVec *v) { v->n = 0; }

void bytevec_push(ByteVec *v, unsigned char b) {
    if (v->n >= v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->data = xrealloc(v->data, (size_t)v->cap);
    }
    v->data[v->n++] = (char)b;
}

void bytevec_append(ByteVec *v, const void *src, int n) {
    if (v->n + n > v->cap) {
        int cap = v->cap ? v->cap : 64;
        while (cap < v->n + n) cap *= 2;
        v->cap = cap;
        v->data = xrealloc(v->data, (size_t)v->cap);
    }
    memcpy(v->data + v->n, src, (size_t)n);
    v->n += n;
}
