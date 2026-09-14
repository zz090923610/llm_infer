#ifndef LLM_UTIL_H
#define LLM_UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>

#define LLM_OK 0
#define LLM_ERR 1

void llm_set_quiet(int quiet);
int llm_steps_enabled(void);

/* Unbuffered progress for gem5 SE / slow hosts. Off by default.
   Enable with LLM_STEPS=1 or --steps; force off with --quiet / LLM_QUIET. */
#define LLM_STEP(...)                          \
    do {                                       \
        if (llm_steps_enabled()) {             \
            printf("STEP: " __VA_ARGS__);      \
            fflush(stdout);                    \
        }                                      \
    } while (0)

void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t sz);
void *xrealloc(void *p, size_t n);
void llm_set_arena_alloc(void *(*fn)(size_t));
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);
void die(const char *fmt, ...);

typedef struct {
    int *data;
    int n;
    int cap;
} IntVec;

void intvec_init(IntVec *v);
void intvec_free(IntVec *v);
void intvec_clear(IntVec *v);
void intvec_push(IntVec *v, int x);
void intvec_extend(IntVec *v, const int *src, int n);

typedef struct {
    char *data;
    int n;
    int cap;
} ByteVec;

void bytevec_init(ByteVec *v);
void bytevec_free(ByteVec *v);
void bytevec_clear(ByteVec *v);
void bytevec_push(ByteVec *v, unsigned char b);
void bytevec_append(ByteVec *v, const void *src, int n);

#endif
