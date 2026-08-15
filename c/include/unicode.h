#ifndef LLM_UNICODE_H
#define LLM_UNICODE_H

#include <stddef.h>
#include <stdint.h>

#define UCAT_OTHER 0
#define UCAT_LETTER 1
#define UCAT_NUMBER 2
#define UCAT_SPACE 3

typedef struct {
    uint32_t lo;
    uint32_t hi;
    uint8_t cat;
} UnicodeRange;

extern const UnicodeRange UNICODE_RANGES[];
extern const int UNICODE_N_RANGES;

int unicode_cat(uint32_t cp);
int is_letter(uint32_t cp);
int is_number(uint32_t cp);
int is_space(uint32_t cp);

/* Decode one UTF-8 codepoint. Returns bytes consumed, or 0 on error. */
int utf8_decode(const char *s, size_t len, size_t i, uint32_t *cp);
int utf8_encode(uint32_t cp, char out[4]);

/* Decode entire string into a newly allocated codepoint array. */
uint32_t *utf8_codepoints(const char *s, int *n_out);

#endif
