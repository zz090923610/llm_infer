#include "unicode.h"
#include "util.h"

int unicode_cat(uint32_t cp) {
    int lo = 0, hi = UNICODE_N_RANGES - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        const UnicodeRange *r = &UNICODE_RANGES[mid];
        if (cp < r->lo) hi = mid - 1;
        else if (cp > r->hi) lo = mid + 1;
        else return r->cat;
    }
    return UCAT_OTHER;
}

int is_letter(uint32_t cp) { return unicode_cat(cp) == UCAT_LETTER; }
int is_number(uint32_t cp) { return unicode_cat(cp) == UCAT_NUMBER; }
int is_space(uint32_t cp) { return unicode_cat(cp) == UCAT_SPACE; }

int utf8_decode(const char *s, size_t len, size_t i, uint32_t *cp) {
    if (i >= len) return 0;
    unsigned char c = (unsigned char)s[i];
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c & 0xe0) == 0xc0 && i + 1 < len) {
        *cp = ((uint32_t)(c & 0x1f) << 6) | (uint32_t)(s[i + 1] & 0x3f);
        return 2;
    }
    if ((c & 0xf0) == 0xe0 && i + 2 < len) {
        *cp = ((uint32_t)(c & 0x0f) << 12) | ((uint32_t)(s[i + 1] & 0x3f) << 6) |
              (uint32_t)(s[i + 2] & 0x3f);
        return 3;
    }
    if ((c & 0xf8) == 0xf0 && i + 3 < len) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3f) << 12) |
              ((uint32_t)(s[i + 2] & 0x3f) << 6) | (uint32_t)(s[i + 3] & 0x3f);
        return 4;
    }
    *cp = 0xfffd;
    return 1;
}

int utf8_encode(uint32_t cp, char out[4]) {
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        out[0] = (char)(0xc0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3f));
        return 2;
    }
    if (cp < 0x10000) {
        out[0] = (char)(0xe0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[2] = (char)(0x80 | (cp & 0x3f));
        return 3;
    }
    out[0] = (char)(0xf0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3f));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3f));
    out[3] = (char)(0x80 | (cp & 0x3f));
    return 4;
}

uint32_t *utf8_codepoints(const char *s, int *n_out) {
    size_t len = strlen(s);
    uint32_t *cps = xmalloc((len + 1) * sizeof(uint32_t));
    int n = 0;
    size_t i = 0;
    while (i < len) {
        uint32_t cp;
        int adv = utf8_decode(s, len, i, &cp);
        if (adv <= 0) break;
        cps[n++] = cp;
        i += (size_t)adv;
    }
    *n_out = n;
    return cps;
}
