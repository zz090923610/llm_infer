#include "rope.h"
#include "util.h"
#include <math.h>

void build_rope_cache(float *cos_tab, float *sin_tab, int seq_len, int head_dim, float theta) {
    int half = head_dim / 2;
    float *inv = xmalloc((size_t)half * sizeof(float));
    for (int i = 0; i < half; i++) {
        inv[i] = (float)pow((double)theta, -(double)i / (double)half);
    }
    for (int t = 0; t < seq_len; t++) {
        for (int i = 0; i < half; i++) {
            float freq = (float)t * inv[i];
            float c = cosf(freq);
            float s = sinf(freq);
            cos_tab[(size_t)t * head_dim + 2 * i] = c;
            cos_tab[(size_t)t * head_dim + 2 * i + 1] = c;
            sin_tab[(size_t)t * head_dim + 2 * i] = s;
            sin_tab[(size_t)t * head_dim + 2 * i + 1] = s;
        }
    }
    free(inv);
}

void apply_rope(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                int B, int n_head, int S, int head_dim) {
    int pairs = head_dim / 2;
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_head; h++) {
            for (int s = 0; s < S; s++) {
                int pos = positions[b * S + s];
                float *row = x + ((((size_t)b * n_head + (size_t)h) * S + (size_t)s) * (size_t)head_dim);
                const float *c = cos_tab + (size_t)pos * (size_t)head_dim;
                const float *si = sin_tab + (size_t)pos * (size_t)head_dim;
                for (int i = 0; i < pairs; i++) {
                    float x0 = row[2 * i];
                    float x1 = row[2 * i + 1];
                    float cv = c[2 * i];
                    float sv = si[2 * i];
                    row[2 * i] = x0 * cv - x1 * sv;
                    row[2 * i + 1] = x0 * sv + x1 * cv;
                }
            }
        }
    }
}
