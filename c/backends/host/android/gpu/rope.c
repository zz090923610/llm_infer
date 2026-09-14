#include "rope.h"
#include "vk.h"
#include "util.h"
#include <math.h>
#include <string.h>

static GpuPC zpc(void) {
    GpuPC p;
    memset(&p, 0, sizeof(p));
    return p;
}

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

void apply_rope(float *x, const float *cos_tab, const float *sin_tab, const int *positions, int B,
                int n_head, int S, int head_dim) {
    if (B <= 0 || n_head <= 0 || S <= 0 || head_dim <= 0) return;
    int maxp = 0;
    for (int i = 0; i < B * S; i++)
        if (positions[i] > maxp) maxp = positions[i];
    size_t tab = (size_t)(maxp + 1) * (size_t)head_dim * sizeof(float);
    size_t xn = (size_t)B * (size_t)n_head * (size_t)S * (size_t)head_dim * sizeof(float);
    GpuPC pc = zpc();
    pc.i[0] = B;
    pc.i[1] = n_head;
    pc.i[2] = S;
    pc.i[3] = head_dim;
    GpuBuf *bX = gpu_intern(x, xn, GPU_BUF_RW);
    GpuBuf *bC = gpu_find(cos_tab);
    GpuBuf *bS = gpu_find(sin_tab);
    if (!bC) bC = gpu_intern(cos_tab, tab, GPU_BUF_WEIGHT);
    if (!bS) bS = gpu_intern(sin_tab, tab, GPU_BUF_WEIGHT);
    GpuBuf *bP = gpu_intern(positions, (size_t)B * (size_t)S * sizeof(int), GPU_BUF_RW);
    GpuBuf *bs[] = {bX, bC, bS, bP};
    gpu_dispatch_1d(GPU_PIPE_ROPE, bs, 4, &pc,
                    (uint32_t)B * (uint32_t)n_head * (uint32_t)S * (uint32_t)(head_dim / 2));
    gpu_wrote(bX);
}
