#include "tensor.h"
#include "vk.h"
#include <float.h>
#include <math.h>
#include <string.h>

/* GPU: linear, RMSNorm, SiLU, vec_add, vec_mul. CPU: softmax.
 * Callers must host_write embeddings and host_read h before sliced logits. */

static GpuPC zpc(void) {
    GpuPC p;
    memset(&p, 0, sizeof(p));
    return p;
}

void linear(const float *W, const float *x, float *y, int n_out, int n_in) {
    linear_rows(W, x, y, 1, n_out, n_in);
    GpuBuf *by = gpu_find(y);
    if (by) gpu_commit(by);
}

static void linear_rows_ex(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in,
                           int acc) {
    if (n_tok <= 0 || n_out <= 0 || n_in <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = n_tok;
    pc.i[1] = n_out;
    pc.i[2] = n_in;
    pc.i[3] = acc;
    size_t xbytes = (size_t)n_tok * (size_t)n_in * sizeof(float);
    size_t xoff = 0;
    size_t wbytes = (size_t)n_out * (size_t)n_in * sizeof(float);
    GpuBuf *bW = gpu_find(W);
    GpuBufKind wkind = GPU_BUF_WEIGHT;
    if (bW && gpu_buf_is_f16(bW)) wkind = GPU_BUF_WEIGHT_F16;
    else if (bW && gpu_buf_is_q8(bW)) wkind = GPU_BUF_WEIGHT_Q8;
    bW = gpu_intern(W, wbytes, wkind);
    GpuBuf *bX = gpu_find_containing(x, xbytes, &xoff);
    if (!bX) {
        /* Unaligned slice of an interned activation: download the parent first
           so the new intern does not upload a stale host copy. */
        gpu_host_read(x);
        bX = gpu_intern(x, xbytes, GPU_BUF_RW);
        xoff = 0;
    }
    GpuBuf *bY = gpu_intern(y, (size_t)n_tok * (size_t)n_out * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bW, bX, bY};
    size_t offs[] = {0, xoff, 0};
    int pipe = GPU_PIPE_LINEAR;
    uint32_t gy = (uint32_t)n_tok;
    if (gpu_buf_is_q8(bW)) {
        pipe = GPU_PIPE_LINEAR_Q8;
    } else if (n_tok >= 2) {
        pipe = GPU_PIPE_LINEAR_GEMM;
        gy = ((uint32_t)n_tok + 3u) / 4u;
    }
    gpu_dispatch_offs(pipe, bs, offs, 3, &pc, ((uint32_t)n_out + 7u) / 8u, gy, 1);
    gpu_wrote(bY);
}

void linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    linear_rows_ex(W, x, y, n_tok, n_out, n_in, 0);
}

void linear_rows_add(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in) {
    linear_rows_ex(W, x, y, n_tok, n_out, n_in, 1);
}

void rmsnorm(const float *x, const float *weight, float *y, int n, float eps) {
    rmsnorm_rows(x, weight, y, 1, n, eps);
}

void rmsnorm_rows(const float *x, const float *weight, float *y, int n_tok, int n, float eps) {
    if (n_tok <= 0 || n <= 0) return;
    if (x == y) {
        gpu_host_read(x);
        for (int t = 0; t < n_tok; t++) {
            const float *xr = x + (size_t)t * (size_t)n;
            float *yr = y + (size_t)t * (size_t)n;
            float ms = 0.0f;
            for (int i = 0; i < n; i++) ms += xr[i] * xr[i];
            ms /= (float)n;
            float inv = 1.0f / sqrtf(ms + eps);
            for (int i = 0; i < n; i++) yr[i] = xr[i] * inv * weight[i];
        }
        gpu_host_write(y);
        return;
    }
    GpuPC pc = zpc();
    pc.i[0] = n;
    pc.i[1] = n_tok;
    pc.f[0] = eps;
    GpuBuf *bX = gpu_intern(x, (size_t)n_tok * (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bW = gpu_intern(weight, (size_t)n * sizeof(float), GPU_BUF_WEIGHT);
    GpuBuf *bY = gpu_intern(y, (size_t)n_tok * (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bX, bW, bY};
    gpu_dispatch(GPU_PIPE_RMSNORM, bs, 3, &pc, (uint32_t)n_tok, 1, 1);
    gpu_wrote(bY);
}

void silu(const float *x, float *y, int n) {
    if (n <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = n;
    GpuBuf *bX = gpu_intern(x, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bY = gpu_intern(y, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bX, bY};
    gpu_dispatch_1d(GPU_PIPE_SILU, bs, 2, &pc, (uint32_t)n);
    gpu_wrote(bY);
}

void silu_mul(const float *x, const float *g, float *y, int n) {
    if (n <= 0) return;
    if (x == y || g == y) {
        gpu_host_read(x);
        gpu_host_read(g);
        for (int i = 0; i < n; i++) {
            float xv = x[i];
            y[i] = (xv / (1.0f + expf(-xv))) * g[i];
        }
        gpu_host_write(y);
        return;
    }
    GpuPC pc = zpc();
    pc.i[0] = n;
    GpuBuf *bX = gpu_intern(x, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bG = gpu_intern(g, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bY = gpu_intern(y, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bX, bG, bY};
    gpu_dispatch_1d(GPU_PIPE_SILU_MUL, bs, 3, &pc, (uint32_t)n);
    gpu_wrote(bY);
}

void softmax_inplace(float *x, int n) {
    float m = -FLT_MAX;
    for (int i = 0; i < n; i++) {
        float v = isfinite(x[i]) ? x[i] : -1e9f;
        x[i] = v;
        if (v > m) m = v;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

void softmax_rows(float *x, int n_rows, int n) {
    for (int r = 0; r < n_rows; r++) softmax_inplace(x + (size_t)r * (size_t)n, n);
}

void vec_add(const float *a, const float *b, float *y, int n) {
    if (n <= 0) return;
    if (a == y || b == y) {
        gpu_host_read(a);
        gpu_host_read(b);
        if (a == y) {
            for (int i = 0; i < n; i++) y[i] += b[i];
        } else {
            for (int i = 0; i < n; i++) y[i] += a[i];
        }
        gpu_host_write(y);
        return;
    }
    GpuPC pc = zpc();
    pc.i[0] = n;
    GpuBuf *bA = gpu_intern(a, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bB = gpu_intern(b, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bY = gpu_intern(y, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bA, bB, bY};
    gpu_dispatch_1d(GPU_PIPE_VEC_ADD, bs, 3, &pc, (uint32_t)n);
    gpu_wrote(bY);
}

void vec_mul(const float *a, const float *b, float *y, int n) {
    if (n <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = n;
    GpuBuf *bA = gpu_intern(a, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bB = gpu_intern(b, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bY = gpu_intern(y, (size_t)n * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bA, bB, bY};
    gpu_dispatch_1d(GPU_PIPE_VEC_MUL, bs, 3, &pc, (uint32_t)n);
    gpu_wrote(bY);
}

void embed_gather(const float *table, const int *ids, float *out, int n_rows, int d) {
    if (n_rows <= 0 || d <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = n_rows;
    pc.i[1] = d;
    GpuBuf *bT = gpu_find(table);
    if (!bT) bT = gpu_intern(table, (size_t)n_rows * (size_t)d * sizeof(float), GPU_BUF_WEIGHT);
    GpuBuf *bI = gpu_intern(ids, (size_t)n_rows * sizeof(int), GPU_BUF_RW);
    GpuBuf *bO = gpu_intern(out, (size_t)n_rows * (size_t)d * sizeof(float), GPU_BUF_RW);
    GpuBuf *bs[] = {bT, bI, bO};
    int pipe = gpu_buf_is_f16(bT) ? GPU_PIPE_GATHER_F16 : GPU_PIPE_GATHER;
    gpu_dispatch_1d(pipe, bs, 3, &pc, (uint32_t)n_rows * (uint32_t)d);
    gpu_wrote(bO);
}

int argmax_f64(const double *x, int n) {
    int best = 0;
    double m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) {
            m = x[i];
            best = i;
        }
    }
    return best;
}

int argmax_f32(const float *x, int n) {
    if (n <= 0) return 0;
    gpu_host_read(x);
    int best = 0;
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) {
            m = x[i];
            best = i;
        }
    }
    return best;
}
