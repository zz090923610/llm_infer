#include "attn.h"
#include "vk.h"
#include <string.h>

static GpuPC zpc(void) {
    GpuPC p;
    memset(&p, 0, sizeof(p));
    return p;
}

void attn_pack_heads(const float *src, float *dst, int B, int S, int n_head, int d) {
    if (B <= 0 || S <= 0 || n_head <= 0 || d <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = B;
    pc.i[1] = S;
    pc.i[2] = n_head;
    pc.i[3] = d;
    size_t n = (size_t)B * (size_t)S * (size_t)n_head * (size_t)d * sizeof(float);
    GpuBuf *bSrc = gpu_intern(src, n, GPU_BUF_RW);
    GpuBuf *bDst = gpu_intern(dst, n, GPU_BUF_RW);
    GpuBuf *bs[] = {bSrc, bDst};
    gpu_dispatch_1d(GPU_PIPE_PACK, bs, 2, &pc, (uint32_t)B * (uint32_t)S * (uint32_t)n_head * (uint32_t)d);
    gpu_wrote(bDst);
}

void attn_merge_heads(const float *src, float *dst, int B, int S, int n_head, int d) {
    if (B <= 0 || S <= 0 || n_head <= 0 || d <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = B;
    pc.i[1] = S;
    pc.i[2] = n_head;
    pc.i[3] = d;
    size_t n = (size_t)B * (size_t)S * (size_t)n_head * (size_t)d * sizeof(float);
    GpuBuf *bSrc = gpu_intern(src, n, GPU_BUF_RW);
    GpuBuf *bDst = gpu_intern(dst, n, GPU_BUF_RW);
    GpuBuf *bs[] = {bSrc, bDst};
    gpu_dispatch_1d(GPU_PIPE_MERGE, bs, 2, &pc, (uint32_t)B * (uint32_t)S * (uint32_t)n_head * (uint32_t)d);
    gpu_wrote(bDst);
}

void attn_cache_store(float *cache_k, float *cache_v, const float *k, const float *v,
                      size_t batch_stride, size_t head_stride, const int *valid_len,
                      const int *starts, int B, int S, int n_head_kv, int d) {
    if (B <= 0 || S <= 0 || n_head_kv <= 0 || d <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = B;
    pc.i[1] = S;
    pc.i[2] = n_head_kv;
    pc.i[3] = d;
    pc.i[4] = (int32_t)batch_stride;
    pc.i[5] = (int32_t)head_stride;
    size_t kv = (size_t)B * (size_t)n_head_kv * (size_t)S * (size_t)d * sizeof(float);
    size_t layer = (size_t)B * batch_stride * sizeof(float);
    GpuBuf *bCK = gpu_intern(cache_k, layer, GPU_BUF_DEVICE);
    GpuBuf *bCV = gpu_intern(cache_v, layer, GPU_BUF_DEVICE);
    GpuBuf *bK = gpu_intern(k, kv, GPU_BUF_RW);
    GpuBuf *bV = gpu_intern(v, kv, GPU_BUF_RW);
    GpuBuf *bVL = gpu_intern(valid_len, (size_t)B * sizeof(int), GPU_BUF_RW);
    GpuBuf *bST = gpu_intern(starts, (size_t)B * sizeof(int), GPU_BUF_RW);
    GpuBuf *bs[] = {bCK, bCV, bK, bV, bVL, bST};
    gpu_dispatch_1d(GPU_PIPE_CACHE_STORE, bs, 6, &pc,
                    (uint32_t)B * (uint32_t)n_head_kv * (uint32_t)S * (uint32_t)d);
    gpu_wrote(bCK);
    gpu_wrote(bCV);
}

void attn_gqa(float *y, float *attn, const float *q, const float *cache_k, const float *cache_v,
              size_t batch_stride, size_t head_stride, const int *positions, const int *valid_len,
              const int *key_len, int B, int n_head, int S, int n_head_kv, int head_dim, int max_k,
              float scale) {
    (void)attn;
    if (B <= 0 || n_head <= 0 || S <= 0 || head_dim <= 0 || max_k <= 0) return;
    GpuPC pc = zpc();
    pc.i[0] = B;
    pc.i[1] = n_head;
    pc.i[2] = S;
    pc.i[3] = n_head_kv;
    pc.i[4] = head_dim;
    pc.i[5] = max_k;
    pc.i[6] = (int32_t)batch_stride;
    pc.i[7] = (int32_t)head_stride;
    pc.f[0] = scale;
    size_t qn = (size_t)B * (size_t)n_head * (size_t)S * (size_t)head_dim * sizeof(float);
    size_t layer = (size_t)B * batch_stride * sizeof(float);
    GpuBuf *bY = gpu_intern(y, qn, GPU_BUF_RW);
    GpuBuf *bQ = gpu_intern(q, qn, GPU_BUF_RW);
    GpuBuf *bK = gpu_intern(cache_k, layer, GPU_BUF_DEVICE);
    GpuBuf *bV = gpu_intern(cache_v, layer, GPU_BUF_DEVICE);
    GpuBuf *bP = gpu_intern(positions, (size_t)B * (size_t)S * sizeof(int), GPU_BUF_RW);
    GpuBuf *bVL = gpu_intern(valid_len, (size_t)B * sizeof(int), GPU_BUF_RW);
    GpuBuf *bKL = gpu_intern(key_len, (size_t)B * sizeof(int), GPU_BUF_RW);
    GpuBuf *bs[] = {bY, bQ, bK, bV, bP, bVL, bKL};
    uint32_t ng = (uint32_t)B * (uint32_t)n_head * (uint32_t)S;
    gpu_dispatch(GPU_PIPE_ATTN, bs, 7, &pc, ng, 1, 1);
    gpu_wrote(bY);
}
