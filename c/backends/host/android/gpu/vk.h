#ifndef LLM_GPU_VK_H
#define LLM_GPU_VK_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    GPU_BUF_WEIGHT = 0, /* upload once as f32, never download */
    GPU_BUF_RW = 1,     /* activations; CPU may write after a wait */
    GPU_BUF_DEVICE = 2, /* KV cache; GPU is source of truth */
    GPU_BUF_WEIGHT_F16 = 3, /* host f32, device packed f16 */
    GPU_BUF_WEIGHT_Q8 = 4  /* host f32 key, device Q8_0 36-byte blocks */
} GpuBufKind;

typedef struct GpuBuf GpuBuf;

typedef struct {
    int32_t i[8];
    float f[4];
} GpuPC;

enum {
    GPU_PIPE_LINEAR = 0,
    GPU_PIPE_RMSNORM,
    GPU_PIPE_SILU,
    GPU_PIPE_VEC_ADD,
    GPU_PIPE_VEC_MUL,
    GPU_PIPE_SOFTMAX,
    GPU_PIPE_ROPE,
    GPU_PIPE_PACK,
    GPU_PIPE_MERGE,
    GPU_PIPE_CACHE_STORE,
    GPU_PIPE_ATTN,
    GPU_PIPE_GATHER,
    GPU_PIPE_LINEAR_F16,
    GPU_PIPE_GATHER_F16,
    GPU_PIPE_LINEAR_GEMM,
    GPU_PIPE_ARGMAX,
    GPU_PIPE_LINEAR_Q8,
    GPU_PIPE_SILU_MUL
};

void gpu_init(void);
void gpu_sync(void); /* submit, wait, download dirty RW buffers */
const char *gpu_device_name(void);

GpuBuf *gpu_intern(const void *host, size_t bytes, GpuBufKind kind);
GpuBuf *gpu_find(const void *host);
GpuBuf *gpu_find_containing(const void *p, size_t bytes, size_t *off_out);
int gpu_buf_is_f16(const GpuBuf *b);
int gpu_buf_is_q8(const GpuBuf *b);
int gpu_has_f16_texel(void);
int gpu_has_rgba16_texel(void);
void gpu_intern_q8(const void *host, const void *q8_blob, int n_elements);
void gpu_upload(GpuBuf *b);
void gpu_wrote(GpuBuf *b);
void gpu_commit(GpuBuf *b); /* wait + download this buffer to host */
void gpu_host_read(const void *p);
void gpu_host_write(void *p);

void gpu_dispatch(int pipe, GpuBuf **bufs, int nbuf, const GpuPC *pc, uint32_t gx, uint32_t gy,
                  uint32_t gz);
void gpu_dispatch_offs(int pipe, GpuBuf **bufs, const size_t *offs, int nbuf, const GpuPC *pc,
                       uint32_t gx, uint32_t gy, uint32_t gz);
void gpu_dispatch_1d(int pipe, GpuBuf **bufs, int nbuf, const GpuPC *pc, uint32_t n_threads);

#endif
