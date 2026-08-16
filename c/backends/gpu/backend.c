#include "backend.h"
#include "vk.h"

const char *llm_backend_name(void) {
    return gpu_device_name();
}

void llm_backend_set_threads(int n) {
    (void)n;
}

void llm_backend_set_prefill_threads(int n) {
    (void)n;
}

void llm_backend_set_decode_threads(int n) {
    (void)n;
}

int llm_backend_n_threads(void) {
    return 1;
}

int llm_backend_n_prefill_threads(void) {
    return 1;
}

int llm_backend_n_decode_threads(void) {
    return 1;
}

void llm_backend_sync(void) {
    gpu_sync();
}

void llm_backend_intern_weight(const void *p, size_t bytes) {
    if (!p || bytes == 0) return;
    GpuBuf *b = gpu_intern(p, bytes, GPU_BUF_WEIGHT);
    gpu_upload(b);
}

void llm_backend_intern_weight_f16(const void *p, size_t bytes) {
    if (!p || bytes == 0) return;
    GpuBufKind kind = gpu_has_f16_texel() ? GPU_BUF_WEIGHT_F16 : GPU_BUF_WEIGHT;
    GpuBuf *b = gpu_intern(p, bytes, kind);
    gpu_upload(b);
}

void llm_backend_intern_weight_q8(const void *host_key, const void *q8_blob, int n_elements) {
    gpu_intern_q8(host_key, q8_blob, n_elements);
}

void llm_backend_host_write(void *p) {
    gpu_host_write(p);
}

void llm_backend_host_read(const void *p) {
    gpu_host_read(p);
}

void llm_backend_intern_rw(const void *p, size_t bytes) {
    if (!p || bytes == 0) return;
    gpu_intern(p, bytes, GPU_BUF_RW);
}

void llm_backend_intern_device(const void *p, size_t bytes) {
    if (!p || bytes == 0) return;
    GpuBuf *b = gpu_intern(p, bytes, GPU_BUF_DEVICE);
    gpu_upload(b);
}
