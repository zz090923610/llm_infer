#include "backend.h"

const char *llm_backend_name(void) {
    return LLM_BACKEND_NAME;
}

void llm_backend_sync(void) {}

void llm_backend_intern_weight(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}

void llm_backend_intern_weight_f16(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}

void llm_backend_intern_weight_q8(const void *host_key, const void *q8_blob, int n_elements) {
    (void)host_key;
    (void)q8_blob;
    (void)n_elements;
}

void llm_backend_intern_weight_quant(const void *host_key, const void *blob, int n_elements,
                                     int ggml_type) {
    (void)host_key;
    (void)blob;
    (void)n_elements;
    (void)ggml_type;
}

int llm_backend_q8_linear(void) {
    return 0;
}

int llm_backend_quant_linear(int ggml_type) {
    (void)ggml_type;
    return 0;
}

void llm_backend_host_write(void *p) {
    (void)p;
}

void llm_backend_host_read(const void *p) {
    (void)p;
}

void llm_backend_intern_rw(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}

void llm_backend_intern_device(const void *p, size_t bytes) {
    (void)p;
    (void)bytes;
}
