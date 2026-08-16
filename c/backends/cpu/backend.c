#include "backend.h"

const char *llm_backend_name(void) {
    return LLM_BACKEND_NAME;
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
