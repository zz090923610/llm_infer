#include "tokenizer.h"
#include "util.h"
#include <stdio.h>

#ifndef LLM_DEFAULT_MODEL
#define LLM_DEFAULT_MODEL "models/smollm2-360m-instruct-q8_0.gguf"
#endif

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

int main(void) {
    const char *path = LLM_DEFAULT_MODEL;
    if (!file_exists(path)) {
        printf("test_tokenizer skipped (missing %s)\n", path);
        return 0;
    }
    Tokenizer *tok = tokenizer_from_file(path);
    const char *text = "Hello, world! 123 cats.";
    IntVec ids;
    intvec_init(&ids);
    tokenizer_encode(tok, text, 0, &ids);
    char *back = tokenizer_decode(tok, ids.data, ids.n, 0);
    if (strcmp(back, text) != 0) {
        fprintf(stderr, "FAIL roundtrip:\n  in:  %s\n  out: %s\n", text, back);
        return 1;
    }
    free(back);
    intvec_free(&ids);

    int has_eos = 0, has_end = 0, has_gend = 0;
    int end_id = -1, gend_id = -1, im_start = -1, im_end = -1;
    strmap_get(&tok->token_to_id, "END", &end_id);
    strmap_get(&tok->token_to_id, "ĠEND", &gend_id);
    strmap_get(&tok->token_to_id, "<|im_start|>", &im_start);
    strmap_get(&tok->token_to_id, "<|im_end|>", &im_end);
    for (int i = 0; i < tok->n_stop; i++) {
        if (tok->stop_ids[i] == tok->hparams.eos_id) has_eos = 1;
        if (tok->stop_ids[i] == end_id) has_end = 1;
        if (tok->stop_ids[i] == gend_id) has_gend = 1;
    }
    if (!has_eos || !has_end || !has_gend) {
        fprintf(stderr, "FAIL stop_ids missing END markers\n");
        return 1;
    }

    ChatMessage msg = {"user", "Hi"};
    char *prompt = apply_chat_template(tok, &msg, 1, 1);
    intvec_init(&ids);
    tokenizer_encode(tok, prompt, 1, &ids);
    if (ids.n < 1 || ids.data[0] != im_start) {
        fprintf(stderr, "FAIL chat template first token\n");
        return 1;
    }
    int found_end = 0;
    for (int i = 0; i < ids.n; i++)
        if (ids.data[i] == im_end) found_end = 1;
    if (!found_end) {
        fprintf(stderr, "FAIL chat template missing im_end\n");
        return 1;
    }
    free(prompt);
    intvec_free(&ids);
    tokenizer_free(tok);
    printf("test_tokenizer ok\n");
    return 0;
}
