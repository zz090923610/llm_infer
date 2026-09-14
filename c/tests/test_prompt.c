/*
 * M0 prompt-level golden test. llm_infer is the correctness anchor.
 *
 * If this test looks wrong (bad tokens, NaNs, backend disagreement),
 * fix llm_infer before changing pim_func.
 *
 * Also dumps a deterministic linear() kernel for pim_func to compare against.
 */
#include "backend.h"
#include "generate.h"
#include "gguf.h"
#include "model.h"
#include "tensor.h"
#include "tokenizer.h"
#include "util.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LLM_DEFAULT_MODEL
#define LLM_DEFAULT_MODEL "models/smollm2-360m-instruct-q8_0.gguf"
#endif

#ifndef LLM_GOLDEN_DIR
#define LLM_GOLDEN_DIR "golden"
#endif

static int fails;

static int checksum_slack(void) {
    const char *b = llm_backend_name();
    return strcmp(b, "plain-cpu") != 0;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static uint32_t rng_state = 1u;

static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

static uint64_t checksum_f32(const float *x, int n) {
    uint64_t cs = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t bits = 0;
        memcpy(&bits, &x[i], sizeof(bits));
        cs ^= (uint64_t)bits;
        cs *= 1099511628211ULL;
    }
    return cs;
}

static int dump_linear_kernel(const char *path) {
    const int n_out = 16;
    const int n_in = 64;
    float *W = malloc((size_t)n_out * (size_t)n_in * sizeof(float));
    float *x = malloc((size_t)n_in * sizeof(float));
    float *y = malloc((size_t)n_out * sizeof(float));
    if (!W || !x || !y) {
        fprintf(stderr, "FAIL: alloc linear dump\n");
        fails++;
        free(W);
        free(x);
        free(y);
        return 1;
    }
    rng_state = 1u;
    for (int i = 0; i < n_out * n_in; i++) W[i] = frand();
    for (int i = 0; i < n_in; i++) x[i] = frand();
    linear(W, x, y, n_out, n_in);
    for (int i = 0; i < n_out; i++) {
        if (!isfinite(y[i])) {
            fprintf(stderr, "FAIL: linear dump y[%d] not finite\n", i);
            fails++;
        }
    }

    uint64_t got_cs = checksum_f32(y, n_out);
    if (file_exists(path)) {
        FILE *g = fopen(path, "r");
        char line[256];
        uint64_t want = 0;
        int found = 0;
        while (g && fgets(line, sizeof(line), g)) {
            unsigned long long tmp = 0;
            if (sscanf(line, "y_checksum 0x%llx", &tmp) == 1) {
                want = (uint64_t)tmp;
                found = 1;
                break;
            }
        }
        if (g) fclose(g);
        if (found && got_cs != want) {
            if (checksum_slack()) {
                printf("linear_dump checksum slack (%s simd) got=0x%016llx want=0x%016llx\n",
                       llm_backend_name(), (unsigned long long)got_cs, (unsigned long long)want);
            } else {
                fprintf(stderr,
                        "FAIL: linear dump checksum mismatch got=0x%016llx want=0x%016llx "
                        "(fix llm_infer first if the kernel is wrong)\n",
                        (unsigned long long)got_cs, (unsigned long long)want);
                fails++;
            }
        }
    } else {
        FILE *f = fopen(path, "w");
        if (!f) {
            fprintf(stderr, "FAIL: cannot write %s\n", path);
            fails++;
            free(W);
            free(x);
            free(y);
            return 1;
        }
        fprintf(f, "# llm_infer %s linear dump\n", llm_backend_name());
        fprintf(f, "n_out %d\n", n_out);
        fprintf(f, "n_in %d\n", n_in);
        fprintf(f, "y_checksum 0x%016llx\n", (unsigned long long)got_cs);
        fprintf(f, "W\n");
        for (int i = 0; i < n_out * n_in; i++) fprintf(f, "%.9g%s", W[i], ((i + 1) % 8) ? " " : "\n");
        if ((n_out * n_in) % 8) fprintf(f, "\n");
        fprintf(f, "x\n");
        for (int i = 0; i < n_in; i++) fprintf(f, "%.9g%s", x[i], ((i + 1) % 8) ? " " : "\n");
        if (n_in % 8) fprintf(f, "\n");
        fprintf(f, "y\n");
        for (int i = 0; i < n_out; i++) fprintf(f, "%.9g%s", y[i], ((i + 1) % 8) ? " " : "\n");
        if (n_out % 8) fprintf(f, "\n");
        fclose(f);
    }

    printf("linear_dump n_out=%d n_in=%d y_checksum=0x%016llx path=%s\n", n_out, n_in,
           (unsigned long long)got_cs, path);
    free(W);
    free(x);
    free(y);
    return fails ? 1 : 0;
}

static int parse_golden(const char *path, int *tokens, int max_tokens, int *n_tokens, char *text,
                        size_t text_cap, uint64_t *logits_cs, int *have_cs) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    *n_tokens = 0;
    text[0] = 0;
    *have_cs = 0;
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "tokens ", 7)) {
            char *p = line + 7;
            int v;
            while (*n_tokens < max_tokens && sscanf(p, "%d", &v) == 1) {
                tokens[(*n_tokens)++] = v;
                while (*p == ' ') p++;
                while (*p && *p != ' ' && *p != '\n') p++;
            }
        } else if (!strncmp(line, "text ", 5)) {
            char *p = line + 5;
            size_t n = strlen(p);
            while (n && (p[n - 1] == '\n' || p[n - 1] == '\r')) p[--n] = 0;
            snprintf(text, text_cap, "%s", p);
        } else if (!strncmp(line, "logits_checksum ", 16)) {
            if (sscanf(line + 16, "0x%llx", (unsigned long long *)logits_cs) == 1) *have_cs = 1;
        }
    }
    fclose(f);
    return 1;
}

static int run_prompt_golden(const char *model_path, const char *golden_path) {
    if (!file_exists(model_path)) {
        printf("test_prompt skipped (missing model %s)\n", model_path);
        return 0;
    }
    if (!file_exists(golden_path)) {
        fprintf(stderr, "FAIL: missing golden %s\n", golden_path);
        fails++;
        return 1;
    }

    int want_tok[64];
    int n_want = 0;
    char want_text[512];
    uint64_t want_cs = 0;
    int have_cs = 0;
    if (!parse_golden(golden_path, want_tok, 64, &n_want, want_text, sizeof(want_text), &want_cs,
                      &have_cs)) {
        fprintf(stderr, "FAIL: cannot parse %s\n", golden_path);
        fails++;
        return 1;
    }

    llm_set_quiet(1);
    LoadedModel *loaded = load_model(model_path, 1, 0);
    LlamaModel *model = llama_model_init(loaded);
    Tokenizer *tok = tokenizer_from_gguf(loaded->gguf, &loaded->hparams);

    ChatMessage msg = {"user", "Say hi in one word."};
    char *prompt = apply_chat_template(tok, &msg, 1, 1);
    IntVec ids;
    intvec_init(&ids);
    tokenizer_encode(tok, prompt, 1, &ids);

    GenerateState st;
    generate_state_init(&st, model, tok, NULL, 8, 0.0f, 1, 1.0f, 1);
    generate_start(&st, ids.data, ids.n);
    uint64_t logits_cs = checksum_f32(st.logits, model->hparams.n_vocab);
    if (!isfinite(st.logits[0])) {
        fprintf(stderr, "FAIL: first logit not finite — fix llm_infer first\n");
        fails++;
    }

    int got_tok[64];
    int n_got = 0;
    int tid = 0;
    while (n_got < 8 && generate_next(&st, &tid) == 0) {
        got_tok[n_got++] = tid;
    }

    char *got_text = tokenizer_decode(tok, got_tok, n_got, 1);

    printf("prompt_tokens n=%d ids=", n_got);
    for (int i = 0; i < n_got; i++) printf("%s%d", i ? "," : "", got_tok[i]);
    printf(" text=%s logits_checksum=0x%016llx backend=%s\n", got_text ? got_text : "",
           (unsigned long long)logits_cs, llm_backend_name());

    if (n_got != n_want) {
        fprintf(stderr, "FAIL: token count got=%d want=%d — if llm_infer looks wrong, fix it first\n",
                n_got, n_want);
        fails++;
    } else {
        for (int i = 0; i < n_got; i++) {
            if (got_tok[i] != want_tok[i]) {
                fprintf(stderr, "FAIL: token[%d] got=%d want=%d — if llm_infer looks wrong, fix it first\n",
                        i, got_tok[i], want_tok[i]);
                fails++;
                break;
            }
        }
    }
    if (got_text && strcmp(got_text, want_text) != 0) {
        fprintf(stderr, "FAIL: text got=%s want=%s — if llm_infer looks wrong, fix it first\n",
                got_text, want_text);
        fails++;
    }
    if (have_cs && logits_cs != want_cs) {
        /* SIMD/FP16 backends may differ in last bits; tokens must still match. */
        if (checksum_slack()) {
            printf("logits_checksum slack (%s) got=0x%016llx want=0x%016llx\n",
                   llm_backend_name(), (unsigned long long)logits_cs,
                   (unsigned long long)want_cs);
        } else {
            fprintf(stderr,
                    "FAIL: logits checksum got=0x%016llx want=0x%016llx — if llm_infer looks wrong, "
                    "fix it first\n",
                    (unsigned long long)logits_cs, (unsigned long long)want_cs);
            fails++;
        }
    }

    free(got_text);
    generate_state_free(&st);
    intvec_free(&ids);
    free(prompt);
    tokenizer_free(tok);
    llama_model_free(model);
    loaded_model_free(loaded);
    return fails ? 1 : 0;
}

int main(void) {
    char dump_path[1024];
    snprintf(dump_path, sizeof(dump_path), "%s/linear_16x64.txt", LLM_GOLDEN_DIR);
    dump_linear_kernel(dump_path);

    char golden_path[1024];
    snprintf(golden_path, sizeof(golden_path), "%s/prompt_hi.txt", LLM_GOLDEN_DIR);
    run_prompt_golden(LLM_DEFAULT_MODEL, golden_path);

    if (fails) {
        fprintf(stderr, "test_prompt FAILED (%d)\n", fails);
        return 1;
    }
    printf("test_prompt ok backend=%s\n", llm_backend_name());
    return 0;
}
