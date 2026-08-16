#ifndef LLM_GENERATE_H
#define LLM_GENERATE_H

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

typedef struct {
    LlamaModel *model;
    Tokenizer *tok;
    KVCache *cache;
    int own_cache;
    float temperature;
    int top_k;
    float top_p;
    int max_tokens;
    int n_generated;
    int done;
    Rng rng;
    float *logits;
} GenerateState;

/* seed 0 uses time(NULL). */
void generate_state_init(GenerateState *st, LlamaModel *model, Tokenizer *tok, KVCache *cache,
                         int max_tokens, float temperature, int top_k, float top_p, uint64_t seed);
void generate_state_free(GenerateState *st);

/* Prefill. Returns 0. */
int generate_start(GenerateState *st, const int *prompt_ids, int n_prompt);
/* Sample one token. Returns 0 and writes token, or 1 if stopped. */
int generate_next(GenerateState *st, int *token_out);

char *generate_text(LlamaModel *model, Tokenizer *tok, const char *prompt, int max_tokens,
                    float temperature, int top_k, float top_p, int parse_special, int stream,
                    KVCache *cache, uint64_t seed);

/* Returns newly allocated array of IntVec (length n_prompts). Caller frees each + the array. */
IntVec *generate_batch(LlamaModel *model, Tokenizer *tok, char **prompts, int n_prompts,
                       int max_tokens, float temperature, int top_k, float top_p, int parse_special);

#endif
