#ifndef LLM_MODEL_H
#define LLM_MODEL_H

#include "cache.h"
#include "gguf.h"

typedef struct {
    const float *attn_norm;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *wo;
    const float *ffn_norm;
    const float *gate;
    const float *up;
    const float *down;
} LayerWeights;

typedef struct {
    LlamaHParams hparams;
    LoadedModel *owned;
    const float *tok_embd;
    const float *output_norm;
    LayerWeights *layers;
    float *rope_cos;
    float *rope_sin;
    int rope_len;
    /* scratch */
    float *x, *h, *q, *k, *v, *attn, *y, *ffn_gate, *ffn_up;
    float *logits;
    int *positions;
    int *key_len;
    int *starts;
    int *valid_buf;
    int *tok_ids;
    int scratch_B, scratch_S, scratch_K;
} LlamaModel;

LlamaModel *llama_model_init(LoadedModel *loaded);
LlamaModel *llama_model_from_file(const char *path, int progress);
void llama_model_free(LlamaModel *m);

KVCache *llama_model_new_cache(LlamaModel *m, int batch, int max_seq);

/* tokens: (B, S) row-major. valid_len NULL means all S.
   logits_out: (B, vocab) or NULL to use m->logits.
   Returns pointer to logits (B, vocab). */
float *llama_model_forward(LlamaModel *m, const int *tokens, int B, int S, KVCache *cache,
                           const int *valid_len, float *logits_out);

#endif
