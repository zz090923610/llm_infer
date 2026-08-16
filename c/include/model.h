#ifndef LLM_MODEL_H
#define LLM_MODEL_H

#include "cache.h"
#include "gguf.h"

typedef struct {
    const float *attn_norm;
    const WeightTensor *wq;
    const WeightTensor *wk;
    const WeightTensor *wv;
    const WeightTensor *wo;
    const float *bq, *bk, *bv;
    const float *q_norm, *k_norm;
    const float *ffn_norm;
    const WeightTensor *gate;
    const WeightTensor *up;
    const WeightTensor *down;
    const WeightTensor *wqkv;
    const WeightTensor *attn_gate;
    const float *ssm_a;
    const WeightTensor *ssm_alpha;
    const WeightTensor *ssm_beta;
    const float *ssm_conv1d;
    const float *ssm_dt;
    const float *ssm_norm;
    const WeightTensor *ssm_out;
    int is_gdn;
    int wq_out;
    int wo_in;
} LayerWeights;

typedef struct {
    LlamaHParams hparams;
    LoadedModel *owned;
    const WeightTensor *tok_embd;
    const WeightTensor *output;
    const float *output_norm;
    LayerWeights *layers;
    float *rope_cos;
    float *rope_sin;
    int rope_len;
    int rope_n_rot;
    /* scratch */
    float *x, *h, *q, *k, *v, *attn, *y, *ffn_gate, *ffn_up;
    float *gate_buf;
    float *gdn_scratch;
    float *logits;
    int *positions;
    int *key_len;
    int *starts;
    int *valid_buf;
    int *tok_ids;
    int scratch_B, scratch_S, scratch_K;
    size_t gdn_scratch_n;
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
