#ifndef LLM_SAMPLER_H
#define LLM_SAMPLER_H

#include <stdint.h>

typedef struct {
    uint64_t s;
} Rng;

void rng_seed(Rng *r, uint64_t seed);
double rng_f64(Rng *r);

void softmax_last(const float *logits, int n, double *probs);
int sample_token(const float *logits, int n, float temperature, int top_k, float top_p, Rng *rng);

#endif
