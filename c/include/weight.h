#ifndef LLM_WEIGHT_H
#define LLM_WEIGHT_H

#include "gguf.h"

/* F32 view of a tensor that was materialized as float32 (1D / unaligned). */
const float *weight_f32(const WeightTensor *w);

/* y = W @ x. W is (n_out, n_in) row-major, possibly still quantized. */
void linear_wt(const WeightTensor *W, const float *x, float *y, int n_out, int n_in);
void linear_rows_wt(const WeightTensor *W, const float *x, float *y, int n_tok, int n_out, int n_in);
void linear_rows_add_wt(const WeightTensor *W, const float *x, float *y, int n_tok, int n_out,
                        int n_in);

void embed_gather_wt(const WeightTensor *table, const int *ids, float *out, int n_rows, int d);

#endif
