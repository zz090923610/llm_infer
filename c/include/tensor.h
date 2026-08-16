#ifndef LLM_TENSOR_H
#define LLM_TENSOR_H

#include <stddef.h>

/* Kernel contract. Implemented by the selected backend (see backend.h). */

/* y = W @ x  for one vector. W is (n_out, n_in) row-major. */
void linear(const float *W, const float *x, float *y, int n_out, int n_in);

/* Batched linear: x (n_tok, n_in) -> y (n_tok, n_out). */
void linear_rows(const float *W, const float *x, float *y, int n_tok, int n_out, int n_in);

void rmsnorm(const float *x, const float *weight, float *y, int n, float eps);
void rmsnorm_rows(const float *x, const float *weight, float *y, int n_tok, int n, float eps);

void silu(const float *x, float *y, int n);
void softmax_inplace(float *x, int n);
void softmax_rows(float *x, int n_rows, int n);

void vec_add(const float *a, const float *b, float *y, int n);
void vec_mul(const float *a, const float *b, float *y, int n);

int argmax_f64(const double *x, int n);
int argmax_f32(const float *x, int n);

#endif
