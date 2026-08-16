#ifndef LLM_GDN_H
#define LLM_GDN_H

#include "gguf.h"
#include "model.h"

int layer_is_gdn(const LlamaHParams *hp, int il);

/* One Gated DeltaNet layer. x: (B, S, D) residual stream (read).
   y: (B, S, D) attention output (write, no residual).
   conv_state: (B, conv_dim, d_conv-1)
   ssm_state:  (B, n_v_heads, d_state, d_state)  (transposed S as in llama.cpp)
   scratch: at least conv_dim + 4 * n_v_heads * d_state + n_v_heads + d_state floats. */
void gdn_layer_forward(const LayerWeights *L, const LlamaHParams *hp, const float *x, float *y,
                       int B, int S, const int *valid_len, float *conv_state, float *ssm_state,
                       float *scratch);

#endif
