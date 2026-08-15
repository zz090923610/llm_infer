#ifndef LLM_ROPE_H
#define LLM_ROPE_H

void build_rope_cache(float *cos_tab, float *sin_tab, int seq_len, int head_dim, float theta);

/* x: (B, n_head, S, head_dim) in-place. positions: (B, S).
   cos/sin: (seq_len, head_dim) with interleaved pairs. */
void apply_rope(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                int B, int n_head, int S, int head_dim);

#endif
