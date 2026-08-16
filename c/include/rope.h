#ifndef LLM_ROPE_H
#define LLM_ROPE_H

/* Kernel contract. Implemented by the selected backend (see backend.h). */

void build_rope_cache(float *cos_tab, float *sin_tab, int seq_len, int head_dim, float theta);

/* x: (B, n_head, S, head_dim) in-place. positions: (B, S).
   cos/sin: (seq_len, head_dim) with interleaved pairs. */
void apply_rope(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                int B, int n_head, int S, int head_dim);

/* Common (not backend): NeoX / partial RoPE. cos/sin are (seq_len, n_rot) interleaved pairs. */
void build_rope_cache_n(float *cos_tab, float *sin_tab, int seq_len, int n_rot, float theta);
void apply_rope_neox(float *x, const float *cos_tab, const float *sin_tab, const int *positions,
                     int B, int n_head, int S, int head_dim, int n_rot);

#endif
