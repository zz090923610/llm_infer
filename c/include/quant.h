#ifndef LLM_QUANT_H
#define LLM_QUANT_H

#include <stddef.h>
#include <stdint.h>

#define QK8_0 32
#define BLOCK_Q8_0 34

float fp16_to_fp32(uint16_t h);

/* Dequantize packed Q8_0 bytes into a flat float32 vector of n_elements.
   Returns 0 on success. */
int dequantize_q8_0(const void *data, int n_elements, float *out);

/* Read a tightly packed float32 tensor. Returns 0 on success. */
int dequantize_f32(const void *data, int n_elements, float *out);

#endif
