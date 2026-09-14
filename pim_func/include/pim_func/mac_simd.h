#pragma once

#include "pim_func/mem.h"

namespace pim_func {

/* Sum_lane f16(w[lane]) * f16(x[lane]). */
float burst_dot_f16(const Burst &w, const Burst &x);

/* Convert 16 f16 lanes to f32 (F16C when available). */
void burst_f16_to_f32(const Burst &b, float *out_f32);

/* Sum_lane f16(w[lane]) * x_f32[lane] with preconverted x. */
float burst_dot_f16_xf32(const Burst &w, const float *x_f32);

/* MAC 16 banks: weight bursts at base, bank_stride bursts apart, add into acc[16]. */
void mac_banks16(const Burst *base, size_t bank_stride, const float *x_f32, float *acc);

/* True unless PIM_GEMV_SCALAR_STEPS is set to a non-zero value. */
bool fast_mac_enabled();

} // namespace pim_func
