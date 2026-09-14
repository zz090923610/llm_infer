#define linear cpu_linear
#define linear_rows cpu_linear_rows
#define linear_rows_add cpu_linear_rows_add
#if defined(PIM_USE_X86_CPU_OPS)
#include "../host/pc/x86_64-simd/tensor.c"
#else
#include "../host/pc/plain-cpu/tensor.c"
#endif
