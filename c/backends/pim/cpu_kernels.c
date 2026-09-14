#define linear cpu_linear
#define linear_rows cpu_linear_rows
#define linear_rows_add cpu_linear_rows_add
#if defined(PIM_USE_X86_CPU_OPS)
#include "../x86_64/tensor.c"
#else
#include "../cpu/tensor.c"
#endif
