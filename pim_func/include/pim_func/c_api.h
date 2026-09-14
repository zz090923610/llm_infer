#ifndef PIM_FUNC_C_API_H
#define PIM_FUNC_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Intern row-major f32 weights under `key` (later passed to pim_prepare).
   Does not take ownership of W. name may be NULL. Returns 0 on success. */
int pim_intern_f32(const void *key, const float *W, int n_elements, const char *name);

/* Intern and take ownership of W (free on shutdown / replace). */
int pim_intern_f32_owned(const void *key, float *W, int n_elements, const char *name);

void pim_invalidate(const void *key);

const float *pim_interned_f32(const void *key, int *n_elements);
const char *pim_interned_name(const void *key);

/* Place interned weights into banks. Returns 0; sets row_base. */
int pim_prepare(const void *key, int n_out, int n_in);
int pim_row_base(const void *key);

typedef struct PimHw {
    int n_channels;
    int n_banks;
    int n_rows;
    int lanes;
    int gb_cols;
    int gpr_count;
    uint64_t pim_chmask;
} PimHw;

void pim_hw(PimHw *out);

typedef struct PimCommand {
    uint32_t opcode;
    int32_t opsize;
    int64_t gpr0;
    int64_t gpr1;
    int64_t chmask;
    int32_t row;
    int32_t col;
} PimCommand;

int pim_load_activation(const float *x, int n_in);
int pim_zero_gpr(int idx);
/* One ISR command. Host Device::execute or MMIO issue. EOC is a no-op. */
int pim_execute(const PimCommand *cmd);
int pim_unpack_wave(float *y, int n_out, int acc, int n_cols, int wave);

int pim_gb_cols_to_f32(float *out, int n_cols);
int pim_mac_wave_into(int ch, int row, int n_cols, float *acc, const float *x_f32);

int pim_hybrid_submit_cmds(const float *x, float *y, int n_out, int n_in, int row_base, int n_cols,
                           int acc, const PimCommand *cmds, int n_cmds);

#define PIM_ISSUE_HOST 0
#define PIM_ISSUE_MMIO 1
#define PIM_ISSUE_SHM 2

void pim_set_issue_mode(int mode);
int pim_issue_mode(void);

void pim_set_threads(int n);
int pim_n_threads(void);

#ifndef PIM_MAX_GEMV_THREADS
#define PIM_MAX_GEMV_THREADS 8
#endif

void pim_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
