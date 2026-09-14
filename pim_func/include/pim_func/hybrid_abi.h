#ifndef PIM_FUNC_HYBRID_ABI_H
#define PIM_FUNC_HYBRID_ABI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fixed VA for the hybrid arena in A, gem5 host, and the gem5 guest. */
#define PIM_HYBRID_V0 0x200000000000ULL
#define PIM_HYBRID_MAGIC 0x314D494148594250ULL
#define PIM_HYBRID_MAX_REGIONS 2048
#define PIM_HYBRID_MAX_CMDS 8192
#define PIM_HYBRID_DEFAULT_SIZE (512ULL * 1024ULL * 1024ULL)
/* Guest only maps this prefix (header + x/y scratch). Banks live past it. */
#define PIM_HYBRID_GUEST_MAP (2ULL * 1024ULL * 1024ULL)
#define PIM_HYBRID_X_MAX 16384
#define PIM_HYBRID_Y_MAX 65536

typedef struct PimHybridRegion {
    int32_t ch;
    int32_t n_banks;
    int32_t row0;
    int32_t n_rows;
    int32_t col0;
    int32_t n_cols;
    uint64_t data_off;
} PimHybridRegion;

typedef struct PimHybridCmd {
    uint32_t opcode;
    int32_t opsize;
    int64_t gpr0;
    int64_t gpr1;
    int64_t chmask;
    int32_t row;
    int32_t col;
} PimHybridCmd;

typedef struct PimHybridHeader {
    uint64_t magic;
    uint64_t size;
    uint64_t v0;
    volatile uint32_t b_ready;
    volatile uint32_t a_ready;
    volatile uint32_t stop;
    uint32_t n_regions;
    int32_t next_row;
    uint32_t _pad0;
    uint64_t bump;
    PimHybridRegion regions[PIM_HYBRID_MAX_REGIONS];

    volatile uint32_t gemv_seq;
    volatile uint32_t gemv_done;
    int32_t n_out;
    int32_t n_in;
    int32_t row_base;
    int32_t n_cols;
    int32_t acc;
    int32_t n_waves;
    int32_t n_banks;
    int32_t ch;
    uint64_t x_ptr;
    uint64_t y_ptr;

    volatile uint32_t isr_seq;
    volatile uint32_t isr_done;
    uint32_t n_cmds;
    uint32_t _pad1;
    PimHybridCmd cmds[PIM_HYBRID_MAX_CMDS];
} PimHybridHeader;

#ifdef __cplusplus
}
#endif

#endif
