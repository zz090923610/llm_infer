#ifndef PIM_FUNC_GUEST_MMIO_H
#define PIM_FUNC_GUEST_MMIO_H

/*
 * Guest-side AXI MMIO helpers. Addresses match develop/include/aim_mmio.h.
 * FUNC mailbox is untimed (completed in the GEM5 frontend).
 */
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AIM_DRAM_SIZE        (16ULL * 1024ULL * 1024ULL * 1024ULL)
#define AIM_MMIO_BASE        0x4000010000ULL
#define AIM_ISR_BASE         (AIM_MMIO_BASE + 0x0000ULL)
#define AIM_GPR_BASE         (AIM_MMIO_BASE + 0x2000ULL)
#define AIM_GPR_STRIDE       32U
#define AIM_GPR_COUNT        1024U

#define AIM_ISR_REG_OPCODE   0x00U
#define AIM_ISR_REG_OPSIZE   0x08U
#define AIM_ISR_REG_GPR0     0x10U
#define AIM_ISR_REG_GPR1     0x18U
#define AIM_ISR_REG_CHMASK   0x20U
#define AIM_ISR_REG_BANK     0x28U
#define AIM_ISR_REG_ROW      0x30U
#define AIM_ISR_REG_COL      0x38U
#define AIM_ISR_REG_STATUS   0x40U
#define AIM_ISR_REG_COMMIT   0x48U

#define AIM_ISR_STATUS_IDLE  0U
#define AIM_ISR_COMMIT_KEY   1U

#define AIM_OP_WR_GB         2U
#define AIM_OP_WR_BIAS       3U
#define AIM_OP_RD_MAC        5U
#define AIM_OP_MAC_ABK       11U
#define AIM_OP_EOC           16U

#define AIM_PIM_CHMASK       0xF0u

#define AIM_FUNC_REG_CMD     0x200U
#define AIM_FUNC_REG_A       0x208U
#define AIM_FUNC_REG_B       0x210U
#define AIM_FUNC_REG_C       0x218U
#define AIM_FUNC_REG_DATA    0x220U
#define AIM_FUNC_REG_STATUS  0x228U

#define AIM_FUNC_CMD_CONFIG        1U
#define AIM_FUNC_CMD_INTERN_BEGIN  2U
#define AIM_FUNC_CMD_INTERN_PTR    3U

static inline volatile uint32_t *aim_mmio32(uint64_t base, uint32_t off) {
    return (volatile uint32_t *)(uintptr_t)(base + off);
}

static inline volatile uint64_t *aim_mmio64(uint64_t base, uint32_t off) {
    return (volatile uint64_t *)(uintptr_t)(base + off);
}

/* Ramulator AiMDRAM: mask bit i selects channel (N-1-i). Functional bit i = HW ch i. */
static inline uint64_t aim_func_to_isr_chmask(uint64_t func_mask, int n_channels) {
    uint64_t out = 0;
    if (n_channels <= 0) n_channels = 8;
    for (int i = 0; i < n_channels; i++) {
        if (func_mask & (1ULL << i)) out |= (1ULL << (n_channels - 1 - i));
    }
    return out;
}

static inline void aim_isr_reset_mailbox(void) {
    *aim_mmio64(AIM_ISR_BASE, AIM_ISR_REG_GPR0) = 0;
    *aim_mmio64(AIM_ISR_BASE, AIM_ISR_REG_GPR1) = 0;
    *aim_mmio64(AIM_ISR_BASE, AIM_ISR_REG_CHMASK) = 0;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_OPCODE) = 0;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_OPSIZE) = (uint32_t)-1;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_BANK) = (uint32_t)-1;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_ROW) = (uint32_t)-1;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_COL) = (uint32_t)-1;
}

static inline uint32_t aim_isr_read_status(void) {
    return *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_STATUS);
}

static inline void aim_isr_wait_idle(void) {
    while (aim_isr_read_status() != AIM_ISR_STATUS_IDLE) {
    }
}

static inline void aim_isr_commit_and_wait(void) {
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_COMMIT) = AIM_ISR_COMMIT_KEY;
    aim_isr_wait_idle();
}

static inline void aim_iss_cmd(uint32_t opcode, int32_t opsize, uint64_t gpr0, uint64_t chmask,
                               int32_t row) {
    aim_isr_reset_mailbox();
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_OPCODE) = opcode;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_OPSIZE) = (uint32_t)opsize;
    *aim_mmio64(AIM_ISR_BASE, AIM_ISR_REG_GPR0) = gpr0;
    *aim_mmio64(AIM_ISR_BASE, AIM_ISR_REG_CHMASK) = chmask;
    *aim_mmio32(AIM_ISR_BASE, AIM_ISR_REG_ROW) = (uint32_t)row;
    aim_isr_commit_and_wait();
}

static inline void aim_gpr_write_bytes(uint32_t idx, unsigned off, const void *data, unsigned n) {
    volatile uint8_t *p =
        (volatile uint8_t *)(uintptr_t)(AIM_GPR_BASE + (uint64_t)idx * AIM_GPR_STRIDE + off);
    const uint8_t *s = (const uint8_t *)data;
    for (unsigned i = 0; i < n; i++) p[i] = s[i];
}

static inline void aim_gpr_read_bytes(uint32_t idx, unsigned off, void *data, unsigned n) {
    volatile uint8_t *p =
        (volatile uint8_t *)(uintptr_t)(AIM_GPR_BASE + (uint64_t)idx * AIM_GPR_STRIDE + off);
    uint8_t *d = (uint8_t *)data;
    for (unsigned i = 0; i < n; i++) d[i] = p[i];
}

static inline void aim_func_cmd(uint32_t cmd, uint64_t a, uint64_t b, uint64_t c) {
    *aim_mmio64(AIM_ISR_BASE, AIM_FUNC_REG_A) = a;
    *aim_mmio64(AIM_ISR_BASE, AIM_FUNC_REG_B) = b;
    *aim_mmio64(AIM_ISR_BASE, AIM_FUNC_REG_C) = c;
    *aim_mmio32(AIM_ISR_BASE, AIM_FUNC_REG_CMD) = cmd;
}

static inline void aim_func_push_u64(uint64_t v) {
    *aim_mmio64(AIM_ISR_BASE, AIM_FUNC_REG_DATA) = v;
}

#ifdef __cplusplus
}
#endif

#endif
