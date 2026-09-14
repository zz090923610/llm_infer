#pragma once

#include "pim_func/hybrid_abi.h"

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
namespace pim_func {
class BankedDram;
class Device;
} // namespace pim_func
#else
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int pim_hybrid_active(void);
void *pim_hybrid_base(void);
PimHybridHeader *pim_hybrid_header(void);
uint64_t pim_hybrid_v0(void);
uint64_t pim_hybrid_size(void);

/* B / gem5 host: create shm and MAP_FIXED at V0. */
int pim_hybrid_create_server(uint64_t size);
/* A: open existing shm and MAP_FIXED at V0. Retries until B is ready. */
int pim_hybrid_attach_client(void);
/* Guest: header already identity-mapped at V0. */
int pim_hybrid_guest_attach(void);

void *pim_hybrid_alloc(size_t n);
int pim_hybrid_contains(const void *p);

int pim_hybrid_record_region(int ch, int n_banks, int row0, int n_rows, int col0, int n_cols,
                             void *data);

void pim_hybrid_start_mac_worker(void);
void pim_hybrid_stop_mac_worker(void);

int pim_hybrid_submit(const float *x, float *y, int n_out, int n_in, int row_base, int n_cols,
                      int acc, const PimHybridCmd *cmds, int n_cmds);
void pim_hybrid_request_stop(void);

/* Guest: poll ISR jobs and issue GPR+ISR MMIO. Returns 0 on stop. */
int pim_hybrid_guest_loop(void);

#ifdef __cplusplus
}

namespace pim_func {
void hybrid_sync_dram(BankedDram *dram);
}
#endif
