#pragma once

#include "pim_func/device.h"

#include <cstdint>

namespace pim_func {

enum class FuncMode { Timing, Token };

struct RetiredAim {
    int opcode = 0;
    int channel_id = 0;
    int bank = -1;
    int row = -1;
    int col = -1;
    int opsize = -1;
    int64_t gpr0 = -1;
    int64_t chmask = -1;
};

FuncMode func_mode();
void set_func_mode(FuncMode m);

HardwareInfo gem5_hw();
void configure_device(const HardwareInfo &hw);
Device &global_device();

bool gpr_store(int idx, unsigned byte_off, const void *data, unsigned size);
bool gpr_load(int idx, unsigned byte_off, void *data, unsigned size);

void intern_begin(int n_out, int n_in, int row_base);
void intern_push_f32(const float *v, int n);
void intern_request_copy(uint64_t guest_va);
bool intern_copy_pending();
uint64_t intern_guest_va();
int intern_n_elements();
int intern_n_out();
int intern_n_in();
int intern_row_base();
bool intern_complete_from_host(const float *W);
int intern_status();

void on_retired_aim(const RetiredAim &cmd);

} // namespace pim_func
