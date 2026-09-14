#pragma once

#include "pim_func/gb.h"
#include "pim_func/isa.h"
#include "pim_func/layout.h"
#include "pim_func/mem.h"
#include "pim_func/pu.h"
#include "pim_func/types.h"

#include <string>
#include <vector>

namespace pim_func {

class MacFile {
public:
    MacFile(int n_channels, int n_banks);
    float read(int ch, int bank) const;
    void write(int ch, int bank, float v);
    void add(int ch, int bank, float v);
    void clear();
    float *bank_accs(int ch);
    const float *bank_accs(int ch) const;
    int n_banks() const { return n_banks_; }

private:
    int n_ch_;
    int n_banks_;
    std::vector<float> acc_;
};

class Device {
public:
    explicit Device(HardwareInfo hw = {});

    BankedDram &dram() { return dram_; }
    GprFile &gpr() { return gpr_; }
    GlobalBuffer &gb() { return gb_; }
    MacFile &mac() { return mac_; }
    const HardwareInfo &hw() const { return hw_; }

    /* Full ISR: unrolls opsize columns when col is unset. */
    std::string execute(const Command &cmd);
    /* One column / one channel, used by the Ramulator retirement hook. */
    std::string execute_micro(const Command &cmd, int channel_id);

    void load_weights_f32(const float *W, int n_out, int n_in, const AllocationPolicy &alloc,
                          int row_base = 0);
    void load_activation_f32(const float *x, int n_in, int gpr0);
    void read_mac_f32(float *y, int n_out, int gpr0) const;
    void read_mac_banks_f32(int ch, float *y, int n) const;

    /* MAC_ABK into private acc (host wave-parallel). Not a GEMV planner. */
    void mac_wave_into(int ch, int row, int n_cols, float *acc_out) const;
    void mac_wave_into(int ch, int row, int n_cols, float *acc_out, const float *x_f32) const;
    void gb_cols_to_f32(int ch, int n_cols, float *out) const;

    void reset();

private:
    void apply_step(const MicroStep &st);
    bool exec_wr_gb_fast(const Command &cmd, int channel_id);
    bool exec_mac_abk_fast(const Command &cmd, int channel_id);
    bool exec_wr_bias_fast(const Command &cmd, int channel_id);
    bool exec_rd_mac_fast(const Command &cmd, int channel_id);
    void mac_abk_one_col(int ch, int row, int col, const DenseBurstRegion *region);
    void mac_abk_one_col_into(int ch, int row, int col, const DenseBurstRegion *region,
                              float *acc) const;
    void mac_abk_one_col_xf32(int ch, int row, int col, const DenseBurstRegion *region, float *acc,
                              const float *xf) const;
    HardwareInfo hw_;
    BankedDram dram_;
    GprFile gpr_;
    GlobalBuffer gb_;
    MacFile mac_;
    FullAbkParallelism par_;
    CommandRegistry reg_;
};

} // namespace pim_func
