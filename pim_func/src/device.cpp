#include "pim_func/device.h"
#include "pim_func/mac_simd.h"

#include <algorithm>
#include <vector>

namespace pim_func {

MacFile::MacFile(int n_channels, int n_banks)
    : n_ch_(n_channels), n_banks_(n_banks),
      acc_(static_cast<size_t>(n_channels) * static_cast<size_t>(n_banks), 0.0f) {}

float MacFile::read(int ch, int bank) const {
    if (ch < 0 || ch >= n_ch_ || bank < 0 || bank >= n_banks_) return 0.0f;
    return acc_[static_cast<size_t>(ch) * static_cast<size_t>(n_banks_) + static_cast<size_t>(bank)];
}

void MacFile::write(int ch, int bank, float v) {
    if (ch < 0 || ch >= n_ch_ || bank < 0 || bank >= n_banks_) return;
    acc_[static_cast<size_t>(ch) * static_cast<size_t>(n_banks_) + static_cast<size_t>(bank)] = v;
}

void MacFile::add(int ch, int bank, float v) {
    if (ch < 0 || ch >= n_ch_ || bank < 0 || bank >= n_banks_) return;
    acc_[static_cast<size_t>(ch) * static_cast<size_t>(n_banks_) + static_cast<size_t>(bank)] += v;
}

void MacFile::clear() { std::fill(acc_.begin(), acc_.end(), 0.0f); }

float *MacFile::bank_accs(int ch) {
    if (ch < 0 || ch >= n_ch_) return nullptr;
    return acc_.data() + static_cast<size_t>(ch) * static_cast<size_t>(n_banks_);
}

const float *MacFile::bank_accs(int ch) const {
    if (ch < 0 || ch >= n_ch_) return nullptr;
    return acc_.data() + static_cast<size_t>(ch) * static_cast<size_t>(n_banks_);
}

Device::Device(HardwareInfo hw)
    : hw_(hw), gpr_(hw.gpr_count), gb_(hw.n_channels, hw.gb_cols), mac_(hw.n_channels, hw.n_banks) {}

void Device::reset() {
    dram_.clear();
    gpr_.clear();
    gb_.clear();
    mac_.clear();
}

void Device::apply_step(const MicroStep &st) {
    switch (st.kind) {
    case StepKind::WrGbLane:
        gb_.write_lane(st.ch, st.col, st.lane, gpr_.read_lane(st.gpr, st.lane));
        break;
    case StepKind::WrBiasBank:
        mac_.write(st.ch, st.bank, f16_to_f32(gpr_.read_lane(st.gpr, st.bank)));
        break;
    case StepKind::MacLane: {
        const float w = f16_to_f32(dram_.read_lane(st.ch, st.bank, st.row, st.col, st.lane));
        const float x = f16_to_f32(gb_.read_lane(st.ch, st.col, st.lane));
        mac_.write(st.ch, st.bank, mac_.read(st.ch, st.bank) + w * x);
        break;
    }
    case StepKind::RdMacBank:
        gpr_.write_lane(st.gpr, st.bank, f32_to_f16(mac_.read(st.ch, st.bank)));
        break;
    case StepKind::WrSbkLane:
        dram_.write_lane(st.ch, st.bank, st.row, st.col, st.lane, gpr_.read_lane(st.gpr, st.lane));
        break;
    case StepKind::RdSbkLane:
        gpr_.write_lane(st.gpr, st.lane, dram_.read_lane(st.ch, st.bank, st.row, st.col, st.lane));
        break;
    case StepKind::Nop:
        break;
    }
}

bool Device::exec_wr_gb_fast(const Command &cmd, int channel_id) {
    const int col = (cmd.col >= 0) ? cmd.col : 0;
    const int gpr = (cmd.gpr0 >= 0) ? static_cast<int>(cmd.gpr0) : 0;
    const Burst *src = gpr_.burst_ptr(gpr + col);
    if (!src) return false;
    auto apply_ch = [&](int ch) {
        Burst *dst = gb_.burst_ptr(ch, col);
        if (dst) *dst = *src;
    };
    if (channel_id >= 0) {
        apply_ch(channel_id);
        return true;
    }
    for (int ch = 0; ch < hw_.n_channels; ch++) {
        if (cmd.chmask < 0 || (cmd.chmask & (1LL << ch))) apply_ch(ch);
    }
    return true;
}

bool Device::exec_wr_bias_fast(const Command &cmd, int channel_id) {
    const int gpr = (cmd.gpr0 >= 0) ? static_cast<int>(cmd.gpr0) : 0;
    const Burst *bias = gpr_.burst_ptr(gpr);
    if (!bias) return false;
    bool all_zero = true;
    for (int i = 0; i < hw_.n_banks; i++) {
        if ((*bias)[static_cast<size_t>(i)] != 0) {
            all_zero = false;
            break;
        }
    }
    auto apply_ch = [&](int ch) {
        float *acc = mac_.bank_accs(ch);
        if (!acc) return;
        if (all_zero) {
            std::fill(acc, acc + hw_.n_banks, 0.0f);
            return;
        }
        for (int bank = 0; bank < hw_.n_banks; bank++) {
            acc[bank] = f16_to_f32((*bias)[static_cast<size_t>(bank)]);
        }
    };
    if (channel_id >= 0) {
        apply_ch(channel_id);
        return true;
    }
    for (int ch = 0; ch < hw_.n_channels; ch++) {
        if (cmd.chmask < 0 || (cmd.chmask & (1LL << ch))) apply_ch(ch);
    }
    return true;
}

void Device::mac_abk_one_col_into(int ch, int row, int col, const DenseBurstRegion *region,
                                  float *acc) const {
    if (!acc) return;
    const Burst *xb = gb_.burst_ptr(ch, col);
    alignas(32) float xf[kLanes];
    if (xb) burst_f16_to_f32(*xb, xf);
    else
        for (int i = 0; i < kLanes; i++) xf[i] = 0.0f;
    mac_abk_one_col_xf32(ch, row, col, region, acc, xf);
}

void Device::mac_abk_one_col_xf32(int ch, int row, int col, const DenseBurstRegion *region,
                                  float *acc, const float *xf) const {
    if (!acc || !xf) return;
    if (region && region->ch == ch && row >= region->row0 && row < region->row0 + region->n_rows &&
        col >= region->col0 && col < region->col0 + region->n_cols) {
        const Burst *base = region->burst(0, row, col);
        if (base) {
            mac_banks16(base, /*bank_stride=*/1, xf, acc);
            return;
        }
    }
    for (int bank = 0; bank < hw_.n_banks; bank++) {
        const Burst *wb = dram_.burst_ptr(ch, bank, row, col);
        Burst empty{};
        acc[bank] += burst_dot_f16_xf32(wb ? *wb : empty, xf);
    }
}

void Device::mac_abk_one_col(int ch, int row, int col, const DenseBurstRegion *region) {
    mac_abk_one_col_into(ch, row, col, region, mac_.bank_accs(ch));
}

void Device::gb_cols_to_f32(int ch, int n_cols, float *out) const {
    if (!out || n_cols <= 0) return;
    for (int col = 0; col < n_cols; col++) {
        float *dst = out + static_cast<size_t>(col) * static_cast<size_t>(kLanes);
        const Burst *xb = gb_.burst_ptr(ch, col);
        if (xb) burst_f16_to_f32(*xb, dst);
        else
            for (int i = 0; i < kLanes; i++) dst[i] = 0.0f;
    }
}

void Device::mac_wave_into(int ch, int row, int n_cols, float *acc_out) const {
    mac_wave_into(ch, row, n_cols, acc_out, nullptr);
}

void Device::mac_wave_into(int ch, int row, int n_cols, float *acc_out, const float *x_f32) const {
    if (!acc_out || n_cols < 0) return;
    for (int b = 0; b < hw_.n_banks; b++) acc_out[b] = 0.0f;
    const DenseBurstRegion *region = dram_.region_for_row(ch, row);
    if (x_f32) {
        for (int col = 0; col < n_cols; col++) {
            mac_abk_one_col_xf32(ch, row, col, region, acc_out,
                                 x_f32 + static_cast<size_t>(col) * static_cast<size_t>(kLanes));
        }
        return;
    }
    for (int col = 0; col < n_cols; col++) {
        mac_abk_one_col_into(ch, row, col, region, acc_out);
    }
}

bool Device::exec_mac_abk_fast(const Command &cmd, int channel_id) {
    const int col = (cmd.col >= 0) ? cmd.col : 0;
    const int row = (cmd.row >= 0) ? cmd.row : 0;
    auto apply_ch = [&](int ch) {
        mac_abk_one_col(ch, row, col, dram_.region_for_row(ch, row));
    };
    if (channel_id >= 0) {
        apply_ch(channel_id);
        return true;
    }
    for (int ch = 0; ch < hw_.n_channels; ch++) {
        if (cmd.chmask < 0 || (cmd.chmask & (1LL << ch))) apply_ch(ch);
    }
    return true;
}

bool Device::exec_rd_mac_fast(const Command &cmd, int channel_id) {
    const int gpr = (cmd.gpr0 >= 0) ? static_cast<int>(cmd.gpr0) : 0;
    Burst *dst = gpr_.burst_ptr(gpr);
    if (!dst) return false;
    int ch = channel_id;
    if (ch < 0) {
        ch = 0;
        for (int i = 0; i < hw_.n_channels; i++) {
            if (cmd.chmask < 0 || (cmd.chmask & (1LL << i))) {
                ch = i;
                break;
            }
        }
    }
    const float *acc = mac_.bank_accs(ch);
    if (!acc) return false;
    for (int bank = 0; bank < hw_.n_banks; bank++) {
        (*dst)[static_cast<size_t>(bank)] = f32_to_f16(acc[bank]);
    }
    return true;
}

std::string Device::execute_micro(const Command &cmd, int channel_id) {
    const std::string err = reg_.validate(cmd);
    if (!err.empty()) return err;
    if (cmd.opcode == Opcode::EOC || cmd.opcode == Opcode::SYNC) return {};

    if (fast_mac_enabled()) {
        switch (cmd.opcode) {
        case Opcode::WR_GB:
            if (exec_wr_gb_fast(cmd, channel_id)) return {};
            break;
        case Opcode::WR_BIAS:
            if (exec_wr_bias_fast(cmd, channel_id)) return {};
            break;
        case Opcode::MAC_ABK:
            if (exec_mac_abk_fast(cmd, channel_id)) return {};
            break;
        case Opcode::RD_MAC:
            if (exec_rd_mac_fast(cmd, channel_id)) return {};
            break;
        default:
            break;
        }
    }

    for (const auto &st : par_.expand(cmd, hw_, channel_id)) {
        apply_step(st);
    }
    return {};
}

std::string Device::execute(const Command &cmd) {
    const CommandDef *def = reg_.find(cmd.opcode);
    if (!def) return "unknown opcode";

    /* MAC_ABK opsize unroll: resolve dense region once; still one col step each. */
    if (fast_mac_enabled() && cmd.opcode == Opcode::MAC_ABK && cmd.col < 0) {
        const std::string err = reg_.validate(cmd);
        if (!err.empty()) return err;
        const int n = (cmd.opsize > 0) ? cmd.opsize : 1;
        const int row = (cmd.row >= 0) ? cmd.row : 0;
        for (int ch = 0; ch < hw_.n_channels; ch++) {
            if (cmd.chmask >= 0 && !(cmd.chmask & (1LL << ch))) continue;
            const DenseBurstRegion *region = dram_.region_for_row(ch, row);
            for (int col = 0; col < n; col++) {
                mac_abk_one_col(ch, row, col, region);
            }
        }
        return {};
    }

    if (def->unrolls_cols && cmd.col < 0) {
        const int n = (cmd.opsize > 0) ? cmd.opsize : 1;
        Command m = cmd;
        m.opsize = 1;
        for (int i = 0; i < n; i++) {
            m.col = i;
            const std::string e = execute_micro(m, -1);
            if (!e.empty()) return e;
        }
        return {};
    }
    return execute_micro(cmd, -1);
}

void Device::load_weights_f32(const float *W, int n_out, int n_in, const AllocationPolicy &alloc,
                              int row_base) {
    const int n_cols = (n_in + hw_.lanes - 1) / hw_.lanes;
    const int n_rows = (n_out + hw_.n_banks - 1) / hw_.n_banks;
    const auto chs = alloc.pim_channels(hw_);
    for (int c : chs) {
        dram_.ensure_dense(c, hw_.n_banks, row_base, n_rows, 0, n_cols > 0 ? n_cols : 1);
    }

    for (int o = 0; o < n_out; o++) {
        const SliceLoc loc0 = alloc.map_weight(hw_, n_out, n_in, o, 0);
        const int bank = loc0.burst.bank;
        const int row = loc0.burst.row + row_base;
        const int ch = loc0.burst.ch;
        for (int col = 0; col < n_cols; col++) {
            Burst b{};
            for (int lane = 0; lane < hw_.lanes; lane++) {
                const int i = col * hw_.lanes + lane;
                if (i < n_in) {
                    b[static_cast<size_t>(lane)] =
                        f32_to_f16(W[static_cast<size_t>(o) * static_cast<size_t>(n_in) +
                                     static_cast<size_t>(i)]);
                }
            }
            dram_.write(ch, bank, row, col, b);
        }
    }
}

void Device::load_activation_f32(const float *x, int n_in, int gpr0) {
    const int n_cols = (n_in + hw_.lanes - 1) / hw_.lanes;
    for (int col = 0; col < n_cols; col++) {
        Burst b{};
        for (int lane = 0; lane < hw_.lanes; lane++) {
            const int i = col * hw_.lanes + lane;
            if (i < n_in) b[static_cast<size_t>(lane)] = f32_to_f16(x[i]);
        }
        gpr_.write(gpr0 + col, b);
    }
}

void Device::read_mac_f32(float *y, int n_out, int gpr0) const {
    for (int o = 0; o < n_out; o++) {
        const int bank = o % hw_.n_banks;
        y[o] = f16_to_f32(gpr_.read_lane(gpr0, bank));
    }
}

void Device::read_mac_banks_f32(int ch, float *y, int n) const {
    const float *acc = mac_.bank_accs(ch);
    if (!acc || n <= 0) return;
    const int m = n < hw_.n_banks ? n : hw_.n_banks;
    /* Match RD_MAC: pack through f16 then back (without GPR traffic). */
    for (int i = 0; i < m; i++) y[i] = f16_to_f32(f32_to_f16(acc[i]));
}

} // namespace pim_func
