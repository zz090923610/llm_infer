#pragma once

#include "pim_func/c_api.h"
#include "pim_func/device.h"
#include "pim_func/isa.h"
#include "pim_func/layout.h"

#include <vector>

inline std::vector<pim_func::Command> centgemv_cmds(int n_out, int n_in,
                                                    const pim_func::HardwareInfo &hw,
                                                    int row_base = 0) {
    using pim_func::Command;
    using pim_func::Opcode;
    std::vector<Command> cmds;
    const int n_cols = (n_in + hw.lanes - 1) / hw.lanes;
    pim_func::CentGemvAllocation alloc;
    const auto chs = alloc.pim_channels(hw);
    uint64_t chmask = 0;
    for (int c : chs) chmask |= (1ULL << c);

    Command wr_gb;
    wr_gb.opcode = Opcode::WR_GB;
    wr_gb.opsize = n_cols;
    wr_gb.gpr0 = 0;
    wr_gb.chmask = static_cast<int64_t>(chmask);
    wr_gb.col = -1;
    cmds.push_back(wr_gb);

    const int n_row_groups = (n_out + hw.n_banks - 1) / hw.n_banks;
    const int n_ch = static_cast<int>(chs.size());
    const int n_waves = (n_row_groups + n_ch - 1) / n_ch;
    const int bias_gpr = n_cols;
    const int y_gpr0 = n_cols + 1;
    for (int wave = 0; wave < n_waves; wave++) {
        Command wr_bias;
        wr_bias.opcode = Opcode::WR_BIAS;
        wr_bias.gpr0 = bias_gpr;
        wr_bias.chmask = static_cast<int64_t>(chmask);
        cmds.push_back(wr_bias);

        Command mac;
        mac.opcode = Opcode::MAC_ABK;
        mac.opsize = n_cols;
        mac.chmask = static_cast<int64_t>(chmask);
        mac.row = row_base + wave;
        mac.col = -1;
        cmds.push_back(mac);

        for (int i = 0; i < n_ch; i++) {
            Command rd;
            rd.opcode = Opcode::RD_MAC;
            rd.gpr0 = y_gpr0 + i;
            rd.chmask = 1LL << chs[static_cast<size_t>(i)];
            cmds.push_back(rd);
        }
    }
    Command eoc;
    eoc.opcode = Opcode::EOC;
    cmds.push_back(eoc);
    return cmds;
}

inline PimCommand to_pim_cmd(const pim_func::Command &c) {
    PimCommand o{};
    o.opcode = static_cast<uint32_t>(c.opcode);
    o.opsize = c.opsize;
    o.gpr0 = c.gpr0;
    o.gpr1 = c.gpr1;
    o.chmask = c.chmask;
    o.row = c.row;
    o.col = c.col;
    return o;
}
