#include "pim_func/pu.h"

namespace pim_func {

static std::vector<int> channels_of(const Command &cmd, const HardwareInfo &hw, int channel_id) {
    std::vector<int> chs;
    if (channel_id >= 0) {
        chs.push_back(channel_id);
        return chs;
    }
    for (int i = 0; i < hw.n_channels; i++) {
        if (cmd.chmask < 0 || (cmd.chmask & (1LL << i))) chs.push_back(i);
    }
    if (chs.empty()) chs.push_back(0);
    return chs;
}

std::vector<MicroStep> FullAbkParallelism::expand(const Command &cmd, const HardwareInfo &hw,
                                                  int channel_id) const {
    std::vector<MicroStep> steps;
    const auto chs = channels_of(cmd, hw, channel_id);
    const int col = (cmd.col >= 0) ? cmd.col : 0;
    const int row = (cmd.row >= 0) ? cmd.row : 0;
    const int gpr = (cmd.gpr0 >= 0) ? static_cast<int>(cmd.gpr0) : 0;
    const int gpr_col = gpr + col;

    auto push = [&](StepKind kind, int ch, int bank, int lane, int g) {
        MicroStep s;
        s.kind = kind;
        s.ch = ch;
        s.bank = bank;
        s.row = row;
        s.col = col;
        s.lane = lane;
        s.gpr = g;
        steps.push_back(s);
    };

    switch (cmd.opcode) {
    case Opcode::WR_GB:
        for (int ch : chs) {
            for (int lane = 0; lane < hw.lanes; lane++) {
                push(StepKind::WrGbLane, ch, 0, lane, gpr_col);
            }
        }
        break;
    case Opcode::WR_BIAS:
        for (int ch : chs) {
            for (int bank = 0; bank < hw.n_banks; bank++) {
                push(StepKind::WrBiasBank, ch, bank, 0, gpr);
            }
        }
        break;
    case Opcode::MAC_ABK:
        for (int ch : chs) {
            for (int bank = 0; bank < hw.n_banks; bank++) {
                for (int lane = 0; lane < hw.lanes; lane++) {
                    push(StepKind::MacLane, ch, bank, lane, gpr);
                }
            }
        }
        break;
    case Opcode::MAC_SBK: {
        const int bank = (cmd.bank >= 0) ? cmd.bank : 0;
        for (int ch : chs) {
            for (int lane = 0; lane < hw.lanes; lane++) {
                push(StepKind::MacLane, ch, bank, lane, gpr);
            }
        }
        break;
    }
    case Opcode::RD_MAC:
        for (int ch : chs) {
            for (int bank = 0; bank < hw.n_banks; bank++) {
                push(StepKind::RdMacBank, ch, bank, 0, gpr);
            }
        }
        break;
    case Opcode::WR_SBK: {
        const int bank = (cmd.bank >= 0) ? cmd.bank : 0;
        for (int ch : chs) {
            for (int lane = 0; lane < hw.lanes; lane++) {
                push(StepKind::WrSbkLane, ch, bank, lane, gpr_col);
            }
        }
        break;
    }
    case Opcode::RD_SBK: {
        const int bank = (cmd.bank >= 0) ? cmd.bank : 0;
        for (int ch : chs) {
            for (int lane = 0; lane < hw.lanes; lane++) {
                push(StepKind::RdSbkLane, ch, bank, lane, gpr_col);
            }
        }
        break;
    }
    case Opcode::EOC:
    case Opcode::SYNC:
        break;
    default:
        break;
    }
    return steps;
}

} // namespace pim_func
