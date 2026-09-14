#include "pim_func/layout.h"

namespace pim_func {

std::vector<int> AllocationPolicy::pim_channels(const HardwareInfo &hw) const {
    std::vector<int> ch;
    for (int i = 0; i < hw.n_channels; i++) {
        if (hw.pim_chmask & (1ULL << i)) ch.push_back(i);
    }
    if (ch.empty()) ch.push_back(0);
    return ch;
}

SliceLoc CentGemvAllocation::map_weight(const HardwareInfo &hw, int n_out, int n_in, int out,
                                        int in) const {
    (void)n_out;
    (void)n_in;
    SliceLoc loc;
    loc.lane = in % hw.lanes;
    loc.burst.col = in / hw.lanes;
    loc.burst.bank = out % hw.n_banks;
    const auto chs = pim_channels(hw);
    const int group = out / hw.n_banks;
    loc.burst.ch = chs[static_cast<size_t>(group) % chs.size()];
    loc.burst.row = group / static_cast<int>(chs.size());
    return loc;
}

BurstAddr CentGemvAllocation::map_activation_gb(const HardwareInfo &hw, int n_in, int in) const {
    (void)n_in;
    BurstAddr a;
    a.col = in / hw.lanes;
    a.ch = pim_channels(hw).front();
    return a;
}

} // namespace pim_func
