#pragma once

#include "pim_func/types.h"

#include <vector>

namespace pim_func {

struct SliceLoc {
    BurstAddr burst;
    int lane = 0;
};

class AllocationPolicy {
public:
    virtual ~AllocationPolicy() = default;
    virtual SliceLoc map_weight(const HardwareInfo &hw, int n_out, int n_in, int out,
                                int in) const = 0;
    virtual BurstAddr map_activation_gb(const HardwareInfo &hw, int n_in, int in) const = 0;
    virtual std::vector<int> pim_channels(const HardwareInfo &hw) const;
};

/* CENT-style GEMV: one output per bank, 16-element input chunks along columns. */
class CentGemvAllocation : public AllocationPolicy {
public:
    SliceLoc map_weight(const HardwareInfo &hw, int n_out, int n_in, int out, int in) const override;
    BurstAddr map_activation_gb(const HardwareInfo &hw, int n_in, int in) const override;
};

struct Placement {
    TensorDesc tensor;
    const AllocationPolicy *policy = nullptr;
};

} // namespace pim_func
