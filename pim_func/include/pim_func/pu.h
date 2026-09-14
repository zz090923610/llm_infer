#pragma once

#include "pim_func/isa.h"
#include "pim_func/types.h"

#include <vector>

namespace pim_func {

enum class StepKind {
    Nop,
    WrGbLane,
    WrBiasBank,
    MacLane,
    RdMacBank,
    WrSbkLane,
    RdSbkLane
};

struct MicroStep {
    StepKind kind = StepKind::Nop;
    int ch = 0;
    int bank = 0;
    int row = 0;
    int col = 0;
    int lane = 0;
    int gpr = 0;
};

class ParallelismPolicy {
public:
    virtual ~ParallelismPolicy() = default;
    virtual std::vector<MicroStep> expand(const Command &cmd, const HardwareInfo &hw,
                                          int channel_id) const = 0;
};

/* Deterministic full-ABK: all 16 banks and 16 lanes on the target channel. */
class FullAbkParallelism : public ParallelismPolicy {
public:
    std::vector<MicroStep> expand(const Command &cmd, const HardwareInfo &hw,
                                  int channel_id) const override;
};

} // namespace pim_func
