#pragma once

#include "pim_func/isa.h"
#include "pim_func/layout.h"
#include "pim_func/types.h"

#include <string>
#include <vector>

namespace llm_pim {

struct GbFill {
    int ch = 0;
    int col = 0;
    int gpr = 0;
};

struct TempCacheSlot {
    std::string tensor;
    pim_func::BurstAddr where;
    std::string reason;
};

struct Plan {
    pim_func::TensorDesc weights;
    pim_func::TensorDesc activations;
    pim_func::TensorDesc result;
    pim_func::Placement weight_place;
    std::vector<GbFill> gb_plan;
    std::vector<TempCacheSlot> temp_cache;
    std::vector<pim_func::Command> cmds;
};

class Planner {
public:
    Planner();
    Plan plan_linear(int n_out, int n_in, const pim_func::HardwareInfo &hw,
                     int row_base = 0) const;

    const pim_func::AllocationPolicy &allocation() const { return *alloc_; }
    const pim_func::CommandRegistry &registry() const { return *reg_; }

private:
    pim_func::CentGemvAllocation default_alloc_;
    const pim_func::CommandRegistry *reg_;
    pim_func::AllocationPolicy *alloc_;
};

} // namespace llm_pim
