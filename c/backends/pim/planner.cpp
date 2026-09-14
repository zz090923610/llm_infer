#include "planner.h"

namespace llm_pim {

Planner::Planner() : reg_(&pim_func::default_registry()), alloc_(&default_alloc_) {}

Plan Planner::plan_linear(int n_out, int n_in, const pim_func::HardwareInfo &hw,
                          int row_base) const {
    using pim_func::Command;
    using pim_func::DType;
    using pim_func::Opcode;
    using pim_func::TensorDesc;

    Plan p;
    p.weights = TensorDesc{"W", DType::FP16, {n_out, n_in}};
    p.activations = TensorDesc{"x", DType::FP16, {n_in}};
    p.result = TensorDesc{"y", DType::FP16, {n_out}};
    p.weight_place.tensor = p.weights;
    p.weight_place.policy = alloc_;

    const int n_cols = (n_in + hw.lanes - 1) / hw.lanes;
    const auto chs = alloc_->pim_channels(hw);
    uint64_t chmask = 0;
    for (int c : chs) chmask |= (1ULL << c);

    for (int col = 0; col < n_cols; col++) {
        for (int c : chs) {
            GbFill f;
            f.ch = c;
            f.col = col;
            f.gpr = col;
            p.gb_plan.push_back(f);
        }
    }

    TempCacheSlot xcache;
    xcache.tensor = "x";
    xcache.where = pim_func::BurstAddr{chs.front(), 0, 0, 0};
    xcache.reason = "keep activation in GPR until GB fill; MAC acc is not a safe x cache";
    p.temp_cache.push_back(xcache);

    Command wr_gb;
    wr_gb.opcode = Opcode::WR_GB;
    wr_gb.opsize = n_cols;
    wr_gb.gpr0 = 0;
    wr_gb.chmask = static_cast<int64_t>(chmask);
    wr_gb.col = -1;
    p.cmds.push_back(wr_gb);

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
        p.cmds.push_back(wr_bias);

        Command mac;
        mac.opcode = Opcode::MAC_ABK;
        mac.opsize = n_cols;
        mac.chmask = static_cast<int64_t>(chmask);
        mac.row = row_base + wave;
        mac.col = -1;
        p.cmds.push_back(mac);

        for (int i = 0; i < n_ch; i++) {
            Command rd;
            rd.opcode = Opcode::RD_MAC;
            rd.gpr0 = y_gpr0 + i;
            rd.chmask = 1LL << chs[static_cast<size_t>(i)];
            p.cmds.push_back(rd);
        }
    }

    Command eoc;
    eoc.opcode = Opcode::EOC;
    p.cmds.push_back(eoc);
    return p;
}

} // namespace llm_pim
