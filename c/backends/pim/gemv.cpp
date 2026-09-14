#include "planner.h"
#include "pim_func/c_api.h"
#include "pim_func/isa.h"
#include "pim_func/pool.h"
#include "pim_func/types.h"

#include <cstring>
#include <vector>

extern "C" int pim_backend_gemv(const void *key, const float *x, float *y, int n_out, int n_in,
                                int acc);

namespace {

pim_func::HardwareInfo hw_from_c(const PimHw &h) {
    pim_func::HardwareInfo hw;
    hw.n_channels = h.n_channels;
    hw.n_banks = h.n_banks;
    hw.n_rows = h.n_rows;
    hw.lanes = h.lanes;
    hw.gb_cols = h.gb_cols;
    hw.gpr_count = h.gpr_count;
    hw.pim_chmask = h.pim_chmask;
    return hw;
}

PimCommand to_c(const pim_func::Command &c) {
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

struct WaveJob {
    const float *x_f32 = nullptr;
    float *y = nullptr;
    int n_out = 0;
    int n_banks = 0;
    int n_cols = 0;
    int row_base = 0;
    int n_waves = 0;
    int acc = 0;
};

void wave_job(int tid, int n_threads, void *ctx) {
    auto *j = static_cast<WaveJob *>(ctx);
    alignas(32) float accv[pim_func::kBanks];
    for (int wave = tid; wave < j->n_waves; wave += n_threads) {
        pim_mac_wave_into(0, j->row_base + wave, j->n_cols, accv, j->x_f32);
        const int o0 = wave * j->n_banks;
        const int n = j->n_out - o0 < j->n_banks ? j->n_out - o0 : j->n_banks;
        if (j->acc) {
            for (int i = 0; i < n; i++) j->y[o0 + i] += accv[i];
        } else {
            std::memcpy(j->y + o0, accv, static_cast<size_t>(n) * sizeof(float));
        }
    }
}

int fire_cmds(const llm_pim::Plan &plan, float *y, int n_out, int acc, int n_cols,
              const std::vector<int> &chs) {
    int wave = -1;
    const int last_ch = chs.back();
    for (const auto &c : plan.cmds) {
        PimCommand pc = to_c(c);
        if (pim_execute(&pc) != 0) return -1;
        if (c.opcode == pim_func::Opcode::WR_BIAS) wave++;
        if (c.opcode == pim_func::Opcode::RD_MAC && c.chmask == (1LL << last_ch) && wave >= 0) {
            if (pim_unpack_wave(y, n_out, acc, n_cols, wave) != 0) return -1;
        }
    }
    return 0;
}

} // namespace

extern "C" int pim_backend_gemv(const void *key, const float *x, float *y, int n_out, int n_in,
                                int acc) {
    if (!key || !x || !y || n_out <= 0 || n_in <= 0) return -1;
    if (!pim_interned_f32(key, nullptr)) return -1;
    if (pim_prepare(key, n_out, n_in) != 0) return -1;
    const int row_base = pim_row_base(key);
    if (row_base < 0) return -1;

    PimHw ph{};
    pim_hw(&ph);
    const pim_func::HardwareInfo hw = hw_from_c(ph);
    const int n_cols = (n_in + hw.lanes - 1) / hw.lanes;
    if (n_cols > hw.gb_cols || n_cols + 8 >= hw.gpr_count) return -1;

    if (pim_load_activation(x, n_in) != 0) return -1;
    if (pim_zero_gpr(n_cols) != 0) return -1;

    pim_func::CentGemvAllocation alloc;
    const auto chs = alloc.pim_channels(hw);
    const int n_row_groups = (n_out + hw.n_banks - 1) / hw.n_banks;
    const int n_waves = (n_row_groups + static_cast<int>(chs.size()) - 1) /
                        static_cast<int>(chs.size());

    if (pim_issue_mode() == PIM_ISSUE_SHM) {
        llm_pim::Planner planner;
        const llm_pim::Plan plan = planner.plan_linear(n_out, n_in, hw, row_base);
        std::vector<PimCommand> cmds;
        cmds.reserve(plan.cmds.size());
        for (const auto &c : plan.cmds) cmds.push_back(to_c(c));
        return pim_hybrid_submit_cmds(x, y, n_out, n_in, row_base, n_cols, acc, cmds.data(),
                                      static_cast<int>(cmds.size()));
    }

    int nth = pim_n_threads();
    if (nth > PIM_MAX_GEMV_THREADS) nth = PIM_MAX_GEMV_THREADS;
    const int use_par =
        (pim_issue_mode() == PIM_ISSUE_HOST && nth > 1 && n_waves > 1 && chs.size() == 1) ? 1 : 0;

    if (use_par) {
        PimCommand wr_gb{};
        wr_gb.opcode = static_cast<uint32_t>(pim_func::Opcode::WR_GB);
        wr_gb.opsize = n_cols;
        wr_gb.gpr0 = 0;
        wr_gb.chmask = static_cast<int64_t>(hw.pim_chmask);
        wr_gb.col = -1;
        if (pim_execute(&wr_gb) != 0) return -1;
        std::vector<float> xf32(static_cast<size_t>(n_cols) * static_cast<size_t>(pim_func::kLanes));
        if (pim_gb_cols_to_f32(xf32.data(), n_cols) != 0) return -1;
        WaveJob job;
        job.x_f32 = xf32.data();
        job.y = y;
        job.n_out = n_out;
        job.n_banks = hw.n_banks;
        job.n_cols = n_cols;
        job.row_base = row_base;
        job.n_waves = n_waves;
        job.acc = acc;
        int nrun = nth;
        if (nrun > n_waves) nrun = n_waves;
        pim_func::pool_run(wave_job, &job, nrun);
        return 0;
    }

    llm_pim::Planner planner;
    const llm_pim::Plan plan = planner.plan_linear(n_out, n_in, hw, row_base);
    return fire_cmds(plan, y, n_out, acc, n_cols, chs);
}
