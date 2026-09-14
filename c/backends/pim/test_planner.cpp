#include "planner.h"
#include "pim_func/isa.h"

#include <cstdio>

using pim_func::Command;
using pim_func::HardwareInfo;
using pim_func::Opcode;
using pim_func::opcode_name;

static void print_cmd(size_t i, const Command &c) {
    std::printf("  [%zu] %-8s", i, opcode_name(c.opcode));
    if (c.opsize >= 0) std::printf(" opsize=%d", c.opsize);
    if (c.gpr0 >= 0) std::printf(" gpr0=%lld", (long long)c.gpr0);
    if (c.gpr1 >= 0) std::printf(" gpr1=%lld", (long long)c.gpr1);
    if (c.chmask >= 0) std::printf(" chmask=0x%llx", (unsigned long long)c.chmask);
    if (c.bank >= 0) std::printf(" bank=%d", c.bank);
    if (c.row >= 0) std::printf(" row=%d", c.row);
    if (c.col >= 0) std::printf(" col=%d", c.col);
    else if (c.col < 0 && (c.opcode == Opcode::WR_GB || c.opcode == Opcode::MAC_ABK ||
                           c.opcode == Opcode::MAC_SBK))
        std::printf(" col=<unroll>");
    if (c.row_start >= 0) std::printf(" row_start=%d", c.row_start);
    if (c.row_end >= 0) std::printf(" row_end=%d", c.row_end);
    std::printf("\n");
}

int main() {
    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    llm_pim::Planner pl;
    const llm_pim::Plan p = pl.plan_linear(16, 64, hw);
    if (p.cmds.empty()) {
        std::fprintf(stderr, "FAIL: empty plan\n");
        return 1;
    }
    if (p.cmds.front().opcode != Opcode::WR_GB || p.cmds.back().opcode != Opcode::EOC) {
        std::fprintf(stderr, "FAIL: expected WR_GB ... EOC\n");
        return 1;
    }
    bool has_mac = false, has_rd = false, has_bias = false;
    for (const auto &c : p.cmds) {
        if (c.opcode == Opcode::MAC_ABK) has_mac = true;
        if (c.opcode == Opcode::RD_MAC) has_rd = true;
        if (c.opcode == Opcode::WR_BIAS) has_bias = true;
        const std::string err = pl.registry().validate(c);
        if (!err.empty()) {
            std::fprintf(stderr, "FAIL: %s\n", err.c_str());
            return 1;
        }
    }
    if (!has_mac || !has_rd || !has_bias) {
        std::fprintf(stderr, "FAIL: missing MAC/RD/BIAS\n");
        return 1;
    }
    if (p.gb_plan.size() != 4) {
        std::fprintf(stderr, "FAIL: expected 4 GB fills, got %zu\n", p.gb_plan.size());
        return 1;
    }
    if (p.temp_cache.empty()) {
        std::fprintf(stderr, "FAIL: expected temp cache plan\n");
        return 1;
    }

    std::printf("plan_linear n_out=16 n_in=64 cmds=%zu gb_fills=%zu\n", p.cmds.size(),
                p.gb_plan.size());
    for (size_t i = 0; i < p.cmds.size(); i++) print_cmd(i, p.cmds[i]);
    std::printf("test_planner ok cmds=%zu gb_fills=%zu\n", p.cmds.size(), p.gb_plan.size());

    HardwareInfo hw4 = hw;
    hw4.n_channels = 8;
    hw4.pim_chmask = 0x0F;
    const llm_pim::Plan p4 = pl.plan_linear(64, 64, hw4, 3);
    int n_rd = 0, n_mac = 0;
    for (const auto &c : p4.cmds) {
        if (c.opcode == Opcode::RD_MAC) n_rd++;
        if (c.opcode == Opcode::MAC_ABK) {
            n_mac++;
            if (c.row != 3) {
                std::fprintf(stderr, "FAIL: multi-ch MAC row=%d want 3\n", c.row);
                return 1;
            }
        }
    }
    if (n_mac != 1 || n_rd != 4) {
        std::fprintf(stderr, "FAIL: 4ch 64-out expected 1 MAC 4 RD, got mac=%d rd=%d\n", n_mac,
                     n_rd);
        return 1;
    }
    std::printf("test_planner multi-ch ok\n");
    return 0;
}
