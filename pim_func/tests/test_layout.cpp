#include "pim_func/fp16.h"
#include "pim_func/layout.h"
#include "pim_func/mem.h"

#include <cmath>
#include <cstdio>

using namespace pim_func;

int main() {
    int fails = 0;
    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    CentGemvAllocation alloc;
    BankedDram dram;

    const int n_out = 16, n_in = 64;
    for (int o = 0; o < n_out; o++) {
        for (int i = 0; i < n_in; i++) {
            const float v = static_cast<float>(o * 100 + i);
            const SliceLoc loc = alloc.map_weight(hw, n_out, n_in, o, i);
            dram.write_lane(loc.burst.ch, loc.burst.bank, loc.burst.row, loc.burst.col, loc.lane,
                            f32_to_f16(v));
        }
    }
    for (int o = 0; o < n_out; o++) {
        for (int i = 0; i < n_in; i++) {
            const SliceLoc loc = alloc.map_weight(hw, n_out, n_in, o, i);
            const float got = f16_to_f32(dram.read_lane(loc.burst.ch, loc.burst.bank, loc.burst.row,
                                                        loc.burst.col, loc.lane));
            const float want = static_cast<float>(o * 100 + i);
            if (std::fabs(got - want) > 0.05f) {
                std::fprintf(stderr, "FAIL: roundtrip o=%d i=%d got=%g want=%g\n", o, i, got, want);
                fails++;
                o = n_out;
                break;
            }
        }
    }
    if (alloc.map_weight(hw, n_out, n_in, 3, 17).burst.bank != 3) {
        std::fprintf(stderr, "FAIL: bank mapping\n");
        fails++;
    }
    if (alloc.map_activation_gb(hw, n_in, 17).col != 1) {
        std::fprintf(stderr, "FAIL: GB col mapping\n");
        fails++;
    }
    if (fails) return 1;
    std::printf("test_layout ok bursts=%zu\n", dram.n_bursts());
    return 0;
}
