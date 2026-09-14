#include "pim_func/fp16.h"
#include "pim_func/device.h"

#include <cmath>
#include <cstdio>

using namespace pim_func;

int main() {
    int fails = 0;
    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    Device dev(hw);
    Burst src{};
    for (int i = 0; i < kLanes; i++) src[static_cast<size_t>(i)] = f32_to_f16(static_cast<float>(i));
    dev.gpr().write(0, src);

    Command wr;
    wr.opcode = Opcode::WR_GB;
    wr.opsize = 1;
    wr.gpr0 = 0;
    wr.chmask = 1;
    wr.col = 0;
    const std::string e = dev.execute(wr);
    if (!e.empty()) {
        std::fprintf(stderr, "FAIL: %s\n", e.c_str());
        return 1;
    }
    for (int i = 0; i < kLanes; i++) {
        const float got = f16_to_f32(dev.gb().read_lane(0, 0, i));
        if (std::fabs(got - static_cast<float>(i)) > 0.01f) {
            std::fprintf(stderr, "FAIL: GB[%d]=%g\n", i, got);
            fails++;
        }
    }
    if (fails) return 1;
    std::printf("test_gb ok\n");
    return 0;
}
