#include "pim_func/device.h"
#include "pim_func/pu.h"

#include <cstdio>

using namespace pim_func;

int main() {
    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    Command mac;
    mac.opcode = Opcode::MAC_ABK;
    mac.col = 0;
    mac.row = 0;
    mac.chmask = 1;
    FullAbkParallelism par;
    const auto steps = par.expand(mac, hw, 0);
    if (static_cast<int>(steps.size()) != hw.n_banks * hw.lanes) {
        std::fprintf(stderr, "FAIL: expected %d MAC lane steps, got %zu\n", hw.n_banks * hw.lanes,
                     steps.size());
        return 1;
    }
    int banks = 0;
    for (const auto &s : steps) {
        if (s.kind != StepKind::MacLane || s.ch != 0) {
            std::fprintf(stderr, "FAIL: bad step\n");
            return 1;
        }
        if (s.bank == 0 && s.lane == 0) banks++;
    }
    std::printf("test_parallelism ok steps=%zu\n", steps.size());
    return 0;
}
