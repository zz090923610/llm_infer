#include "pim_func/hook.h"

#include <cstdio>

using namespace pim_func;

int main() {
    set_func_mode(FuncMode::Token);
    Device &d = global_device();
    d.reset();
    Burst b{};
    b[0] = 0x3c00; /* 1.0 f16 */
    if (!gpr_store(0, 0, b.data(), 32)) {
        std::fprintf(stderr, "FAIL: gpr_store\n");
        return 1;
    }
    Burst rdb{};
    if (!gpr_load(0, 0, rdb.data(), 32) || rdb[0] != 0x3c00) {
        std::fprintf(stderr, "FAIL: gpr_load\n");
        return 1;
    }
    d.gpr().write(0, b);
    RetiredAim wr;
    wr.opcode = static_cast<int>(Opcode::WR_GB);
    wr.channel_id = 0;
    wr.col = 0;
    wr.gpr0 = 0;
    wr.chmask = 1;
    on_retired_aim(wr);
    if (d.gb().read_lane(0, 0, 0) != 0x3c00) {
        std::fprintf(stderr, "FAIL: hook WR_GB\n");
        return 1;
    }
    std::printf("test_hook ok\n");
    return 0;
}
