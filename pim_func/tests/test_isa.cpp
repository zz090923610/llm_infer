#include "pim_func/isa.h"

#include <cstdio>
#include <cstdlib>

using namespace pim_func;

int main() {
    int fails = 0;
    const CommandRegistry &r = default_registry();
    if (!r.find(Opcode::MAC_ABK) || r.find(Opcode::MAC_ABK)->compat != Compat::Supported) {
        std::fprintf(stderr, "FAIL: MAC_ABK missing\n");
        fails++;
    }
    if (!r.find("WR_GB")) {
        std::fprintf(stderr, "FAIL: WR_GB by name\n");
        fails++;
    }
    if (r.find(Opcode::SYNC)->compat != Compat::Removed) {
        std::fprintf(stderr, "FAIL: SYNC should be removed\n");
        fails++;
    }
    Command c;
    c.opcode = Opcode::MAC_ABK;
    c.row_start = 4;
    const std::string err = r.validate(c);
    if (err.empty()) {
        std::fprintf(stderr, "FAIL: row_start should be rejected until the schema allows it\n");
        fails++;
    }
    Command eoc;
    eoc.opcode = Opcode::EOC;
    if (!r.validate(eoc).empty()) {
        std::fprintf(stderr, "FAIL: EOC should validate\n");
        fails++;
    }
    if (fails) return 1;
    std::printf("test_isa ok\n");
    return 0;
}
