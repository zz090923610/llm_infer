#pragma once

#include "pim_func/types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace pim_func {

enum class Opcode : uint32_t {
    MIN = 0,
    WR_SBK = 1,
    WR_GB = 2,
    WR_BIAS = 3,
    WR_AFLUT = 4,
    RD_MAC = 5,
    RD_AF = 6,
    RD_SBK = 7,
    COPY_BKGB = 8,
    COPY_GBBK = 9,
    MAC_SBK = 10,
    MAC_ABK = 11,
    AF = 12,
    EWMUL = 13,
    EWADD = 14,
    WR_ABK = 15,
    EOC = 16,
    SYNC = 17,
    MAX = 18
};

enum class Compat { Supported, Experimental, Deprecated, Removed };

enum Field : uint32_t {
    F_OPSIZE = 1u << 0,
    F_GPR0 = 1u << 1,
    F_GPR1 = 1u << 2,
    F_CHMASK = 1u << 3,
    F_BANK = 1u << 4,
    F_ROW = 1u << 5,
    F_COL = 1u << 6,
    F_ROW_START = 1u << 7,
    F_ROW_END = 1u << 8
};

enum Affected : uint32_t {
    A_DRAM = 1u << 0,
    A_GB = 1u << 1,
    A_GPR = 1u << 2,
    A_MAC = 1u << 3,
    A_TEMP = 1u << 4
};

struct Command {
    Opcode opcode = Opcode::MIN;
    int32_t opsize = -1;
    int64_t gpr0 = -1;
    int64_t gpr1 = -1;
    int64_t chmask = -1;
    int16_t bank = -1;
    int32_t row = -1;
    int32_t col = -1;
    int32_t row_start = -1; /* future sub-row compute */
    int32_t row_end = -1;
};

struct CommandDef {
    Opcode opcode = Opcode::MIN;
    const char *name = "";
    uint32_t fields = 0;
    uint32_t affected = 0;
    Compat compat = Compat::Removed;
    bool unrolls_cols = false;
};

class CommandRegistry {
public:
    CommandRegistry();
    const CommandDef *find(Opcode op) const;
    const CommandDef *find(const std::string &name) const;
    std::string validate(const Command &cmd) const;
    const std::vector<CommandDef> &all() const { return defs_; }

private:
    std::vector<CommandDef> defs_;
};

const CommandRegistry &default_registry();
const char *opcode_name(Opcode op);

} // namespace pim_func
