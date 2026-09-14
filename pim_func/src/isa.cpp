#include "pim_func/isa.h"

#include <stdexcept>

namespace pim_func {

CommandRegistry::CommandRegistry() {
    auto add = [this](Opcode op, const char *name, uint32_t fields, uint32_t affected, Compat c,
                      bool unroll) {
        CommandDef d;
        d.opcode = op;
        d.name = name;
        d.fields = fields;
        d.affected = affected;
        d.compat = c;
        d.unrolls_cols = unroll;
        defs_.push_back(d);
    };

    const uint32_t gb_fields = F_OPSIZE | F_GPR0 | F_CHMASK | F_COL;
    const uint32_t mac_fields = F_OPSIZE | F_CHMASK | F_ROW | F_COL;
    const uint32_t bias_fields = F_GPR0 | F_CHMASK;
    const uint32_t rdmac_fields = F_GPR0 | F_CHMASK;
    const uint32_t sbk_fields = F_OPSIZE | F_GPR0 | F_CHMASK | F_BANK | F_ROW | F_COL;

    add(Opcode::WR_SBK, "WR_SBK", sbk_fields, A_DRAM | A_GPR, Compat::Supported, true);
    add(Opcode::WR_GB, "WR_GB", gb_fields, A_GB | A_GPR, Compat::Supported, true);
    add(Opcode::WR_BIAS, "WR_BIAS", bias_fields, A_MAC | A_GPR, Compat::Supported, false);
    add(Opcode::WR_AFLUT, "WR_AFLUT", F_GPR0, A_TEMP, Compat::Removed, false);
    add(Opcode::RD_MAC, "RD_MAC", rdmac_fields, A_MAC | A_GPR, Compat::Supported, false);
    add(Opcode::RD_AF, "RD_AF", rdmac_fields, A_GPR, Compat::Experimental, false);
    add(Opcode::RD_SBK, "RD_SBK", sbk_fields, A_DRAM | A_GPR, Compat::Supported, true);
    add(Opcode::COPY_BKGB, "COPY_BKGB", F_OPSIZE | F_CHMASK | F_BANK | F_ROW | F_COL, A_DRAM | A_GB,
        Compat::Experimental, true);
    add(Opcode::COPY_GBBK, "COPY_GBBK", F_OPSIZE | F_CHMASK | F_BANK | F_ROW | F_COL, A_DRAM | A_GB,
        Compat::Experimental, true);
    add(Opcode::MAC_SBK, "MAC_SBK", mac_fields | F_BANK, A_DRAM | A_GB | A_MAC, Compat::Supported,
        true);
    add(Opcode::MAC_ABK, "MAC_ABK", mac_fields, A_DRAM | A_GB | A_MAC, Compat::Supported, true);
    add(Opcode::AF, "AF", F_CHMASK, A_MAC, Compat::Experimental, false);
    add(Opcode::EWMUL, "EWMUL", F_OPSIZE | F_CHMASK | F_ROW, A_DRAM | A_GB, Compat::Experimental,
        true);
    add(Opcode::EWADD, "EWADD", F_OPSIZE | F_GPR0 | F_GPR1, A_GPR, Compat::Experimental, false);
    add(Opcode::WR_ABK, "WR_ABK", F_GPR0 | F_CHMASK | F_ROW, A_DRAM | A_GPR, Compat::Supported,
        false);
    add(Opcode::EOC, "EOC", 0, 0, Compat::Supported, false);
    add(Opcode::SYNC, "SYNC", 0, 0, Compat::Removed, false);
}

const CommandDef *CommandRegistry::find(Opcode op) const {
    for (const auto &d : defs_) {
        if (d.opcode == op) return &d;
    }
    return nullptr;
}

const CommandDef *CommandRegistry::find(const std::string &name) const {
    for (const auto &d : defs_) {
        if (name == d.name) return &d;
    }
    return nullptr;
}

std::string CommandRegistry::validate(const Command &cmd) const {
    const CommandDef *d = find(cmd.opcode);
    if (!d) return "unknown opcode";
    if (d->compat == Compat::Removed) return std::string(d->name) + " is removed";
    if ((d->fields & F_ROW_START) == 0 && cmd.row_start >= 0)
        return std::string(d->name) + " does not accept row_start yet";
    if ((d->fields & F_ROW_END) == 0 && cmd.row_end >= 0)
        return std::string(d->name) + " does not accept row_end yet";
    return {};
}

const CommandRegistry &default_registry() {
    static CommandRegistry r;
    return r;
}

const char *opcode_name(Opcode op) {
    const CommandDef *d = default_registry().find(op);
    return d ? d->name : "UNKNOWN";
}

} // namespace pim_func
