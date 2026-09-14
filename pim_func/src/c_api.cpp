#include "pim_func/c_api.h"
#include "pim_func/device.h"
#include "pim_func/fp16.h"
#include "pim_func/guest_mmio.h"
#include "pim_func/hook.h"
#include "pim_func/hybrid.h"
#include "pim_func/isa.h"
#include "pim_func/pool.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct FreeFn {
    void operator()(float *p) const { std::free(p); }
};

struct Interned {
    const float *f32 = nullptr;
    std::unique_ptr<float, FreeFn> owned;
    int n_elements = 0;
    int n_out = 0;
    int n_in = 0;
    int row_base = -1;
    bool stale = true;
    std::string name;
};

int g_issue = -1;

int issue_from_env() {
#ifdef PIM_DEFAULT_ISSUE_MMIO
    int def = PIM_ISSUE_MMIO;
#else
    int def = PIM_ISSUE_HOST;
#endif
    const char *e = std::getenv("PIM_ISSUE");
    if (!e || !e[0]) return def;
    if (std::strcmp(e, "mmio") == 0 || std::strcmp(e, "gem5") == 0) return PIM_ISSUE_MMIO;
    if (std::strcmp(e, "shm") == 0 || std::strcmp(e, "hybrid") == 0) return PIM_ISSUE_SHM;
    return PIM_ISSUE_HOST;
}

int issue_mode() {
    if (g_issue < 0) g_issue = issue_from_env();
    return g_issue;
}

pim_func::HardwareInfo host_hw() {
    pim_func::HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    hw.n_banks = pim_func::kBanks;
    hw.n_rows = 65536;
    hw.n_cols = 4096;
    hw.gb_cols = 4096;
    hw.gpr_count = 2048;
    hw.lanes = pim_func::kLanes;
    return hw;
}

pim_func::HardwareInfo active_hw() {
    if (issue_mode() == PIM_ISSUE_MMIO || issue_mode() == PIM_ISSUE_SHM)
        return pim_func::gem5_hw();
    return host_hw();
}

struct Runtime {
    pim_func::HardwareInfo hw = host_hw();
    pim_func::Device dev{hw};
    pim_func::CentGemvAllocation alloc;
    std::unordered_map<const void *, Interned> intern;
    int next_row = 0;
    bool hw_ready = false;
};

Runtime &rt() {
    static Runtime *r = new Runtime();
    static bool once = [] {
        std::atexit(pim_shutdown);
        return true;
    }();
    (void)once;
    if (!r->hw_ready) {
        r->hw = active_hw();
        r->dev = pim_func::Device{r->hw};
        r->hw_ready = true;
        if (issue_mode() == PIM_ISSUE_SHM) pim_hybrid_attach_client();
    }
    return *r;
}

int intern_store(const void *key, const float *W, int n_elements, const char *name, float *owned) {
    if (!key || !W || n_elements <= 0) return -1;
    Interned e;
    if (owned && !pim_hybrid_active()) {
        e.owned.reset(owned);
        e.f32 = e.owned.get();
    } else {
        e.f32 = owned ? owned : W;
    }
    e.n_elements = n_elements;
    e.stale = true;
    if (name) e.name = name;
    rt().intern[key] = std::move(e);
    return 0;
}

int mmio_install_weights(Interned &e, int n_out, int n_in, int row_base) {
    aim_func_cmd(AIM_FUNC_CMD_CONFIG, static_cast<uint64_t>(pim_func::gem5_hw().n_channels),
                 pim_func::gem5_hw().pim_chmask, static_cast<uint64_t>(pim_func::gem5_hw().n_rows));
    aim_func_cmd(AIM_FUNC_CMD_INTERN_BEGIN, static_cast<uint64_t>(n_out),
                 static_cast<uint64_t>(n_in), static_cast<uint64_t>(row_base));
    aim_func_cmd(AIM_FUNC_CMD_INTERN_PTR, reinterpret_cast<uint64_t>(e.f32), 0, 0);

    const int n = n_out * n_in;
    const uint32_t st = *aim_mmio32(AIM_ISR_BASE, AIM_FUNC_REG_STATUS);
    if (st == 0) return 0;
    for (int i = 0; i < n;) {
        uint64_t word = 0;
        float pair[2] = {e.f32[i], 0.0f};
        int k = 1;
        if (i + 1 < n) {
            pair[1] = e.f32[i + 1];
            k = 2;
        }
        std::memcpy(&word, pair, static_cast<size_t>(k) * sizeof(float));
        aim_func_push_u64(word);
        i += k;
    }
    return 0;
}

int ensure_banks(Interned &e, int n_out, int n_in) {
    if (!e.f32) return -1;
    if (static_cast<long long>(n_out) * n_in != e.n_elements) return -1;
    Runtime &r = rt();
    const bool shape_ok = (e.n_out == n_out && e.n_in == n_in && e.row_base >= 0);
    if (shape_ok && !e.stale) return 0;
    if (!shape_ok) {
        e.row_base = r.next_row;
        e.n_out = n_out;
        e.n_in = n_in;
        const auto chs = r.alloc.pim_channels(r.hw);
        const int n_ch = static_cast<int>(chs.size());
        const int n_row_groups = (n_out + r.hw.n_banks - 1) / r.hw.n_banks;
        const int n_waves = (n_row_groups + n_ch - 1) / n_ch;
        r.next_row += n_waves;
        if (r.next_row > r.hw.n_rows) return -2;
    }
    if (issue_mode() == PIM_ISSUE_MMIO) {
        if (mmio_install_weights(e, n_out, n_in, e.row_base) != 0) return -1;
    } else {
        r.dev.load_weights_f32(e.f32, n_out, n_in, r.alloc, e.row_base);
    }
    e.stale = false;
    return 0;
}

void mmio_pack_x(const float *x, int n_in, int n_cols, int lanes) {
    for (int col = 0; col < n_cols; col++) {
        uint16_t lanes16[16] = {};
        for (int lane = 0; lane < lanes && lane < 16; lane++) {
            const int i = col * lanes + lane;
            if (i < n_in) lanes16[lane] = pim_func::f32_to_f16(x[i]);
        }
        aim_gpr_write_bytes(static_cast<uint32_t>(col), 0, lanes16, 32);
    }
}

int issue_one_mmio(const pim_func::Command &c, const pim_func::HardwareInfo &hw) {
    const uint64_t isr_mask = aim_func_to_isr_chmask(hw.pim_chmask, hw.n_channels);
    uint64_t mask = isr_mask;
    if (c.opcode == pim_func::Opcode::RD_MAC && c.chmask > 0 &&
        (c.chmask & (c.chmask - 1)) == 0) {
        mask = aim_func_to_isr_chmask(static_cast<uint64_t>(c.chmask), hw.n_channels);
    }
    const int32_t opsize =
        (c.opcode == pim_func::Opcode::WR_GB || c.opcode == pim_func::Opcode::MAC_ABK) ? c.opsize
                                                                                       : -1;
    const int32_t row = (c.row >= 0) ? c.row : -1;
    const uint64_t gpr0 = (c.gpr0 >= 0) ? static_cast<uint64_t>(c.gpr0) : 0;
    aim_iss_cmd(static_cast<uint32_t>(c.opcode), opsize, gpr0, mask, row);
    return 0;
}

pim_func::Command from_c(const PimCommand *cmd) {
    pim_func::Command c;
    c.opcode = static_cast<pim_func::Opcode>(cmd->opcode);
    c.opsize = cmd->opsize;
    c.gpr0 = cmd->gpr0;
    c.gpr1 = cmd->gpr1;
    c.chmask = cmd->chmask;
    c.row = cmd->row;
    c.col = cmd->col;
    return c;
}

void unpack_wave_host(float *y, int n_out, int acc, int n_cols, int wave) {
    Runtime &r = rt();
    const auto chs = r.alloc.pim_channels(r.hw);
    const int n_ch = static_cast<int>(chs.size());
    const int y_gpr0 = n_cols + 1;
    for (int ci = 0; ci < n_ch; ci++) {
        const int group = wave * n_ch + ci;
        const int o0 = group * r.hw.n_banks;
        if (o0 >= n_out) break;
        const int n = n_out - o0 < r.hw.n_banks ? n_out - o0 : r.hw.n_banks;
        const pim_func::Burst b = r.dev.gpr().read(y_gpr0 + ci);
        for (int i = 0; i < n; i++) {
            const float v = pim_func::f16_to_f32(b[static_cast<size_t>(i)]);
            if (acc) y[o0 + i] += v;
            else y[o0 + i] = v;
        }
    }
}

void unpack_wave_mmio(float *y, int n_out, int acc, int n_cols, int wave) {
    Runtime &r = rt();
    const auto chs = r.alloc.pim_channels(r.hw);
    const int n_ch = static_cast<int>(chs.size());
    const int y_gpr0 = n_cols + 1;
    for (int ci = 0; ci < n_ch; ci++) {
        const int group = wave * n_ch + ci;
        const int o0 = group * r.hw.n_banks;
        if (o0 >= n_out) break;
        uint16_t lanes16[16] = {};
        aim_gpr_read_bytes(static_cast<uint32_t>(y_gpr0 + ci), 0, lanes16, 32);
        const int n = n_out - o0 < r.hw.n_banks ? n_out - o0 : r.hw.n_banks;
        for (int i = 0; i < n; i++) {
            const float v = pim_func::f16_to_f32(lanes16[i]);
            if (acc) y[o0 + i] += v;
            else y[o0 + i] = v;
        }
    }
}

} // namespace

extern "C" {

void pim_set_threads(int n) { pim_func::pool_set_threads(n); }

int pim_n_threads(void) { return pim_func::pool_n_threads(); }

void pim_set_issue_mode(int mode) {
    if (mode == PIM_ISSUE_MMIO) g_issue = PIM_ISSUE_MMIO;
    else if (mode == PIM_ISSUE_SHM) g_issue = PIM_ISSUE_SHM;
    else g_issue = PIM_ISSUE_HOST;
    rt().hw_ready = false;
}

int pim_issue_mode(void) { return issue_mode(); }

int pim_intern_f32(const void *key, const float *W, int n_elements, const char *name) {
    return intern_store(key, W, n_elements, name, nullptr);
}

int pim_intern_f32_owned(const void *key, float *W, int n_elements, const char *name) {
    if (!W) return -1;
    return intern_store(key, W, n_elements, name, W);
}

void pim_invalidate(const void *key) {
    auto it = rt().intern.find(key);
    if (it != rt().intern.end()) it->second.stale = true;
}

const float *pim_interned_f32(const void *key, int *n_elements) {
    auto it = rt().intern.find(key);
    if (it == rt().intern.end()) {
        if (n_elements) *n_elements = 0;
        return nullptr;
    }
    if (n_elements) *n_elements = it->second.n_elements;
    return it->second.f32;
}

const char *pim_interned_name(const void *key) {
    auto it = rt().intern.find(key);
    if (it == rt().intern.end() || it->second.name.empty()) return nullptr;
    return it->second.name.c_str();
}

int pim_prepare(const void *key, int n_out, int n_in) {
    if (!key || n_out <= 0 || n_in <= 0) return -1;
    auto it = rt().intern.find(key);
    if (it == rt().intern.end()) return -1;
    return ensure_banks(it->second, n_out, n_in);
}

int pim_row_base(const void *key) {
    auto it = rt().intern.find(key);
    if (it == rt().intern.end()) return -1;
    return it->second.row_base;
}

void pim_hw(PimHw *out) {
    if (!out) return;
    Runtime &r = rt();
    out->n_channels = r.hw.n_channels;
    out->n_banks = r.hw.n_banks;
    out->n_rows = r.hw.n_rows;
    out->lanes = r.hw.lanes;
    out->gb_cols = r.hw.gb_cols;
    out->gpr_count = r.hw.gpr_count;
    out->pim_chmask = r.hw.pim_chmask;
}

int pim_load_activation(const float *x, int n_in) {
    if (!x || n_in <= 0) return -1;
    Runtime &r = rt();
    const int n_cols = (n_in + r.hw.lanes - 1) / r.hw.lanes;
    if (n_cols > r.hw.gb_cols || n_cols + 8 >= r.hw.gpr_count) return -1;
    if (issue_mode() == PIM_ISSUE_MMIO) {
        mmio_pack_x(x, n_in, n_cols, r.hw.lanes);
        return 0;
    }
    const int n_in_pad = n_cols * r.hw.lanes;
    std::vector<float> xpad;
    const float *xp = x;
    if (n_in_pad != n_in) {
        xpad.assign(static_cast<size_t>(n_in_pad), 0.0f);
        std::memcpy(xpad.data(), x, static_cast<size_t>(n_in) * sizeof(float));
        xp = xpad.data();
    }
    r.dev.load_activation_f32(xp, n_in_pad, 0);
    return 0;
}

int pim_zero_gpr(int idx) {
    Runtime &r = rt();
    if (issue_mode() == PIM_ISSUE_MMIO) {
        uint8_t z[32] = {};
        aim_gpr_write_bytes(static_cast<uint32_t>(idx), 0, z, 32);
        return 0;
    }
    pim_func::Burst z{};
    r.dev.gpr().write(idx, z);
    return 0;
}

int pim_execute(const PimCommand *cmd) {
    if (!cmd) return -1;
    if (cmd->opcode == static_cast<uint32_t>(pim_func::Opcode::EOC) ||
        cmd->opcode == static_cast<uint32_t>(pim_func::Opcode::SYNC))
        return 0;
    pim_func::Command c = from_c(cmd);
    Runtime &r = rt();
    if (issue_mode() == PIM_ISSUE_MMIO) return issue_one_mmio(c, r.hw);
    if (!r.dev.execute(c).empty()) return -1;
    return 0;
}

int pim_unpack_wave(float *y, int n_out, int acc, int n_cols, int wave) {
    if (!y || n_out <= 0 || n_cols <= 0 || wave < 0) return -1;
    if (issue_mode() == PIM_ISSUE_MMIO) unpack_wave_mmio(y, n_out, acc, n_cols, wave);
    else unpack_wave_host(y, n_out, acc, n_cols, wave);
    return 0;
}

int pim_gb_cols_to_f32(float *out, int n_cols) {
    if (!out || n_cols <= 0) return -1;
    rt().dev.gb_cols_to_f32(0, n_cols, out);
    return 0;
}

int pim_mac_wave_into(int ch, int row, int n_cols, float *acc, const float *x_f32) {
    if (!acc || n_cols < 0) return -1;
    rt().dev.mac_wave_into(ch, row, n_cols, acc, x_f32);
    return 0;
}

int pim_hybrid_submit_cmds(const float *x, float *y, int n_out, int n_in, int row_base, int n_cols,
                           int acc, const PimCommand *cmds, int n_cmds) {
    if (!x || !y || n_out <= 0 || n_in <= 0) return -1;
    PimHybridCmd buf[PIM_HYBRID_MAX_CMDS];
    int n = n_cmds;
    if (n > PIM_HYBRID_MAX_CMDS) n = PIM_HYBRID_MAX_CMDS;
    if (cmds && n > 0) {
        for (int i = 0; i < n; i++) {
            buf[i].opcode = cmds[i].opcode;
            buf[i].opsize = cmds[i].opsize;
            buf[i].gpr0 = cmds[i].gpr0;
            buf[i].gpr1 = cmds[i].gpr1;
            buf[i].chmask = cmds[i].chmask;
            buf[i].row = cmds[i].row;
            buf[i].col = cmds[i].col;
        }
    } else {
        n = 0;
    }
    return pim_hybrid_submit(x, y, n_out, n_in, row_base, n_cols, acc, n > 0 ? buf : nullptr, n);
}

void pim_shutdown(void) {
    static Runtime *dying = nullptr;
    Runtime *r = &rt();
    if (dying == r) return;
    dying = r;
    pim_func::pool_shutdown();
    pim_hybrid_request_stop();
    r->intern.clear();
    r->dev.reset();
    r->next_row = 0;
    r->hw_ready = false;
}

} // extern "C"
