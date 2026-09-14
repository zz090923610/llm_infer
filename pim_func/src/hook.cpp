#include "pim_func/hook.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace pim_func {

namespace {

FuncMode g_mode = FuncMode::Timing;
bool g_mode_set = false;

std::unique_ptr<Device> g_dev;

struct InternJob {
    int n_out = 0;
    int n_in = 0;
    int row_base = 0;
    int filled = 0;
    uint64_t guest_va = 0;
    bool copy_pending = false;
    std::vector<float> buf;
};

InternJob g_intern;

FuncMode mode_from_env() {
    const char *e = std::getenv("PIM_FUNC_MODE");
    if (e && (std::strcmp(e, "token") == 0 || std::strcmp(e, "timing+token") == 0 ||
              std::strcmp(e, "timing_token") == 0)) {
        return FuncMode::Token;
    }
    return FuncMode::Timing;
}

} // namespace

FuncMode func_mode() {
    if (!g_mode_set) {
        g_mode = mode_from_env();
        g_mode_set = true;
    }
    return g_mode;
}

void set_func_mode(FuncMode m) {
    g_mode = m;
    g_mode_set = true;
}

HardwareInfo gem5_hw() {
    HardwareInfo hw;
    hw.n_channels = 8;
    hw.pim_chmask = 0x0F; /* functional HW ch 0-3; guest ISRs use AIM mask 0xF0 */
    hw.n_banks = kBanks;
    hw.n_rows = 65536;
    hw.n_cols = 1024;
    hw.gb_cols = 4096;
    hw.gpr_count = kGprCount;
    hw.lanes = kLanes;
    hw.burst_bytes = kBurstBytes;
    return hw;
}

void configure_device(const HardwareInfo &hw) {
    if (g_dev) {
        const HardwareInfo &cur = g_dev->hw();
        if (cur.n_channels == hw.n_channels && cur.pim_chmask == hw.pim_chmask &&
            cur.n_rows == hw.n_rows && cur.n_banks == hw.n_banks &&
            cur.gpr_count == hw.gpr_count) {
            return;
        }
    }
    g_dev = std::make_unique<Device>(hw);
}

Device &global_device() {
    if (!g_dev) g_dev = std::make_unique<Device>(gem5_hw());
    return *g_dev;
}

bool gpr_store(int idx, unsigned byte_off, const void *data, unsigned size) {
    if (!data || size == 0) return false;
    Burst *b = global_device().gpr().burst_ptr(idx);
    if (!b || byte_off + size > sizeof(Burst)) return false;
    std::memcpy(reinterpret_cast<uint8_t *>(b->data()) + byte_off, data, size);
    return true;
}

bool gpr_load(int idx, unsigned byte_off, void *data, unsigned size) {
    if (!data || size == 0) return false;
    const Burst *b = global_device().gpr().burst_ptr(idx);
    if (!b || byte_off + size > sizeof(Burst)) return false;
    std::memcpy(data, reinterpret_cast<const uint8_t *>(b->data()) + byte_off, size);
    return true;
}

void intern_begin(int n_out, int n_in, int row_base) {
    g_intern.n_out = n_out;
    g_intern.n_in = n_in;
    g_intern.row_base = row_base;
    g_intern.filled = 0;
    g_intern.guest_va = 0;
    g_intern.copy_pending = false;
    const long long n = static_cast<long long>(n_out) * n_in;
    if (n_out <= 0 || n_in <= 0 || n > 1LL << 30) {
        g_intern.buf.clear();
        return;
    }
    g_intern.buf.assign(static_cast<size_t>(n), 0.0f);
}

void intern_push_f32(const float *v, int n) {
    if (!v || n <= 0 || g_intern.buf.empty()) return;
    const int room = static_cast<int>(g_intern.buf.size()) - g_intern.filled;
    if (room <= 0) return;
    if (n > room) n = room;
    std::memcpy(g_intern.buf.data() + g_intern.filled, v, static_cast<size_t>(n) * sizeof(float));
    g_intern.filled += n;
    if (g_intern.filled >= static_cast<int>(g_intern.buf.size())) {
        intern_complete_from_host(g_intern.buf.data());
    }
}

bool intern_copy_pending() { return g_intern.copy_pending; }

uint64_t intern_guest_va() { return g_intern.guest_va; }

int intern_n_elements() { return static_cast<int>(g_intern.buf.size()); }

int intern_n_out() { return g_intern.n_out; }

int intern_n_in() { return g_intern.n_in; }

int intern_row_base() { return g_intern.row_base; }

void intern_request_copy(uint64_t guest_va) {
    g_intern.guest_va = guest_va;
    g_intern.copy_pending = !g_intern.buf.empty();
}

bool intern_complete_from_host(const float *W) {
    if (!W || g_intern.n_out <= 0 || g_intern.n_in <= 0) return false;
    CentGemvAllocation alloc;
    global_device().load_weights_f32(W, g_intern.n_out, g_intern.n_in, alloc, g_intern.row_base);
    g_intern.copy_pending = false;
    g_intern.filled = static_cast<int>(g_intern.buf.size());
    return true;
}

int intern_status() {
    if (g_intern.buf.empty()) return 0;
    if (g_intern.filled >= static_cast<int>(g_intern.buf.size())) return 0;
    return 1;
}

void on_retired_aim(const RetiredAim &cmd) {
    if (func_mode() != FuncMode::Token) return;
    Command c;
    c.opcode = static_cast<Opcode>(cmd.opcode);
    c.opsize = 1;
    c.gpr0 = cmd.gpr0;
    c.chmask = cmd.chmask;
    c.bank = cmd.bank;
    c.row = cmd.row;
    c.col = (cmd.col >= 0) ? cmd.col : 0;
    global_device().execute_micro(c, cmd.channel_id);
}

} // namespace pim_func
