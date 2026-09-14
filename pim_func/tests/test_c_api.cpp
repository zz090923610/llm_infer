#include "centgemv_seq.h"
#include "pim_func/c_api.h"
#include "pim_func/fp16.h"
#include "pim_func/isa.h"

#include <cmath>
#include <cstdio>
#include <vector>

using pim_func::f16_to_f32;
using pim_func::f32_to_f16;

static int fails;

static void f16_gemv(const float *W, const float *x, float *y, int n_out, int n_in) {
    for (int o = 0; o < n_out; o++) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            acc += f16_to_f32(f32_to_f16(W[o * n_in + i])) * f16_to_f32(f32_to_f16(x[i]));
        }
        y[o] = f16_to_f32(f32_to_f16(acc));
    }
}

static int run_gemv(const void *key, const float *x, float *y, int n_out, int n_in) {
    if (pim_prepare(key, n_out, n_in) != 0) return -1;
    const int row = pim_row_base(key);
    if (pim_load_activation(x, n_in) != 0) return -1;
    PimHw hw{};
    pim_hw(&hw);
    const int n_cols = (n_in + hw.lanes - 1) / hw.lanes;
    if (pim_zero_gpr(n_cols) != 0) return -1;
    pim_func::HardwareInfo h;
    h.n_channels = hw.n_channels;
    h.n_banks = hw.n_banks;
    h.lanes = hw.lanes;
    h.pim_chmask = hw.pim_chmask;
    const auto cmds = centgemv_cmds(n_out, n_in, h, row);
    int wave = -1;
    for (const auto &c : cmds) {
        PimCommand pc = to_pim_cmd(c);
        if (pim_execute(&pc) != 0) return -1;
        if (c.opcode == pim_func::Opcode::WR_BIAS) wave++;
        if (c.opcode == pim_func::Opcode::RD_MAC && wave >= 0) {
            pim_func::CentGemvAllocation alloc;
            const auto chs = alloc.pim_channels(h);
            if (c.chmask == (1LL << chs.back())) {
                if (pim_unpack_wave(y, n_out, 0, n_cols, wave) != 0) return -1;
            }
        }
    }
    return 0;
}

static void check_gemv(const char *tag, int n_out, int n_in) {
    std::vector<float> W(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    std::vector<float> x(static_cast<size_t>(n_in));
    std::vector<float> y(static_cast<size_t>(n_out));
    std::vector<float> ref(static_cast<size_t>(n_out));
    for (size_t i = 0; i < W.size(); i++) W[i] = ((int)i % 17) * 0.01f - 0.08f;
    for (int i = 0; i < n_in; i++) x[static_cast<size_t>(i)] = ((i % 9) * 0.03f) - 0.1f;
    if (pim_intern_f32(W.data(), W.data(), n_out * n_in, tag) != 0) {
        std::fprintf(stderr, "FAIL: intern %s\n", tag);
        fails++;
        return;
    }
    if (run_gemv(W.data(), x.data(), y.data(), n_out, n_in) != 0) {
        std::fprintf(stderr, "FAIL: gemv %s\n", tag);
        fails++;
        return;
    }
    f16_gemv(W.data(), x.data(), ref.data(), n_out, n_in);
    float max_e = 0.0f;
    for (int i = 0; i < n_out; i++) {
        const float e = std::fabs(y[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)]);
        if (e > max_e) max_e = e;
        if (e > 1e-3f) {
            std::fprintf(stderr, "FAIL: %s y[%d] pim=%g ref=%g err=%g\n", tag, i, y[i], ref[i], e);
            fails++;
            return;
        }
    }
    std::printf("ok %s max_err=%g\n", tag, max_e);
}

int main() {
    check_gemv("16x64", 16, 64);
    check_gemv("32x48", 32, 48);
    check_gemv("17x32", 17, 32);
    check_gemv("960x960", 960, 960);
    check_gemv("32x2560", 32, 2560);

    std::vector<float> A(16 * 32, 1.0f);
    std::vector<float> B(16 * 32, 2.0f);
    std::vector<float> x(32, 1.0f);
    std::vector<float> ya(16), yb(16);
    pim_intern_f32(A.data(), A.data(), 16 * 32, "A");
    pim_intern_f32(B.data(), B.data(), 16 * 32, "B");
    if (run_gemv(A.data(), x.data(), ya.data(), 16, 32) != 0 ||
        run_gemv(B.data(), x.data(), yb.data(), 16, 32) != 0) {
        std::fprintf(stderr, "FAIL: intern pair exec\n");
        fails++;
    }
    if (std::fabs(ya[0] - 32.0f) > 0.5f || std::fabs(yb[0] - 64.0f) > 0.5f) {
        std::fprintf(stderr, "FAIL: intern collision ya0=%g yb0=%g\n", ya[0], yb[0]);
        fails++;
    }

    if (fails) return 1;
    std::printf("test_c_api ok\n");
    return 0;
}
