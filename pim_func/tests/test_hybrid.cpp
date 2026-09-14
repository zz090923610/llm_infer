#include "pim_func/c_api.h"
#include "pim_func/fp16.h"
#include "pim_func/hybrid.h"

#include <cmath>
#include <cstdint>
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

int main() {
    if (pim_hybrid_create_server(64ULL << 20) != 0) {
        std::fprintf(stderr, "FAIL: create_server\n");
        return 1;
    }
    pim_set_issue_mode(PIM_ISSUE_SHM);
    pim_hybrid_start_mac_worker();

    const int n_out = 16, n_in = 64;
    std::vector<float> W(static_cast<size_t>(n_out) * static_cast<size_t>(n_in));
    std::vector<float> x(static_cast<size_t>(n_in));
    std::vector<float> y(static_cast<size_t>(n_out), 0.0f);
    std::vector<float> ref(static_cast<size_t>(n_out));
    for (size_t i = 0; i < W.size(); i++) W[i] = ((int)i % 17) * 0.01f - 0.08f;
    for (int i = 0; i < n_in; i++) x[static_cast<size_t>(i)] = ((i % 9) * 0.03f) - 0.1f;

    if (pim_intern_f32(W.data(), W.data(), n_out * n_in, "hybrid") != 0 ||
        pim_prepare(W.data(), n_out, n_in) != 0) {
        std::fprintf(stderr, "FAIL: intern/prepare\n");
        pim_hybrid_stop_mac_worker();
        return 1;
    }
    PimHw hw{};
    pim_hw(&hw);
    const int n_cols = (n_in + hw.lanes - 1) / hw.lanes;
    const int row = pim_row_base(W.data());
    if (pim_hybrid_submit_cmds(x.data(), y.data(), n_out, n_in, row, n_cols, 0, nullptr, 0) != 0) {
        std::fprintf(stderr, "FAIL: submit\n");
        pim_hybrid_stop_mac_worker();
        return 1;
    }
    f16_gemv(W.data(), x.data(), ref.data(), n_out, n_in);
    float max_e = 0.0f;
    for (int i = 0; i < n_out; i++) {
        const float e = std::fabs(y[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)]);
        if (e > max_e) max_e = e;
        if (e > 1e-3f) {
            std::fprintf(stderr, "FAIL: y[%d] got=%g ref=%g err=%g\n", i, y[i], ref[i], e);
            fails++;
        }
    }
    PimHybridHeader *h = pim_hybrid_header();
    if (!h || reinterpret_cast<uintptr_t>(h) != PIM_HYBRID_V0) {
        std::fprintf(stderr, "FAIL: header not at V0 (%p)\n", static_cast<void *>(h));
        fails++;
    }
    pim_hybrid_stop_mac_worker();
    if (fails) return 1;
    std::printf("test_hybrid 16x64 ok max_err=%g V0=%#llx\n", max_e,
                (unsigned long long)PIM_HYBRID_V0);
    return 0;
}
