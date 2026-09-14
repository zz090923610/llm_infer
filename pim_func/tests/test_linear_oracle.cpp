#include "centgemv_seq.h"
#include "pim_func/device.h"
#include "pim_func/fp16.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pim_func;

#ifndef LLM_LINEAR_DUMP
#define LLM_LINEAR_DUMP ""
#endif

static int fails;

static std::vector<float> read_block(std::istream &in, int n) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        if (!(in >> v[static_cast<size_t>(i)])) {
            std::fprintf(stderr, "FAIL: dump truncated at %d/%d\n", i, n);
            fails++;
            break;
        }
    }
    return v;
}

int main() {
    const char *path = LLM_LINEAR_DUMP;
    if (!path || !path[0]) {
        std::fprintf(stderr, "FAIL: LLM_LINEAR_DUMP not set\n");
        return 1;
    }
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "FAIL: cannot open %s — run llm_infer test_prompt first\n", path);
        return 1;
    }
    int n_out = 0, n_in = 0;
    std::string tok;
    while (in >> tok) {
        if (tok == "n_out") in >> n_out;
        else if (tok == "n_in") in >> n_in;
        else if (tok == "W") break;
    }
    if (n_out != 16 || n_in != 64) {
        std::fprintf(stderr, "FAIL: unexpected dump shape %d x %d\n", n_out, n_in);
        return 1;
    }
    std::vector<float> W = read_block(in, n_out * n_in);
    in >> tok;
    if (tok != "x") {
        std::fprintf(stderr, "FAIL: expected x, got %s\n", tok.c_str());
        return 1;
    }
    std::vector<float> x = read_block(in, n_in);
    in >> tok;
    if (tok != "y") {
        std::fprintf(stderr, "FAIL: expected y, got %s\n", tok.c_str());
        return 1;
    }
    std::vector<float> y_ref = read_block(in, n_out);
    if (fails) return 1;

    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    hw.gb_cols = 64;
    CentGemvAllocation alloc;
    const auto cmds = centgemv_cmds(n_out, n_in, hw);
    Device dev(hw);
    dev.load_weights_f32(W.data(), n_out, n_in, alloc);
    dev.load_activation_f32(x.data(), n_in, 0);
    for (const auto &c : cmds) {
        const std::string e = dev.execute(c);
        if (!e.empty()) {
            std::fprintf(stderr, "FAIL: exec %s\n", e.c_str());
            return 1;
        }
    }
    std::vector<float> y(static_cast<size_t>(n_out));
    const int y_gpr = static_cast<int>(cmds[cmds.size() - 2].gpr0);
    dev.read_mac_f32(y.data(), n_out, y_gpr);

    float max_err = 0.0f;
    const float tol = 0.2f; /* fp16 MAC vs llm_infer f32 */
    for (int i = 0; i < n_out; i++) {
        const float e = std::fabs(y[static_cast<size_t>(i)] - y_ref[static_cast<size_t>(i)]);
        if (e > max_err) max_err = e;
        if (e > tol) {
            std::fprintf(stderr, "FAIL: y[%d] pim=%g llm_infer=%g err=%g\n", i, y[i], y_ref[i], e);
            fails++;
        }
    }

    /* Tighter check against fp16-rounded llm_infer operands. */
    float max_f16 = 0.0f;
    for (int o = 0; o < n_out; o++) {
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            acc += f16_to_f32(f32_to_f16(W[static_cast<size_t>(o) * n_in + i])) *
                   f16_to_f32(f32_to_f16(x[static_cast<size_t>(i)]));
        }
        const float acc_f16 = f16_to_f32(f32_to_f16(acc));
        const float e = std::fabs(y[static_cast<size_t>(o)] - acc_f16);
        if (e > max_f16) max_f16 = e;
        if (e > 1e-3f) {
            std::fprintf(stderr, "FAIL: y[%d] pim=%g f16_ref=%g err=%g\n", o, y[o], acc_f16, e);
            fails++;
        }
    }

    if (fails) return 1;
    std::printf("test_linear_oracle ok max_err_vs_llm=%g max_err_vs_f16=%g dump=%s\n", max_err,
                max_f16, path);
    return 0;
}
