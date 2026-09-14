#include "pim_func/c_api.h"
#include "pim_func/isa.h"
#include "planner.h"

extern "C" {
#include "gguf.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef LLM_DEFAULT_MODEL
#define LLM_DEFAULT_MODEL ""
#endif

using pim_func::Command;
using pim_func::HardwareInfo;
using pim_func::Opcode;
using pim_func::opcode_name;

extern "C" int pim_backend_gemv(const void *key, const float *x, float *y, int n_out, int n_in,
                                int acc);

struct GemvShape {
    std::string name;
    int n_out = 0;
    int n_in = 0;
    int n_tensors = 1;
};

static uint32_t rng_state = 1u;

static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)(rng_state >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
}

static bool file_ok(const char *path) {
    if (!path || !path[0]) return false;
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
}

static int dump_all(void) {
    const char *e = std::getenv("LLM_PIM_DUMP_ALL");
    return e && e[0] && e[0] != '0';
}

static void print_cmd(size_t i, const Command &c) {
    std::printf("  [%zu] %-8s", i, opcode_name(c.opcode));
    if (c.opsize >= 0) std::printf(" opsize=%d", c.opsize);
    if (c.gpr0 >= 0) std::printf(" gpr0=%lld", (long long)c.gpr0);
    if (c.gpr1 >= 0) std::printf(" gpr1=%lld", (long long)c.gpr1);
    if (c.chmask >= 0) std::printf(" chmask=0x%llx", (unsigned long long)c.chmask);
    if (c.bank >= 0) std::printf(" bank=%d", c.bank);
    if (c.row >= 0) std::printf(" row=%d", c.row);
    if (c.col >= 0) std::printf(" col=%d", c.col);
    else if (c.col < 0 && (c.opcode == Opcode::WR_GB || c.opcode == Opcode::MAC_ABK ||
                           c.opcode == Opcode::MAC_SBK))
        std::printf(" col=<unroll>");
    if (c.row_start >= 0) std::printf(" row_start=%d", c.row_start);
    if (c.row_end >= 0) std::printf(" row_end=%d", c.row_end);
    std::printf("\n");
}

static int n_pim_ch(const HardwareInfo &hw) {
    pim_func::CentGemvAllocation alloc;
    return static_cast<int>(alloc.pim_channels(hw).size());
}

static int n_cols_of(int n_in, const HardwareInfo &hw) {
    return (n_in + hw.lanes - 1) / hw.lanes;
}

static int n_waves_of(int n_out, const HardwareInfo &hw) {
    const int n_ch = n_pim_ch(hw);
    const int n_row_groups = (n_out + hw.n_banks - 1) / hw.n_banks;
    return (n_row_groups + n_ch - 1) / n_ch;
}

static int expected_cmds(int n_out, int n_in, const HardwareInfo &hw) {
    (void)n_in;
    const int n_ch = n_pim_ch(hw);
    return 1 + n_waves_of(n_out, hw) * (2 + n_ch) + 1;
}

static int validate_plan(const llm_pim::Planner &pl, const llm_pim::Plan &p, const char *tag) {
    if (p.cmds.empty()) {
        std::fprintf(stderr, "FAIL: %s empty plan\n", tag);
        return 1;
    }
    if (p.cmds.front().opcode != Opcode::WR_GB || p.cmds.back().opcode != Opcode::EOC) {
        std::fprintf(stderr, "FAIL: %s expected WR_GB ... EOC\n", tag);
        return 1;
    }
    bool has_mac = false, has_rd = false, has_bias = false;
    for (const auto &c : p.cmds) {
        if (c.opcode == Opcode::MAC_ABK) has_mac = true;
        if (c.opcode == Opcode::RD_MAC) has_rd = true;
        if (c.opcode == Opcode::WR_BIAS) has_bias = true;
        const std::string err = pl.registry().validate(c);
        if (!err.empty()) {
            std::fprintf(stderr, "FAIL: %s %s\n", tag, err.c_str());
            return 1;
        }
    }
    if (!has_mac || !has_rd || !has_bias) {
        std::fprintf(stderr, "FAIL: %s missing MAC/RD/BIAS\n", tag);
        return 1;
    }
    return 0;
}

static void print_plan_cmds(const llm_pim::Plan &p, int n_ch, int n_waves) {
    if (p.cmds.empty()) return;
    if (dump_all() || n_waves <= 1 || p.cmds.size() <= 12) {
        for (size_t i = 0; i < p.cmds.size(); i++) print_cmd(i, p.cmds[i]);
        return;
    }
    const size_t wave_len = static_cast<size_t>(2 + n_ch);
    print_cmd(0, p.cmds[0]);
    const size_t first_end = 1 + wave_len;
    for (size_t i = 1; i < first_end && i + 1 < p.cmds.size(); i++) print_cmd(i, p.cmds[i]);
    if (n_waves > 1 && first_end + 1 < p.cmds.size()) {
        const size_t second_end = first_end + wave_len;
        for (size_t i = first_end; i < second_end && i + 1 < p.cmds.size(); i++)
            print_cmd(i, p.cmds[i]);
        if (n_waves > 2)
            std::printf("  ... %d more waves (WR_BIAS + MAC_ABK row++ + %d RD_MAC) ...\n",
                        n_waves - 2, n_ch);
    }
    print_cmd(p.cmds.size() - 1, p.cmds.back());
}

static int is_decode_gemv_name(const char *n) {
    if (!n || !n[0]) return 0;
    if (std::strstr(n, "norm") || std::strstr(n, "bias") || std::strstr(n, "conv")) return 0;
    if (std::strstr(n, "ssm_alpha") || std::strstr(n, "ssm_beta") || std::strstr(n, "ssm_dt") ||
        std::strstr(n, "ssm_a"))
        return 0;
    if (std::strstr(n, "attn_q") || std::strstr(n, "attn_k") || std::strstr(n, "attn_v") ||
        std::strstr(n, "attn_output") || std::strstr(n, "attn_qkv") || std::strstr(n, "attn_gate"))
        return 1;
    if (std::strstr(n, "ffn_gate") || std::strstr(n, "ffn_up") || std::strstr(n, "ffn_down"))
        return 1;
    if (std::strstr(n, "ssm_out")) return 1;
    if (std::strcmp(n, "output.weight") == 0 || std::strcmp(n, "token_embd.weight") == 0) return 1;
    return 0;
}

static std::vector<GemvShape> unique_gemvs(const GGUFFile *f) {
    std::vector<GemvShape> out;
    for (int i = 0; i < f->n_tensors; i++) {
        const TensorInfo *t = &f->tensors[i];
        if (t->n_dims != 2 || !is_decode_gemv_name(t->name)) continue;
        const int n_out = static_cast<int>(t->dims[1]);
        const int n_in = static_cast<int>(t->dims[0]);
        if (n_out <= 0 || n_in <= 0) continue;
        bool seen = false;
        for (auto &u : out) {
            if (u.n_out == n_out && u.n_in == n_in) {
                u.n_tensors++;
                seen = true;
                break;
            }
        }
        if (!seen) {
            GemvShape s;
            s.name = t->name;
            s.n_out = n_out;
            s.n_in = n_in;
            out.push_back(s);
        }
    }
    std::sort(out.begin(), out.end(), [](const GemvShape &a, const GemvShape &b) {
        const long long ea = (long long)a.n_out * a.n_in;
        const long long eb = (long long)b.n_out * b.n_in;
        if (ea != eb) return ea < eb;
        return a.n_out < b.n_out;
    });
    return out;
}

static void cpu_gemv(const float *W, const float *x, float *y, int n_out, int n_in) {
    for (int i = 0; i < n_out; i++) {
        const float *w = W + (size_t)i * (size_t)n_in;
        float acc = 0.0f;
        for (int j = 0; j < n_in; j++) acc += w[j] * x[j];
        y[i] = acc;
    }
}

static int exec_plan(const char *name, int n_out, int n_in) {
    const long long n = (long long)n_out * n_in;
    /* Keep ctest / smoke runs off the vocab projection (tens of millions of MACs). */
    if (n > 8000000LL) {
        std::printf("  exec skip (n_out*n_in=%lld); set a smaller tensor or intern via generate-pim\n",
                    n);
        return 0;
    }
    std::vector<float> W(static_cast<size_t>(n));
    std::vector<float> x(static_cast<size_t>(n_in));
    std::vector<float> y(static_cast<size_t>(n_out));
    std::vector<float> ref(static_cast<size_t>(n_out));
    rng_state = 1u;
    for (long long i = 0; i < n; i++) W[static_cast<size_t>(i)] = frand();
    for (int i = 0; i < n_in; i++) x[static_cast<size_t>(i)] = frand();
    if (pim_intern_f32(W.data(), W.data(), (int)n, name) != 0) {
        std::fprintf(stderr, "FAIL: intern %s %dx%d\n", name, n_out, n_in);
        return 1;
    }
    if (pim_backend_gemv(W.data(), x.data(), y.data(), n_out, n_in, 0) != 0) {
        std::fprintf(stderr, "FAIL: pim_backend_gemv %s %dx%d\n", name, n_out, n_in);
        return 1;
    }
    cpu_gemv(W.data(), x.data(), ref.data(), n_out, n_in);
    const float tol = 5e-3f * (float)n_in + 0.05f;
    float max_abs = 0.0f;
    for (int i = 0; i < n_out; i++) {
        const float e = std::fabs(y[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)]);
        if (e > max_abs) max_abs = e;
        if (e > tol) {
            std::fprintf(stderr, "FAIL: %s[%d] got=%g ref=%g err=%g tol=%g\n", name, i,
                         y[static_cast<size_t>(i)], ref[static_cast<size_t>(i)], e, tol);
            return 1;
        }
    }
    std::printf("  exec ok row_base=%d max_abs=%g tol=%g\n", pim_row_base(W.data()), max_abs, tol);
    return 0;
}

static int test_model(const char *path, const llm_pim::Planner &pl, const HardwareInfo &host_hw,
                      const HardwareInfo &ch4) {
    std::printf("\nmodel %s\n", path);
    LoadedModel *m = load_model(path, 0, 0);
    const LlamaHParams *hp = &m->hparams;
    std::printf("  hparams n_layer=%d n_embd=%d n_head=%d n_head_kv=%d head_dim=%d n_ff=%d n_vocab=%d\n",
                hp->n_layer, hp->n_embd, hp->n_head, hp->n_head_kv, hp->head_dim, hp->n_ff,
                hp->n_vocab);
    const auto shapes = unique_gemvs(m->gguf);
    if (shapes.empty()) {
        std::fprintf(stderr, "FAIL: no decode GEMV tensors in %s\n", path);
        loaded_model_free(m);
        return 1;
    }
    int rc = 0;
    for (const auto &s : shapes) {
        const int n_cols = n_cols_of(s.n_in, host_hw);
        const int n_waves = n_waves_of(s.n_out, host_hw);
        const int n_ch = n_pim_ch(host_hw);
        char tag[256];
        if (s.n_tensors > 1)
            std::snprintf(tag, sizeof(tag), "%s %dx%d (%d tensors)", s.name.c_str(), s.n_out,
                          s.n_in, s.n_tensors);
        else
            std::snprintf(tag, sizeof(tag), "%s %dx%d", s.name.c_str(), s.n_out, s.n_in);
        if (n_cols > host_hw.gb_cols || n_cols + 8 >= host_hw.gpr_count) {
            std::fprintf(stderr, "FAIL: %s does not fit GB/GPR (n_cols=%d gb=%d gpr=%d)\n", tag,
                         n_cols, host_hw.gb_cols, host_hw.gpr_count);
            rc = 1;
            continue;
        }
        const llm_pim::Plan p = pl.plan_linear(s.n_out, s.n_in, host_hw, 0);
        if (validate_plan(pl, p, tag)) {
            rc = 1;
            continue;
        }
        const int want = expected_cmds(s.n_out, s.n_in, host_hw);
        if (static_cast<int>(p.cmds.size()) != want) {
            std::fprintf(stderr, "FAIL: %s cmds=%zu want %d\n", tag, p.cmds.size(), want);
            rc = 1;
            continue;
        }
        const llm_pim::Plan p4 = pl.plan_linear(s.n_out, s.n_in, ch4, 0);
        if (validate_plan(pl, p4, tag)) {
            rc = 1;
            continue;
        }
        std::printf("plan %s n_cols=%d n_waves=%d cmds=%zu gb_fills=%zu | 4ch cmds=%zu n_waves=%d\n",
                    tag, n_cols, n_waves, p.cmds.size(), p.gb_plan.size(), p4.cmds.size(),
                    n_waves_of(s.n_out, ch4));
        print_plan_cmds(p, n_ch, n_waves);
        if (exec_plan(s.name.c_str(), s.n_out, s.n_in)) rc = 1;
    }
    loaded_model_free(m);
    return rc;
}

static int test_toy(const llm_pim::Planner &pl) {
    HardwareInfo hw;
    hw.n_channels = 1;
    hw.pim_chmask = 0x1;
    const llm_pim::Plan p = pl.plan_linear(16, 64, hw);
    if (validate_plan(pl, p, "16x64")) return 1;
    if (p.gb_plan.size() != 4) {
        std::fprintf(stderr, "FAIL: expected 4 GB fills, got %zu\n", p.gb_plan.size());
        return 1;
    }
    if (p.temp_cache.empty()) {
        std::fprintf(stderr, "FAIL: expected temp cache plan\n");
        return 1;
    }
    std::printf("plan_linear n_out=16 n_in=64 cmds=%zu gb_fills=%zu\n", p.cmds.size(),
                p.gb_plan.size());
    for (size_t i = 0; i < p.cmds.size(); i++) print_cmd(i, p.cmds[i]);
    std::printf("test_planner ok cmds=%zu gb_fills=%zu\n", p.cmds.size(), p.gb_plan.size());

    HardwareInfo hw4 = hw;
    hw4.n_channels = 8;
    hw4.pim_chmask = 0x0F;
    const llm_pim::Plan p4 = pl.plan_linear(64, 64, hw4, 3);
    int n_rd = 0, n_mac = 0;
    for (const auto &c : p4.cmds) {
        if (c.opcode == Opcode::RD_MAC) n_rd++;
        if (c.opcode == Opcode::MAC_ABK) {
            n_mac++;
            if (c.row != 3) {
                std::fprintf(stderr, "FAIL: multi-ch MAC row=%d want 3\n", c.row);
                return 1;
            }
        }
    }
    if (n_mac != 1 || n_rd != 4) {
        std::fprintf(stderr, "FAIL: 4ch 64-out expected 1 MAC 4 RD, got mac=%d rd=%d\n", n_mac,
                     n_rd);
        return 1;
    }
    std::printf("test_planner multi-ch ok\n");
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    pim_set_threads(1);

    llm_pim::Planner pl;
    if (test_toy(pl) != 0) return 1;

    PimHw ph{};
    pim_hw(&ph);
    HardwareInfo host_hw;
    host_hw.n_channels = ph.n_channels;
    host_hw.n_banks = ph.n_banks;
    host_hw.n_rows = ph.n_rows;
    host_hw.lanes = ph.lanes;
    host_hw.gb_cols = ph.gb_cols;
    host_hw.gpr_count = ph.gpr_count;
    host_hw.pim_chmask = ph.pim_chmask;
    std::printf("host hw n_channels=%d n_banks=%d lanes=%d gb_cols=%d gpr=%d pim_chmask=0x%llx\n",
                host_hw.n_channels, host_hw.n_banks, host_hw.lanes, host_hw.gb_cols,
                host_hw.gpr_count, (unsigned long long)host_hw.pim_chmask);

    HardwareInfo ch4 = host_hw;
    ch4.n_channels = 8;
    ch4.pim_chmask = 0x0F;

    std::vector<const char *> models;
    for (int i = 1; i < argc; i++) models.push_back(argv[i]);
    if (models.empty() && file_ok(LLM_DEFAULT_MODEL)) models.push_back(LLM_DEFAULT_MODEL);

    int rc = 0;
    if (models.empty()) {
        std::printf("no GGUF given; pass a model path to plan real layer GEMVs\n");
    } else {
        for (const char *path : models) {
            if (!file_ok(path)) {
                std::fprintf(stderr, "FAIL: cannot open %s\n", path);
                rc = 1;
                continue;
            }
            if (test_model(path, pl, host_hw, ch4) != 0) rc = 1;
        }
    }
    if (rc) return 1;
    std::printf("test_planner model ok\n");
    return 0;
}
