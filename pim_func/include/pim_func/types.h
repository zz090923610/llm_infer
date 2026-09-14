#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace pim_func {

inline constexpr int kLanes = 16;
inline constexpr int kBanks = 16;
inline constexpr int kBurstBytes = 32;
inline constexpr int kGprCount = 1024;

enum class DType { FP16, FP32 };

struct HardwareInfo {
    int n_channels = 8;
    int n_banks = kBanks;
    int n_rows = 32768;
    int n_cols = 128;
    int gpr_count = kGprCount;
    int gb_cols = 256;
    int lanes = kLanes;
    int burst_bytes = kBurstBytes;
    uint64_t pim_chmask = 0x01; /* tests default to channel 0 */
};

struct BurstAddr {
    int ch = 0;
    int bank = 0;
    int row = 0;
    int col = 0;
};

struct TensorDesc {
    std::string name;
    DType dtype = DType::FP16;
    std::vector<int64_t> shape;
};

inline int64_t tensor_numel(const TensorDesc &t) {
    int64_t n = 1;
    for (int64_t d : t.shape) n *= d;
    return n;
}

} // namespace pim_func
