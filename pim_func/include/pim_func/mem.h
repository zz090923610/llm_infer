#pragma once

#include "pim_func/fp16.h"
#include "pim_func/hybrid_abi.h"
#include "pim_func/types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace pim_func {

using Burst = std::array<uint16_t, kLanes>;

struct BurstKey {
    uint8_t ch = 0;
    uint8_t bank = 0;
    uint32_t row = 0;
    uint16_t col = 0;
    bool operator==(const BurstKey &o) const {
        return ch == o.ch && bank == o.bank && row == o.row && col == o.col;
    }
};

struct BurstKeyHash {
    size_t operator()(const BurstKey &k) const {
        return (static_cast<size_t>(k.ch) << 48) ^ (static_cast<size_t>(k.bank) << 40) ^
               (static_cast<size_t>(k.row) << 16) ^ k.col;
    }
};

/* Dense rectangular slab for CentGemv GEMV weights: O(1) full-burst access. */
struct DenseBurstRegion {
    int ch = 0;
    int n_banks = kBanks;
    int row0 = 0;
    int n_rows = 0;
    int col0 = 0;
    int n_cols = 0;
    std::vector<Burst> data; /* [row_off][col_off][bank] — banks contiguous for MAC */
    Burst *ext = nullptr;    /* hybrid arena; when set, used instead of data */

    const Burst *storage() const { return ext ? ext : data.data(); }
    Burst *storage() { return ext ? ext : data.data(); }

    bool contains(int c, int bank, int row, int col) const {
        return c == ch && bank >= 0 && bank < n_banks && row >= row0 && row < row0 + n_rows &&
               col >= col0 && col < col0 + n_cols;
    }

    size_t index(int bank, int row, int col) const {
        return (static_cast<size_t>(row - row0) * static_cast<size_t>(n_cols) +
                static_cast<size_t>(col - col0)) *
                   static_cast<size_t>(n_banks) +
               static_cast<size_t>(bank);
    }

    const Burst *burst(int bank, int row, int col) const {
        if (bank < 0 || bank >= n_banks || row < row0 || row >= row0 + n_rows || col < col0 ||
            col >= col0 + n_cols)
            return nullptr;
        return &storage()[index(bank, row, col)];
    }
};

class BankedDram {
public:
    Burst read(int ch, int bank, int row, int col) const;
    void write(int ch, int bank, int row, int col, const Burst &b);
    void write_lane(int ch, int bank, int row, int col, int lane, uint16_t v);
    uint16_t read_lane(int ch, int bank, int row, int col, int lane) const;
    void clear();
    size_t n_bursts() const;

    /* Allocate/replace a dense region for GEMV weights on one channel. Returns the region. */
    DenseBurstRegion *ensure_dense(int ch, int n_banks, int row0, int n_rows, int col0, int n_cols);

    /* O(1) lookup by (ch, row) — CentGemv rows are unique across interned matrices. */
    const DenseBurstRegion *region_for_row(int ch, int row) const;
    DenseBurstRegion *region_for_row(int ch, int row);

    const Burst *burst_ptr(int ch, int bank, int row, int col) const;
    Burst *burst_ptr(int ch, int bank, int row, int col);

    /* Rebuild dense slabs from a shared hybrid arena (does not copy bursts). */
    void adopt_hybrid(void *arena_base, const struct PimHybridRegion *regions, uint32_t n);

private:
    static uint64_t row_key(int ch, int row) {
        return (static_cast<uint64_t>(static_cast<uint32_t>(ch)) << 32) |
               static_cast<uint32_t>(row);
    }
    void index_region(DenseBurstRegion *r);

    std::vector<std::unique_ptr<DenseBurstRegion>> dense_;
    std::unordered_map<uint64_t, DenseBurstRegion *> row_index_;
    std::unordered_map<BurstKey, Burst, BurstKeyHash> store_;
};

class GprFile {
public:
    explicit GprFile(int count = kGprCount);
    Burst read(int idx) const;
    void write(int idx, const Burst &b);
    uint16_t read_lane(int idx, int lane) const;
    void write_lane(int idx, int lane, uint16_t v);
    void clear();
    int count() const { return count_; }
    const Burst *burst_ptr(int idx) const;
    Burst *burst_ptr(int idx);

private:
    int count_;
    std::vector<Burst> regs_;
};

} // namespace pim_func
