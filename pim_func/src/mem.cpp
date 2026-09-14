#include "pim_func/mem.h"
#include "pim_func/hybrid.h"

#include <algorithm>
#include <cstring>

namespace pim_func {

void BankedDram::index_region(DenseBurstRegion *r) {
    for (int row = r->row0; row < r->row0 + r->n_rows; row++) {
        row_index_[row_key(r->ch, row)] = r;
    }
}

DenseBurstRegion *BankedDram::ensure_dense(int ch, int n_banks, int row0, int n_rows, int col0,
                                           int n_cols) {
    if (n_banks <= 0 || n_rows <= 0 || n_cols <= 0) return nullptr;
    for (auto &up : dense_) {
        DenseBurstRegion *r = up.get();
        if (r->ch == ch && r->row0 == row0 && r->n_rows == n_rows && r->col0 == col0 &&
            r->n_cols == n_cols && r->n_banks == n_banks) {
            Burst *s = r->storage();
            if (s) {
                const size_t n = static_cast<size_t>(n_banks) * static_cast<size_t>(n_rows) *
                                 static_cast<size_t>(n_cols);
                std::fill(s, s + n, Burst{});
            }
            return r;
        }
    }
    auto up = std::make_unique<DenseBurstRegion>();
    DenseBurstRegion *r = up.get();
    r->ch = ch;
    r->n_banks = n_banks;
    r->row0 = row0;
    r->n_rows = n_rows;
    r->col0 = col0;
    r->n_cols = n_cols;
    const size_t n = static_cast<size_t>(n_banks) * static_cast<size_t>(n_rows) *
                     static_cast<size_t>(n_cols);
    if (pim_hybrid_active()) {
        void *p = pim_hybrid_alloc(n * sizeof(Burst));
        if (!p) return nullptr;
        r->ext = static_cast<Burst *>(p);
        if (pim_hybrid_record_region(ch, n_banks, row0, n_rows, col0, n_cols, p) != 0)
            return nullptr;
    } else {
        r->data.assign(n, Burst{});
    }
    dense_.push_back(std::move(up));
    index_region(r);
    return r;
}

void BankedDram::adopt_hybrid(void *arena_base, const PimHybridRegion *regions, uint32_t n) {
    if (!arena_base || !regions) return;
    dense_.clear();
    row_index_.clear();
    for (uint32_t i = 0; i < n; i++) {
        const PimHybridRegion &m = regions[i];
        if (m.n_banks <= 0 || m.n_rows <= 0 || m.n_cols <= 0) continue;
        auto up = std::make_unique<DenseBurstRegion>();
        DenseBurstRegion *r = up.get();
        r->ch = m.ch;
        r->n_banks = m.n_banks;
        r->row0 = m.row0;
        r->n_rows = m.n_rows;
        r->col0 = m.col0;
        r->n_cols = m.n_cols;
        r->ext = reinterpret_cast<Burst *>(static_cast<uint8_t *>(arena_base) + m.data_off);
        dense_.push_back(std::move(up));
        index_region(r);
    }
}

const DenseBurstRegion *BankedDram::region_for_row(int ch, int row) const {
    auto it = row_index_.find(row_key(ch, row));
    return it == row_index_.end() ? nullptr : it->second;
}

DenseBurstRegion *BankedDram::region_for_row(int ch, int row) {
    auto it = row_index_.find(row_key(ch, row));
    return it == row_index_.end() ? nullptr : it->second;
}

const Burst *BankedDram::burst_ptr(int ch, int bank, int row, int col) const {
    if (const DenseBurstRegion *r = region_for_row(ch, row)) {
        return r->burst(bank, row, col);
    }
    BurstKey k{static_cast<uint8_t>(ch), static_cast<uint8_t>(bank), static_cast<uint32_t>(row),
               static_cast<uint16_t>(col)};
    auto it = store_.find(k);
    if (it == store_.end()) return nullptr;
    return &it->second;
}

Burst *BankedDram::burst_ptr(int ch, int bank, int row, int col) {
    if (DenseBurstRegion *r = region_for_row(ch, row)) {
        if (bank < 0 || bank >= r->n_banks || col < r->col0 || col >= r->col0 + r->n_cols)
            return nullptr;
        return &r->storage()[r->index(bank, row, col)];
    }
    BurstKey k{static_cast<uint8_t>(ch), static_cast<uint8_t>(bank), static_cast<uint32_t>(row),
               static_cast<uint16_t>(col)};
    return &store_[k];
}

Burst BankedDram::read(int ch, int bank, int row, int col) const {
    if (const Burst *p = burst_ptr(ch, bank, row, col)) return *p;
    return Burst{};
}

void BankedDram::write(int ch, int bank, int row, int col, const Burst &b) {
    if (Burst *p = burst_ptr(ch, bank, row, col)) {
        *p = b;
        return;
    }
    BurstKey k{static_cast<uint8_t>(ch), static_cast<uint8_t>(bank), static_cast<uint32_t>(row),
               static_cast<uint16_t>(col)};
    store_[k] = b;
}

void BankedDram::write_lane(int ch, int bank, int row, int col, int lane, uint16_t v) {
    Burst *p = burst_ptr(ch, bank, row, col);
    if (!p) {
        BurstKey k{static_cast<uint8_t>(ch), static_cast<uint8_t>(bank), static_cast<uint32_t>(row),
                   static_cast<uint16_t>(col)};
        p = &store_[k];
    }
    (*p)[static_cast<size_t>(lane)] = v;
}

uint16_t BankedDram::read_lane(int ch, int bank, int row, int col, int lane) const {
    return read(ch, bank, row, col)[static_cast<size_t>(lane)];
}

void BankedDram::clear() {
    dense_.clear();
    row_index_.clear();
    store_.clear();
}

size_t BankedDram::n_bursts() const {
    size_t n = store_.size();
    for (const auto &up : dense_) {
        const size_t n_r = static_cast<size_t>(up->n_banks) * static_cast<size_t>(up->n_rows) *
                           static_cast<size_t>(up->n_cols);
        n += n_r;
    }
    return n;
}

GprFile::GprFile(int count) : count_(count), regs_(static_cast<size_t>(count), Burst{}) {}

Burst GprFile::read(int idx) const {
    if (idx < 0 || idx >= count_) return Burst{};
    return regs_[static_cast<size_t>(idx)];
}

void GprFile::write(int idx, const Burst &b) {
    if (idx < 0 || idx >= count_) return;
    regs_[static_cast<size_t>(idx)] = b;
}

uint16_t GprFile::read_lane(int idx, int lane) const {
    return read(idx)[static_cast<size_t>(lane)];
}

void GprFile::write_lane(int idx, int lane, uint16_t v) {
    Burst b = read(idx);
    b[static_cast<size_t>(lane)] = v;
    write(idx, b);
}

void GprFile::clear() { std::fill(regs_.begin(), regs_.end(), Burst{}); }

const Burst *GprFile::burst_ptr(int idx) const {
    if (idx < 0 || idx >= count_) return nullptr;
    return &regs_[static_cast<size_t>(idx)];
}

Burst *GprFile::burst_ptr(int idx) {
    if (idx < 0 || idx >= count_) return nullptr;
    return &regs_[static_cast<size_t>(idx)];
}

} // namespace pim_func
