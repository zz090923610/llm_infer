#include "pim_func/gb.h"

#include <algorithm>

namespace pim_func {

GlobalBuffer::GlobalBuffer(int n_channels, int n_cols)
    : n_ch_(n_channels), n_cols_(n_cols),
      data_(static_cast<size_t>(n_channels) * static_cast<size_t>(n_cols), Burst{}) {}

Burst GlobalBuffer::read(int ch, int col) const {
    if (ch < 0 || ch >= n_ch_ || col < 0 || col >= n_cols_) return Burst{};
    return data_[static_cast<size_t>(ch) * static_cast<size_t>(n_cols_) + static_cast<size_t>(col)];
}

void GlobalBuffer::write(int ch, int col, const Burst &b) {
    if (ch < 0 || ch >= n_ch_ || col < 0 || col >= n_cols_) return;
    data_[static_cast<size_t>(ch) * static_cast<size_t>(n_cols_) + static_cast<size_t>(col)] = b;
}

uint16_t GlobalBuffer::read_lane(int ch, int col, int lane) const {
    return read(ch, col)[static_cast<size_t>(lane)];
}

void GlobalBuffer::write_lane(int ch, int col, int lane, uint16_t v) {
    Burst b = read(ch, col);
    b[static_cast<size_t>(lane)] = v;
    write(ch, col, b);
}

void GlobalBuffer::clear() { std::fill(data_.begin(), data_.end(), Burst{}); }

const Burst *GlobalBuffer::burst_ptr(int ch, int col) const {
    if (ch < 0 || ch >= n_ch_ || col < 0 || col >= n_cols_) return nullptr;
    return &data_[static_cast<size_t>(ch) * static_cast<size_t>(n_cols_) + static_cast<size_t>(col)];
}

Burst *GlobalBuffer::burst_ptr(int ch, int col) {
    if (ch < 0 || ch >= n_ch_ || col < 0 || col >= n_cols_) return nullptr;
    return &data_[static_cast<size_t>(ch) * static_cast<size_t>(n_cols_) + static_cast<size_t>(col)];
}

} // namespace pim_func
