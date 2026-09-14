#pragma once

#include "pim_func/mem.h"

#include <vector>

namespace pim_func {

class GlobalBuffer {
public:
    GlobalBuffer(int n_channels, int n_cols);
    Burst read(int ch, int col) const;
    void write(int ch, int col, const Burst &b);
    uint16_t read_lane(int ch, int col, int lane) const;
    void write_lane(int ch, int col, int lane, uint16_t v);
    void clear();
    int n_channels() const { return n_ch_; }
    int n_cols() const { return n_cols_; }
    const Burst *burst_ptr(int ch, int col) const;
    Burst *burst_ptr(int ch, int col);

private:
    int n_ch_;
    int n_cols_;
    std::vector<Burst> data_; /* [ch * n_cols + col] */
};

} // namespace pim_func
