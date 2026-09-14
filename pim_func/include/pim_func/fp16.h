#pragma once

#include <cstdint>
#include <cstring>

namespace pim_func {

inline uint16_t f32_to_f16(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) {
        return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0));
    }
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        const uint32_t m = (mant | 0x800000u) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((m + 0x1000u) >> 13));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | ((mant + 0x1000u) >> 13));
}

inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u) << 16);
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                e--;
            }
            m &= 0x3ffu;
            x = sign | ((e + 127 - 15) << 23) | (m << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

} // namespace pim_func
