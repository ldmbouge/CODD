#pragma once

#include <cstdint>

namespace gfl
{
    using i8 = int8_t;
    using u8 = uint8_t;
    using i16 = int16_t;
    using u16 = uint16_t;
    using i32 = int32_t;
    using u32 = uint32_t;
    using i64 = int64_t;
    using u64 = uint64_t;

    using ull = unsigned long long;

    // Reference: https://developer.nvidia.com/blog/implementing-high-precision-decimal-arithmetic-with-cuda-int128
    using i128 = __int128;
    using u128 = unsigned __int128;

    using f32 = float;
    using f64 = double;
}