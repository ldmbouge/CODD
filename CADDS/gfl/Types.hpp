#pragma once

#include <FunQual.hpp>
#include <cstdint>

namespace gfl {
using i8 = int8_t;
using u8 = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;
using i128 = __int128;
using u128 = unsigned __int128;

using f32 = float;
using f64 = double;

using uptr = std::uintptr_t;
using usize = std::size_t;

using lld = long long int;
using llu = unsigned long long int;

template<typename To, typename From>
GFL_HOST_DEVICE constexpr
To scast(From f) noexcept {
  return static_cast<To>(f);
}

template<typename To, typename From>
GFL_HOST_DEVICE constexpr
To rcast(From f) noexcept {
  return reinterpret_cast<To>(f);
}
} // namespace gfl