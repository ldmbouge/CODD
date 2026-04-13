#pragma once

#include <cassert>
#include <type_traits>

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl {

template<typename TOut, typename T1, typename T2>
GFL_HOST_DEVICE constexpr
TOut ceilDiv(T1 const a, T2 const b) {
  static_assert(std::is_integral_v<T1>);
  static_assert(std::is_integral_v<T2>);
  static_assert(std::is_integral_v<TOut>);
  assert(a >= 0);
  assert(b > 0);
  using TComm = std::common_type_t<T1, T2>;
  TComm const a_ = scast<TComm>(a);
  TComm const b_ = scast<TComm>(b);
  return scast<TOut>((a_ + b_ - 1) / b_);
}

template<typename TOut, typename T1, typename T2>
GFL_HOST_DEVICE constexpr
TOut roundUp(T1 const a, T2 const b) {
  static_assert(std::is_integral_v<T1>);
  static_assert(std::is_integral_v<T2>);
  static_assert(std::is_integral_v<TOut>);
  assert(a >= 0);
  assert(b > 0);
  using TComm = std::common_type_t<T1, T2>;
  TComm const a_ = scast<TComm>(a);
  TComm const b_ = scast<TComm>(b);
  return scast<TOut>(((a_ + b_ - 1) / b_) * b_);
}

template<typename TOut, typename T1, typename T2>
GFL_HOST_DEVICE constexpr
TOut min(T1 const a, T2 const b) {
  static_assert(std::is_arithmetic_v<T1>);
  static_assert(std::is_arithmetic_v<T2>);
  static_assert(std::is_arithmetic_v<TOut>);
  using TComm = std::common_type_t<T1, T2>;
  TComm const a_ = scast<TComm>(a);
  TComm const b_ = scast<TComm>(b);
  return scast<TOut>(a_ < b_ ? a_ : b_);
}

template<typename TOut, typename T1, typename T2>
GFL_HOST_DEVICE constexpr
TOut max(T1 const a, T2 const b) {
  static_assert(std::is_arithmetic_v<T1>);
  static_assert(std::is_arithmetic_v<T2>);
  static_assert(std::is_arithmetic_v<TOut>);
  using TComm = std::common_type_t<T1, T2>;
  TComm const a_ = scast<TComm>(a);
  TComm const b_ = scast<TComm>(b);
  return scast<TOut>(a_ > b_ ? a_ : b_);
}
} // namespace gfl