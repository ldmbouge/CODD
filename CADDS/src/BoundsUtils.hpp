#pragma once

#include <GFL.hpp>

template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 best() noexcept {
  constexpr gfl::f64 big_value = 1000000000.0; // 10^9
  if constexpr (Model::is_maximization)
    return big_value;
  else
    return -big_value;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 worst() noexcept {
  return -best<Model>();
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isBetter(gfl::f64 const a, gfl::f64 const b) noexcept {
  if constexpr (Model::is_maximization)
    return a > b;
  else
    return a < b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isBetterEq(gfl::f64 const a, gfl::f64 const b) noexcept {
  if constexpr (Model::is_maximization)
    return a >= b;
  else
    return a <= b;
}

// Check if a cost is valid
template<typename Model>
GFL_HOST_DEVICE constexpr
bool isBest(gfl::f64 const a) noexcept {
  return isBetterEq<Model>(a, best<Model>());
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isWorse(gfl::f64 const a, gfl::f64 const b) noexcept {
  if constexpr (Model::is_maximization)
    return a < b;
  else {
    return a > b;
  }
}

// Check if a cost is valid
template<typename Model>
GFL_HOST_DEVICE constexpr
bool isWorseEq(gfl::f64 const a, gfl::f64 const b) noexcept {
  if constexpr (Model::is_maximization)
    return a <= b;
  else {
    return a >= b;
  }
}

// Check if a cost is valid
template<typename Model>
GFL_HOST_DEVICE constexpr
bool isWorst(gfl::f64 const a) noexcept {
  return isWorseEq<Model>(a, worst<Model>());
}

template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 better(gfl::f64 const a, gfl::f64 const b) noexcept {
  return isBetter<Model>(a, b) ? a : b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 worse(gfl::f64 const a, gfl::f64 const b) noexcept {

  return isWorse<Model>(a, b) ? a : b;
}

// Transform cost to score, used to sort by better cost
template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 score(gfl::f64 const a) noexcept {
  if constexpr (Model::is_maximization)
    return -a;
  else
    return a;
}