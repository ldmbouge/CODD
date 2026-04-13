#pragma once

#include <array>
#include <vector>

#include "Types.hpp"

namespace gfl {

template<typename T>
struct DummyDecomposer64 {
  GFL_HOST_DEVICE
  tuple<f64 &> operator()(T & t) const {
    f64 f;
    return {f};
  }
};

template<typename... Containers>
void swapAll(i32 i, i32 j, Containers &... arrays) {
  (
      [&] {
        auto tmp = arrays[i];
        arrays[i] = arrays[j];
        arrays[j] = tmp;
      }(),
      ...);
}

template<typename KeyContainer, typename... Containers>
int partition(KeyContainer & keys, i32 low, i32 high, Containers &... arrays) {
  auto pivotKey = keys[high];
  i32 i = low - 1;
  for (i32 j = low; j < high; ++j) {
    if (keys[j] <= pivotKey) {
      ++i;
      swapAll(i, j, keys, arrays...);
    }
  }
  swapAll(i + 1, high, keys, arrays...);
  return i + 1;
}

template<typename KeyContainer, typename... Containers>
void quickSort(KeyContainer & keys, i32 low, i32 high, Containers &... arrays) {
  if (low >= high)
    return;
  i32 const pivot = partition(keys, low, high, arrays...);
  quickSort(keys, low, pivot - 1, arrays...);
  quickSort(keys, pivot + 1, high, arrays...);
}

template<typename KeyContainer, typename... Containers>
void sortByKey(KeyContainer keys, Containers &... arrays) {
  usize const n = keys.size();
  assert(((scast<usize>(arrays.size()) == n) && ...));
  quickSort(keys, 0, scast<i32>(n - 1), arrays...);
}

template<typename KeyFn, typename... Containers>
void sortByKeyFn(KeyFn keyFn, Containers &... arrays) {
  using Key = decltype(keyFn(arrays[0]...));
  usize const n = scast<usize>(std::get<0>(std::tie(arrays...)).size());
  std::vector<Key> keys;
  keys.reserve(n);
  for (usize i = 0; i < n; ++i)
    keys.push_back(keyFn(arrays[i]...));
  sortByKey(std::move(keys), arrays...);
}
} // namespace gfl