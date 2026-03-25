#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <type_traits>

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
template <typename... Containers>
void swapAll(int i, int j, Containers&... arrays) {
    ([&]{
        auto tmp  = arrays[i];
        arrays[i] = arrays[j];
        arrays[j] = tmp;
    }(), ...);
}

template <typename KeyContainer, typename... Containers>
int partition(KeyContainer& keys, int low, int high, Containers&... arrays) {
    auto pivotKey = keys[high];
    int i = low - 1;
    for (int j = low; j < high; ++j) {
        if (keys[j] <= pivotKey) {
            ++i;
            swapAll(i, j, keys, arrays...);
        }
    }
    swapAll(i + 1, high, keys, arrays...);
    return i + 1;
}

template <typename KeyContainer, typename... Containers>
void quickSort(KeyContainer& keys, int low, int high, Containers&... arrays) {
    if (low >= high) return;
    int pivot = partition(keys, low, high, arrays...);
    quickSort(keys, low,       pivot - 1, arrays...);
    quickSort(keys, pivot + 1, high,      arrays...);
}

// ─────────────────────────────────────────────
// VERSION 1: explicit key container + N containers
// ─────────────────────────────────────────────
template <typename KeyContainer, typename... Containers>
void sortByKey(KeyContainer keys, Containers&... arrays) {
    size_t n = keys.size();
    if (!((arrays.size() == n) && ...))
        throw std::invalid_argument("All containers must have the same size as keys");

    if (n > 1)
        quickSort(keys, 0, (int)n - 1, arrays...);
}

// ─────────────────────────────────────────────
// VERSION 2: key lambda receives one element from EACH container
// ─────────────────────────────────────────────
template <typename KeyFn, typename... Containers>
void sortByKeyFn(KeyFn keyFn, Containers&... arrays) {
    std::array<size_t, sizeof...(Containers)> sizes = { arrays.size()... };
    for (size_t s : sizes)
        if (s != sizes[0])
            throw std::invalid_argument("All containers must have the same size");

    size_t n = sizes[0];

    using Key = decltype(keyFn(arrays[0]...));
    std::vector<Key> keys;
    keys.reserve(n);
    for (size_t i = 0; i < n; ++i)
        keys.push_back(keyFn(arrays[i]...));

    if (n > 1)
        quickSort(keys, 0, (int)n - 1, arrays...);
}