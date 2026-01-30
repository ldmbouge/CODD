#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <cstring>

template<typename T>
class RadixSort {
public:
    // Type alias for the key extractor function
    using KeyExtractor = std::function<uint64_t(const T&)>;

    /**
     * Radix sort for arrays using pre-allocated buffers
     * @param input Pointer to input array pointer (will be swapped)
     * @param output Pointer to output array pointer (sorted result will be here, may be swapped)
     * @param temp Pointer to temporary buffer pointer (may be swapped)
     * @param size Number of elements
     * @param getKey Function to extract 64-bit key from element
     *
     * After sorting, *output will point to the sorted array.
     * The buffer pointers may be swapped among themselves.
     */
    static void sort(T** input, T** output, T** temp, size_t size, KeyExtractor getKey) {
        if (size <= 1) {
            if (size == 1 && *input != *output) {
                (*output)[0] = (*input)[0];
            }
            return;
        }

        T* src;
        T* dst;

        // First pass: input -> output
        countingSortByByte(*input, *output, size, 0, getKey);

        src = *output;
        dst = *temp;

        // Remaining 7 passes, ping-ponging between output and temp
        for (int shift = 8; shift < 64; shift += 8) {
            countingSortByByte(src, dst, size, shift, getKey);
            std::swap(src, dst);
        }

        // After 8 passes total, src points to the sorted data
        // Update the original pointers so *output points to sorted data
        if (src == *output) {
            // Sorted data is already in output, nothing to do
        } else if (src == *temp) {
            // Sorted data is in temp, swap output and temp pointers
            std::swap(*output, *temp);
        } else if (src == *input) {
            // Sorted data is in input (shouldn't happen with current logic, but handle it)
            std::swap(*output, *input);
        }
    }

    /**
     * Overload for uint64_t arrays (no key extractor needed)
     */
    static void sort(uint64_t** input, uint64_t** output, uint64_t** temp, size_t size) {
        sort(input, output, temp, size, [](const uint64_t& val) { return val; });
    }

    /**
     * Sort by signed 64-bit field
     */
    static void sortSigned(T** input, T** output, T** temp, size_t size,
                          std::function<int64_t(const T&)> getKey) {
        if (size <= 1) {
            if (size == 1 && *input != *output) {
                (*output)[0] = (*input)[0];
            }
            return;
        }

        // Wrapper that flips the sign bit for proper sorting
        auto adjustedKey = [getKey](const T& item) -> uint64_t {
            int64_t signedVal = getKey(item);
            uint64_t unsignedVal = *reinterpret_cast<const uint64_t*>(&signedVal);
            return unsignedVal ^ 0x8000000000000000ULL;
        };

        sort(input, output, temp, size, adjustedKey);
    }

    /**
     * Overload for signed int64_t arrays
     */
    static void sortSigned(int64_t** input, int64_t** output, int64_t** temp, size_t size) {
        sortSigned(input, output, temp, size, [](const int64_t& val) { return val; });
    }

private:
    static void countingSortByByte(const T* input, T* output, size_t size,
                                   int shift, KeyExtractor getKey) {
        const int RADIX = 256;
        int count[RADIX] = {0};

        // Count occurrences of each byte value
        for (size_t i = 0; i < size; i++) {
            uint64_t key = getKey(input[i]);
            uint8_t byte = (key >> shift) & 0xFF;
            count[byte]++;
        }

        // Convert to cumulative count
        for (int i = 1; i < RADIX; i++) {
            count[i] += count[i - 1];
        }

        // Build output array (iterate backwards for stability)
        for (int i = size - 1; i >= 0; i--) {
            uint64_t key = getKey(input[i]);
            uint8_t byte = (key >> shift) & 0xFF;
            output[--count[byte]] = input[i];
        }
    }
};

// Specialized RadixSort32 for 32-bit keys
template<typename T>
class RadixSort32 {
public:
    using KeyExtractor = std::function<uint32_t(const T&)>;

    /**
     * Radix sort for 32-bit keys
     */
    static void sort(T** input, T** output, T** temp, size_t size, KeyExtractor getKey) {
        if (size <= 1) {
            if (size == 1 && *input != *output) {
                (*output)[0] = (*input)[0];
            }
            return;
        }

        T* src;
        T* dst;

        // First pass: input -> output
        countingSortByByte(*input, *output, size, 0, getKey);

        src = *output;
        dst = *temp;

        // Remaining 3 passes (4 total for 32 bits), ping-ponging between output and temp
        for (int shift = 8; shift < 32; shift += 8) {
            countingSortByByte(src, dst, size, shift, getKey);
            std::swap(src, dst);
        }

        // After 4 passes total, update pointers so *output points to sorted data
        if (src == *output) {
            // Already in output
        } else if (src == *temp) {
            std::swap(*output, *temp);
        } else if (src == *input) {
            std::swap(*output, *input);
        }
    }

    /**
     * Overload for uint32_t arrays
     */
    static void sort(uint32_t** input, uint32_t** output, uint32_t** temp, size_t size) {
        sort(input, output, temp, size, [](const uint32_t& val) { return val; });
    }

    /**
     * Sort by signed 32-bit field
     */
    static void sortSigned(T** input, T** output, T** temp, size_t size,
                          std::function<int32_t(const T&)> getKey) {
        if (size <= 1) {
            if (size == 1 && *input != *output) {
                (*output)[0] = (*input)[0];
            }
            return;
        }

        auto adjustedKey = [getKey](const T& item) -> uint32_t {
            int32_t signedVal = getKey(item);
            uint32_t unsignedVal = *reinterpret_cast<const uint32_t*>(&signedVal);
            return unsignedVal ^ 0x80000000U;
        };

        sort(input, output, temp, size, adjustedKey);
    }

    /**
     * Overload for signed int32_t arrays
     */
    static void sortSigned(int32_t** input, int32_t** output, int32_t** temp, size_t size) {
        sortSigned(input, output, temp, size, [](const int32_t& val) { return val; });
    }

private:
    static void countingSortByByte(const T* input, T* output, size_t size,
                                   int shift, KeyExtractor getKey) {
        const int RADIX = 256;
        int count[RADIX] = {0};

        for (size_t i = 0; i < size; i++) {
            uint32_t key = getKey(input[i]);
            uint8_t byte = (key >> shift) & 0xFF;
            count[byte]++;
        }

        for (int i = 1; i < RADIX; i++) {
            count[i] += count[i - 1];
        }

        for (int i = size - 1; i >= 0; i--) {
            uint32_t key = getKey(input[i]);
            uint8_t byte = (key >> shift) & 0xFF;
            output[--count[byte]] = input[i];
        }
    }
};

// Specialized RadixSort for doubles (IEEE 754 float64)
template<typename T>
class RadixSortFloat64 {
public:
    using KeyExtractor = std::function<double(const T&)>;

    /**
     * Radix sort for double (f64) fields
     */
    static void sort(T** input, T** output, T** temp, size_t size, KeyExtractor getKey) {
        if (size <= 1) {
            if (size == 1 && *input != *output) {
                (*output)[0] = (*input)[0];
            }
            return;
        }

        // Convert doubles to sortable uint64_t representation
        auto floatToSortable = [](double val) -> uint64_t {
            uint64_t bits;
            std::memcpy(&bits, &val, sizeof(uint64_t));

            // Check sign bit (bit 63)
            if (bits & 0x8000000000000000ULL) {
                // Negative number: flip all bits
                return ~bits;
            } else {
                // Positive number: flip sign bit only
                return bits ^ 0x8000000000000000ULL;
            }
        };

        auto adjustedKey = [getKey, floatToSortable](const T& item) -> uint64_t {
            return floatToSortable(getKey(item));
        };

        RadixSort<T>::sort(input, output, temp, size, adjustedKey);
    }

    /**
     * Overload for double arrays
     */
    static void sort(double** input, double** output, double** temp, size_t size) {
        sort(input, output, temp, size, [](const double& val) { return val; });
    }
};