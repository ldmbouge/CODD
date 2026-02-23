#pragma once

#include <cassert>
#include <cstdio>

#include "FunQual.hpp"
#include "Types.hpp"
#include "Backend.hpp"

namespace gfl
{
    // Count set bits (population count)
    GFL_HOST_DEVICE
    constexpr
    i32 popcount(u32 const x) noexcept
    {
#ifdef __CUDA_ARCH__
        i32 const count = __popc(x);
#else
        i32 const count = __builtin_popcount(x);
#endif
        assert(count >= 0);
        assert(count <= 32);
        return count;
    }

    GFL_HOST_DEVICE
    constexpr
    i32 popcount(u64 const x) noexcept
    {
#ifdef __CUDA_ARCH__
        i32 const count = __popcll(x);
#else
        i32 const count = __builtin_popcountll(x);
#endif
        assert(count >= 0);
        assert(count <= 64);
        return count;
    }

    // Count leading zeros
    GFL_HOST_DEVICE
    constexpr
    i32 clz(u32 const x) noexcept
    {
        assert(x != 0);
#ifdef __CUDA_ARCH__
        i32 const count = __clz(x);
#else
        i32 const count = __builtin_clz(x);
#endif
        assert(count >= 0);
        assert(count < 32);
        return count;
    }

    GFL_HOST_DEVICE
    constexpr
    i32 clz(u64 const x) noexcept
    {
        assert(x != 0);
#ifdef __CUDA_ARCH__
        i32 const count = __clzll(x);
#else
        i32 const count = __builtin_clzll(x);
#endif
        assert(count >= 0);
        assert(count < 64);
        return count;
    }

    // Count trailing zeros (find first set - 1)
    GFL_HOST_DEVICE
    constexpr
    i32 ctz(u32 const x) noexcept
    {
        assert(x != 0);
#ifdef __CUDA_ARCH__
        i32 const count = __ffs(x) - 1;
#else
        i32 const count = __builtin_ctz(x);
#endif
        assert(count >= 0);
        assert(count < 32);
        return count;
    }

    GFL_HOST_DEVICE
    constexpr
    i32 ctz(u64 const x) noexcept
    {
        assert(x != 0);
#ifdef __CUDA_ARCH__
        i32 const count = __ffsll(x) - 1;
#else
        i32 const count = __builtin_ffsll(x) - 1;
#endif
        assert(count >= 0);
        assert(count < 64);
        return count;
    }

    // Bit reverse
    GFL_HOST_DEVICE
    constexpr
    u32 bitreverse(u32 const x) noexcept
    {
#ifdef __CUDA_ARCH__
        u32 const result = __brev(x);
#else
        u32 result = x;
        result = ((result >> 1) & 0x55555555u) | ((result & 0x55555555u) << 1);
        result = ((result >> 2) & 0x33333333u) | ((result & 0x33333333u) << 2);
        result = ((result >> 4) & 0x0F0F0F0Fu) | ((result & 0x0F0F0F0Fu) << 4);
        result = ((result >> 8) & 0x00FF00FFu) | ((result & 0x00FF00FFu) << 8);
        result = (result >> 16) | (result << 16);
#endif
        return result;
    }

    GFL_HOST_DEVICE
    constexpr
    u64 bitreverse(u64 const x) noexcept
    {
#ifdef __CUDA_ARCH__
        u64 const result = __brevll(x);
#else
        u64 result = x;
        result = ((result >> 1) & 0x5555555555555555ull) | ((result & 0x5555555555555555ull) << 1);
        result = ((result >> 2) & 0x3333333333333333ull) | ((result & 0x3333333333333333ull) << 2);
        result = ((result >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((result & 0x0F0F0F0F0F0F0F0Full) << 4);
        result = ((result >> 8) & 0x00FF00FF00FF00FFull) | ((result & 0x00FF00FF00FF00FFull) << 8);
        result = ((result >> 16) & 0x0000FFFF0000FFFFull) | ((result & 0x0000FFFF0000FFFFull) << 16);
        result = (result >> 32) | (result << 32);
#endif
        return result;
    }

    // Create bit mask for specific bit index
    template <typename T>
    GFL_HOST_DEVICE
    constexpr
    T mask(i32 const bitIdx) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(bitIdx >= 0);
        assert(bitIdx < gfl::numeric_limits<T>::digits);

        T const mask = scast<T>(1) << bitIdx;
        return mask;
    }

    // Most significant bit index
    template <typename T>
    GFL_HOST_DEVICE
    constexpr
    i32 msb(T const x) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(x != 0);

        constexpr i32 bitSize = gfl::numeric_limits<T>::digits;
        i32 const idx = bitSize - 1 - clz(x);

        assert(idx >= 0);
        assert(idx < bitSize);
        return idx;
    }

    // Least significant bit index
    template <typename T>
    GFL_HOST_DEVICE
    constexpr
    i32 lsb(T const x) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(x != 0);

        i32 const idx = ctz(x);

        assert(idx >= 0);
        assert(idx < gfl::numeric_limits<T>::digits);
        return idx;
    }

    // Flip most significant bit
    template <typename T>
    GFL_HOST_DEVICE
    constexpr
    T flipMsb(T const x) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(x != 0);

        T const result = x ^ mask<T>(msb(x));
        return result;
    }

    // Flip least significant bit
    template <typename T>
    GFL_HOST_DEVICE
    constexpr
    T flipLsb(T const x) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(x != 0);

        T const result = x ^ mask<T>(lsb(x));
        return result;
    }



    template<typename T>
    GFL_HOST_DEVICE constexpr
    bool test(T const x, i32 const bitIdx) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);
        assert(bitIdx >= 0);
        assert(bitIdx < gfl::numeric_limits<T>::digits);
        return (x & mask<T>(bitIdx)) != 0;
    }

    // Suffix mask inclusive (bits from bitIdx to LSB)
    template<typename T>
    GFL_HOST_DEVICE
    constexpr
    T suffixMaskInclusive(i32 const bitIdx) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);

        i32 constexpr bitSize = numeric_limits<T>::digits;
        T constexpr fullMask = numeric_limits<T>::max();
        T constexpr emptyMask = scast<T>(0);

        bool const isLow = bitIdx < 0;
        T const isLowMask = isLow ? fullMask : emptyMask;

        bool const isHigh = bitSize <= bitIdx;
        //T const isHighMask = isHigh ? emptyMask : emptyMask;

        bool const isFine = (not isLow) and (not isHigh);
        T const isFineMask = isFine ? fullMask << bitIdx : emptyMask;

        return isLowMask | isFineMask;
    }

    // Prefix mask inclusive (bits from MSB to bitIdx)
    template<typename T>
    GFL_HOST_DEVICE
    constexpr
    T prefixMaskInclusive(i32 const bitIdx) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);

        i32 constexpr bitSize = numeric_limits<T>::digits;
        T constexpr fullMask = numeric_limits<T>::max();
        T constexpr emptyMask = scast<T>(0);

        bool const isLow = bitIdx < 0;
        // T const isLowMask = isLow ? emptyMask : emptyMask;

        bool const isHigh = bitSize <= bitIdx;
        T const isHighMask = isHigh ? fullMask : emptyMask;

        bool const isFine = (not isLow) and (not isHigh);
        T const isFineMask = isFine ? fullMask >> (bitSize - bitIdx - 1) : emptyMask;

        return isHighMask | isFineMask;
    }

    template<typename T>
    GFL_HOST_DEVICE
    static
    void printBits(T const x) noexcept
    {
        static_assert(std::is_integral_v<T>);
        static_assert(std::is_unsigned_v<T>);

        i32 constexpr bitSize = numeric_limits<T>::digits;
        for(i32 bitIdx = 0;bitIdx < bitSize; ++bitIdx)
        {
            printf(test(x,bitIdx) ? "1" : "0");
        }
    }
}