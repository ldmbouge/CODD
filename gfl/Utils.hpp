#pragma once

#include <cassert>
#include <cstdio>
#include <type_traits>
#include <cmath>

#include "Types.hpp"
#include "Common.hpp"
#include "Backend.hpp"

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace gfl
{
    // Formatting
    template <typename T>
    GFL_HOST_DEVICE
    void printVal(T const v) noexcept
    {
        if constexpr (std::is_integral_v<T>)
        {
            if constexpr (std::is_signed_v<T>)
            {
                printf("%ld", static_cast<i64>(v));
            }
            else
            {
                printf("%lu", static_cast<u64>(v));
            }
        }
        else if constexpr (std::is_floating_point_v<T>)
        {
            printf("%f", static_cast<f64>(v));
        }
        else if constexpr (std::is_pointer_v<T>)
        {
            printf("%p", static_cast<void const*>(v));
        }
        else
        {
            static_assert(std::is_arithmetic_v<T> or std::is_pointer_v<T>, "Unsupported type");
        }
    }

    GFL_HOST_DEVICE inline
    void printMemSize(i64 const size) noexcept
    {
        char const* const units[] = {" B", "KB", "MB", "GB"};
        auto uIdx = 0;
        auto dSize = static_cast<double>(size);
        while (dSize >= 1024 and uIdx < 3)
        {
            dSize /= 1024.0;
            uIdx += 1;
        }
        printf("%7.2f %s", dSize, units[uIdx]);
    }

    // Math
    template <typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    double div(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_integral_v<TIn1>);
        static_assert(std::is_integral_v<TIn2>);
        assert(a >= 0);
        assert(b > 0);
        auto const _a = static_cast<double>(a);
        auto const _b = static_cast<double>(b);
        return _a / _b;
    }

    template <typename TOut, typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    TOut roundUpDivPosInt(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_integral_v<TIn1>);
        static_assert(std::is_integral_v<TIn2>);
        assert(a >= 0);
        assert(b > 0);
        using CommonT = std::common_type_t<TIn1, TIn2>;
        CommonT const _a = static_cast<CommonT>(a);
        CommonT const _b = static_cast<CommonT>(b);
        return static_cast<TOut>((_a + _b - 1) / _b);
    }

    template <typename TOut, typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    TOut roundDownToMultiple(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_integral_v<TIn1>);
        static_assert(std::is_integral_v<TIn2>);
        static_assert(std::is_integral_v<TOut>);
        assert(a >= 0);
        assert(b > 0);
        using CommonT = std::common_type_t<TIn1, TIn2>;
        CommonT const _a = static_cast<CommonT>(a);
        CommonT const _b = static_cast<CommonT>(b);
        return static_cast<TOut>((_a / _b) * _b);
    }

    template <typename TOut, typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    TOut roundUpToMultiple(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_integral_v<TIn1>);
        static_assert(std::is_integral_v<TIn2>);
        static_assert(std::is_integral_v<TOut>);
        assert(a >= 0);
        assert(b > 0);
        using CommonT = std::common_type_t<TIn1, TIn2>;
        CommonT const _a = static_cast<CommonT>(a);
        CommonT const _b = static_cast<CommonT>(b);
        return static_cast<TOut>(((_a + _b - 1) / _b) * _b);
    }

    template <typename TOut, typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    TOut min(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_arithmetic_v<TIn1>);
        static_assert(std::is_arithmetic_v<TIn2>);
        static_assert(std::is_arithmetic_v<TOut>);
        using CommonT = std::common_type_t<TIn1, TIn2>;
        CommonT const _a = static_cast<CommonT>(a);
        CommonT const _b = static_cast<CommonT>(b);
        return static_cast<TOut>( _a < _b ? _a : _b);
    }

    template <typename TOut, typename TIn1, typename TIn2, typename... TRest>
    GFL_HOST_DEVICE constexpr TOut min(TIn1 const a, TIn2 const b, TRest const... rest)
    {
        TOut const ab = min<TOut>(a, b);
        return min<TOut>(ab, rest...);
    }

    template <typename TOut, typename TIn1, typename TIn2>
    GFL_HOST_DEVICE constexpr
    TOut max(TIn1 const a, TIn2 const b)
    {
        static_assert(std::is_arithmetic_v<TIn1>);
        static_assert(std::is_arithmetic_v<TIn2>);
        static_assert(std::is_arithmetic_v<TOut>);
        using CommonT = std::common_type_t<TIn1, TIn2>;
        CommonT const _a = static_cast<CommonT>(a);
        CommonT const _b = static_cast<CommonT>(b);
        return static_cast<TOut>( _a > _b ? _a : _b);
    }

    template <typename TOut, typename TIn1, typename TIn2, typename... TRest>
    GFL_HOST_DEVICE constexpr TOut max(TIn1 const a, TIn2 const b, TRest const... rest)
    {
        TOut const ab = max<TOut>(a, b);
        return max<TOut>(ab, rest...);
    }

    // Bits
    template <typename T>
    GFL_HOST_DEVICE constexpr
    i32 popcount(T x);

    template <>
    GFL_HOST_DEVICE constexpr
    i32 popcount<u32>(u32 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __popc(x);
#else
        result = __builtin_popcount(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 popcount(u64 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __popcll(x);
#else
        result = __builtin_popcountll(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 popcount<unsigned long long>(unsigned long long const x)
    {
        return popcount(static_cast<u64>(x));
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    i32 clz(T x);

    template <>
    GFL_HOST_DEVICE constexpr
    i32 clz<u32>(u32 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __clz(x);
#else
        result = __builtin_clz(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 clz<u64>(u64 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __clzll(x);
#else
        result = __builtin_clzll(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 clz<unsigned long long>(unsigned long long const x)
    {
        return clz(static_cast<u64>(x));
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr i32 ffs(T x);

    template <>
    GFL_HOST_DEVICE constexpr
    i32 ffs<u32>(u32 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __ffs(x);
#else
        result = __builtin_ffs(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 ffs<u64>(u64 const x)
    {
        i32 result;
#ifdef __CUDA_ARCH__
        result = __ffsll(x);
#else
        result = __builtin_ffsll(x);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    i32 ffs<unsigned long long>(unsigned long long const x)
    {
        static_assert(sizeof(unsigned long long) == sizeof(u64));
        return ffs(static_cast<u64>(x));
    }

    template <typename T>
    GFL_HOST_DEVICE
    T bitreverse(T x);

    template <>
    GFL_HOST_DEVICE constexpr
    u32 bitreverse<u32>(u32 const x)
    {
        u32 result;
#ifdef __CUDA_ARCH__
        result = __brev(x);
#else
        result = x;
        result = ((result >> 1) & 0x55555555u) | ((result & 0x55555555u) << 1);
        result = ((result >> 2) & 0x33333333u) | ((result & 0x33333333u) << 2);
        result = ((result >> 4) & 0x0F0F0F0Fu) | ((result & 0x0F0F0F0Fu) << 4);
        result = ((result >> 8) & 0x00FF00FFu) | ((result & 0x00FF00FFu) << 8);
        result = (result >> 16) | (result << 16);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    u64 bitreverse<u64>(u64 const x)
    {
        u64 result;
#ifdef __CUDA_ARCH__
        result = __brevll(x);
#else
        result = x;
        result = ((result >> 1) & 0x5555555555555555ull) | ((result & 0x5555555555555555ull) << 1);
        result = ((result >> 2) & 0x3333333333333333ull) | ((result & 0x3333333333333333ull) << 2);
        result = ((result >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((result & 0x0F0F0F0F0F0F0F0Full) << 4);
        result = ((result >> 8) & 0x00FF00FF00FF00FFull) | ((result & 0x00FF00FF00FF00FFull) << 8);
        result = ((result >> 16) & 0x0000FFFF0000FFFFull) | ((result & 0x0000FFFF0000FFFFull) << 16);
        result = (result >> 32) | (result << 32);
#endif
        return result;
    }

    template <>
    GFL_HOST_DEVICE constexpr
    unsigned long long bitreverse<unsigned long long>(unsigned long long const x)
    {
        static_assert(sizeof(unsigned long long) == sizeof(u64));
        return bitreverse(static_cast<u64>(x));
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    i32 msb(T const x) noexcept
    {
        assert(x != 0);
        constexpr i32 biggestIdx = (sizeof(T) * 8) - 1;
        i32 const idx = biggestIdx - clz<T>(x);
        return idx;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    i32 lsb(T const x) noexcept
    {
        assert(x != 0);
        i32 const idx = ffs<T>(x) - 1;
        return idx;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T flipMsb(T const x) noexcept
    {
        assert(x != 0);
        T result = x;
        T const mask = T{1} << msb(x);
        result ^= mask;
        return result;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T flipLsb(T const x) noexcept
    {
        assert(x != 0);
        T result = x;
        T const mask = T{1} << lsb(x);
        result ^= mask;
        return result;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T maskFilledFrom(i32 const bIdx) noexcept
    {
        static_assert(std::is_unsigned_v<T>);
        T constexpr fullMask = numeric_limits<T>::max();
        i32 constexpr maskBitSize = sizeof(T) * 8;

        T const c0 = bIdx < 0 ? fullMask : 0;
        T const r0 = fullMask;

        T const c1 = 0 <= bIdx and bIdx < maskBitSize ? fullMask : 0;
        T const r1 = fullMask << bIdx;

        T const c2 = maskBitSize <= bIdx ? fullMask : 0;
        T const r2 = 0;

        return (c0 & r0) | (c1 & r1) | (c2 & r2);
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T maskFilledThrough(i32 const bIdx) noexcept
    {
        static_assert(std::is_unsigned_v<T>);

        T constexpr fullMask = numeric_limits<T>::max();
        i32 constexpr maskBitSize = sizeof(T) * 8;

        T const c0 = bIdx < 0 ? fullMask : 0;
        T const r0 = 0;

        T const c1 = 0 <= bIdx and bIdx < maskBitSize ? fullMask : 0;
        T const r1 = fullMask >> maskBitSize - bIdx;

        T const c2 = maskBitSize <= bIdx ? fullMask : 0;
        T const r2 = fullMask;

        return (c0 & r0) | (c1 & r1) | (c2 & r2);
    }

    // Kernels
#ifdef __CUDACC__
    GFL_DEVICE inline
    i32 getSharedMemSize() noexcept
    {
        i32 size;
        asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(size));
        return size;
    }
#endif

    // Misc
    GFL_HOST_DEVICE inline
    void abort()
    {
#ifdef __CUDA_ARCH__
        __trap();
#else
        std::abort();
#endif
    }

    template<typename T>
    GFL_HOST_DEVICE inline
    void getBeginEnd(T & begin, T & end, i64 index, i64 workers, i64 jobs) noexcept
    {
        auto const jobsPerWorker = roundUpDivPosInt<T>(jobs, workers);
        begin = jobsPerWorker * index;
        end = min<T>(jobs, begin + jobsPerWorker);
    }

    inline
    std::tuple<i32,i32,i32> calcGridSize(i64 const jobs) noexcept
    {
        i32 constexpr xMax = 2147483647;
        i32 constexpr yMax = 65535;
        i32 constexpr zMax = 65535;

        i64 xDim = min<i64>(jobs, xMax);
        i64 remain = roundUpDivPosInt<i64>(jobs, xMax);
        i64 yDim = min<i64>(remain, yMax);
        remain = roundUpDivPosInt<i64>(remain, yMax);
        i64 zDim = min<i64>(remain, zMax);
        remain = roundUpDivPosInt<i64>(remain, zMax);

        assert(remain == 1);

        return {xDim,yDim,zDim};
    }

#ifdef __CUDACC__
    GFL_DEVICE inline
    i64 calcBlockIdx() noexcept
    {
        i64 const xDim = gridDim.x;
        i64 const yDim = gridDim.y;

        return blockIdx.x +
               xDim * blockIdx.y +
               xDim * yDim * blockIdx.z;
    }

    GFL_DEVICE inline
    i32 laneIdx() noexcept
    {
        unsigned int lIdx;
        asm volatile("mov.u32 %0, %laneid;" : "=r"(lIdx));
        return lIdx;
    }

    inline
    void checkCudaError(cudaError_t err, const char *file, const int line)
    {
        if (err != cudaSuccess)
        {
            const char * errorStr = NULL;
            errorStr = cudaGetErrorString(err);
            fprintf(stderr, "CUDA API error = %04d \"%s\" from %s:%i\n", err, errorStr, file, line);
            exit(EXIT_FAILURE);
        }
    }

#define CHECK_CUDA_ERROR(err) gfl::checkCudaError(err, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() CHECK_CUDA_ERROR(cudaGetLastError())
#endif
}
