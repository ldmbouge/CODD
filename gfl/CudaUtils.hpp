#pragma once

#include <cassert>
#include <cstdio>
#include <type_traits>
#include <cmath>

#include "Types.hpp"
#include "FunQual.hpp"
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
    GFL_HOST_DEVICE constexpr
    TOut max(TIn1 const a, TIn2 const b, TRest const... rest)
    {
        TOut const ab = max<TOut>(a, b);
        return max<TOut>(ab, rest...);
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    f32 minOverMax(T const a, T const b)
    {
        return static_cast<f32>(min(a,b)) /
               static_cast<f32>(max(a,b));
    }

    // Ordered pairs
    template <typename T>
    GFL_HOST_DEVICE constexpr
    T countOrderedPairs(T const n)
    {
        return n * (n + 1) / 2;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T offsetFor(T const i, T const n)
    {
        // Number of pairs with first element < i
        // Pairs starting with 0: (0,1),(0,2),...,(0,n-1) -> n-1 -> n-0-1
        // Pairs starting with 1: (1,2),(1,3),...,(1,n-1) -> n-2 -> n-1-1
        // Pairs starting with 2: (2,3),(2,4),...,(2,n-1) -> n-3 -> n-2- 1
        // ...
        // Sum for 0 < i < n-1 of (n-i-1) -> i*(2*n-i-1)/2;
        return i * (2 * n - i - 1) / 2;
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    T pairToIdx(T const i, T const j, T const n)
    {
        return offsetFor(i, n) + (j - i - 1);
    }

    template <typename T>
    GFL_HOST_DEVICE constexpr
    tuple<T,T> idxToPair(T const idx, T const n)
    {
        // Since offsetFor(i,n) <= idx < offsetFor(i+1,n) and offsetFor(i,n) = i*(2*n-i-1)/2
        // We solve i*(2*n-i-1)/2 = idx
        T const b = 2 * n - 1;
        T const s = floor(sqrt(b * b - 8 * idx));
        T const i = (b - s) / 2;
        T const j = i + 1 + (idx - offsetFor(i, n));
        return {i,j};
    }

    // Bits


    // Kernels
#ifdef __CUDACC__
    GFL_DEVICE inline
    i32 sharedMemSize() noexcept
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
        assert(index < workers);
        assert(0 <= workers);
        assert(0 <= jobs);

        auto const jobsPerWorker = jobs / workers;
        auto const remainder = jobs % workers;

        // First 'remainder' workers get one extra job
        auto const extra = (index < remainder) ? 1 : 0;

        begin = index * jobsPerWorker + min<i64>(index, remainder);
        end = begin + jobsPerWorker + extra;
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
