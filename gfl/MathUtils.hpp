#pragma once

#include <cassert>
#include <type_traits>

#include "Types.hpp"
#include "FunQual.hpp"

namespace gfl
{
    // Integer division returning double
    template <typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    double div(T1 const a, T2 const b)
    {
        static_assert(std::is_integral_v<T1>);
        static_assert(std::is_integral_v<T2>);
        assert(b != 0);
        double const a_ = scast<double>(a);
        double const b_ = scast<double>(b);
        return a_ / b_;
    }

    // Ceiling division: (a + b - 1) / b
    template <typename TOut, typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    TOut ceil(T1 const a, T2 const b)
    {
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

    // Round down to nearest multiple of b
    template <typename TOut, typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    TOut roundDown(T1 const a, T2 const b)
    {
        static_assert(std::is_integral_v<T1>);
        static_assert(std::is_integral_v<T2>);
        static_assert(std::is_integral_v<TOut>);
        assert(a >= 0);
        assert(b > 0);
        using TComm = std::common_type_t<T1, T2>;
        TComm const a_ = scast<TComm>(a);
        TComm const b_ = scast<TComm>(b);
        return scast<TOut>((a_ / b_) * b_);
    }

    // Round up to nearest multiple of b
    template <typename TOut, typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    TOut roundUp(T1 const a, T2 const b)
    {
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

    // Minimum of two values
    template <typename TOut, typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    TOut min(T1 const a, T2 const b)
    {
        static_assert(std::is_arithmetic_v<T1>);
        static_assert(std::is_arithmetic_v<T2>);
        static_assert(std::is_arithmetic_v<TOut>);
        using TComm = std::common_type_t<T1, T2>;
        TComm const a_ = scast<TComm>(a);
        TComm const b_ = scast<TComm>(b);
        return scast<TOut>(a_ < b_ ? a_ : b_);
    }

    // Minimum of multiple values
    template <typename TOut, typename T1, typename T2, typename... TRest>
    GFL_HOST_DEVICE
    constexpr
    TOut min(T1 const a, T2 const b, TRest const... rest)
    {
        static_assert((std::is_arithmetic_v<TRest> && ...));
        TOut const pMin = min<TOut>(a, b);
        return min<TOut>(pMin, rest...);
    }

    // Maximum of two values
    template <typename TOut, typename T1, typename T2>
    GFL_HOST_DEVICE
    constexpr
    TOut max(T1 const a, T2 const b)
    {
        static_assert(std::is_arithmetic_v<T1>);
        static_assert(std::is_arithmetic_v<T2>);
        static_assert(std::is_arithmetic_v<TOut>);
        using TComm = std::common_type_t<T1, T2>;
        TComm const a_ = scast<TComm>(a);
        TComm const b_ = scast<TComm>(b);
        return scast<TOut>(a_ > b_ ? a_ : b_);
    }

    // Maximum of multiple values
    template <typename TOut, typename T1, typename T2, typename... TRest>
    GFL_HOST_DEVICE
    constexpr
    TOut max(T1 const a, T2 const b, TRest const... rest)
    {
        static_assert((std::is_arithmetic_v<TRest> && ...));
        TOut const pMax = max<TOut>(a, b);
        return max<TOut>(pMax, rest...);
    }
}