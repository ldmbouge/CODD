#pragma once

#include "Common.hpp"

//  Minimization TSP (Opt 4)
//  Primal 5,7,+inf (Better 5) Better is Smaller
//  Dual -inf,2,3 (Better 3) Better is Larger

// Maximization KS (Opt 10)
// Primal -inf,7,9 (Better 9) Better is Larger
// Dual 13,14,+inf (Better 13) Better is Smaller

template<typename Model>
GFL_HOST_DEVICE
constexpr
bool isBetter(double const & c1, double const & c2) noexcept
{
    if constexpr (Model::is_maximization)
        return c1 > c2;
    else
        return c1 < c2;
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
bool isBetterEq(double const & c1, double const & c2)  noexcept
{
    if constexpr (Model::is_maximization)
        return c1 >= c2;
    else
        return c1 <= c2;
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
bool isWorst(double const & c1, double const & c2) noexcept
{
    if constexpr (Model::is_maximization)
        return c1 < c2;
    else
        return c1 > c2;
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
bool isWorstEq(double const & c1, double const & c2) noexcept
{
    if constexpr (Model::is_maximization)
        return c1 <= c2;
    else
        return c1 >= c2;
}

template<typename Model>
constexpr
double calcBetter(double const & c1, double const & c2) noexcept
{
    return isBetter<Model>(c1, c2) ? c1 : c2;
}

template<typename Model>
constexpr
double calcBetterEq(double const & c1, double const & c2) noexcept
{
    return isBetterEq<Model>(c1, c2) ? c1 : c2;
}


template<typename Model>
constexpr
double calcWorst(double const & c1, double const & c2) noexcept
{
    return isWorst<Model>(c1, c2) ? c1 : c2;
}

template<typename Model>
constexpr
double calcWorstEq(double const & c1, double const & c2) noexcept
{
    return isWorstEq<Model>(c1, c2) ? c1 : c2;
}


template<typename Model>
GFL_HOST_DEVICE
constexpr
double bestValue() noexcept
{
    // IMPORTANT
    // Do not use numeric_limits<double>::max() and/or
    // numeric_limits<double>::lowest(). They are out of scale,
    // and do not change when bounds are added/subtracted.
    constexpr double bigValue = 10000000000.0; // 1M
    if constexpr (Model::is_maximization)
    {
        return bigValue;
    }
    else
    {
        return -bigValue;
    }
}

template<typename Model>
constexpr
double worstValue() noexcept
{
   return -bestValue<Model>();
}

template<typename Model>
constexpr
bool isValid(double const c) noexcept
{
    return c != worstValue<Model>() and c != bestValue<Model>();
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
double absDiffWithBest(double const c) noexcept
{
    double const diff = bestValue<Model>() - (bestValue<Model>() - c);
    return diff >= 0 ? diff : -diff;
}




