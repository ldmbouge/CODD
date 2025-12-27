#pragma once

#include "Common.hpp"

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
bool isWorse(double const & c1, double const & c2) noexcept
{
    if constexpr (Model::is_maximization)
        return c1 < c2;
    else
        return c1 > c2;
}

template<typename Model>
constexpr
double calcBetter(double const & c1, double const & c2) noexcept
{
    return isBetter<Model>(c1, c2) ? c1 : c2;
}

template<typename Model>
constexpr
double calcWorst(double const & c1, double const & c2) noexcept
{
    return not isBetter<Model>(c1, c2) ? c1 : c2;
}

template<typename Model>
constexpr
double calcTighterBound(double const & c1, double const & c2) noexcept
{
    return calcWorst<Model>(c1, c2);
}

template<typename Model>
constexpr
double isTighterBound(double const & c1, double const & c2) noexcept
{
    return isWorse<Model>(c1, c2);
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
double bestBound() noexcept
{
    // IMPORTANT
    // Do not use numeric_limits<double>::max() and/or
    // numeric_limits<double>::lowest(). They are out of scale,
    // and do not change when bounds are added/subtracted.
    constexpr double bigValue = 10000000000.0; // 1 Billion
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
double worstBound() noexcept
{
   return -bestBound<Model>();
}

template<typename Model>
constexpr
bool isValid(double const c) noexcept
{
    return c != worstBound<Model>() and c != bestBound<Model>();
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
double absDiffWithBest(double const c) noexcept
{
    return abs(bestBound<Model>() - c);
}




