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
GFL_HOST_DEVICE
constexpr
double bestBound() noexcept
{
    if constexpr (Model::is_maximization)
        return gfl::numeric_limits<double>::max();
    else
        return gfl::numeric_limits<double>::lowest();
}

template<typename Model>
constexpr
double worstBound() noexcept
{
    if constexpr (Model::is_maximization)
        return gfl::numeric_limits<double>::lowest();
    else
        return gfl::numeric_limits<double>::max();
}

template<typename Model>
constexpr
bool isValid(double const c) noexcept
{
    return c != worstBound<Model>();
}

template<typename Model>
GFL_HOST_DEVICE
constexpr
double absDiffWithBest(double const c) noexcept
{
    return abs(bestBound<Model>() - c);
}




