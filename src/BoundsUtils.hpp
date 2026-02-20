#pragma once

#include <GFL.hpp>

// Minimization TSP (Opt 4)
// Primal 5,7,+inf (Better 5) Better is Smaller
// Dual -inf,2,3 (Better 3) Better is Larger

// Maximization KS (Opt 10)
// Primal -inf,7,9 (Better 9) Better is Larger
// Dual 13,14,+inf (Better 13) Better is Smaller

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isBetter(gfl::f64 const a, gfl::f64 const b) noexcept
{
    if constexpr (Model::is_maximization)
        return a > b;
    else
        return a < b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isBetterEq(gfl::f64 const a, gfl::f64 const b) noexcept
{
    if constexpr (Model::is_maximization)
        return a >= b;
    else
        return a <= b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isWorse(gfl::f64 const a, gfl::f64 const b) noexcept
{
    if constexpr (Model::is_maximization)
        return a < b;
    else
        return a > b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
bool isWorseEq(gfl::f64 const a, gfl::f64 const b) noexcept
{
    if constexpr (Model::is_maximization)
        return a <= b;
    else
        return a >= b;
}

template<typename Model>
constexpr
gfl::f64 better(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isBetter<Model>(a, b) ? a : b;
}

template<typename Model>
constexpr
gfl::f64 betterEq(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isBetterEq<Model>(a, b) ? a : b;
}

template<typename Model>
constexpr
gfl::f64 worse(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isWorse<Model>(a, b) ? a : b;
}

template<typename Model>
constexpr
gfl::f64 worseEq(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isWorseEq<Model>(a, b) ? a : b;
}

template<typename Model>
GFL_HOST_DEVICE constexpr
gfl::f64 best() noexcept
{
    // Do NOT use numeric_limits<gfl::f64>::max() and/or
    // numeric_limits<gfl::f64>::lowest(). They are out of scale,
    // and do not change when bounds are added/subtracted.
    constexpr gfl::f64 big_value = 1000000000.0; // 10^9
    if constexpr (Model::is_maximization)
        return big_value;
    else
        return -big_value;
}

template<typename Model>
constexpr
gfl::f64 worst() noexcept
{
    return -best<Model>();
}

template<typename Model>
constexpr
bool isValid(gfl::f64 const a) noexcept
{
    return a != worst<Model>() and a != best<Model>();
}

template<typename Model>
constexpr
gfl::f64 score(gfl::f64 const a) noexcept
{
    if constexpr (Model::is_maximization)
        return -a;
    else
        return a;
}

template<typename Model>
constexpr
gfl::f64 isTighter(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isWorse<Model>(a, b);
}

template<typename Model>
constexpr
gfl::f64 tighter(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return worse<Model>(a, b);
}


template<typename Model>
constexpr
gfl::f64 isLooser(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return isBetter<Model>(a, b);
}

template<typename Model>
constexpr
gfl::f64 looser(gfl::f64 const a, gfl::f64 const b) noexcept
{
    return better<Model>(a, b);
}

