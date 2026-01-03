#pragma once

#include <concepts>
#include <Backend.hpp>
#include <Types.hpp>

enum DDContext : int;
enum LocalContext : int;

template<typename S>
concept IsState = requires(S const & s)
{
    { S::equal(s,s) } -> std::same_as<bool>;
    { S::hash(s) }    -> std::same_as<std::size_t>;
};

template<typename L>
concept IsLabels = requires(L const & l)
{
    { l.begin() };
    { l.end() };
};

template<typename M, typename S, typename L>
concept IsDP = requires(M const & m, S const & s, int l, DDContext c, double pBound, double dBound)
{
    { M::is_maximization } -> std::convertible_to<bool>;

    { m.initial() }   -> std::same_as<S>;
    { m.target() }    -> std::same_as<S>;
    { m.isTarget(s) } -> std::same_as<bool>;

    { m.lgf(s,c,pBound,dBound) }   -> std::same_as<L>;
    { m.stf(s,l) }   -> std::same_as<gfl::optional<S>>;
    { m.scf(s,l) }   -> std::same_as<double>;
};

template<typename M>
concept HasObj = requires(double const & d)
{
    { M::isBetter(d,d) }   -> std::same_as<bool>;
    { M::isBetterEq(d,d) } -> std::same_as<bool>;
    { M::calcBetter(d,d) } -> std::same_as<double>;
    { M::calcWorst(d,d) }  -> std::same_as<double>;
    { M::bestValue() }     -> std::same_as<double>;
    { M::worstValue() }    -> std::same_as<double>;
};

template<typename M, typename S>
concept HasMerge =
    (M::has_merge == false) or
    requires(M const & m, S const & s)
    {
        { m.smf(s,s) } -> std::same_as<S>;
        { m.ssf(s,s) } -> std::same_as<gfl::f32>;
    };

template<typename M, typename S>
concept HasLocal =
    (M::has_local == false) or
    requires(M const & m, S const & s, LocalContext c)
    {
        { m.local(s,c) } -> std::same_as<double>;
    };

template<typename M, typename S>
concept HasDom =
    (M::has_dom == false) or
    requires(S const & s)
    {
        { M::dom(s,s) }   -> std::same_as<bool>;
        { M::domHash(s) } -> std::same_as<std::size_t>;
    };


template<typename M>
concept IsModel =
    std::is_trivially_copyable_v<typename M::State>  and IsState<typename M::State> and
    std::is_trivially_copyable_v<typename M::Labels> and IsLabels<typename M::Labels> and
    IsDP<M,typename M::State, typename M::Labels> and
    requires { { M::has_merge } -> std::convertible_to<bool>; } and HasMerge<M,typename M::State> and
    requires { { M::has_local } -> std::convertible_to<bool>; } and HasLocal<M,typename M::State> and
    requires { { M::has_dom }   -> std::convertible_to<bool>; } and HasDom<M,typename M::State>;
