#pragma once

#include <concepts>
#include <Backend.hpp>

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

template<typename M, typename S>
concept IsDP = requires(M const & m, S const & s, int l, DDContext c)
{
    { m.initial() }   -> std::same_as<S>;
    { m.target() }    -> std::same_as<S>;
    { m.isTarget(s) } -> std::same_as<bool>;

    { m.stf(s,l) }   -> std::same_as<gfl::optional<S>>;
    { m.scf(s,l) }   -> std::same_as<double>;
};

template<typename M,  typename S, typename L>
concept HasSimpleLGF = requires(M const & m, S const & s, DDContext c)
{
    { m.lgf(s,c) }   -> std::same_as<L>;
};

template<typename M,  typename S, typename L>
concept HasComplexLGF = requires(M const & m, S const & s, DDContext c, double pBound, double dBound)
{
    { m.lgf(s,c,pBound,dBound) }   -> std::same_as<L>;
};

template<typename M,  typename S, typename L>
concept HasLGF =
    (M::has_simple_lgf == true  and HasSimpleLGF<M,S,L>) or
    (M::has_simple_lgf == false and HasComplexLGF<M,S,L>);

template<typename M>
concept HasCmp = requires(double const & d)
{
    { M::better(d,d) }    -> std::same_as<bool>;
    { M::betterEq(d, d) } -> std::same_as<bool>;
    { M::bestValue() }    -> std::same_as<double>;
    { M::worstValue() }   -> std::same_as<double>;
};

template<typename M, typename S>
concept HasMerge =
    (M::has_merge == false) or
    requires(M const & m, S const & s)
    {
        { m.smf(s,s) } -> std::same_as<gfl::optional<S>>;
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
        { M::domEq(s,s) } -> std::same_as<bool>;
    };

template<typename M>
concept IsModel =
    IsState<typename M::State> and
    IsLabels<typename M::Labels> and
    IsDP<M,typename M::State> and HasCmp<M> and HasLGF<M,typename M::State,typename M::Labels> and
    requires { { M::has_merge }; } and HasMerge<M,typename M::State> and
    requires { { M::has_local }; } and HasLocal<M,typename M::State> and
    requires { { M::has_dom }; }   and HasDom<M,typename M::State>;