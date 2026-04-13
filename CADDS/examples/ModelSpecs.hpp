#pragma once

#include <Contexts.hpp>
#include <GFL.hpp>
#include <concepts>

template<typename S>
concept IsState = requires(S const & s) {
  { S::equal(s, s) } -> std::same_as<bool>;
  { S::hash(s) }     -> std::same_as<std::size_t>;
};

template<typename L>
concept IsLabels = requires(L const & l) {
  { l.begin() };
  { l.end() };
};

template<typename M, typename S, typename L>
concept IsDP = requires(M const & m,
                        S const & s,
                        gfl::i32 l,
                        gfl::f64 p,
                        gfl::f64 d,
                        DDContext c) {
  { M::is_maximization } -> std::convertible_to<bool>;

  { m.initial() }   -> std::same_as<S>;
  { m.isTarget(s) } -> std::same_as<bool>;

  { m.lgf(s, p, d, c) } -> std::same_as<L>;
  { m.stf(s, l) }       -> std::same_as<gfl::optional<S>>;
  { m.scf(s, l) }       -> std::same_as<gfl::f64>;
};

template<typename M, typename S>
concept HasMerge = (M::has_merge == false) or
                   requires(M const & m, S const & s) {
  { m.smf(s, s) } -> std::same_as<S>;
};

template<typename M, typename S>
concept HasHeur = (M::has_heur == false) or
                  requires(M const & m, S const & s, HContext c) {
  { m.h(s, c) } -> std::same_as<double>;
};

template<typename M, typename S>
concept HasDom = (M::has_dom == false) or
                 requires(S const & s) {
  { M::dom(s, s) }  -> std::same_as<bool>;
  { M::domHash(s) } -> std::same_as<gfl::u64>;
};

template<typename M>
concept IsModel = IsDP<M, typename M::State, typename M::OutLabels> and
  requires { { M::has_merge } -> std::convertible_to<bool>;} and HasMerge<M, typename M::State> and
  requires { { M::has_heur }  -> std::convertible_to<bool>;} and HasHeur<M, typename M::State> and
  requires { { M::has_dom }   -> std::convertible_to<bool>;} and HasDom<M, typename M::State>;
