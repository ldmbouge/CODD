#pragma once

#include <GFL.hpp>
#include "ModelSpecs.hpp"
#include "TsptwData.hpp"

class Tsptw : public TsptwData {

public:
  static constexpr bool is_maximization = false;
  using OutLabels = gfl::BitSet<gfl::BitSet<>::numWords(BranchFactor)>;

  template<typename Allocator>
  Tsptw(std::string const & instancePath, Allocator & alloc) :
    TsptwData(instancePath, alloc)
  {}

  class State {
    CitiesSet U;
    gfl::i32 e;
    gfl::i32 t;
    gfl::i32 hops;

  public:
    GFL_HOST_DEVICE
    State() : U(), e(0), t(0), hops(0) {}

    GFL_HOST_DEVICE
    State(CitiesSet const & U, gfl::i32 const e, gfl::i32 const t, gfl::i32 const hops) :
      U(U), e(e), t(t), hops(hops)
    {}

    GFL_HOST_DEVICE static
    bool equal(State const & s1, State const & s2) noexcept {
      return s1.U == s2.U and
             s1.e == s2.e and
             s1.t == s2.t and
             s1.hops == s2.hops;
    }

    GFL_HOST_DEVICE static
    gfl::u64 hash(State const & s) noexcept {
      using namespace gfl;
      u64 seed = 0;
      hashCombine(seed, s.U.hash());
      hashCombine(seed, s.e);
      hashCombine(seed, s.t);
      hashCombine(seed, s.hops);
      return seed;
    }

    friend
    std::ostream & operator<<(std::ostream & os, State const & s) {
      return os << "<"
                << "U=" << s.U << ","
                << "E=" << s.e << ","
                << "T=" << s.t << ","
                << "HOPS=" << s.hops
                << ">";
    }

    friend class Tsptw;
  };

  State initial() const noexcept {
    State const s = State(CitiesSet(depot + 1, n - 1), depot, 0, 0);
    return s;
  }

  GFL_HOST_DEVICE
  bool isTarget(State const & s) const noexcept {
    bool const c1 = s.e == depot;
    bool const c2 = s.hops == n;
    return c1 and c2;
  }

  GFL_HOST_DEVICE
  OutLabels lgf(State const & s, double pBound, double dBound, DDContext ddCtx) const noexcept {
    using namespace gfl;
    if (s.hops >= n - 1)
      return (s.t + d[s.e][depot] <= tw[depot].e) and s.e != depot ? CitiesSet(depot) : CitiesSet();
    else
      return filter(s.U, [this, s](i32 const & u) { return s.t + d[s.e][u] <= tw[u].e; });
  }

  GFL_HOST_DEVICE
  gfl::optional<State> stf(State const & s, int l) const noexcept {
    using namespace gfl;
    if (l == depot) {
      return State(CitiesSet(), depot, 0, n);
    } else {
      const auto & [U, e, t, hops] = s;
      const int nextT = gfl::max<int>(t + d[e][l], tw[l].b);
      CitiesSet nextU = U - l;
      for (auto u : nextU)
        if (nextT + d[l][u] > tw[u].e)
          return nullopt;
      return State(nextU, l, nextT, hops + 1);
    }
  }

  GFL_HOST_DEVICE
  double scf(State const & s, int l) const noexcept { return d[s.e][l]; }

  constexpr static bool has_merge = false;

  constexpr static bool has_heur = true;
  GFL_HOST_DEVICE
  gfl::f64 h(State const & s, HContext ctx) const noexcept {
    using namespace gfl;
    i32 sumIn = 0, sumOut = 0, n1 = 0, n2 = 0;
    for (int i = 0; (i < n) and (n1 < n - s.hops); i++) {
      if (s.U.contains(permIn[i]) or permIn[i] == depot) {
        sumIn += dIn[i];
        n1++;
      }
    }
    for (int i = 0; (i < n) and (n2 < n - s.hops); i++) {
      if (s.U.contains(permOut[i]) or permOut[i] == s.e) {
        sumOut += dOut[i];
        n2++;
      }
    }
    return gfl::max<f64>(sumIn, sumOut);
  }

  constexpr static bool has_dom = true;
  GFL_HOST_DEVICE static
  bool dom(State const & s1, State const & s2) noexcept {
    return s1.U == s2.U and
           s1.e == s2.e and
           s1.hops == s2.hops and
           s1.t < s2.t;
  }

  GFL_HOST_DEVICE static
  gfl::u64 domHash(State const & s) noexcept {
    using namespace gfl;
    u64 seed = 0;
    hashCombine(seed, s.U.hash());
    hashCombine(seed, s.e);
    hashCombine(seed, s.hops);
    return seed;
  }

  template<typename T>
  bool validate(gfl::ArrayView<T> const chooses, gfl::f64 const expectedCost) const noexcept {
    using namespace gfl;
    if (chooses.size() != n)
      return false;

    if (chooses[chooses.size() - 1] != depot)
      return false;

    CitiesSet visited;
    for (i32 i = 0; i < chooses.size() - 1; ++i) {
      i32 const city = scast<i32>(chooses[i]);

      if (city == depot)
        return false;

      if (visited.contains(city))
        return false;

      visited.insert(city);
    }

    for (i32 c = 1; c < n; ++c) {
      if (not visited.contains(c))
        return false;
    }

    i32 cur = depot;
    i32 t = 0;
    f64 cost = 0.0;

    for (i32 i = 0; i < chooses.size(); ++i) {
      i32 const next = scast<i32>(chooses[i]);
      i32 const travel = d[cur][next];
      i32 const arrival = t + travel;

      if (arrival > tw[next].e)
        return false;

      t = max<i32>(arrival, tw[next].b);
      cost += travel;
      cur = next;
    }

    if (cost != expectedCost)
      return false;

    return true;
  }
};

static_assert(IsModel<Tsptw>);