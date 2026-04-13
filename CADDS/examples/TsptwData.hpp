#pragma once

#include <fstream>
#include <GFL.hpp>

class TsptwData {
public:
  static constexpr gfl::i32 BranchFactor = 32;
  static constexpr gfl::i32 Cities = 32;
  static constexpr gfl::i32 Depth = 32;

protected:
  using CitiesSet = gfl::BitSet<gfl::BitSet<>::numWords(Cities)>;

  struct TimeWindow {
    gfl::i32 b, e;
    TimeWindow(gfl::i32 const b, gfl::i32 const e) noexcept : b(b), e(e) {}
    TimeWindow() noexcept : b(gfl::numeric_limits<gfl::i32>::max()), e(gfl::numeric_limits<gfl::i32>::min()) {}
  };

  gfl::i32 depot;
  gfl::i32 n;
  gfl::ArrayView<gfl::ArrayView<gfl::i32>> d;
  gfl::ArrayView<TimeWindow> tw;
  gfl::ArrayView<gfl::i32> dIn;
  gfl::ArrayView<gfl::i32> dOut;
  gfl::ArrayView<gfl::i32> permIn;
  gfl::ArrayView<gfl::i32> permOut;

public:
  template<typename Allocator>
  TsptwData(std::string const & instancePath, Allocator & alloc) {
    using namespace gfl;

    std::ifstream file(instancePath);
    assert(file.is_open());

    depot = 0;
    file >> n;
    d = ArrayView<ArrayView<i32>>(n, alloc);
    for (auto i = 0; i < n; i++) {
      d[i] = ArrayView<i32>(n, alloc);
      for (auto j = 0; j < n; j++) {
        file >> d[i][j];
      }
    }
    tw = ArrayView<TimeWindow>(n, alloc);
    for (auto i = 0; i < n; i++) {
      i32 b, e;
      file >> b >> e;
      tw[i] = TimeWindow(b, e);
    }
    file.close();

    dIn = ArrayView<i32>(n, alloc);
    dOut = ArrayView<i32>(n, alloc);
    permIn = ArrayView<i32>(n, alloc);
    permOut = ArrayView<i32>(n, alloc);

    auto allCities = CitiesSet(0, n - 1);
    for (i32 j : allCities) {
      auto [e1, minIn] = argmin(allCities - j, [j, this] (i32 const k) { return d[k][j]; });
      auto [e2, minOut] = argmin(allCities - j, [j, this] (i32 const k) { return d[j][k]; });
      dIn[j] = minIn;
      dOut[j] = minOut;
      permIn[j] = j;
      permOut[j] = j;
    }

    auto const cmp = [](double const a, double const b) { return a < b; };
    sortByKeyFn(cmp, dOut, permOut);
    sortByKeyFn(cmp, dIn, permIn);
  }
};