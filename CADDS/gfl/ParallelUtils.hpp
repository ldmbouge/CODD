#pragma once

#include <cassert>

#include "Backend.hpp"
#include "FunQual.hpp"
#include "MathUtils.hpp"
#include "Types.hpp"

namespace gfl {
template<typename T>
GFL_HOST_DEVICE constexpr
tuple<T, T> calcSlice(i64 const wIdx, i64 const nWorkers, i64 const nJobs) noexcept {
  assert(wIdx < nWorkers);
  assert(0 <= nWorkers);
  T begin = 0;
  T end = -1;
  if (nJobs > 0) {
    auto const jobsPerWorker = nJobs / nWorkers;
    auto const remainder = nJobs % nWorkers;
    auto const extra = (wIdx < remainder) ? 1 : 0;
    begin = wIdx * jobsPerWorker + min<i64>(wIdx, remainder);
    end = begin + jobsPerWorker + extra;
  }
  return make_tuple(begin, end);
}
} // namespace gfl