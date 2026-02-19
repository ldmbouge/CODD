#pragma once

#include <cassert>

#include "Backend.hpp"
#include "Types.hpp"
#include "FunQual.hpp"
#include "MathUtils.hpp"

namespace gfl
{

    template<typename T>
    GFL_HOST_DEVICE
    constexpr
    tuple<T,T> calcSlice(i64 const wIdx, i64 const nWorkers, i64 const nJobs) noexcept
    {
        assert(wIdx < nWorkers);
        assert(0 <= nWorkers);
        assert(0 <= nJobs);

        auto const jobsPerWorker = nJobs / nWorkers;
        auto const remainder = nJobs % nWorkers;

        // First 'remainder' workers get one extra job
        auto const extra = (wIdx < remainder) ? 1 : 0;

        T const begin = wIdx * jobsPerWorker + min<i64>(wIdx, remainder);
        T const end = begin + jobsPerWorker + extra;

        return make_tuple(begin, end);
    }
}