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

        T begin = 0;
        T end = -1;
        if (nJobs > 0) // There are reasonable cases where nJobs can be negative
        {
            auto const jobsPerWorker = nJobs / nWorkers;
            auto const remainder = nJobs % nWorkers;

            // First 'remainder' workers get one extra job
            auto const extra = (wIdx < remainder) ? 1 : 0;

            begin = wIdx * jobsPerWorker + min<i64>(wIdx, remainder);
            end = begin + jobsPerWorker + extra;
        }
        return make_tuple(begin,end);
    }

#ifdef __CUDACC__
    GFL_DEVICE inline
    i32 getSharedMemSize() noexcept
    {
        i32 size;
        asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r"(size));
        return size;
    }
#endif
}