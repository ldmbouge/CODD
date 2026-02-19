#pragma once

#include <chrono>

#include <Types.hpp>

namespace gfl
{
    using TimePoint = std::chrono::steady_clock::time_point;
    using sec  = std::chrono::seconds;
    using msec = std::chrono::milliseconds;
    using usec = std::chrono::microseconds;

    inline
    TimePoint now() noexcept
    {
        using namespace std::chrono;
        return steady_clock::now();
    }

    template <typename TUnit>
    f64 elapsed(TimePoint const & start, TimePoint const & end) noexcept
    {
        using namespace std::chrono;
        return duration_cast<duration<f64, typename TUnit::period>>(end - start).count();
    }

    template <typename TUnit>
    f64 elapsed(TimePoint const & start) noexcept { return elapsed<TUnit>(start, now()); }
}