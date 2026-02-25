#pragma once

#include <GFL.hpp>

class StatsManager
{

    // TODO manage edge cases with std::optional
    gfl::TimePoint start_{};
    gfl::TimePoint end_{};
    gfl::f64 timeout_{};

public:
    explicit
    StatsManager(gfl::f64 const timeout) : timeout_(timeout){};
    ~StatsManager() = default;

    StatsManager(StatsManager const &) = delete;
    StatsManager(StatsManager &&) = delete;
    StatsManager & operator=(StatsManager const &) = delete;
    StatsManager & operator=(StatsManager &&) = delete;

    void start() noexcept { start_ = gfl::now(); }
    void end() noexcept { end_ = gfl::now(); }

    template <typename TUnit>
    gfl::f64 duration() const noexcept { return gfl::elapsed<TUnit>(start_, end_); }

    template <typename TUnit>
    gfl::f64 elapsed() const noexcept { return gfl::elapsed<TUnit>(start_); }

    bool timeout() const noexcept { return duration<gfl::sec>() > timeout_; }
};