#pragma once

#include <GFL.hpp>

class StatsManager
{
    gfl::TimePoint start_{};
    gfl::TimePoint end_{};
    gfl::i64 extracted_{0};
    gfl::i64 inserted_{0};

public:
    StatsManager() = default;
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

    gfl::i64 extracted() const noexcept { return extracted_; }
    void extracted(gfl::i64 const count) noexcept { extracted_ += count; }

    gfl::i64 inserted() const noexcept { return inserted_; }
    void inserted(gfl::i64 const count) noexcept { inserted_ += count; }
};