#pragma once

#include <fmt/core.h>

#include "BnBManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node>
class LogManager
{
    using BnBManager = BnBManager<Model, Node>;

    static constexpr gfl::i32 wTime     = 10;
    static constexpr gfl::i32 wPrimal   = 10;
    static constexpr gfl::i32 wDual     = 10;
    static constexpr gfl::i32 wGap      = 10;
    static constexpr gfl::i32 wExpanded = 12;
    static constexpr gfl::i32 wQueue    = 12;

    static constexpr auto fmtStr =
        "{:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}\n";

    gfl::f64 solutionTime_{0.0};
    gfl::f64 lastPrintTime_{0.0};
    gfl::f64 logInterval_{5.0};

public:
    LogManager() noexcept = default;

    explicit
    LogManager(gfl::f64 const logInterval) noexcept : logInterval_(logInterval)
    {}

    ~LogManager() = default;

    LogManager(LogManager const &) = delete;
    LogManager(LogManager &&) = delete;
    LogManager & operator=(LogManager const &) = delete;
    LogManager & operator=(LogManager &&) = delete;

    void header()
    {
        fmt::print(fmtStr,
            "Time [s]", wTime,
            "Primal",   wPrimal,
            "Dual",     wDual,
            "Gap [%]",  wGap,
            "Expanded", wExpanded,
            "Queue",    wQueue
        );
    }

    void primal(StatsManager const & stats, BnBManager const & bnb)
    {
        using namespace gfl;
        solutionTime_ = stats.elapsed<sec>();
        log(bnb, stats);
    }

    void dual(StatsManager const & stats, BnBManager const & bnb)
    {
        log(bnb, stats);
    }

    void progress(StatsManager const & stats, BnBManager const & bnb)
    {
        using namespace gfl;
        f64 const time = stats.elapsed<sec>();
        if (time - lastPrintTime_ >= logInterval_) log(bnb, stats);
    }

    void summary(StatsManager const & stats, BnBManager const & bnb, gfl::f64 const timeout)
    {
        using namespace gfl;

        f64 const searchTime = stats.duration<sec>();
        auto const statusStr =
            bnb.solved()          ? "Completed" :
            searchTime >= timeout ? "Timeout"   :
                                    "Error";
        fmt::print("Status        = {}\n", statusStr);
        fmt::print("Extracted     = {}\n", stats.extracted());
        fmt::print("Queue         = {}\n", stats.inserted() - stats.extracted());
        fmt::print("Search Time   = {:.2f}\n", searchTime);
        fmt::print("Solution Time = {:.2f}\n", solutionTime_);
        fmt::print("Solution Cost = {:.2f}\n", bnb.primal());
        fmt::print("Solution Path = "); bnb.solution().print();
        fmt::print("\n");
    }

private:

    void log(BnBManager const & bnb, StatsManager const & stats)
    {
        using namespace gfl;

        lastPrintTime_ = stats.elapsed<sec>();

        bool const hasPrimal = isValid<Model>(bnb.primal());
        bool const hasDual   = isValid<Model>(bnb.dual());
        bool const hasGap    = hasPrimal and hasDual;

        auto const timeStr     = fmt::format("{:.2f}",lastPrintTime_);
        auto const primalStr   = hasPrimal ? fmt::format("{:.2f}", bnb.primal()) : "-";
        auto const dualStr     = hasDual   ? fmt::format("{:.2f}", bnb.dual())   : "-";
        auto const gapStr      = hasGap    ? fmt::format("{:.2f}", bnb.gap())    : "-";
        auto const expandedStr = fmt::format("{}", stats.extracted());
        auto const queueStr    = fmt::format("{}", stats.inserted() - stats.extracted());

        fmt::print(fmtStr,
            timeStr,     wTime,
            primalStr,   wPrimal,
            dualStr,     wDual,
            gapStr,      wGap,
            expandedStr, wExpanded,
            queueStr,    wQueue
        );
    }
};