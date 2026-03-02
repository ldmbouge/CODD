#pragma once

#include <fmt/core.h>

#include "BnBManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node>
class LogManager
{
    using BnBManager = BnBManager<Model, Node>;
    using Queue = Queue<Model, Node>;

    static constexpr gfl::i32 wTime     = 10;
    static constexpr gfl::i32 wPrimal   = 10;
    static constexpr gfl::i32 wDual     = 10;
    static constexpr gfl::i32 wGap      = 10;
    static constexpr gfl::i32 wExpanded = 12;
    static constexpr gfl::i32 wQueue    = 12;
    static constexpr gfl::i32 wNps      = 10;

    static constexpr auto fmtStr =
        "{:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}\n";

    gfl::f64 solutionTime_{0.0};
    gfl::f64 lastPrintTime_{0.0};
    gfl::f64 logInterval_{5.0};
    gfl::i64 lastExtracted_{0};

    BnBManager & bnb_;
    Queue const & q_;
    StatsManager const & stats_;

public:
    LogManager(BnBManager & bnb, Queue const & q, StatsManager const & stats, gfl::f64 const logInterval = 5.0) :
        bnb_(bnb), q_(q), stats_(stats), logInterval_(logInterval)
    {
        bnb.onPrimal([this]{this->primal();});
        bnb.onDual([this]{this->dual();});
    }

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
            "Queue",    wQueue,
            "Nodes/s",      wNps
        );
    }

    void primal()
    {
        using namespace gfl;
        solutionTime_ = stats_.elapsed<sec>();
        log();
    }

    void dual()
    {
log();
    }

    void progress()
    {
        using namespace gfl;
        if (stats_.elapsed<sec>() - lastPrintTime_ >= logInterval_) log();
    }

    void summary()
    {
        using namespace gfl;

        f64 const searchTime = stats_.duration<sec>();
        auto const statusStr =
            bnb_.solved()          ? "Solved" :
            q_.empty()             ? "Completed" :
            stats_.timeout()       ? "Timeout"   :
                                     "Error";
        fmt::print("Status        = {}\n", statusStr);
        fmt::print("Extracted     = {}\n", q_.pulled());
        fmt::print("Queue         = {}\n", q_.size());
        fmt::print("Search Time   = {:.2f}\n", searchTime);
        fmt::print("Solution Time = {:.2f}\n", solutionTime_);
        fmt::print("Solution Cost = {:.2f}\n", bnb_.primal());
        fmt::print("Solution    = "); bnb_.printSolution();
        fmt::print("\n");
    }

private:

    void log()
    {
        using namespace gfl;

        f64 const time      = stats_.elapsed<sec>();
        f64 const dt        = time - lastPrintTime_;
        i64 const extracted = q_.pulled();
        i64 const dn        = extracted - lastExtracted_;
        f64 const nps       = dt > 0.0 ? scast<f64>(dn) / dt : 0.0;

        lastPrintTime_ = time;
        lastExtracted_ = extracted;

        auto const timeStr     = fmt::format("{:.2f}", time);
        auto const primalStr   = bnb_.hasPrimal() ? fmt::format("{:.2f}", bnb_.primal()) : "-";
        auto const dualStr     = bnb_.hasDual()   ? fmt::format("{:.2f}", bnb_.dual())   : "-";
        auto const gapStr      = bnb_.hasGap()    ? fmt::format("{:.2f}", bnb_.gap())    : "-";
        auto const expandedStr = fmt::format("{}", extracted);
        auto const queueStr    = fmt::format("{}", q_.size());
        auto const npsStr      = dt > 0.0 ? fmt::format("{:.0f}", nps) : "-";

        fmt::print(fmtStr,
            timeStr,     wTime,
            primalStr,   wPrimal,
            dualStr,     wDual,
            gapStr,      wGap,
            expandedStr, wExpanded,
            queueStr,    wQueue,
            npsStr,      wNps
        );
    }
};