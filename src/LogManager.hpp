#pragma once

#include <fmt/core.h>

#include "BlockingQueue.cuh"
#include "BnBManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node, typename Q1, typename Q2>
class LogManager
{
    using BnBManager = BnBManager<Model, Node>;

    static constexpr gfl::i32 wTime     = 10;
    static constexpr gfl::i32 wPrimal   = 10;
    static constexpr gfl::i32 wDual     = 10;
    static constexpr gfl::i32 wGap      = 10;
    static constexpr gfl::i32 wExpanded = 20;
    static constexpr gfl::i32 wQueue    = 20;
    static constexpr gfl::i32 wNps      = 20;

    static constexpr auto fmtStr =
        "{:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}\n";

    gfl::f64 solutionTime_{0.0};
    gfl::f64 lastPrintTime_{0.0};
    gfl::f64 logInterval_{5.0};
    gfl::i64 lastExtractedReady_{0};
    gfl::i64 lastExtractedStash_{0};

    BnBManager &         bnb_;
    Q1 const &           ready_;
    Q2 const &           stash_;
    StatsManager const & stats_;

public:
    LogManager(BnBManager & bnb, Q1 const & ready, Q2 const & stash, StatsManager const & stats, gfl::f64 const logInterval = 5.0) :
        bnb_(bnb), ready_(ready), stash_(stash), stats_(stats), logInterval_(logInterval)
    {
        bnb.onPrimal([this]{ this->primal(); });
        bnb.onDual([this]{ this->dual(); });
    }

    ~LogManager() = default;

    LogManager(LogManager const &) = delete;
    LogManager(LogManager &&) = delete;
    LogManager & operator=(LogManager const &) = delete;
    LogManager & operator=(LogManager &&) = delete;

    void header()
    {
        fmt::print(fmtStr,
            "Time [s]",        wTime,
            "Primal",          wPrimal,
            "Dual",            wDual,
            "Gap [%]",         wGap,
            "Expanded (R/S)",  wExpanded,
            "Queue (R/S)",     wQueue,
            "Nodes/s (R/S)",   wNps
        );
    }

    void primal()
    {
        solutionTime_ = stats_.elapsed<gfl::sec>();
        log();
    }

    void dual()
    {
        log();
    }

    void progress()
    {
        if (stats_.elapsed<gfl::sec>() - lastPrintTime_ >= logInterval_) log();
    }

    void summary()
    {
        using namespace gfl;

        f64 const searchTime = stats_.duration<sec>();
        auto const statusStr =
            bnb_.solved()                    ? "Solved"    :
            ready_.empty() && stash_.empty() ? "Completed" :
            stats_.timeout()                 ? "Timeout"   :
                                               "Error";
        fmt::print("Status           = {}\n",   statusStr);
        fmt::print("Extracted (R/S)  = {}/{}\n", ready_.pulled(), stash_.pulled());
        fmt::print("Queue (R/S)      = {}/{}\n", ready_.size(),   stash_.size());
        fmt::print("Search Time      = {:.2f}\n", searchTime);
        fmt::print("Solution Time    = {:.2f}\n", solutionTime_);
        fmt::print("Solution Cost    = {:.2f}\n", bnb_.primal());
        fmt::print("Solution         = "); bnb_.printSolution();
        fmt::print("\n");
    }

private:

    void log()
    {
        using namespace gfl;

        f64 const time         = stats_.elapsed<sec>();
        f64 const dt           = time - lastPrintTime_;
        i64 const extractReady = ready_.pulled();
        i64 const extractStash = stash_.pulled();
        i64 const dnReady      = extractReady - lastExtractedReady_;
        i64 const dnStash      = extractStash - lastExtractedStash_;
        f64 const npsReady     = dt > 0.0 ? scast<f64>(dnReady) / dt : 0.0;
        f64 const npsStash     = dt > 0.0 ? scast<f64>(dnStash) / dt : 0.0;

        lastPrintTime_      = time;
        lastExtractedReady_ = extractReady;
        lastExtractedStash_ = extractStash;

        auto const timeStr     = fmt::format("{:.2f}", time);
        auto const primalStr   = bnb_.hasPrimal() ? fmt::format("{:.2f}", bnb_.primal()) : "-";
        auto const dualStr     = bnb_.hasDual()   ? fmt::format("{:.2f}", bnb_.dual())   : "-";
        auto const gapStr      = bnb_.hasGap()    ? fmt::format("{:.2f}", bnb_.gap())    : "-";
        auto const expandedStr = fmt::format("{}/{}", extractReady, extractStash);
        auto const queueStr    = fmt::format("{}/{}", ready_.size(), stash_.size());
        auto const npsStr      = dt > 0.0 ? fmt::format("{:.0f}/{:.0f}", npsReady, npsStash) : "-/-";

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