#pragma once

#include <fmt/core.h>

#include "BnBManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node>
class LogManager
{
    using BnBManager = BnBManager<Model, Node>;

    enum Event : gfl::i32
    {
        None   = 0,
        Primal = 1,
        Dual   = 2
    };

    static constexpr gfl::i32 w_time     = 10;
    static constexpr gfl::i32 w_event    =  6;
    static constexpr gfl::i32 w_primal   = 10;
    static constexpr gfl::i32 w_dual     = 10;
    static constexpr gfl::i32 w_gap      = 10;
    static constexpr gfl::i32 w_expanded = 12;
    static constexpr gfl::i32 w_queue    = 12;

    static constexpr auto fmt_str =
        "{:>{}}   {:<{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}\n";

    gfl::f64 solution_time_{0.0};
    gfl::f64 last_print_time_{0.0};
    gfl::f64 log_interval_{5.0};
    bool log_events_{true};
    Event pending_event_{None};

public:
    LogManager() noexcept = default;

    explicit
    LogManager(gfl::f64 const log_interval, bool const log_events) noexcept :
        log_interval_(log_interval),
        log_events_(log_events)
    {}

    ~LogManager() = default;

    LogManager(LogManager const &) = delete;
    LogManager(LogManager &&) = delete;
    LogManager & operator=(LogManager const &) = delete;
    LogManager & operator=(LogManager &&) = delete;

    void header()
    {
        fmt::print(fmt_str,
            "Time [s]", w_time,
            "Event",    w_event,
            "Primal",   w_primal,
            "Dual",     w_dual,
            "Gap [%]",  w_gap,
            "Expanded", w_expanded,
            "Queue",    w_queue
        );
    }

    void log(StatsManager const & stats_mgr, BnBManager & bnb_mgr)
    {
        if (bnb_mgr.primalUpdated())
        {
            pending_event_ = static_cast<Event>(pending_event_ | Primal);
            solution_time_ = stats_mgr.elapsed<gfl::sec>();
        }
        if (bnb_mgr.dualUpdated())
        {
            pending_event_ = static_cast<Event>(pending_event_ | Dual);
        }

        bool const some_event = log_events_ and pending_event_ != None;
        bool const log_interval_elapsed = stats_mgr.elapsed<gfl::sec>() - last_print_time_ >= log_interval_;

        if (some_event or log_interval_elapsed)
        {
            gfl::f64 const time = stats_mgr.elapsed<gfl::sec>();
            log_row(bnb_mgr, event_to_str(pending_event_), time, stats_mgr);
            last_print_time_ = time;
            pending_event_   = None;
        }
    }

    void summary(StatsManager const & stats_mgr, BnBManager const & bnb_mgr, gfl::f64 const timeout)
    {
        gfl::f64 const search_time = stats_mgr.duration<gfl::sec>();
        auto const status_str=
            bnb_mgr.gapClosed()    ? "Completed" :
            search_time >= timeout ? "Timeout"   :
                                     "Interrupted";

        fmt::print("\n");
        fmt::print("Search Status = {}\n", status_str);
        fmt::print("Search Time   = {:.2f} s\n", search_time);
        fmt::print("Solution Time = {:.2f} s\n", solution_time_);
        fmt::print("Solution Cost = {:.2f}\n", bnb_mgr.primal());
        fmt::print("Solution Path = "); bnb_mgr.solution().print();
        fmt::print("\n");
    }

private:
    static
    char const * event_to_str(Event const e) noexcept
    {
        switch (e)
        {
            case Primal:        return "P";
            case Dual:          return "D";
            case Primal | Dual: return "P,D";
            default:            return "";
        }
    }

    void log_row(BnBManager const & bnb,
                 char const * const event_str,
                 gfl::f64 const time,
                 StatsManager const & stats)
    {
        bool const has_primal = isValid<Model>(bnb.primal());
        bool const has_dual   = isValid<Model>(bnb.dual());
        bool const has_gap    = has_primal and has_dual;

        auto const time_str     = fmt::format("{:.2f}", time);
        auto const primal_str   = has_primal ? fmt::format("{:.2f}", bnb.primal()) : "";
        auto const dual_str     = has_dual   ? fmt::format("{:.2f}", bnb.dual())   : "";
        auto const gap_str      = has_gap    ? fmt::format("{:.2f}", bnb.gap())    : "";
        auto const expanded_str = fmt::format("{}", stats.extracted());
        auto const queue_str    = fmt::format("{}", stats.inserted() - stats.extracted());

        fmt::print(fmt_str,
            time_str,     w_time,
            event_str,    w_event,
            primal_str,   w_primal,
            dual_str,     w_dual,
            gap_str,      w_gap,
            expanded_str, w_expanded,
            queue_str,    w_queue
        );
    }
};