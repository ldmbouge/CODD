#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <sstream>
#include <iomanip>

#include "ChronoUtils.hpp"

namespace gfl
{
    // ── TimingData ────────────────────────────────────────────────────────────────

    class TimingData
    {
    public:
        gfl::TimePoint begin_time     = gfl::now();
        double         total_run_time = 0.0;
        int            count          = 0;
    };

    // ── Timer ─────────────────────────────────────────────────────────────────────

    class Timer
    {
    private:
        inline static Timer* instance_ = nullptr;

        std::mutex                                  mutex_;
        TimingData                                  program_data_;
        std::unordered_map<std::string, TimingData> functions_data_;

        Timer() { program_data_.begin_time = gfl::now(); }

        static Timer& get()
        {
            if (instance_ == nullptr)
                instance_ = new Timer();
            return *instance_;
        }

    public:

        // ── Scoped ────────────────────────────────────────────────────────────────

        class Scoped
        {
        public:
            explicit Scoped(std::string const& name) : name_(name) { Timer::begin(name_); }
            ~Scoped() { Timer::end(name_); }

            Scoped(Scoped const&)            = delete;
            Scoped& operator=(Scoped const&) = delete;

        private:
            std::string name_;
        };

        // ── API ───────────────────────────────────────────────────────────────────

        static void begin(std::string const& name = "Anonymous")
        {
            auto t = gfl::now();
            auto& inst = get();
            std::lock_guard<std::mutex> lock(inst.mutex_);
            inst.functions_data_[name].begin_time = t;
        }

        static void end(std::string const& name = "Anonymous", bool print_elapsed = false)
        {
            auto t = gfl::now();
            auto& inst = get();
            std::lock_guard<std::mutex> lock(inst.mutex_);

            auto it = inst.functions_data_.find(name);
            if (it == inst.functions_data_.end()) {
                std::cerr << "Timer::end — unknown timer '" << name << "'" << std::endl;
                return;
            }

            double ms = gfl::elapsed<gfl::msec>(it->second.begin_time, t);
            it->second.total_run_time += ms;
            it->second.count++;

            if (print_elapsed)
                std::cout << name << ": " << ms << " ms" << std::endl;
        }

        static std::string summary()
        {
            auto&  inst  = get();
            double total = gfl::elapsed<gfl::msec>(inst.program_data_.begin_time);

            const int W_NAME = 30;
            const int W_AVG  = 15;
            const int W_TOT  = 15;
            const int W_CNT  = 15;
            const int W_PCT  = 15;

            auto row = [&](std::stringstream& ss,
                           const std::string& name,
                           double avg, double tot, int cnt, double pct)
            {
                ss << "%% | ";
                ss << std::left  << std::setw(W_NAME)    << name << " |";
                ss << std::right << std::setw(W_AVG - 3) << avg  << " ms |";
                ss << std::right << std::setw(W_TOT - 3) << tot  << " ms |";
                ss << std::right << std::setw(W_CNT - 7) << cnt  << " times |";
                ss << std::right << std::setw(W_PCT - 2) << pct  << " %" << std::endl;
            };

            std::stringstream ss;
            ss << std::fixed << std::setprecision(3);
            ss << "%% +-------------------"  << std::endl;
            ss << "%% | Profiling Summary"   << std::endl;
            ss << "%% +-------------------"  << std::endl;
            ss << "%% | ";
            ss << std::left  << std::setw(W_NAME) << "Function Name" << " |";
            ss << std::right << std::setw(W_AVG)  << "Average"        << " |";
            ss << std::right << std::setw(W_TOT)  << "Total"          << " |";
            ss << std::right << std::setw(W_CNT)  << "Called"         << " |";
            ss << std::right << std::setw(W_PCT)  << "Percentage"     << " |" << std::endl;

            row(ss, "Total Run Time", total, total, 1, 100.0);

            std::lock_guard<std::mutex> lock(inst.mutex_);
            for (auto& [name, d] : inst.functions_data_) {
                double avg = d.count > 0 ? d.total_run_time / d.count : 0.0;
                double pct = total   > 0 ? 100.0 * d.total_run_time / total : 0.0;
                row(ss, name, avg, d.total_run_time, d.count, pct);
            }

            return ss.str();
        }
    };
}

// Time the current scope using the function name as the timer key.
// Usage: void myFunction() { TIMED_SCOPE(); ... }
#define TIMED_SCOPE()       Timer::Scoped _scoped_timer_(__func__)

// Time the current scope using a custom name as the timer key.
// Usage: void myFunction() { TIMED_SCOPE_N("my section"); ... }
#define TIMED_SCOPE_N(name) Timer::Scoped _scoped_timer_(name)