#pragma once

#include <GFL.hpp>

class StatsManager {
  gfl::TimePoint _start;
  gfl::TimePoint _end;
  gfl::f64       _timeout;

public:
  explicit
  StatsManager(gfl::f64 const timeout) :
    _timeout(timeout)
  {}

  void startTime() noexcept { _start = gfl::now(); }

  void endTime() noexcept { _end = gfl::now(); }

  template<typename TUnit = gfl::sec>
  gfl::f64 duration() const noexcept { return gfl::elapsed<TUnit>(_start, _end); }

  template<typename TUnit = gfl::sec>
  gfl::f64 elapsed() const noexcept { return gfl::elapsed<TUnit>(_start); }

  bool timeout() const noexcept { return duration<gfl::sec>() > _timeout; }
};