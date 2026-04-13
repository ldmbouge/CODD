#pragma once

#include <chrono>
#include <thread>

#include "LayeredQueue.hpp"
#include "SolutionManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node>
class LogManager {
  using SolMgr = SolutionsManager<Model, Node>;

  static constexpr gfl::i32 wTime = 8;
  static constexpr gfl::i32 wPrimal = 12;
  static constexpr gfl::i32 wExpanded = 12;
  static constexpr gfl::i32 wQueue = 12;
  static constexpr gfl::i32 wNps = 12;
  static constexpr auto fmtStr = "{:>{}}   {:>{}}   {:>{}}   {:>{}}   {:>{}}\n";

  gfl::f64 _solutionTime = 0.0;
  gfl::f64 _lastPrintTime = 0.0;
  gfl::f64 _logInterval = 5.0;
  gfl::i64 _lastExtracted = 0;
  SolMgr & _solMgr;
  LayeredQueue<Node> const & _queue;
  StatsManager const & _stats;
  bool _stop = false;
  std::thread _logThread;

public:
  LogManager(SolMgr & solMgr,
             LayeredQueue<Node> const & queue,
             StatsManager const & stats,
             gfl::f64 const logInterval = 5.0)
      : _logInterval(logInterval), _solMgr(solMgr), _queue(queue), _stats(stats) {
    _solMgr.onSolution([this] { this->solution(); });
  }

  void run() {
    _logThread = std::thread([this]() {
      header();
      while (not _stop) {
        progress();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    });
  }

  void stop() {
    _stop = true;
    _logThread.join();
    summary();
  }

  void header() {
    std::cout << "---\n";
    std::cout << std::format(fmtStr, "Time [s]", wTime, "Primal", wPrimal, "Expanded", wExpanded, "Queue", wQueue, "Nodes/s", wNps);
  }

  void solution() {
    _solutionTime = _stats.elapsed<gfl::sec>();
    log();
  }

  void progress() {
    if (_stats.elapsed<gfl::sec>() - _lastPrintTime >= _logInterval)
      log();
  }

  void summary() {
    using namespace gfl;

    f64 const searchTime = _stats.duration<sec>();
    auto const statusStr = _queue.empty() ? "Completed" : _stats.timeout() ? "Timeout" : "Error";
    std::cout  << "---\n";
    std::cout << std::format("Status           = {}\n", statusStr);
    std::cout << std::format("Extracted        = {}\n", _queue.pulled());
    std::cout << std::format("Queue            = {}\n", _queue.size());
    std::cout << std::format("Search Time      = {:.2f}\n", searchTime);
    std::cout << std::format("Solution Time    = {}\n", _solMgr.hasSolution() ? std::format("{:.2f}", _solutionTime) : "-");
    std::cout << std::format("Solution Cost    = {}\n", _solMgr.hasSolution() ? std::format("{:.2f}", _solMgr.solutionCost()) : "-");
    std::cout << "Solution         = ";
    if (_solMgr.hasSolution())
      _solMgr.printSolution();
    else
        (std::cout << "-");
    std::cout << "\n";
  }

private:
  void log() {
    using namespace gfl;

    f64 const time = _stats.elapsed<sec>();
    f64 const deltaTime = time - _lastPrintTime;
    i64 const extract = _queue.pulled();
    i64 const deltaNodes = extract - _lastExtracted;
    f64 const nps = deltaTime > 0.0 ? scast<f64>(deltaNodes) / deltaTime : 0.0;
    _lastPrintTime = time;
    _lastExtracted = extract;
    auto const timeStr = std::format("{:.2f}", time);
    auto const primalStr = _solMgr.hasSolution() ? std::format("{:.2f}", _solMgr.solutionCost()) : "-";
    auto const expandedStr = std::format("{}", extract);
    auto const queueStr = std::format("{}", _queue.size());
    auto const npsStr = deltaTime > 0.0 ? std::format("{:.0f}", nps) : "-";
    std::cout << std::format(fmtStr, timeStr, wTime, primalStr, wPrimal, expandedStr, wExpanded, queueStr, wQueue, npsStr, wNps);
    std::flush(std::cout);
  }
};