#pragma once

#include <functional>
#include "BoundsUtils.hpp"
#include <GFL.hpp>

template<typename Model, typename Node>
class SolutionsManager {
  using SolutionListener = std::function<void()>;
  std::vector<SolutionListener> _solutionListeners;
  Node _solution;
  Model const * _model;

public:
  SolutionsManager(Model const * model) :
    _model(model)
  {
    _solution.g(worst<Model>());
  }

  void onSolution(SolutionListener l) { _solutionListeners.emplace_back(std::move(l)); }

  gfl::f64 solutionCost() const noexcept { return _solution.g(); }

  bool hasSolution() const noexcept { return not isWorst<Model>(_solution.g()); }

  void printSolution() const noexcept { _solution.chooses().print(); }

  void notifySolution() const {
    for (auto const & l : _solutionListeners)
      l();
  }
  bool solution(Node const & node) {
    if (_model->isTarget(node.state())) {
      if (isBetter<Model>(node.g(), _solution.g())) {
        _solution = node;
        notifySolution();
        return true;
      }
    }
    return false;
  }
};
