#pragma once

#include <GFL.hpp>

#include "ExpansionData.hpp"
#include "ExpansionFunctionsSeq.hpp"

template<typename Node>
class ExpansionEngine {
protected:
  ExpansionData<Node> _data;

public:
  Node const * getBestTargetNode() const noexcept {
    Node const * const node = _data.bestTargetNode.has_value() ? &_data.bestTargetNode.value() : nullptr;
    return node;
  }
  gfl::i32 getBranchingFactor() const noexcept { return _data.branchingFactor; }
  gfl::ArrayView<Node> const & getExpansion() const noexcept { return _data.children; }
};