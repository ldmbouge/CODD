#pragma once

#include <GFL.hpp>

#include "ExpansionData.hpp"
#include "ExpansionFunctionsSeq.hpp"

template<typename Node>
class ExpansionEngine {
protected:
  ExpansionData<Node> _data;

public:
  gfl::optional<Node> const & getBestTargetNode() const noexcept {
    return _data.bestTargetNode;
  }
  gfl::i32 getBranchingFactor() const noexcept { return _data.branchingFactor; }
  gfl::ArrayView<Node> const & getExpansion() const noexcept { return _data.children; }
};