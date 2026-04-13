#pragma once

#include <GFL.hpp>

#include "ExpansionData.hpp"
#include "ExpansionEngine.hpp"
#include "ExpansionFunctionsSeq.hpp"

template<typename Model, typename Node>
class ExpansionEngineSeq : public ExpansionEngine<Node> {
  using ExpansionEngine<Node>::_data;
public:
  void allocData(gfl::ArenaAllocator & alloc, gfl::i32 const nodes, gfl::i32 const maxBranchFactor) {
    using namespace gfl;
    i32 const nParents = nodes;
    i32 const nChildren = nParents * maxBranchFactor;
    _data.init(nParents, nChildren, alloc);
  }

  void expand(Model const * model, gfl::ArrayView<Node> const & parents, gfl::f64 const primal) {
    using namespace gfl;
    _data.parents.pushBack(parents);
    expandParents(model, _data, primal);
    filterRepresentedChildren<Model, Node>(_data);
    _data.tmpNodes.resizeTo(_data.childrenInfo.size());
    copyByInfoIdx(_data.tmpNodes, _data.children, _data.childrenInfo);
    VectorView<Node>::swap(_data.tmpNodes, _data.children);
    calcOutLabels<Model, Node>(model,
                              _data.children,
                              _data.childrenInfo,
                              _data.branchingFactor,
                              primal,
                              best<Model>(),
                              DDRestricted);
    copyBestTargetNode<Model,Node>(model,_data);
  }

  static gfl::i64 getBatchSize(gfl::i64 const maxMemSize, gfl::i32 const branchingFactor) {
    using namespace gfl;

    i64 lbNodes = 0;
    i64 ubNodes = 1;

    auto const calcMemSize = [branchingFactor](i32 const nodes) {
      i32 const nParents = nodes;
      i32 const nChildren = nParents * branchingFactor;
      return ExpansionData<Node>::dataMemSize(nParents, nChildren) + DefaultAlign;
    };

    while (calcMemSize(ubNodes) <= maxMemSize) {
      lbNodes = ubNodes;
      ubNodes *= 2;
    }

    while (lbNodes < ubNodes) {
      i64 const mid = lbNodes + (ubNodes - lbNodes + 1) / 2;
      i64 const memSize = calcMemSize(mid);
      if (memSize <= maxMemSize)
        lbNodes = mid;
      else
        ubNodes = mid - 1;
    }
    return lbNodes;
  }
};
