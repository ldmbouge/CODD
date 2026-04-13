#pragma once

#include "GFL.hpp"
#include <Node.hpp>

template <typename Node>
struct ExpansionData {

  gfl::i32 branchingFactor;
  gfl::optional<Node> bestTargetNode;
  gfl::VectorView<Node> parents;
  gfl::VectorView<Node> children;
  gfl::VectorView<NodeInfo> childrenInfo;
  gfl::VectorView<Node> tmpNodes;
  gfl::VectorView<NodeInfo> tmpInfo;
  gfl::ArrayView<Node> tmpView;
  gfl::ArrayView<NodeInfo> tmpInfoView;

  void init(gfl::i32 const nParents, gfl::i32 const nChildren, gfl::ArenaAllocator &alloc) noexcept {
    using namespace gfl;
    parents = VectorView<Node>(nParents, alloc);
    children = VectorView<Node>(nChildren, alloc);
    childrenInfo = VectorView<NodeInfo>(nChildren, alloc);
    tmpNodes = VectorView<Node>(nChildren, alloc);
    tmpInfo = VectorView<NodeInfo>(nChildren, alloc);
  }

  void clear() noexcept {
    branchingFactor = 0;
    bestTargetNode.reset();
    parents.clear();
    children.clear();
    childrenInfo.clear();
    tmpNodes.clear();
    tmpInfo.clear();
  }

  static
  gfl::i64 dataMemSize(gfl::i32 const nParents, gfl::i32 const nChildren) {
    using namespace gfl;
    i64 memSize = 0;
    memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // parents
    memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // children
    memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // childrenInfo
    memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // tmpNodes
    memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // tmpInfo
    return memSize;
  }
};