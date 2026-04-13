#pragma once

#include <GFL.hpp>

#include "BoundsUtils.hpp"
#include "ExpansionData.hpp"
#include "Node.hpp"

template<typename Model, typename Node>
void expandParents(Model const * const model, ExpansionData<Node> & data, gfl::f64 const primal) {
  using namespace gfl;
  auto const & parents = data.parents;
  auto & children = data.children;
  auto & childrenInfo = data.childrenInfo;

  for (i32 pIdx = 0; pIdx < parents.size(); pIdx += 1) {
    Node const & pNode = parents[pIdx];
    auto const & pLabels = pNode.outLabels();
    auto const & [minl, maxl, nLabels] = pLabels.summary();
    for (i32 label = minl; label <= maxl; label += 1) {
      if (pLabels.contains(label)) {
        auto const cState = model->stf(pNode.state(), label);
        if (cState.has_value()) {
          f64 const tCost = model->scf(pNode.state(), label);
          f64 const cG = pNode.g() + tCost;
          f64 cH = pNode.f() - cG;
          if constexpr (Model::has_heur) {
            f64 const h = model->h(cState.value(), BBCtx);
            cH = worse<Model>(cH, h);
          }
          if (isBetter<Model>(cG + cH, primal)) {
            i32 const cInfoIdx = childrenInfo.resizeBy(1);
            i32 const cIdx = children.resizeBy(1);
            childrenInfo[cInfoIdx] = NodeInfo(cIdx);
            children[cIdx] = Node(cState.value(), cG, cH, label, pNode);
          }
        }
      }
    }
  }
}

template<typename Model, typename Node>
void calcHash(gfl::VectorView<Node> & nodes, gfl::VectorView<NodeInfo> const & nodesInfo) {
  using namespace gfl;
  assert(nodes.size() == nodesInfo.size());
  for (i32 i = 0; i < nodesInfo.size(); i += 1) {
    NodeInfo & info = nodesInfo[i];
    Node const & node = nodes[info.idx];
    if constexpr (Model::has_dom)
      info.hash = Model::domHash(node.state());
    else
      info.hash = Model::State::hash(node.state());
  }
}

template<typename Fn>
void sort(gfl::ArrayView<NodeInfo> & nodesInfo, Fn cmp, bool reverse = false) {
  using namespace gfl;
  if (reverse) {
    auto const rCmp = [&](auto const & a, auto const & b) { return cmp(b, a); };
    std::sort(nodesInfo.begin(), nodesInfo.end(), rCmp);
  } else {
    std::sort(nodesInfo.begin(), nodesInfo.end(), cmp);
  }
}

template<typename Model, typename Node>
GFL_HOST_DEVICE void
flagRepresented(NodeInfo & iInfo, NodeInfo & jInfo, Node const & iNode, Node const & jNode, gfl::i64 const flag) {
  using namespace gfl;
  if (Model::State::equal(iNode.state(), jNode.state())) {
    if (iNode.g() != jNode.g()) {
      if (isBetterEq<Model>(iNode.g(), jNode.g()))
        jInfo.flag = flag;
      else
        iInfo.flag = flag;
    } else
      jInfo.flag = flag;
  } else if constexpr (Model::has_dom) {
    if (isBetterEq<Model>(iNode.g(), jNode.g()) and Model::dom(iNode.state(), jNode.state()))
      jInfo.flag = flag;
    else if (isBetterEq<Model>(jNode.g(), iNode.g()) and Model::dom(jNode.state(), iNode.state()))
      iInfo.flag = flag;
  }
}

inline
void setFlag(gfl::u8 const flag, gfl::VectorView<NodeInfo> const & nodesInfo) {
  using namespace gfl;
  for (auto i = 0; i < nodesInfo.size(); ++i) {
    NodeInfo & info = nodesInfo[i];
    info.flag = flag;
  }
}

template<typename Model, typename Node>
void flagRepresented(gfl::i64 const flag, gfl::VectorView<Node> const & nodes, gfl::VectorView<NodeInfo> & nodesInfo) {
  using namespace gfl;

  assert(nodes.size() >= nodesInfo.size());
  for (i32 i = 0; i < nodesInfo.size(); i += 1) {
    NodeInfo & iInfo = nodesInfo[i];
    Node const & iNode = nodes[iInfo.idx];
    for (i32 j = i + 1; j < nodesInfo.size(); j += 1) {
      NodeInfo & jInfo = nodesInfo[j];
      Node const & jNode = nodes[jInfo.idx];
      if (iInfo.hash == jInfo.hash) {
        flagRepresented<Model>(iInfo, jInfo, iNode, jNode, flag);
      } else {
        break;
      }
    }
  }
}

inline gfl::i64 countFlagged(gfl::i64 const flag, gfl::ArrayView<NodeInfo> const & nodesInfo) {
  using namespace gfl;
  i64 count = 0;
  for (auto i = 0; i < nodesInfo.size(); ++i) {
    NodeInfo const & info = nodesInfo[i];
    count += info.flag == flag;
  }
  return count;
}

template<typename Node>
void copyByInfoIdx(gfl::ArrayView<Node> const & dst,
                   gfl::ArrayView<Node> const & src,
                   gfl::ArrayView<NodeInfo> const & nodesInfo) {
  using namespace gfl;
  assert(nodesInfo.size() <= src.size());
  assert(nodesInfo.size() <= dst.size());

  for (i32 i = 0; i < nodesInfo.size(); ++i) {
    NodeInfo & info = nodesInfo[i];
    Node const & sNode = src[info.idx];
    Node & dNode = dst[i];
    dNode = sNode;
    info.idx = i;
  }
}

template<typename Model, typename Node>
void filterRepresentedChildren(ExpansionData<Node> & data) {
  using namespace gfl;
  auto & children = data.children;
  auto & childrenInfo = data.childrenInfo;
  constexpr u8 Represented = 1;
  constexpr u8 NotRepresented = 0;

  assert(childrenInfo.size() <= children.size());
  calcHash<Model>(children, childrenInfo);
  sort(childrenInfo, NodeInfo::cmpByHash);
  setFlag(NotRepresented, childrenInfo);
  flagRepresented<Model, Node>(Represented, children, childrenInfo);
  i32 const nRepresentatives = countFlagged(NotRepresented, childrenInfo);
  sort(childrenInfo, NodeInfo::cmpByFlag);
  childrenInfo.resizeTo(nRepresentatives);
}

inline void initInfoIdx(gfl::ArrayView<NodeInfo> const & nodesInfo) {
  using namespace gfl;
  for (auto i = 0; i < nodesInfo.size(); ++i) {
    nodesInfo[i].idx = i;
  }
}

template<typename Model, typename Node>
void calcOutLabels(Model const * const model,
                   gfl::ArrayView<Node> const & nodes,
                   gfl::ArrayView<NodeInfo> const & nodesInfo,
                   gfl::i32 & branchingFactor,
                   gfl::f64 pBound,
                   gfl::f64 dBound,
                   DDContext const ddCtx) {
  using namespace gfl;

  branchingFactor = 0;
  for (i64 i = 0; i < nodesInfo.size(); i += 1) {
    NodeInfo const & info = nodesInfo[i];
    Node & node = nodes[info.idx];
    node.outLabels(model->lgf(node.state(), pBound, dBound, ddCtx));
    branchingFactor = max<i32>(branchingFactor, node.outLabels().size());
  }
}

template<typename Model, typename Node>
void copyBestTargetNode(Model const * const model, ExpansionData<Node> & data) {
 
  auto & bestTargetNode = data.bestTargetNode;
  auto & children = data.children;

  bestTargetNode.reset();
  if (not children.empty() and model->isTarget(children.front().state())) {
    for (auto i  = 0; i < children.size(); i += 1) {
      Node const & node = children[i];
      if (not bestTargetNode.has_value() or isBetter<Model>(node.g(), bestTargetNode.value().g())) {
        bestTargetNode = node;
      }
    }
  }
}
