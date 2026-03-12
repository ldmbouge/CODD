#pragma once

#include <algorithm>
#include <GFL.hpp>

#include "Debug.hpp"
#include "Node.hpp"
#include "BoundsUtils.hpp"
#include "CutsetData.hpp"
#include "ExpansionData.hpp"

template<typename Node>
bool infoConsistent(
     gfl::VectorView<Node> const & nodes,
     gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    for (i32 i = 0;  i < nodesInfo.size(); ++i)
    {
        NodeInfo const & info = nodesInfo[i];
        assert(info.idx >= 0);
        assert(info.idx < nodes.size());
    }
    return true;
}

inline
void printInfo(gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    for (i32 i = 0;  i < nodesInfo.size(); ++i)
    {
        NodeInfo const & info = nodesInfo[i];
        NodeInfo::print(info); printf("\n");
    }
    printf("\n");
    fflush(stdout);
}

template<typename Model, typename Node>
void expandParents(
       Model const * const model,
       ExpansionData<Node> & expData,
       gfl::f64 const primal)
{
    using namespace gfl;

    auto const & parents = expData.parents;
    auto const & parentsInfo = expData.parentInfo;
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;

    for (i32 iIdx = 0; iIdx < parentsInfo.size(); iIdx += 1)
    {
        NodeInfo const & info = parentsInfo[iIdx];
        i32 const pIdx = info.idx;
        Node const & pNode = parents[pIdx];
        auto const & pLabels = pNode.outLabels();
        auto const & [minl, maxl, nLabels] = pLabels.summary();
        for (i32 label = minl; label <= maxl; label += 1)
        {
            if (pLabels.contains(label))
            {
                // Transition
                auto const cState = model->stf(pNode.state(), label);
                if (cState.has_value())
                {
                    f64 const tCost = model->scf(pNode.state(), label);
                    f64 const cG = pNode.g() + tCost;
                    f64 cH = pNode.f() - cG;
                    if constexpr (Model::has_heur)
                    {
                        f64 const h = model->h(cState.value(), BBCtx);
                        cH = worse<Model>(cH,h);
                    }
                    // Conditions to keep the child
                    if (isBetterEq<Model>(cG + cH, primal))
                    {
                        // Node
                        i32 const iIdx = childrenInfo.resizeBy(1);
                        i32 const cIdx = children.resizeBy(1);
                        childrenInfo[iIdx] = NodeInfo(cIdx, pIdx);
                        children[cIdx] = Node(cState.value(), cG, cH, label, pNode);
                    }
                }
            }
        }
    }
}

inline
void resetInfoIdx(gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    for (i64 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        info.idx = scast<i32>(i);
    }
}

template<typename Model, typename Node>
void calcHash(gfl::VectorView<Node> & nodes, gfl::VectorView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(nodes.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        Node const & node = nodes[info.idx];
        if constexpr (Model::has_dom)
            info.hash = Model::domHash(node.state());
        else
            info.hash = Model::State::hash(node.state());
    }
}

template<typename Fn>
void sort(gfl::ArrayView<NodeInfo> & nodesInfo, Fn cmp)
{
    using namespace gfl;
    std::sort(nodesInfo.begin(),nodesInfo.end(), cmp);
}

template<typename Model, typename Node>
GFL_HOST_DEVICE
void flagRepresented(NodeInfo & iInfo, NodeInfo & jInfo, Node const & iNode, Node const & jNode, gfl::i64 const flag)
{
    using namespace gfl;

    if (Model::State::equal(iNode.state(), jNode.state()))
    {
        if (iNode.approximated() == jNode.approximated() or iNode.g() != jNode.g())
        {
            if (isBetterEq<Model>(iNode.g(), jNode.g()))
                jInfo.flag = flag;
            else
                iInfo.flag = flag;
        }
        else // iNode.approximated() != jNode.approximated() and iNode.g() == jNode.g()
        {
            if (iNode.approximated())
                iInfo.flag = flag;
            else
                jInfo.flag = flag;
       }
    }

    if constexpr (Model::has_dom)
    {
        // TODO Every approximated-aware policy leads to a different trade-off
        if (isBetterEq<Model>(iNode.g(), jNode.g()) and Model::dom(iNode.state(), jNode.state()))
            jInfo.flag = flag;
        else if (isBetterEq<Model>(jNode.g(), iNode.g()) and Model::dom(jNode.state(), iNode.state()))
            iInfo.flag = flag;
    }
}

inline
void setFlag(
    gfl::u8 const flag,
    gfl::VectorView<NodeInfo> & nodesInfo)
{
    using namespace gfl;

    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        NodeInfo & info = nodesInfo[i];
        info.flag = flag;
    }
}

template<typename Model, typename Node>
void flagRepresented(
    gfl::i64 const flag,
    gfl::VectorView<Node> const & nodes,
    gfl::VectorView<NodeInfo> & nodesInfo)
{
    using namespace gfl;

    assert(nodes.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & iInfo = nodesInfo[i];
        Node const & iNode = nodes[iInfo.idx];
        for (i32 j = i + 1; j < nodesInfo.size(); j += 1)
        {
            NodeInfo & jInfo = nodesInfo[j];
            Node const & jNode = nodes[jInfo.idx];
            if (iInfo.hash == jInfo.hash)
            {
                flagRepresented<Model>(iInfo,jInfo, iNode, jNode, flag);
            }
            else
            {
                break;
            }
        }
    }
}

inline
gfl::i64  countFlagged(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    i64 count = 0;
    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        NodeInfo const & info = nodesInfo[i];
        count += info.flag == flag;
    }
    return count;
}

template<typename Node>
void copyByInfoIdx(gfl::ArrayView<Node> const & dst, gfl::ArrayView<Node> const & src, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(nodesInfo.size() <= src.size());
    assert(nodesInfo.size() <= dst.size());

    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        NodeInfo & info = nodesInfo[i];
        Node const & sNode = src[info.idx];
        Node & dNode = dst[i];
        dNode = sNode;
        info.idx = i;
    }
}

template<typename Model, typename Node>
void filterRepresentedChildren(ExpansionData<Node> & expData)
{
    using namespace gfl;
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;
    auto & tmpNodes = expData.tmpNodes;
    auto & tmpInfo = expData.tmpInfo;
    constexpr u8 Represented = 1;
    constexpr u8 NotRepresented = 0;

    assert(childrenInfo.size() == children.size());
    calcHash<Model>(children, childrenInfo);
    sort(childrenInfo, NodeInfo::cmpByHash);

    setFlag(NotRepresented,childrenInfo);
    flagRepresented<Model,Node>(Represented,children,childrenInfo);
    i32 const nRepresentatives = countFlagged(NotRepresented, childrenInfo);

    sort(childrenInfo, NodeInfo::cmpByFlag);
    childrenInfo.resizeTo(nRepresentatives);
}

template<typename Model, typename Node>
void sortChildrenByG(ExpansionData<Node> & expData,
    gfl::i32 const lambda = 1.0)
{
    using namespace gfl;

    using namespace gfl;
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;

    assert(childrenInfo.size() <= children.size());

    for (i32 i = 0; i < childrenInfo.size(); i += 1)
    {
        NodeInfo & info = childrenInfo[i];
        Node const & node = children[info.idx];
        info.score = score<Model>(node.g());
        info.score =
         node.approximated() ? info.score
                             : boostScore<Model>(info.score, lambda);
    }
    sort(childrenInfo, NodeInfo::cmpByScore);
}

template<typename Node>
void flagToSave(
    gfl::i64 const width,
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const & parentsInfo,
    gfl::ArrayView<Node> const & children,
    gfl::ArrayView<NodeInfo> const & childrenInfo
    )
{
    using namespace gfl;

    assert(children.size() >= childrenInfo.size());

    for (i32 i = width - 1; i < children.size(); ++i)
    {
        NodeInfo const & info = childrenInfo[i];
        Node const & node = children[info.idx];
        if (not node.ancestorInCutset())
        {
            parentsInfo[info.pIdx].flag = flag;
        }
    }
}

template< typename Node>
void updateAncInCut(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const & parentInfo,
    gfl::ArrayView<Node> const & children,
    gfl::ArrayView<NodeInfo> const & childrenInfo)
{
    assert(children.size() >= childrenInfo.size());

    for (auto i = 0; i < childrenInfo.size(); i += 1)
    {
        NodeInfo const & info = childrenInfo[i];
        Node & node = children[info.idx];
        if (parentInfo[info.pIdx].flag == flag)
        {
            node.ancestorInCutset(true);
        }
    }
}

template<typename Node>
void saveCutset(
    gfl::i64 const width,
    ExpansionData<Node> & expData,
    CutsetData<Node> & cutData)
{
    using namespace gfl;
    auto const & parents = expData.parents;
    auto & parentsInfo = expData.parentInfo;
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;
    auto & cutset = cutData;
    constexpr u8 ToNotSave = 1;
    constexpr u8 ToSave = 0;

    resetInfoIdx(parentsInfo);
    setFlag(ToNotSave, parentsInfo);
    flagToSave(width, ToSave, parentsInfo, children, childrenInfo);
    updateAncInCut(ToSave, parentsInfo, children, childrenInfo);
    sort(parentsInfo, NodeInfo::cmpByFlag);
    i64 nFlagged = countFlagged(ToSave, parentsInfo);
    parentsInfo.resizeTo(nFlagged);
    cutset.markAndResizeBy(nFlagged);
    copyByInfoIdx(cutset.mark(), parents, parentsInfo);
}

inline
void initInfoIdx(gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        nodesInfo[i].idx = i;
    }
}


template<typename Model, typename Node>
void mergeChildren(
    gfl::i64 const width,
    ExpansionData<Node> const & expData)
{
    using namespace gfl;

    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;

    // Merge the last children - (width - 1) nodes
    i32 const prefixSize = width - 1;
    ArrayView<NodeInfo> const suffixInfo = childrenInfo.slice(prefixSize, childrenInfo.size());

    NodeInfo const & rInfo = suffixInfo[0];
    Node & rNode = children[rInfo.idx];
    rNode.approximated(true);
    for (i32 i = 1; i < suffixInfo.size(); i += 1)
    {
        NodeInfo const & info = suffixInfo[i];
        Node const & node = children[info.idx];
        rNode.state(Model::smf(rNode.state(), node.state()));
        rNode.g(better<Model>(rNode.g(), node.g()));
        rNode.h(worse<Model>(rNode.h(), node.h()));
    }
}

template<typename Model, typename Node>
void calcOutLabels(
        Model const * const model,
        gfl::ArrayView<Node> const & nodes,
        gfl::ArrayView<NodeInfo> const & nodesInfo,
        gfl::f64 pBound,
        gfl::f64 dBound,
        DDContext const ddCtx)
{
    using namespace gfl;

    for (i64 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo const & info = nodesInfo[i];
        Node & node = nodes[info.idx];
        node.outLabels(model->lgf(node.state(), pBound, dBound, ddCtx));
    }
}

template<typename Model, typename Node>
void copyBestTargets(ExpansionData<Node> & expData)
{
    using namespace gfl;

    auto & targets = expData.children;
    auto & targetsInfo = expData.childrenInfo;
    optional<Node> & bestTarget = expData.bestTargetNode;
    optional<Node> & bestExactTarget = expData.bestExactTargetNode;

    bestTarget.reset();

    for (i64 i = 0; i < targetsInfo.size(); ++i)
    {
        NodeInfo const & info = targetsInfo[i];
        Node const & node = targets[info.idx];

        // Best overall (any node)
        f64 const g = node.g();
        if (not bestTarget.has_value() or
             isBetter<Model>(g, bestTarget.value().g()))
        {
            bestTarget = node;
            //printf("[DBG] Best found with value %.2f\n", node.g());
        }

        // Best exact (non-approximated)
        if (not node.approximated())
        {
            if (not bestExactTarget.has_value() or
                isBetter<Model>(node.g(), bestExactTarget.value().g()))
            {
                //printf("[DBG] Exact found with value %.2f\n", node.f());
                bestExactTarget = node;
            }
        }
    }
}

template<typename Model, typename Node>
void checkForTarget(
    Model const * const model,
    ExpansionData<Node> & expData
    )
{
    using namespace gfl;

    auto & targets = expData.children;
    auto & targetsInfo = expData.childrenInfo;
    optional<Node> & bestTarget = expData.bestTargetNode;

    if (not targets.empty())
    {
        NodeInfo const & info = targetsInfo[0];
        Node const & node = targets[info.idx];
        if (node.isTarget(model)) bestTarget = node;
    }
}