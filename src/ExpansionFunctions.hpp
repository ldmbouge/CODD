#pragma once

#include <algorithm>
#include <GFL.hpp>

#include "Node.hpp"
#include "BoundsUtils.hpp"
#include "CutsetData.hpp"
#include "ExpansionData.hpp"
#include "ExpansionEngine.hpp"

template<typename Model, typename Node>
void expandParents(
        Model const * const model,
        ExpansionData<Node> * const expData,
        gfl::f64 const pBound)
{
    using namespace gfl;
    using State = Model::State;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & nodesInfo = expData->nodesInfo;

    for (i32 pIdx = 0; pIdx < parents.size(); pIdx += 1)
    {
        Node const & pNode = parents[pIdx];
        auto const & pLabels = pNode.labels();
        auto const & [minLabel, maxLabel, nLabels] = pLabels.summary();
        if (nLabels > 0)
        {
            for (i32 label = minLabel; label <= maxLabel; label += 1)
            {
                if (pLabels.contains(label))
                {
                    // Transition
                    auto const cState = model->stf(pNode.state(), label);
                    if (cState.has_value())
                    {
                        f64 const tCost = model->scf(pNode.state(), label);
                        f64 const cPrimal = pNode.primal() + tCost;
                        f64 cDual = pNode.dual();
                        if constexpr (Model::has_heur)
                        {
                            f64 const hChild = model->h(cState.value(), BBCtx);
                            assert(isBetterEq<Model>(pNode.dual(), cPrimal + hChild));
                            cDual = cPrimal + hChild;
                        }

                        // Conditions to keep the child
                        if (isBetter<Model>(cDual,pBound))
                        {
                            // Node
                            i32 const cIdx = children.resizeBy(1);
                            Node & cNode = children[cIdx];
                            new (&cNode) Node(cState.value(), cDual, cPrimal, label, pNode);
                            NodeInfo & cInfo = nodesInfo[cIdx];
                            new (&cInfo) NodeInfo(cIdx, pIdx);
                        }
                    }
                }
            }
        }
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
void flagRepresented(NodeInfo & iInfo, NodeInfo & jInfo, Node const & iNode, Node const & jNode)
{
    using namespace gfl;

    if (Model::State::equal(iNode.state(), jNode.state()))
    {
        if (isBetterEq<Model>(iNode.primal(), jNode.primal()))
        {
            jInfo.flag = 1;
        }
        else
        {
            iInfo.flag = 1;
        }
    }
    if constexpr (Model::has_dom)
    {
        if (isBetterEq<Model>(iNode.primal(), jNode.primal()) and Model::dom(iNode.state(), jNode.state()))
        {
            jInfo.flag = 1;
        }
        else if (isBetterEq<Model>(jNode.primal(), iNode.primal()) and Model::dom(jNode.state(), iNode.state()))
        {
            iInfo.flag = 1;
        }
    }
}

inline
void setFlag(gfl::VectorView<NodeInfo> & nodesInfo, gfl::u64 const flag)
{
    using namespace gfl;

    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        info.flag = flag;
    }
}

template<typename Model, typename Node>
void flagRepresented(gfl::VectorView<NodeInfo> & nodesInfo, gfl::VectorView<Node> const & nodes)
{
    using namespace gfl;

    assert(nodes.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & iInfo = nodesInfo[i];
        Node const & iNode = nodes[iInfo.idx];
        for (i64 j = i + 1; j < nodesInfo.size(); j += 1)
        {
            NodeInfo & jInfo = nodesInfo[j];
            Node const & jNode = nodes[jInfo.idx];
            if (iInfo.hash == jInfo.hash)
            {
                flagRepresented<Model>(iInfo,jInfo, iNode, jNode);
            }
            else
            {
                break;
            }
        }
    }
}

inline
void countFlagged(gfl::i32 * const count, gfl::ArrayView<NodeInfo> const & nodesInfo, gfl::u64 const flag)
{
    using namespace gfl;

    *count = 0;
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo const & info = nodesInfo[i];
        *count += info.flag == flag;
    }
}

template<typename Node>
void copyByInfo(gfl::ArrayView<Node> & dst, gfl::ArrayView<Node> const & src, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(src.size() >= nodesInfo.size());
    assert(dst.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        dst[i] = src[info.idx];
        info.idx = i;
    }
}

template<typename Node>
void copyByInfoAndSwap(gfl::VectorView<Node> & dst, gfl::VectorView<Node> & src, gfl::VectorView<NodeInfo> & nodesInfo)
{
    using namespace gfl;

    copyByInfo(dst,src, nodesInfo);
    VectorView<Node>::swap(dst, src);
}

template<typename T>
void copy(gfl::ArrayView<T> & dst, gfl::ArrayView<T> const & src)
{
    using namespace gfl;

    assert(src.size() <= dst.size());
    for (i32 i = 0; i < src.size(); i += 1)
    {
        dst[i] = src[i];
    }
}


template<typename Model, typename Node>
void filterChildren(ExpansionData<Node> * const expData)
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & tmpChildren = expData->tmpNodes;
    auto & nodesInfo = expData->nodesInfo;

    calcHash<Model>(children, nodesInfo);
    sort(nodesInfo, NodeInfo::cmpByHash);

    i32 nRepresentatives = 0;
    setFlag(nodesInfo, 0);
    flagRepresented<Model,Node>(nodesInfo, children);
    countFlagged(&nRepresentatives, nodesInfo, 0);
    sort(nodesInfo, NodeInfo::cmpByFlag);

    nodesInfo.resizeTo(nRepresentatives);
    tmpChildren.resizeTo(nRepresentatives);
    copyByInfoAndSwap(tmpChildren, children, nodesInfo);
}

template<typename Model, typename Node>
void calcScoreFromPrimal(gfl::ArrayView<Node> & nodes, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(nodes.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        Node const & node = nodes[info.idx];
        info.score = score<Model>(node.primal());
    }
}

template<typename Model, typename Node>
void mergeChildrenInplace(gfl::ArrayView<Node> & children)
{
    using namespace gfl;

    Node mNode = children[0];
    mNode.approximated(true);
    for (i32 nIdx = 1; nIdx < children.size(); nIdx += 1)
    {
        Node const & tNode = children[nIdx];
        mNode.state(Model::smf(mNode.state(), tNode.state()));
        mNode.primal(better<Model>(mNode.primal(), tNode.primal()));
        mNode.dual(better<Model>(mNode.dual(), tNode.dual()));
    }
}

template<typename Node>
void flagParentsToSave(gfl::ArrayView<Node> & children, gfl::ArrayView<NodeInfo> const & childrenInfo, gfl::ArrayView<NodeInfo> & parentInfo)
{
    using namespace gfl;
    assert(children.size() == childrenInfo.size());

    for (i32 i = 0; i < children.size(); i += 1)
    {
        NodeInfo const & info = childrenInfo[i];
        Node const & node = children[info.idx];
        if (not node.ancestorInCutset())
        {
            parentInfo[info.pIdx].flag = 1;
        }
    }
}

template< typename Node>
void updateAncestorFlag(gfl::ArrayView<Node> & children, gfl::ArrayView<NodeInfo> const & childrenInfo, gfl::ArrayView<NodeInfo> & parentInfo)
{
    for (auto i = 0; i < children.size(); i += 1)
    {
        auto const & info = childrenInfo[i];
        auto & node = children[info.idx];
        if (parentInfo[info.pIdx].flag == 1)
        {
            node.ancestorInCutset(true);
        }
    }
}

template<typename Node>
void saveCutset(
    gfl::ArrayView<Node> const & parents,
    gfl::ArrayView<NodeInfo> & parentInfo,
    CutsetData<Node> * const cutData)
{
    using namespace gfl;

    i32 nParentsToCopy = 0;
    countFlagged(&nParentsToCopy, parentInfo, 1);
    sort(parentInfo, NodeInfo::cmpByFlag);

    ArrayView<Node> segment = cutData->addSegment(nParentsToCopy);
    copyByInfo(segment, parents, parentInfo);
}

template<typename Model, typename Node>
void mergeChildren(gfl::i64 const width, ExpansionData<Node> * const expData, CutsetData<Node> * const cutData)
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & tmpChildren = expData->tmpNodes;
    auto & childrenInfo = expData->nodesInfo;
    auto & parentInfo = expData->tmpNodesInfo;

    assert(width < children.size());

    tmpChildren.resizeTo(children.size());
    calcScoreFromPrimal<Model>(children, childrenInfo);
    sort(childrenInfo, NodeInfo::cmpByScore);
    copyByInfoAndSwap(tmpChildren, children, childrenInfo);

    // Copy the best width - 1 nodes, merge down to 1 the remaining
    tmpChildren.resizeTo(width);
    i32 const prefixSize = width - 1;
    auto const prefixInfo = childrenInfo.slice(0,prefixSize);
    auto const prefixSrc = children.slice(0,prefixSize);
    auto prefixDst = tmpChildren.slice(0,prefixSize);
    copy(prefixDst, prefixSrc);

    auto const suffixInfo = childrenInfo.slice(prefixSize, children.size());
    auto suffixNodes = children.slice(prefixSize, children.size());
    mergeChildrenInplace<Model>(suffixNodes);

    parentInfo.resizeTo(parents.size());
    setFlag(parentInfo,0);
    flagParentsToSave(suffixNodes, suffixInfo, parentInfo);
    updateAncestorFlag(prefixDst,prefixInfo,parentInfo);
    saveCutset(parents, parentInfo, cutData);

    tmpChildren.back() = suffixNodes.front();
    VectorView<Node>::swap(tmpChildren, children);
}


template<typename Model, typename Node>
void calcOutLabels(
        Model const * const model,
        gfl::ArrayView<Node> const & nodes,
        gfl::f64 pBound,
        gfl::f64 dBound,
        DDContext const ddCtx)
{
    using namespace gfl;

    for (i64 i = 0; i < nodes.size(); i += 1)
    {
        Node & node = nodes[i];
        node.labels(model->lgf(node.state(), pBound, dBound, ddCtx));
    }
}


template<typename Model, typename Node>
void onlyBestTarget(Model const * const model, gfl::VectorView<Node> & nodes, gfl::ArrayView<NodeInfo> & nodesInfo)
{
    using namespace gfl;

    if (not nodes.empty())
    {
        assert(nodes.front().isTarget(model)); // All of them should be targets
        calcScoreFromPrimal<Model>(nodes, nodesInfo);
        sort(nodesInfo, NodeInfo::cmpByScore);
        NodeInfo const & info = nodesInfo[0];
        Node const & node = nodes[info.idx];
        nodes[0] = node;
        nodes.resizeTo(1);
    }
}


template<typename Node>
void setDual(gfl::ArrayView<Node> const & nodes, gfl::f64 const dual)
{
    using namespace gfl;

    for (i64 i = 0; i < nodes.size(); i += 1)
    {
        Node & node = nodes[i];
        node.dual(dual);
    }
}

template<typename Model, typename Node>
void finializeCutset(
    Model const * const model,
    ExpansionData<Node> * const expData,
    CutsetData<Node> * const cutData,
    gfl::f64 pBound,
    gfl::f64 dBound,
    DDContext const ddCtx)
{
    using namespace gfl;

    if (expData->hasTarget())
    {
        calcOutLabels(model, cutData->nodes(), pBound, dBound, ddCtx);
        i64 const dual = expData->getTarget().dual();
        setDual(cutData->nodes(), dual);
    }
}