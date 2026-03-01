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
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;

    for (i32 pIdx = 0; pIdx < parents.size(); pIdx += 1)
    {
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
                        cH = tighter<Model>(cH,h);
                    }
                    // Conditions to keep the child
                    if (not isValid<Model>(cG + cH) or isBetter<Model>(cG + cH,primal))
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
gfl::i32 countFlagged(
    gfl::i64 const flag,

    gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    i32 count = 0;
    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        NodeInfo const & info = nodesInfo[i];
        count += info.flag == flag;
    }
    return count;
}

template<typename Node>
void copyByInfo(gfl::ArrayView<Node> const & dst, gfl::ArrayView<Node> const & src, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(src.size() >= nodesInfo.size());
    assert(dst.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); ++i)
    {
        NodeInfo & info = nodesInfo[i];
        Node & sNode = src[info.idx];
        Node & dNode = dst[i];
        dNode = sNode;
        info.idx = i;
    }
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
void sortChildrenByF(ExpansionData<Node> & expData)
{
    using namespace gfl;

    using namespace gfl;
    auto & children = expData.children;
    auto & childrenInfo = expData.childrenInfo;

    assert(childrenInfo.size() == children.size());

    for (i32 i = 0; i < childrenInfo.size(); i += 1)
    {
        NodeInfo & info = childrenInfo[i];
        Node const & node = children[info.idx];
        info.score = score<Model>(node.f());
    }
    sort(childrenInfo, NodeInfo::cmpByScore);
}


template<typename Model, typename Node>
void sortChildrenByG(ExpansionData<Node> & expData)
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
    }
    sort(childrenInfo, NodeInfo::cmpByScore);
}

template<typename Model, typename Node>
void setRevScoreF(gfl::ArrayView<Node> const & nodes, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(nodes.size() >= nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        Node const & node = nodes[info.idx];
        info.score = -score<Model>(node.f());
    }
}

template<typename Model, typename Node>
void mergeChildrenInplace(gfl::ArrayView<Node> & children)
{
    using namespace gfl;

    Node & mNode = children[0];
    mNode.approximated(true);
    for (i32 nIdx = 1; nIdx < children.size(); nIdx += 1)
    {
        Node const & tNode = children[nIdx];
        mNode.state(Model::smf(mNode.state(), tNode.state()));
        mNode.g(better<Model>(mNode.g(), tNode.g()));
        mNode.h(looser<Model>(mNode.h(), tNode.h()));
    }
}

template<typename Node>
void flagParentsToSave(gfl::i32 const width, gfl::ArrayView<Node> & children, gfl::ArrayView<NodeInfo> const & childrenInfo, gfl::ArrayView<NodeInfo> & parentInfo)
{
    using namespace gfl;

    assert(children.size() == childrenInfo.size());

    for (i32 i = width - 1; i < children.size(); ++i)
    {
        NodeInfo const & info = childrenInfo[i];
        assert(info.idx == i);
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
    assert(children.size() == childrenInfo.size());
    for (auto i = 0; i < children.size(); i += 1)
    {
        auto const & info = childrenInfo[i];
        assert(info.idx == i);
        auto & node = children[info.idx];
        if (parentInfo[info.pIdx].flag == 1)
        {
            node.ancestorInCutset(true);
        }
    }
}

template<typename Model, typename Node>
void saveCutset(ExpansionData<Node> & expData)
{
    // using namespace gfl;
    // auto const & parents = expData.parents;
    // auto & parentsInfo = expData.parentInfo;
    // auto & children = expData.children;
    // auto & childrenInfo = expData.childrenInfo;
    // auto & tmpNodes = expData.tmpNodes;
    // auto & tmpInfo = expData.tmpInfo;
    // auto & cutset = cutData;
    // constexpr u8 ToNotSave = 1;
    // constexpr u8 ToSave = 0;
    //
    // i32 const maxChildren = width_ * branchFactor_;
    // i32 const blockSize = 256;
    // i32 const gridSize = ceil<i32>(maxChildren, blockSize);
    //
    // resetInfoIdxKernel<<<gridSize,blockSize>>>(&parentsInfo);
    // CHECK_LAST_CUDA_ERROR();
    // setFlagKernel<<<gridSize,blockSize>>>(ToNotSave, &parentsInfo);
    // CHECK_LAST_CUDA_ERROR();
    // flagToSaveKernel<<<gridSize,blockSize>>>(ToSave, &parentsInfo, width_, &children, &childrenInfo);
    // CHECK_LAST_CUDA_ERROR();
    // updateAncInCutKernel<<<gridSize,blockSize>>>(ToSave, &parentsInfo, &children, &childrenInfo);
    // CHECK_LAST_CUDA_ERROR();
    // resizeToKernel<<<1,1>>>(&tmpInfo, parentsInfo.sizePtr());
    // CHECK_LAST_CUDA_ERROR();
    // sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo, &tmpInfo, &cubAuxMem);
    // CHECK_LAST_CUDA_ERROR();
    // swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
    // CHECK_LAST_CUDA_ERROR();
    // setValueKernel<<<1,1>>>(&nFlagged, scast<i64>(0));
    // CHECK_LAST_CUDA_ERROR();
    // countFlaggedKernel<<<gridSize,blockSize>>>(ToSave, &nFlagged, &parentsInfo);
    // CHECK_LAST_CUDA_ERROR();
    // resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
    // CHECK_LAST_CUDA_ERROR();
    // resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
    // CHECK_LAST_CUDA_ERROR();
    // markAndResizeByKernel<<<1,1>>>(&cutset, &nFlagged);
    // CHECK_LAST_CUDA_ERROR();
    // copyByInfoIdxKernel<<<gridSize,blockSize>>>(cutset.mark(), &parents, &parentsInfo);
    // CHECK_LAST_CUDA_ERROR();
    //
    //
    //
    //
    // using namespace gfl;
    //
    // sort(parentInfo, NodeInfo::cmpByFlag);
    // i32 nParentsToCopy = 0;
    // countFlagged(1, &nParentsToCopy, parentInfo);
    //
    // if (nParentsToCopy > 0)
    // {
    //     ArrayView<NodeInfo> parentsToCopyInfo = parentInfo.slice(-nParentsToCopy);
    //     setRevScoreF<Model,Node>(parents, parentsToCopyInfo);
    //     sort(parentsToCopyInfo, NodeInfo::cmpByScore);
    //     cutData->markAndResizeBy(nParentsToCopy);
    //     ArrayView<Node> const * segment = cutData->mark();
    //     copyByInfo(*segment, parents, parentsToCopyInfo);
    //
    //     printf("CUTSET:\n");
    //     for(auto const & c : *segment) {Node::print(c);printf("\n");}
    //     printf("\n");
    // }
    //
    // DEBUG_CUT (
    //
    // )
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
    ExpansionData<Node> * const expData,
    CutsetData<Node> * const cutData)
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & tmpChildren = expData->tmpNodes;
    auto & childrenInfo = expData->childrenInfo;
    auto & parentInfo = expData->parentInfo;

    assert(width < children.size());

    assert(infoConsistent(children, childrenInfo));

    tmpChildren.resizeTo(children.size());
    setScoreG<Model>(children, childrenInfo);
    sort(childrenInfo, NodeInfo::cmpByScore);
    copyByInfo(tmpChildren,children, childrenInfo);
    VectorView<Node>::swap(tmpChildren, children);



    // Flag the parents of the nodes that will be merged
    parentInfo.resizeTo(parents.size());
    initInfoIdx(parentInfo);
    setFlag(0,parentInfo);
    flagParentsToSave(width, children, childrenInfo, parentInfo);

    // Merge the last children - (width - 1) nodes
    i32 const prefixSize = width - 1;
    auto suffix = children.slice(prefixSize, children.size());
    mergeChildrenInplace<Model>(suffix);
    children.resizeTo(width);
    childrenInfo.resizeTo(width);


    DEBUG_MRG(
        printf("AFTER MRG:\n");
        for(auto const & c : expData->children) {Node::print(c);printf("\n");}
        printf("\n");
    )

    // Save cutset
    updateAncestorFlag(children,childrenInfo,parentInfo);
    saveCutset<Model>(parents, parentInfo, cutData);

    assert(infoConsistent(children, childrenInfo));
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
void onlyBestTargets(Model const * const model, ExpansionData<Node> * const expData)
{
    using namespace gfl;

    auto & targets = expData->children;
    auto & targetsInfo = expData->childrenInfo;

    if (not targets.empty())
    {
        assert(targets.front().isTarget(model)); // All of them should be targets
        setScoreG<Model>(targets, targetsInfo);
        sort(targetsInfo, NodeInfo::cmpByScore);
        NodeInfo const & bestInfo = targetsInfo[0];
        expData->bestTargetNode = targets[bestInfo.idx];
        for (i32 i = 0; i < targets.size(); ++i)
        {
            NodeInfo const & bestExactInfo = targetsInfo[i];
            Node const & node = targets[bestExactInfo.idx];
            if (not node.approximated())
            {
                expData->bestExactTargetNode = node;
                break;
            }
        }
    }
}


template<typename Model, typename Node>
void setH(gfl::ArrayView<Node> const & nodes, gfl::f64 const f)
{
    using namespace gfl;

    for (i64 i = 0; i < nodes.size(); i += 1)
    {
        Node & node = nodes[i];
        f64 const h = f - node.g();
        node.h(tighter<Model>(node.h(), h));
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
        f64 const f = expData->getTarget().g();
        setH<Model,Node>(*cutData->nodes(), f);
    }

}

template<typename Model, typename Node>
void checkForTarget(
    Model const * const model,
    gfl::optional<Node> & target,
    gfl::ArrayView<Node> const & nodes,
    gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    if (not nodesInfo.empty())
    {
        NodeInfo const & info = nodesInfo[0];
        Node const & node = nodes[info.idx];
        if (node.isTarget(model)) target = node;
    }
}