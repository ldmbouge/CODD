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
        ExpansionData<Node> * const expData,
        gfl::f64 const primal)
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & childrenInfo = expData->nodesInfo;

    assert(infoConsistent(children, childrenInfo));

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
                        f64 const cG = pNode.g() + tCost;
                        f64 cH =  pNode.h() > 0 ? pNode.f() - cG : 0; // Deal with shallow target states
                        if constexpr (Model::has_heur)
                        {
                            f64 const h = model->h(cState.value(), BBCtx);
                            cH = tighter<Model>(cH,h);
                        }
                        assert(cH >= 0);

                        // Conditions to keep the child
                        if (isBetter<Model>(cG + cH,primal))
                        {
                            // Node
                            assert(childrenInfo.size() == children.size());
                            childrenInfo.resizeBy(1);
                            i32 const cIdx = children.resizeBy(1);
                            Node & cNode = children[cIdx];
                            new (&cNode) Node(cState.value(), cG, cH, label, pNode);
                            NodeInfo & cInfo = childrenInfo[cIdx];
                            new (&cInfo) NodeInfo(cIdx, pIdx);
                        }
                    }
                }
            }
        }
    }
    assert(infoConsistent(children, childrenInfo));
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
        if (isBetterEq<Model>(iNode.g(), jNode.g()))
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
        if (isBetterEq<Model>(iNode.g(), jNode.g()) and Model::dom(iNode.state(), jNode.state()))
        {
            jInfo.flag = 1;
        }
        else if (isBetterEq<Model>(jNode.g(), iNode.g()) and Model::dom(jNode.state(), iNode.state()))
        {
            iInfo.flag = 1;
        }
    }
}

inline
void setFlag(gfl::VectorView<NodeInfo> & nodesInfo, gfl::u64 const flag)
{
    using namespace gfl;

    for (i32 i = 0; i < nodesInfo.size(); ++i)
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
        for (i32 j = i + 1; j < nodesInfo.size(); j += 1)
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
void filterChildren(ExpansionData<Node> * const expData)
{
    using namespace gfl;

    auto & children = expData->children;
    auto & tmpChildren = expData->tmpNodes;
    auto & childrenInfo = expData->nodesInfo;

    assert(infoConsistent(children, childrenInfo));

    calcHash<Model>(children, childrenInfo);
    sort(childrenInfo, NodeInfo::cmpByHash);

    i32 nRepresentatives = 0;
    setFlag(childrenInfo, 0);
    flagRepresented<Model,Node>(childrenInfo, children);
    countFlagged(&nRepresentatives, childrenInfo, 0);
    sort(childrenInfo, NodeInfo::cmpByFlag);

    childrenInfo.resizeTo(nRepresentatives);
    tmpChildren.resizeTo(nRepresentatives);
    copyByInfo(tmpChildren,children, childrenInfo);
    VectorView<Node>::swap(tmpChildren, children);

    assert(infoConsistent(children, childrenInfo));
}

template<typename Model, typename Node>
void setScoreG(gfl::ArrayView<Node> & nodes, gfl::ArrayView<NodeInfo> const & nodesInfo)
{
    using namespace gfl;

    assert(nodes.size() == nodesInfo.size());
    for (i32 i = 0; i < nodesInfo.size(); i += 1)
    {
        NodeInfo & info = nodesInfo[i];
        Node const & node = nodes[info.idx];
        info.score = score<Model>(node.g());
    }
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
void saveCutset(
    gfl::ArrayView<Node> const & parents,
    gfl::ArrayView<NodeInfo> & parentInfo,
    CutsetData<Node> * const cutData)
{
    using namespace gfl;

    sort(parentInfo, NodeInfo::cmpByFlag);
    i32 nParentsToCopy = 0;
    countFlagged(&nParentsToCopy, parentInfo, 1);

    if (nParentsToCopy > 0)
    {
        ArrayView<NodeInfo> parentsToCopyInfo = parentInfo.slice(-nParentsToCopy);
        setRevScoreF<Model,Node>(parents, parentsToCopyInfo);
        sort(parentsToCopyInfo, NodeInfo::cmpByScore);
        ArrayView<Node> segment = cutData->addSegment(nParentsToCopy);
        copyByInfo(segment, parents, parentsToCopyInfo);
    }

    DEBUG_CUT (
        printf("CUTSET:\n");
        for(auto const & c : cutData->nodes()) {Node::print(c);printf("\n");}
        printf("\n");
    )

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
void mergeChildren(gfl::i64 const width, ExpansionData<Node> * const expData, CutsetData<Node> * const cutData)
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & tmpChildren = expData->tmpNodes;
    auto & childrenInfo = expData->nodesInfo;
    auto & parentInfo = expData->tmpNodesInfo;

    assert(width < children.size());

    assert(infoConsistent(children, childrenInfo));

    tmpChildren.resizeTo(children.size());
    setScoreG<Model>(children, childrenInfo);
    sort(childrenInfo, NodeInfo::cmpByScore);
    copyByInfo(tmpChildren,children, childrenInfo);
    VectorView<Node>::swap(tmpChildren, children);

    DEBUG_MRG(
        printf("BEFORE MRG:\n");
        for(auto const & c : expData->children) {Node::print(c);printf("\n");}
        printf("\n");
    )

    // Flag the parents of the nodes that will be merged
    parentInfo.resizeTo(parents.size());
    initInfoIdx(parentInfo);
    setFlag(parentInfo,0);
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
        setScoreG<Model>(nodes, nodesInfo);
        sort(nodesInfo, NodeInfo::cmpByScore);
        NodeInfo const & info = nodesInfo[0];
        Node const & node = nodes[info.idx];
        nodes[0] = node;
        nodes.resizeTo(1);
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
        assert(h >= 0);
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
        i64 const f = expData->getTarget().f();
        //setH<Model,Node>(cutData->nodes(), f);
    }

}