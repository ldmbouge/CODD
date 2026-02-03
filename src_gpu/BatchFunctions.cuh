#pragma once

#include "Utils.hpp"
#include "BatchInfo.cuh"
#include "ctx.hpp"

#include <algorithm>

#include "BoundsHelpers.cuh"
#include "../examples_gpu/bro_base.cuh"

#include "RadixSort.hpp"

template <typename T>
GFL_HOST_DEVICE
void swapPtr(T** a, T** b)
{
    assert(a != nullptr);
    assert(b != nullptr);
    T * tmp = *a;
    *a = *b;
    *b = tmp;
}

template <typename T>
GFL_HOST_DEVICE
void swapVal(T* a, T* b)
{
    assert(a != nullptr);
    assert(b != nullptr);
    T tmp = *a;
    *a = *b;
    *b = tmp;
}

inline
void printNodesInfo(gfl::i64 const nNodes, NodeInfo const * const nodesInfo)
{
    using namespace gfl;
    for (i64 nIdx = 0; nIdx < nNodes; nIdx += 1)
    {
        NodeInfo::print(nodesInfo[nIdx]);
        printf("\n");
    }
    fflush(stdout);
}

template<typename Node>
void printNodes(gfl::i64 const nNodes, Node const * const nodes)
{
    using namespace gfl;
    for (i64 nIdx = 0; nIdx < nNodes; nIdx += 1)
    {
        Node::print(nodes[nIdx]);
        printf("\n");
    }
    fflush(stdout);

}


template<typename Model, typename Node>
GFL_HOST_DEVICE
gfl::f32 calcMergeScore(Node const & baseNode, Node const & toEvalNode, gfl::f32 const alpha = 0.5)
{
    using namespace gfl;

    f64 const normBaseScore =  absDiffWithBest<Model>(baseNode.heuristicBound);
    f64 const normToEvalScore =  absDiffWithBest<Model>(toEvalNode.heuristicBound);

    assert(normBaseScore <= normToEvalScore);
    assert(0.0 <= alpha);

    f32 const simScore = Model::ssf(baseNode.state, toEvalNode.state);        // 1.0 = identical, 0.0 = very different
    f32 const costRatio = static_cast<f32>(normToEvalScore / normBaseScore);  // How much worse than best (≥ 1.0)
    f32 const score = costRatio + (alpha * simScore);
    return score;
}

template<typename Model, typename Node>
GFL_HOST_DEVICE
void calcRep(NodeInfo & iInfo, NodeInfo & jInfo, Node const & iNode, Node const & jNode)
{
    using namespace gfl;

    if (Model::State::equal(iNode.state, jNode.state))
    {
        if (isBetterEq<Model>(iNode.gValue, jNode.gValue))
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
        if (isBetterEq<Model>(iNode.gValue, jNode.gValue) and Model::dom(iNode.state, jNode.state))
        {
            jInfo.flag = 1;
        }
        else if (isBetterEq<Model>(jNode.gValue, iNode.gValue) and Model::dom(jNode.state, iNode.state))
        {
            iInfo.flag = 1;
        }
    }
}

template<typename Node>
GFL_HOST_DEVICE
void assertInfoConsistency(BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for(i64 i = 0; i < bi.nChildren; i+=1)
    {
        assert(0 <= bi.childrenInfo[i].idx);
        assert(bi.childrenInfo[i].idx < bi.nChildren);
    }
}


template<typename Model, typename Node>
void calcRep(BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo & iInfo = bi.nodesInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < bi.nChildren);
        Node const & iChild = bi.children[iInfo.idx];
        for (i64 j = i + 1; j < bi.nChildren; j += 1)
        {
            NodeInfo & jInfo = bi.nodesInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < bi.nChildren);
            Node const & jChild = bi.children[jInfo.idx];
            if (iInfo.hash == jInfo.hash)
            {
                calcRep<Model>(iInfo,jInfo, iChild, jChild);
            }
            else
            {
                break;
            }
        }
    }
}


template<typename Model, typename Node>
void calcMergeScore(BatchInfo<Node> * const batchInfo, gfl::i64 const width)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    NodeInfo & baseInfo = bi.childrenInfo[0];
    Node const & baseNode = bi.children[baseInfo.idx];

    for(i64 i = 0; i < width; i += 1)
    {
        i64 begin,end;
        getBeginEnd(begin,end,i,width,bi.nChildren);
        NodeInfo & toCompareInfo = bi.childrenInfo[begin];
        Node const & toCompareNode = bi.children[toCompareInfo.idx];
        for (i64 j = begin+1; j < end; j += 1)
        {
            NodeInfo & toScoreInfo = bi.childrenInfo[j];
            Node const & toScoreNode = bi.children[toScoreInfo.idx];
            toScoreInfo.score = calcMergeScore<Model,Node>(baseNode, toScoreNode);
        }
    }
    baseInfo.score = 1.0; // Manually adjust the base state
}


template<typename Node>
void countFlagged(BatchInfo<Node> * const batchInfo, gfl::u32 const flag)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for (i64 cIdx = 0; cIdx < bi.nChildren; cIdx += 1)
    {
       bi.nFlagged += bi.nodesInfo[cIdx].flag == flag;
    }
}

template<typename Node>
void copyNodes(gfl::i64 const * const nNodes, Node * const dst,  Node const * const src, NodeInfo const * const nodesInfo)
{
    using namespace gfl;

    for (i64 i = 0; i < *nNodes; i += 1)
    {
        i64 const srcIdx = nodesInfo[i].idx;
        dst[i] = src[srcIdx];
    }
}

template<typename Node>
void copyNodes(gfl::i64 const * const nNodes, Node * const dst,  Node const * const src)
{
    using namespace gfl;

    for (i64 i = 0; i < *nNodes; i += 1)
    {
        dst[i] = src[i];
    }
}

template<typename Model, typename Node>
void filterChildren(BatchInfo<Node> * const batchInfo, bool sort)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    std::sort(bi.nodesInfo,
              bi.nodesInfo + bi.nChildren,
              NodeInfo::cmpByHash);

    calcRep<Model,Node>(batchInfo);

    std::sort(bi.nodesInfo,
              bi.nodesInfo + bi.nChildren,
              NodeInfo::cmpByFlag);

    countFlagged<Node>(batchInfo, 0);

    if (sort)
    {
        for (i64 i = 0; i < bi.nFlagged; i += 1)
        {
            NodeInfo & cInfo = bi.nodesInfo[i];
            cInfo.score = bi.children[cInfo.idx].fValue;
        }

        std::sort(bi.nodesInfo,
                  bi.nodesInfo + bi.nFlagged,
                  NodeInfo::cmpByScore);
    }

    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nFlagged, bi.children, bi.tmpChildren, bi.nodesInfo);

    bi.nChildren = bi.nFlagged;
    bi.nFlagged = 0;
}

template<typename Model, typename Node>
void calcChildren(
        Model const * const model,
        gfl::f64 pBound,
        BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    using State = Model::State;

    BatchInfo<Node> & bi = *batchInfo;
    LabelsInfo const & li = bi.labelsInfoParents;

    for (i64 pIdx = 0; pIdx < bi.nParents; pIdx += 1)
    {
        if (li.nLabels > 0)
        {
            Node const & pNode = bi.parents[pIdx];
            for (i32 label = li.minLabel; label <= li.maxLabel; label += 1)
            {
                if (pNode.labels.contains(label))
                {
                    Node & cNode = bi.children[bi.nChildren];
                    NodeInfo & cInfo = bi.nodesInfo[bi.nChildren];

                    // Transition
                    auto cState = model->stf(pNode.state, label);
                    if (cState.has_value())
                    {
                        // Current cost
                        f64 const tCost = model->scf(pNode.state, label);
                        cNode.gValue = pNode.gValue + tCost;

                        // Lower/Upper bound
                        if constexpr (Model::has_local)
                        {
                            double const hValue =  model->local(cState.value(), DDCtx);
                            cNode.fValue = cNode.gValue + hValue;
                        }
                        else
                        {
                            cNode.fValue = pNode.fValue;
                        }
                        // Conditions to keep the child
                        if (isBetter<Model>(cNode.fValue,pBound))
                        {
                            // Node
                            cNode.state = cState.value();
                            cNode.isApproximated = pNode.isApproximated;
                            cNode.hasAncestorInCutset = pNode.hasAncestorInCutset;
                            memcpy(cNode.labelsSrcToNode, pNode.labelsSrcToNode, sizeof(cNode.labelsSrcToNode));
                            cNode.labelsSrcToNode[pNode.nEdgesSrcToNode] = label;
                            cNode.nEdgesSrcToNode = pNode.nEdgesSrcToNode + 1;

                            // NodeInfo
                            if constexpr (Model::has_dom)
                                cInfo.hash = Model::domHash(cNode.state);
                            else
                                cInfo.hash = State::hash(cNode.state);
                            cInfo.flag = 0;
                            cInfo.pIdx = pIdx;
                            cInfo.idx = bi.nChildren;

                            bi.nChildren += 1;
                            //bi.children[cInfo.idx] = cNode;
                            //bi.childrenInfo[cInfo.idx] = cInfo;
                        }
                    }
                }
            }
        }
    }
}

template<typename Model, typename Node>
void calcMergePartition(gfl::i64 const width, BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    assert(width < bi.nChildren);

    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo & cInfo = bi.nodesInfo[i];
        cInfo.idx = i;
        cInfo.score = absDiffWithBest<Model>(bi.children[i].fValue);
    }

    std::sort(bi.nodesInfo,
              bi.nodesInfo + bi.nChildren,
              NodeInfo::cmpByScore);

    bi.nChildrenToCopy = roundUpDivPosInt<i64>(width, 2);
    bi.nChildrenToMerge = bi.nChildren - bi.nChildrenToCopy;
}


template<typename Model, typename Node>
void copyAndMergeSuffix(
        gfl::i64 const width,
        BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    assert(bi.nChildren > width);

    auto const nBinds = width - bi.nChildrenToCopy;
    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nChildrenToCopy, bi.children, bi.tmpChildren, bi.nodesInfo);
    for(i32 i = 0; i < bi.nChildrenToCopy; i += 1)
    {
        bi.nodesInfo[i].idx = i;
    }

    bi.nChildren = bi.nChildrenToCopy;
    for(i32 bIdx = 0; bIdx < nBinds; bIdx += 1)
    {
        i64 bBegin, bEnd;
        getBeginEnd(bBegin,bEnd, bIdx, nBinds, bi.nChildrenToMerge);
        bBegin += bi.nChildrenToCopy;
        bEnd += bi.nChildrenToCopy;
        NodeInfo & repInfo = bi.nodesInfo[bBegin];

        Node repNode = bi.tmpChildren[repInfo.idx];
        repNode.isApproximated = 1;
        for (i64 i = bBegin+1; i < bEnd; i += 1)
        {
            NodeInfo const & toMergeInfo = bi.nodesInfo[i];
            Node const & toMergeNode = bi.tmpChildren[toMergeInfo.idx];
            repNode.state = Model::smf(repNode.state, toMergeNode.state);
            repNode.gValue = calcBetter<Model>(repNode.gValue, toMergeNode.gValue);
            repNode.fValue = calcBetter<Model>(repNode.fValue, toMergeNode.fValue);
        }
        bi.children[bi.nChildrenToCopy + bIdx] = repNode;

        repInfo.idx = bi.nChildrenToCopy + bIdx;
        bi.nodesInfo[bi.nChildrenToCopy + bIdx] = repInfo;

        bi.nChildren += 1;
    }
}
template<typename Model, typename Node>
void mergeChildren(gfl::i64 const width, BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    copyAndMergeSuffix<Model,Node>(width,batchInfo);

    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo & cInfo = bi.nodesInfo[i];
        auto const & cIdx = cInfo.idx;
        assert(0 <= cIdx);
        assert(cIdx < bi.nChildren);
        auto & cNode = bi.children[cIdx];
        cInfo.score = absDiffWithBest<Model>(cNode.fValue);
    }
}

template<typename Model, typename Node>
void keepOnlyBestChild(BatchInfo<Node> * batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    assert(bi.nChildren > 0);

    Node & bestChild = bi.children[0];
    for(i64 cIdx = 1; cIdx < bi.nChildren; cIdx += 1)
    {
        Node const & child = bi.children[cIdx];
        if (isBetter<Model>(child.fValue,bestChild.fValue))
        {
            bestChild = child;
        }
    }
    bi.nChildren = 1;
}

template<typename Model, typename Node>
void calcChildrenLabels(
        Model const * const model,
        BatchInfo<Node> * const batchInfo,
        DDContext const ddCtx,
        gfl::f64 pBound,
        gfl::f64 dBound)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for (i64 cIdx = 0; cIdx < bi.nChildren; cIdx += 1)
    {
        auto & cNode = bi.children[cIdx];
        cNode.labels = model->lgf(cNode.state, ddCtx, pBound, dBound);
        bi.labelsInfoChildren.update(cNode.labels.slc());
    }
}


template<typename Model, typename Node>
void saveCutset(gfl::i64 const width, BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    //assert(std::is_sorted(bi.parents, bi.parents + bi.nParents, Node::cmpByFDec));
    assert(std::is_sorted(bi.nodesInfo,bi.nodesInfo + bi.nChildren, NodeInfo::cmpByScore));

    for (auto i = 0; i < bi.nParents; i += 1)
    {
        bi.tmpNodesInfo[i].flag = 0;
    }

    for (auto i = 0; i < bi.nChildrenToMerge; i += 1)
    {
        auto const & cInfo = bi.nodesInfo[bi.nChildrenToCopy + i];
        auto const & cIdx = cInfo.idx;
        assert(0 <= cIdx);
        assert(cIdx < bi.nChildren);
        auto & cNode = bi.children[cIdx];
        if (cNode.hasAncestorInCutset == 0)
        {
            cNode.hasAncestorInCutset = 1;
            auto const & pIdx = cInfo.pIdx;
            assert(0 <= pIdx);
            assert(pIdx < bi.nParents);
            bi.tmpNodesInfo[pIdx].flag = 1;
        }
    }

    for (auto i = 0; i < bi.nChildrenToCopy; i += 1)
    {
        auto const & cInfo = bi.nodesInfo[i];
        auto const & cIdx = cInfo.idx;
        assert(0 <= cIdx);
        assert(cIdx < bi.nChildren);
        auto & cNode = bi.children[cIdx];
        auto const & pIdx = cInfo.pIdx;
        assert(0 <= pIdx);
        assert(pIdx < bi.nParents);
        if (bi.tmpNodesInfo[pIdx].flag == 1)
        {
            cNode.hasAncestorInCutset = 1;
        }
    }

    i64 nSavedNodes = 0;
    for (auto i = 0; i < bi.nParents; i += 1)
    {
        if (bi.tmpNodesInfo[i].flag == 1)
        {
            bi.cutset[bi.cutsetSize] = bi.parents[i];
            bi.cutsetSize += 1;
            nSavedNodes += 1;
        }
        assert(bi.cutsetSize <= width * bi.labelsInfoParents.nLabels);
    }
    assert(bi.cutsetSize <= width * bi.labelsInfoParents.nLabels);
    if (nSavedNodes > 0)
    {
        //printf("Saved %d nodes of layer %d in cutset\n", nSavedNodes, bi.cutset[bi.cutsetSize-1].nEdgesSrcToNode);
    }
}