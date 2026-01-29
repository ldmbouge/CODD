#pragma once

#include "Utils.hpp"
#include "BatchInfo.cuh"
#include "ctx.hpp"

#include <algorithm>

#include "BoundsHelpers.cuh"
#include "../examples_gpu/bro_base.cuh"

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
        if (isBetterEq<Model>(iNode.sumEdgesSrcToNode, jNode.sumEdgesSrcToNode))
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
        if (isBetterEq<Model>(iNode.sumEdgesSrcToNode, jNode.sumEdgesSrcToNode) and Model::dom(iNode.state, jNode.state))
        {
            jInfo.flag = 1;
        }
        else if (isBetterEq<Model>(jNode.sumEdgesSrcToNode, iNode.sumEdgesSrcToNode) and Model::dom(jNode.state, iNode.state))
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
        NodeInfo & iInfo = bi.childrenInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < bi.nChildren);
        Node const iChild = bi.children[iInfo.idx];
        for (i64 j = i + 1; j < bi.nChildren; j += 1)
        {
            NodeInfo & jInfo = bi.childrenInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < bi.nChildren);
            Node const jChild = bi.children[jInfo.idx];
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
       bi.nFlagged += bi.childrenInfo[cIdx].flag == flag;
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

template<typename Model, typename Node>
void updateChildrenBound(BatchInfo<Node> * const batchInfo, gfl::f64 const hBound, bool sort)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        Node & cNode = bi.children[i];
        cNode.heuristicBound = calcWorst<Model>(cNode.heuristicBound, hBound);
    }

    if (sort)
    {
        for (i64 i = 0; i < bi.nChildren; i += 1)
        {
            NodeInfo & cInfo = bi.childrenInfo[i];
            cInfo.score = bi.children[i].heuristicBound;
            cInfo.idx = i;
        }

        auto cmpByScore = [](NodeInfo const & a, NodeInfo const & b){return isWorstEq<Model>(a.score,b.score);};
        std::sort(bi.childrenInfo,
                  bi.childrenInfo + bi.nChildren,
                  cmpByScore);
    }

    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nChildren, bi.children, bi.tmpChildren, bi.childrenInfo);
}

template<typename Model, typename Node>
void filterChildren(BatchInfo<Node> * const batchInfo, bool sort)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    auto cmpByHash = [](NodeInfo const & a, NodeInfo const & b){return a.hash < b.hash;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByHash);

    calcRep<Model,Node>(batchInfo);

    auto cmpByFlag = [](NodeInfo const & a, NodeInfo const & b){return a.flag < b.flag;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByFlag);

    countFlagged<Node>(batchInfo, 0);

    if (sort)
    {
        for (i64 i = 0; i < bi.nFlagged; i += 1)
        {
            NodeInfo & cInfo = bi.childrenInfo[i];
            cInfo.score = bi.children[cInfo.idx].heuristicBound;
        }

        //printNodesInfo(bi.nFlagged,bi.childrenInfo);
        auto cmpByScore = [](NodeInfo const & a, NodeInfo const & b){return isWorst<Model>(a.score,b.score);};
        std::sort(bi.childrenInfo,
                  bi.childrenInfo + bi.nFlagged,
                  cmpByScore);
        //printNodesInfo(bi.nFlagged,bi.childrenInfo);
    }

    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nFlagged, bi.children, bi.tmpChildren, bi.childrenInfo);

    bi.nChildren = bi.nFlagged;
    bi.nFlagged = 0;
}


template<typename Node>
bool isOptAnc(Node const & n)
{
    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
    bool isPrefix = true;
    for (gfl::i32 i = 0; i < n.nEdgesSrcToNode; i += 1)
    {
        isPrefix = isPrefix and  n.labelsSrcToNode[i] == opt[i];
    }
    return isPrefix;
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
    LabelsInfo const & li = bi.labelsInfo;

    for (i64 pIdx = 0; pIdx < bi.nParents; pIdx += 1)
    {
        if (li.nLabels > 0)
        {
            Node cNode;
            NodeInfo cInfo;
            Node const pNode = bi.parents[pIdx];
            for (i32 label = li.minLabel; label <= li.maxLabel; label += 1)
            {
                if (pNode.labels.contains(label))
                {
                    // Transition
                    auto cState = model->stf(pNode.state, label);
                    if (cState.has_value())
                    {
                        // Current cost
                        f64 const tCost = model->scf(pNode.state, label);
                        cNode.sumEdgesSrcToNode = pNode.sumEdgesSrcToNode + tCost;

                        // Lower/Upper bound
                        if constexpr (Model::has_local)
                        {
                            double const h =  model->local(cState.value(), DDCtx);
                            cNode.heuristicBound = cNode.sumEdgesSrcToNode + h;
                        }
                        else
                        {
                            cNode.heuristicBound = pNode.heuristicBound;
                        }
                        // Conditions to keep the child
                        if (isBetter<Model>(cNode.heuristicBound,pBound))
                        {
                            // Node
                            cNode.state = cState.value();
                            cNode.isNotExact = pNode.isNotExact;
                            memcpy(cNode.labelsSrcToNode, pNode.labelsSrcToNode, sizeof(cNode.labelsSrcToNode));
                            cNode.labelsSrcToNode[pNode.nEdgesSrcToNode] = label;
                            cNode.nEdgesSrcToNode = pNode.nEdgesSrcToNode + 1;

                            // NodeInfo
                            if constexpr (Model::has_dom)
                                cInfo.hash = Model::domHash(cNode.state);
                            else
                                cInfo.hash = State::hash(cNode.state);
                            cInfo.flag = 0;
                            cInfo.idx = bi.nChildren;

                            bi.nChildren += 1;
                            bi.children[cInfo.idx] = cNode;
                            bi.childrenInfo[cInfo.idx] = cInfo;
                        }
                    }
                }
            }
        }
    }
}

template<typename Model, typename Node>
void copyAndMergeSuffix(
        gfl::i64 const width,
        BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    assert(bi.nChildren > width);

    i64 const nToCopy = bi.nChildren <= 2 * width ? 2 * width - bi.nChildren : roundUpDivPosInt<i64>(width, 2);
    i64 const nBinds = width - nToCopy;
    i64 const nToMerge = bi.nChildren - nToCopy;
    assert(nToMerge >= 2 * nBinds);

    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&nToCopy, bi.children, bi.tmpChildren, bi.childrenInfo);

    bi.nChildren = nToCopy;
    for(i32 bIdx = 0; bIdx < nBinds; bIdx += 1)
    {
        i64 bBegin, bEnd;
        getBeginEnd(bBegin,bEnd, bIdx, nBinds, nToMerge);
        bBegin += nToCopy;
        bEnd += nToCopy;
        NodeInfo const & repInfo = bi.childrenInfo[bBegin];
        Node repNode = bi.tmpChildren[repInfo.idx];
        repNode.isNotExact = 1;
        for (i64 i = bBegin+1; i < bEnd; i += 1)
        {
            NodeInfo const & toMergeInfo = bi.childrenInfo[i];
            Node const & toMergeNode = bi.tmpChildren[toMergeInfo.idx];
            repNode.state = Model::smf(repNode.state, toMergeNode.state);
            repNode.sumEdgesSrcToNode = calcBetter<Model>(repNode.sumEdgesSrcToNode, toMergeNode.sumEdgesSrcToNode);
            repNode.heuristicBound = calcBetter<Model>(repNode.heuristicBound, toMergeNode.heuristicBound);
        }
        bi.children[nToCopy + bIdx] = repNode;
        bi.nChildren += 1;
    }
}
template<typename Model, typename Node>
void mergeChildren(gfl::i64 const width, BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo & cInfo = bi.childrenInfo[i];
        cInfo.idx = i;
        cInfo.score = absDiffWithBest<Model>(bi.children[i].heuristicBound);
    }

    //printNodesInfo(bi.nChildren, bi.childrenInfo);

    auto cmpByScore = [](NodeInfo const & a, NodeInfo const & b){return a.score < b.score;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByScore);

    //printNodesInfo(bi.nChildren, bi.childrenInfo);

    //calcMergeScore<Model,Node>(batchInfo, width);

    //printNodesInfo(bi.nChildren, bi.childrenInfo);

    // std::sort(bi.childrenInfo,
    //          bi.childrenInfo + bi.nChildren,
    //          cmpByScore);

    // printNodesInfo(bi.nChildren, bi.childrenInfo);
    //
    // printf("Before (%d)\n", bi.nChildren);
    // printNodes(bi.nChildren, bi.children);

    copyAndMergeSuffix<Model,Node>(width,batchInfo);

    // printf("After (%d)\n", bi.nChildren);
    // printNodes(bi.nChildren, bi.children);
}


template<typename Model, typename Node>
bool someTarget(Model const * const model, BatchInfo<Node> * batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    assert(bi.nChildren > 0);

    for(i64 cIdx = 0; cIdx < bi.nChildren; cIdx += 1)
    {
       if (model->isTarget(bi.children[cIdx].state))
       {
           return true;
       }
    }
    return false;
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
        if (isBetter<Model>(child.sumEdgesSrcToNode,bestChild.sumEdgesSrcToNode))
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
        bi.labelsInfo.update(cNode.labels.slc());
    }
}

