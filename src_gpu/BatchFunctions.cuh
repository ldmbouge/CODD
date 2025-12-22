#pragma once

#include "Utils.hpp"
#include "BatchInfo.cuh"
#include "ctx.hpp"

#include <algorithm>

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

template<typename Model, typename Node>
GFL_HOST_DEVICE
gfl::f32 calcMergeScore(Node const & baseNode, Node const & toEvalNode, gfl::f32 const alpha = 1.0)
{
    using namespace gfl;

    assert(baseNode.boundSrcToNode <= toEvalNode.boundSrcToNode);
    assert(0.0 <= alpha);
    assert(alpha <= 1.0);

    f32 const simScore = Model::ssf(baseNode.state, toEvalNode.state);  // 1.0 = identical, 0.0 = very different
    f32 const costRatio = toEvalNode.boundSrcToNode / baseNode.boundSrcToNode;              // How much worse than best (≥ 1.0)
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
        if (Model::betterEq(iNode.boundSrcToNode, jNode.boundSrcToNode))
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
        if (Model::betterEq(iNode.boundSrcToNode, jNode.boundSrcToNode) and Model::dom(iNode.state, jNode.state))
        {
            jInfo.flag = 1;
        }
        else if (Model::betterEq(jNode.boundSrcToNode, iNode.boundSrcToNode) and Model::dom(jNode.state, iNode.state))
        {
            iInfo.flag = 1;
        }
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
        for (i64 j = begin; j < end; j += 1)
        {
            NodeInfo const & toScoreInfo = bi.childrenInfo[j];
            Node const & toScoreNode = bi.children[toScoreInfo.idx];
            toScoreInfo.score = calcMergeScore<Model,Node>(baseNode,toScoreNode);
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
        i64 const nIdx = nodesInfo[i].idx;
        dst[i] = src[nIdx];
    }
}

template<typename Model, typename Node>
void filterChildren(BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    auto cmpByHash = [](NodeInfo const & a, NodeInfo const & b)
        {return a.hash < b.hash;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByHash);

    calcRep<Model,Node>(batchInfo);

    auto cmpByFlag = [](NodeInfo const & a, NodeInfo const & b)
        {return a.flag < b.flag;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByFlag);

    swapPtr(&bi.children, &bi.tmpChildren);
    countFlagged<Node>(batchInfo, 0);
    copyNodes<Node>(&bi.nFlagged, bi.children, bi.tmpChildren, bi.childrenInfo);

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
                        f64 const cBoundSrcToNode = pNode.boundSrcToNode + model->scf(pNode.state, label);
                        f64 cDualBound = cBoundSrcToNode;
                        if constexpr (Model::has_local)
                            cDualBound += model->local(cState.value(), DDCtx);

                        if (Model::better(cDualBound, pBound))
                        {
                            // Node
                            cNode.state = cState.value();
                            cNode.boundSrcToNode = cBoundSrcToNode;
                            cNode.dualBound = cDualBound;
                            cNode.isNotExact = 0;
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
                            bi.children[cInfo.idx] = cNode;
                            bi.childrenInfo[cInfo.idx] = cInfo;

                            bi.nChildren += 1;
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

    copyNodes<Node>(&width, &bi.children, &bi.tmpChildren, &bi.childrenInfo);

    Node & lastNode = bi.children[width-1];
    for(i64 i = width; i < bi.nChildren; i += 1)
    {
        NodeInfo const  & toMergeInfo = bi.childrenInfo[i];
        Node const & toMergeNode = bi.tmpChildren[toMergeInfo.idx];
        lastNode.state = Model::smf(lastNode.state, toMergeNode.state);
        lastNode.boundSrcToNode = Model::better(lastNode.boundSrcToNode, toMergeNode.boundSrcToNode) ?
                                  toMergeNode.boundSrcToNode : lastNode.boundSrcToNode;
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
        cInfo.score = bi.children[cInfo.idx].boundSrcToNode;
    }

    auto cmpByScore = [](NodeInfo const & a, NodeInfo const & b)
    {return a.score < b.score;};
    std::sort(bi.childrenInfo,
              bi.childrenInfo + bi.nChildren,
              cmpByScore);

    calcMergeScore<Model,Node>(batchInfo);

    std::sort(bi.childrenInfo,
             bi.childrenInfo + bi.nChildren,
             cmpByScore);

    swapPtr(&bi.children, &bi.tmpChildren);
    copyAndMergeSuffix<Model,Node>(width,batchInfo);
    bi.nChildren = width;
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




template<typename Model, typename Node>
void ProcessBatchRelaxed(
      Model const * const model,
      gfl::f64 pBound,
      gfl::i64 width,
      BatchInfo<Node> * const batchInfo)
{
    ProcessBatchExact<Model,Node>(model,pBound,batchInfo);
    if (batchInfo->nChildren > width)
    {
        mergeChildren<Model,Node>(width,batchInfo);
        batchInfo->isExact = false;
    }
}


