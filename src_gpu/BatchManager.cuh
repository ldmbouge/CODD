#pragma once

#include "Utils.hpp"
#include "BatchInfo.cuh"
#include "ctx.hpp"

#include <algorithm>

template<typename Model, typename State>
GFL_HOST_DEVICE
void checkPair(NodeInfo & iInfo, NodeInfo & jInfo, State const & iState, State const & jState)
{
    using namespace gfl;

    if (Model::State::equal(iState, jState))
    {
        if (Model::betterEq(iState.boundSrcToNode, jState.boundSrcToNode))
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
        if (Model::betterEq(iState.boundSrcToNode, jState.boundSrcToNode) and Model::dom(iState.state, jState.state))
        {
            jInfo.flag = 1;
        }
        else if (Model::betterEq(jState.boundSrcToNode, iState.boundSrcToNode) and Model::dom(jState.state, iState.state))
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
        Node const iChild = bi.children[iInfo.idx].state;
        for (i64 j = i + 1; j < bi.nChildren; j += 1)
        {
            NodeInfo & jInfo = bi.childrenInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < bi.nChildren);
            Node const jChild = bi.children[jInfo.idx].state;
            if (iInfo.hash == jInfo.hash)
            {
                checkPair<Model>(iInfo,jInfo, iChild, jChild);
            }
            else
            {
                break;
            }
        }
    }
}

template<typename Node>
void countFlagged(BatchInfo<Node> * const batchInfo, gfl::u32 const flag)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    for (i64 cIdx = 0; cIdx < bi.nChildren; cIdx += 1)
    {
       bi.nFlagged += bi.childrenInfo[cIdx].isFlagged == flag;
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
    copyNodes<Node>(&bi.nFlagged, &bi.children, &bi.tmpChildren, &bi.childrenInfo);

    bi.nChildren = bi.nFlagged;
    bi.nChildren = 0;
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

    for (i64 pIdx = 0; pIdx < bi.nParents; pIdx += 1)
    {
        if (bi.labelsInfo.nLabels > 0)
        {
            Node cNode;
            NodeInfo cInfo;
            Node const pNode = bi.parents[pIdx];
            for (i32 label = bi.labelsInfo.minLabel; label <= bi.labelsInfo.maxLabel; label += 1)
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
void ProcessBatchRelaxed(
      Model const * const model,
      gfl::f64 pBound,
      gfl::i64 width,
      BatchInfo<Node> * const batchInfo
    )
{
    using namespace gfl;
    using State = Model::State;

    BatchInfo<Node> & bi = *batchInfo;

    calcChildren(model,pBound,bi);

    Array<Node> inNodes(inNodes, *nInNodes);
    // 1) Calculate children
    // 2) Filter children by eq/dom
    // 3) Merge children until $nChildren <= width$

            // 2) If $dualBound(d) > primalBound$ return no children, otherwise go to 3).

            // 3) Calculate the children of the $n$ roots and return them.
            // 3.1) Restore the roots
            // 3.2) Calculate children
            // 3.3) Filter children by eq/dom
            // 3.4) Set dual bound of each child.
            // 3.5) Return the children.
            // ###

            // 1.1) Backup the roots (see 3)
            memcpy(layerInfo->tmpParents,layerInfo->parents,sizeof(Node) * layerInfo->nParents);

            while (true)
            {
                // 1.2) Calculate children
                calcChildren(model,layerInfo,pBound,DDCtx);

                // 1.3) If no children, set the worst possible dual bound and go to 2)
                if (layerInfo->nChildren == 0)
                {
                    layerInfo->dBound = model->worstValue();
                    break;
                }

                // 1.4) If solution layer, save the best dual bound and go to 2)
                if (model->isTarget(layerInfo->children[0]))
                {
                    for (i64 cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
                    {
                        i64 const cBound = layerInfo->children[cIdx].boundSrcToNode;
                        layerInfo->dBound = model->isBetter(cBound, layerInfo->dBound) ?
                                            cBound:
                                            layerInfo->dBound;
                    }
                    break;
                }
                // 1.5) Filter children by eq/dom
                filterChildren(layerInfo,sort);

                // 1.6) Merge children
                layerInfo->nChildren = layerInfo->nRepresentatives;
                if (layerInfo->nChildren > width)
                {
                    mergeChildren(layerInfo, width);
                    layerInfo->nChildren = width;
                }

                // 1.7) Parents <- Children and go to 1.2)
                copyNodes(&layerInfo->nChildren, layerInfo->parents, layerInfo->children, layerInfo->childrenInfo);
                layerInfo->nParents = layerInfo->nChildren;
                layerInfo->nChildren = 0;
                layerInfo->nRepresentatives = 0;
            }

            // 2) If $dualBound(d) > primalBound$ return no children, otherwise go to 3).
            if (layerInfo->dBound > pBound)
            {
}

void ProcessBatchExact()
{

}