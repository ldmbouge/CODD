#pragma once

#include "Utils.hpp"
#include "BatchInfo.cuh"
#include "ctx.hpp"

#include <algorithm>

#include "BoundsHelpers.cuh"
#include "../examples_gpu/bro_base.cuh"

#include "RadixSort.hpp"

#include <Debug.cuh>

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

template<typename T>
void copyItems(gfl::i64 const * const nItems, T * const dst,  T const * const src)
{
    using namespace gfl;

    for (i64 i = 0; i < *nItems; i += 1)
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
            cInfo.score = fValueToScore<Model>(bi.children[cInfo.idx].fValue);
        }

        std::sort(bi.nodesInfo,
                  bi.nodesInfo + bi.nFlagged,
                  NodeInfo::cmpByScore);
    }

    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nFlagged, bi.children, bi.tmpChildren, bi.nodesInfo);
    for (i32 i = 0; i < bi.nFlagged; i += 1)
    {
        bi.nodesInfo[i].idx = i;
    }

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

#ifdef G_DEBUG
    std::vector<int> const opt = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};


    auto checkParent = [&](Node const & pNode)
    {
        if (pNode.isAncestorOf(opt))
        {
            printf("Ancestor of OPT found!\n");
            assert(pNode.fValue >= 17);
        }
    };
#endif
    for (i64 pIdx = 0; pIdx < bi.nParents; pIdx += 1)
    {
        Node const & pNode = bi.parents[pIdx];

        if (li.nLabels > 0)
        {
            for (i32 label = li.minLabel; label <= li.maxLabel; label += 1)
            {
                if (pNode.labels.contains(label))
                {
                    Node & cNode = bi.children[bi.nChildren];
                    NodeInfo & cInfo = bi.nodesInfo[bi.nChildren];

#ifdef G_DEBUG
                    checkParent(pNode);
#endif

                    // Transition
                    auto cState = model->stf(pNode.state, label);
                    if (cState.has_value())
                    {
#ifdef G_DEBUG
                        checkParent(pNode);
#endif
                        // Current cost
                        f64 const tCost = model->scf(pNode.state, label);
                        cNode.gValue = pNode.gValue + tCost;
#ifdef G_DEBUG
                        checkParent(pNode);
#endif

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
#ifdef G_DEBUG
                        checkParent(pNode);
#endif
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
#ifdef G_DEBUG
                            checkParent(pNode);
#endif

                            // NodeInfo
                            if constexpr (Model::has_dom)
                                cInfo.hash = Model::domHash(cNode.state);
                            else
                                cInfo.hash = State::hash(cNode.state);
                            cInfo.flag = 0;
                            cInfo.pIdx = pIdx;
                            cInfo.idx = bi.nChildren;
#ifdef G_DEBUG
                            checkParent(pNode);
#endif

                            bi.nChildren += 1;
                            //bi.children[cInfo.idx] = cNode;
                            //bi.childrenInfo[cInfo.idx] = cInfo;
                        }
                        else
                        {
#ifdef G_DEBUG
                            memcpy(cNode.labelsSrcToNode, pNode.labelsSrcToNode, sizeof(cNode.labelsSrcToNode));
                            cNode.labelsSrcToNode[pNode.nEdgesSrcToNode] = label;
                            cNode.nEdgesSrcToNode = pNode.nEdgesSrcToNode + 1;
                            if (cNode.isAncestorOf(opt))
                            {

                                printf("P = "); Node::printLabels(pNode); printf("\n");
                                printf("C = "); Node::printLabels(cNode); printf("\n");
                                fflush(stdout);
                                assert(pNode.isAncestorOf(opt));
                                printf("Ancestor of OPT DOB because bounds!\n");
                            }
#endif
                        }

                    }
#ifdef G_DEBUG
                    checkParent(pNode);
#endif
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
        NodeInfo & iInfo = bi.nodesInfo[i];
        iInfo.idx = i;
        Node const & iNode = bi.children[i];
        iInfo.score = fValueToScore<Model>(iNode.fValue);
    }

    // printf("SCORES =");
    // for (i32 i = 0; i < bi.nChildren; i += 1)
    // {
    //     if( i == bi.nChildrenToCopy-1) printf(" |");
    //     printf(" %.2f", bi.nodesInfo[i].score);
    // }
    // printf("\n");
    // fflush(stdout);

    std::sort(bi.nodesInfo,
              bi.nodesInfo + bi.nChildren,
              NodeInfo::cmpByScore);

    // printf("SSCORES =");
    // for (i32 i = 0; i < bi.nChildren; i += 1)
    // {
    //     if( i == bi.nChildrenToCopy-1) printf(" |");
    //     printf(" %.2f", bi.nodesInfo[i].score);
    // }
    // printf("\n");
    // fflush(stdout);



    bi.nChildrenToCopyPrefix = (width - 1) / 2;
    bi.nChildrenToCopySuffix = (width - 1) / 2;
    bi.nChildrenToMerge = bi.nChildren - bi.nChildrenToCopyPrefix -  bi.nChildrenToCopySuffix;
    assert(bi.nChildrenToCopyPrefix + bi.nChildrenToCopySuffix < width);
    assert(bi.nChildrenToMerge > 0);
    assert(bi.nChildren == bi.nChildrenToCopyPrefix + bi.nChildrenToCopySuffix + bi.nChildrenToMerge);
}


template<typename Model, typename Node>
void copyAndMerge(
        gfl::i64 const width,
        BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;

    BatchInfo<Node> & bi = *batchInfo;

    // assert(bi.nChildren > width);
    // printf("BFVALUES =");
    // for (i32 i = 0; i < bi.nChildren; i += 1)
    // {
    //     if( i == bi.nChildrenToCopy-1) printf(" |");
    //     printf(" %.2f", bi.children[bi.nodesInfo[i].idx].fValue);
    // }
    // printf("\n");

    // Copy
    swapPtr(&bi.children, &bi.tmpChildren);
    swapPtr(&bi.nodesInfo, &bi.tmpNodesInfo);
    copyNodes<Node>(&bi.nChildrenToCopyPrefix,
                    bi.children,
                    bi.tmpChildren,
                    bi.tmpNodesInfo);
    copyItems(&bi.nChildrenToCopyPrefix,
                    bi.nodesInfo,
                    bi.tmpNodesInfo);
    copyNodes<Node>(&bi.nChildrenToCopySuffix,
                    bi.children + bi.nChildrenToCopyPrefix,
                    bi.tmpChildren,
                    bi.tmpNodesInfo + bi.nChildren - bi.nChildrenToCopySuffix);
    copyItems(&bi.nChildrenToCopySuffix,
                    bi.nodesInfo + bi.nChildrenToCopyPrefix,
                    bi.tmpNodesInfo+ bi.nChildren - bi.nChildrenToCopySuffix);
    bi.nChildren = bi.nChildrenToCopyPrefix + bi.nChildrenToCopySuffix;

    i32 const nBins = width - bi.nChildren;
    for(i32 bIdx = 0; bIdx < nBins; bIdx += 1)
    {
        i64 bBegin, bEnd;
        getBeginEnd(bBegin,bEnd, bIdx, nBins, bi.nChildrenToMerge);
        bBegin += bi.nChildrenToCopyPrefix;
        bEnd += bi.nChildrenToCopyPrefix;

        NodeInfo & repInfo = bi.tmpNodesInfo[bBegin];
        Node repNode = bi.tmpChildren[repInfo.idx];
        repNode.isApproximated = 1;
        for (i64 i = bBegin+1; i < bEnd; i += 1)
        {
            NodeInfo const & toMergeInfo = bi.tmpNodesInfo[i];
            Node const & toMergeNode = bi.tmpChildren[toMergeInfo.idx];
            repNode.state = Model::smf(repNode.state, toMergeNode.state);
            repNode.gValue = calcBetter<Model>(repNode.gValue, toMergeNode.gValue);
            repNode.fValue = calcBetter<Model>(repNode.fValue, toMergeNode.fValue);
        }
        bi.children[bi.nChildren] = repNode;
        bi.nodesInfo[bi.nChildren] = repInfo;
        bi.nChildren += 1;
    }

    // Fix nodes info
    for (i32 i = 0; i < bi.nChildren; i += 1)
    {
        bi.nodesInfo[i].idx = i;
    }

    // printf("AFVALUES =");
    // for (i32 i = 0; i < bi.nChildren; i += 1)
    // {
    //     if( i == bi.nChildrenToCopy-1) printf(" |");
    //     printf(" %.2f", bi.children[bi.nodesInfo[i].idx].fValue);
    // }
    // printf("\n");
}
template<typename Model, typename Node>
void mergeChildren(gfl::i64 const width, BatchInfo<Node> * const batchInfo)
{
    using namespace gfl;
    BatchInfo<Node> & bi = *batchInfo;

    assert(bi.nChildren > width);

    // Sort by fValue
    for (i64 i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo & iInfo = bi.nodesInfo[i];
        i64 const iIdx = iInfo.idx;
        assert(0 <= iIdx);
        assert(iIdx <  bi.nChildren);
        Node const & iNode = bi.children[iIdx];
        iInfo.score = fValueToScore<Model>(iNode.gValue);
    }
    std::sort(bi.nodesInfo,
              bi.nodesInfo + bi.nChildren,
              NodeInfo::cmpByScore);

    // printf("BEFORE MERGING\n");
    // for (i64 i = 0; i < bi.nChildren; i += 1)
    // {
    //     NodeInfo & iInfo = bi.nodesInfo[i];
    //     i64 const iIdx = iInfo.idx;
    //     assert(0 <= iIdx);
    //     assert(iIdx <  bi.nChildren);
    //     Node const & iNode = bi.children[iIdx];
    //     Node::print(iNode);
    // }

    // Parents to back up
    for (auto i = 0; i < bi.nParents; i += 1)
    {
        bi.tmpNodesInfo[i].flag = 0;
    }

    i64 mergedSuffixSize = 0;
    while(bi.nChildren - mergedSuffixSize > width)
    {
        // Pick last node
        i64 const toMergeInfoIdx = bi.nChildren-1-mergedSuffixSize;
        NodeInfo const & toMergeInfo = bi.nodesInfo[toMergeInfoIdx];
        i64 const toMergeNodeIdx = toMergeInfo.idx;
        assert(0 <= toMergeNodeIdx);
        assert(toMergeNodeIdx < bi.nChildren);
        Node const & toMergeNode = bi.children[toMergeNodeIdx];

        // Find better candidate
        // i64 bestCandidateInfoIdx = -1;
        // double bestCandidateScore = 0.0;
        // for (i64 candidateInfoIdx = width / 2; candidateInfoIdx < toMergeInfoIdx; candidateInfoIdx += 1)
        // {
        //     NodeInfo const & candidateInfo = bi.nodesInfo[candidateInfoIdx];
        //     i64 const & candidateNodeIdx = candidateInfo.idx;
        //     assert(0 <= candidateNodeIdx);
        //     assert(candidateNodeIdx < bi.nChildren);
        //     Node const & candidateNode = bi.children[candidateNodeIdx];
        //     double const fValueMin = min(candidateNode.gValue, toMergeNode.gValue);
        //     double const fValueMax = max(candidateNode.gValue, toMergeNode.gValue);
        //     double const fValueScore = (fValueMax - fValueMin) / fValueMax;
        //     double const simScore =
        //         Model::ssf(toMergeNode.state, candidateNode.state) +
        //         1.0 * fValueScore;
        //     if (bestCandidateScore <= simScore) // We use <= to tie-break with fValue
        //     {
        //         bestCandidateScore = simScore;
        //         bestCandidateInfoIdx = candidateInfoIdx;
        //     }
        // }

        i64 bestCandidateInfoIdx = toMergeInfoIdx -1;

        // Merge nodes
        assert(bestCandidateInfoIdx >= 0);
        assert(bestCandidateInfoIdx < bi.nChildren);
        NodeInfo const & bestInfo = bi.nodesInfo[bestCandidateInfoIdx];
        i64 const bestNodeIdx = bestInfo.idx;
        assert(0 <= bestNodeIdx);
        assert(bestNodeIdx < bi.nChildren);
        Node & bestNode = bi.children[bestNodeIdx];

        //printf("Merging %ld <-> %ld, with fValues %.2f %.2f\n", bestCandidateInfoIdx, toMergeInfoIdx, bestNode.fValue, toMergeNode.fValue);
        bestNode.state = Model::smf(bestNode.state, toMergeNode.state);
        bestNode.gValue = calcBetter<Model>(bestNode.gValue, toMergeNode.gValue);
        bestNode.fValue = calcBetter<Model>(bestNode.fValue, toMergeNode.fValue);

        bestNode.isApproximated = 1;
        if (not bestNode.hasAncestorInCutset) bi.tmpNodesInfo[bestInfo.pIdx].flag = 1;
        if (not toMergeNode.hasAncestorInCutset) bi.tmpNodesInfo[toMergeInfo.pIdx].flag = 1;
        mergedSuffixSize += 1;
    }

    // Sort the nodes
    bi.nChildren -= mergedSuffixSize;
    swapPtr(&bi.children, &bi.tmpChildren);
    copyNodes<Node>(&bi.nChildren, bi.children, bi.tmpChildren, bi.nodesInfo);
    for (i32 i = 0; i < bi.nChildren; i += 1)
    {
        bi.nodesInfo[i].idx = i;
    }

    // Save cutset
    for (auto i = 0; i < bi.nChildren; i += 1)
    {
        NodeInfo const & nodeInfo = bi.nodesInfo[i];
        i64 const & nodeIdx = nodeInfo.idx;
        assert(0 <= nodeIdx);
        assert(nodeIdx < bi.nChildren);
        auto & node = bi.children[nodeIdx];
        if (bi.tmpNodesInfo[nodeInfo.pIdx].flag == 1)
        {
            node.hasAncestorInCutset = 1;
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
        assert(bi.cutsetSize <= width * width);
    }
    assert(bi.cutsetSize <= width * width);
    if (nSavedNodes > 0)
    {
        //printf("Saved %d nodes of layer %d in cutset\n", nSavedNodes, bi.cutset[bi.cutsetSize-1].nEdgesSrcToNode);
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
   // assert(std::is_sorted(bi.nodesInfo,bi.nodesInfo + bi.nChildren, NodeInfo::cmpByScore));

    for (auto i = 0; i < bi.nParents; i += 1)
    {
        bi.tmpNodesInfo[i].flag = 0;
    }

    for (auto i = 0; i < bi.nChildrenToMerge; i += 1)
    {
        auto const & cInfo = bi.nodesInfo[bi.nChildrenToCopyPrefix + i];
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

    for (auto i = 0; i < bi.nChildrenToCopyPrefix; i += 1)
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
    for (auto i = 0; i < bi.nChildrenToCopySuffix; i += 1)
    {
        auto const & cInfo = bi.nodesInfo[bi.nChildren - 1 - i];
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
        assert(bi.cutsetSize <= width * width);
    }
    assert(bi.cutsetSize <= width * width);
    if (nSavedNodes > 0)
    {
        //printf("Saved %d nodes of layer %d in cutset\n", nSavedNodes, bi.cutset[bi.cutsetSize-1].nEdgesSrcToNode);
    }
}

