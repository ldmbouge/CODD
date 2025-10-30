#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "LayerInfo.cuh"
#include "StackAllocator.hpp"

enum DDContext : int;
enum LocalContext : int;

template<typename Model, typename Node>
GFL_GLOBAL
void calcLabelsKernel(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        DDContext const ctx,
        gfl::f64 pBound,
        gfl::f64 dBound)
{
    using namespace gfl;

    assert(gridDim.x * blockDim.x >= layerInfo->nChildren);

    __shared__ i32 minLabel_s;
    __shared__ i32 maxLabel_s;
    __shared__ i32 labelsPerParent_s;

    if (threadIdx.x == 0)
    {
        minLabel_s = numeric_limits<i32>::max();
        maxLabel_s = numeric_limits<i32>::min();
        labelsPerParent_s  = 0;
    }
    __syncthreads();

    i64 const cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx < layerInfo->nChildren)
    {
        auto & pNode = layerInfo->children[cIdx];
        pNode.labels = model->lgf(pNode.state, ctx, pBound, dBound);
        auto [smallest,largest,count] = pNode.labels.slc();
        atomicMin_block(&minLabel_s, smallest);
        atomicMax_block(&maxLabel_s, largest);
        atomicMax_block(&labelsPerParent_s, count);
        //printf("Parent %d has %d labels\n", pIdx, count);
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicMin(&layerInfo->labelsInfo.minLabel,minLabel_s);
        atomicMax(&layerInfo->labelsInfo.maxLabel,maxLabel_s);
        atomicMax(&layerInfo->labelsInfo.nLabels, labelsPerParent_s);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcChildrenKernel(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        NodeInfo * const childrenInfo,
        Node * children,
        gfl::f64 pBound,
        LocalContext localCtx)
{
    using State = typename Model::State;
    using namespace gfl;

    assert(blockDim.x == warpSize);

    i64 const pIdx = blockIdx.x;
    if (pIdx < layerInfo->nParents)
    {
        Node cNode;
        NodeInfo cInfo;
        Node const pNode = layerInfo->parents[pIdx];
        for (i32 label = layerInfo->labelsInfo.minLabel + laneIdx(); label <= layerInfo->labelsInfo.maxLabel; label += warpSize)
        {
            u32 const maskActiveThreads = __activemask();
            i32 haveChild = 0;
            if (pNode.labels.contains(label))
            {
                // Transition
                auto cState = model->stf(pNode.state, label);
                if (cState.has_value())
                {
                    f64 cBoundSrcToNode = pNode.boundSrcToNode + model->scf(pNode.state, label);
                    f64 cHeuristicNodeToSink;
                    if constexpr (Model::has_local)
                        cHeuristicNodeToSink = model->local(cState.value(), localCtx);
                    else
                        cHeuristicNodeToSink = 0;
                    f64 cCost = cBoundSrcToNode + cHeuristicNodeToSink;
                    if (Model::better(cCost, pBound))
                    {
                        haveChild = 1;

                        // Node
                        cNode.state = cState.value();
                        cNode.boundSrcToNode = cBoundSrcToNode;
                        memcpy(cNode.labelsSrcToNode, pNode.labelsSrcToNode, sizeof(cNode.labelsSrcToNode));
                        cNode.labelsSrcToNode[pNode.nEdgesSrcToNode] = label;
                        cNode.nEdgesSrcToNode = pNode.nEdgesSrcToNode + 1;

                        // NodeInfo
                        if constexpr (Model::has_dom)
                            cInfo.hash = Model::domHash(cNode.state);
                        else
                            cInfo.hash = State::hash(cNode.state);
                        cInfo.boundSrcToNode = cBoundSrcToNode;
                        cInfo.isRepresented = false;
                    }
                }
            }

            u32 const maskThreadsWithChild = __ballot_sync(maskActiveThreads, haveChild);
            i32 const nThreadsWithChild = popcount(maskThreadsWithChild);
            if (nThreadsWithChild > 0)
            {
                i64 nChildrenInGlobal = laneIdx() == 0 ? atomicAdd((ull *) &layerInfo->nChildren, (ull) nThreadsWithChild) : 0;
                nChildrenInGlobal = __shfl_sync(maskActiveThreads, nChildrenInGlobal, 0);
                if (haveChild)
                {
                    u32 const maskThreadsBefore = maskFilledThrough<u32>(laneIdx());
                    i64 const offset = popcount(maskThreadsWithChild & maskThreadsBefore);
                    cInfo.idx = nChildrenInGlobal + offset;
                    children[cInfo.idx] = cNode;
                    childrenInfo[cInfo.idx] = cInfo;
                }

//                if (laneIdx() == 1)
//                {
//                    printf("P %ld has %d children. I see %ld (%ld) children in global\n", pIdx, nThreadsWithChild, layerInfo->nChildren,nChildrenInGlobal);
//                }
            }
        }
    }
}


template<typename Model, typename Node>
GFL_GLOBAL
void cpyChildrenKernel(
        Model const * const m,
        LayerInfo<Node> * const layerInfo,
        NodeInfo * const childrenInfo,
        Node * const childrenIn,
        Node * const childrenOut)
{
    using namespace gfl;
    i64 const cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx < layerInfo->tmpInt)
    {
        u32 const maskActiveThreads = __activemask();
        i32 haveChild = not childrenInfo[cIdx].isRepresented;
        u32 const maskThreadsWithChild = __ballot_sync(maskActiveThreads, haveChild);
        i32 const nThreadsWithChild = popcount(maskThreadsWithChild);
        if (nThreadsWithChild > 0)
        {
            i64 nChildrenInGlobal = laneIdx() == 0 ? atomicAdd((ull *) &layerInfo->nChildren, (ull) nThreadsWithChild) : 0;
            nChildrenInGlobal = __shfl_sync(maskActiveThreads, nChildrenInGlobal, 0);
            if (haveChild)
            {
                Node cNode = childrenIn[childrenInfo[cIdx].idx];
                u32 const maskThreadsBefore = maskFilledThrough<u32>(laneIdx());
                i64 const offset = popcount(maskThreadsWithChild & maskThreadsBefore);
                childrenOut[nChildrenInGlobal + offset] = cNode;
                if (m->isTarget(cNode.state))
                {
                    layerInfo->hasTarget = true;
                }
            }
        }
    }
//    if(blockIdx.x == 0 and threadIdx.x == 0)
//    {
//        printf("I see %lu children\n", layerInfo->tmpInt);
//        for(i64 cIdx = 0; cIdx < layerInfo->tmpInt; cIdx += 1)
//        {
//            if(not childrenInfo[cIdx].isRepresented)
//            {
//
//            }
//        }
//    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcChildrenBlockKernel(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        NodeInfo * const childrenInfo,
        gfl::f64 pBound,
        LocalContext localCtx)
{
    using State = typename Model::State;
    using namespace gfl;

    //assert(gridDim.x >= layerInfo->nParents);

    __shared__ i32 nChildrenInShared_s;
    __shared__ i32 nChildrenInGlobal_s;
    __shared__ Node * children_s;
    __shared__ NodeInfo * childrenInfo_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    i32 const pIdx = blockIdx.x;
    if (pIdx < layerInfo->nParents)
    {
        if (threadIdx.x == 0)
        {
            nChildrenInShared_s = 0;
            StackAllocator allocator(shrMem, getSharedMemSize());
            children_s = allocator.allocateArray<Node>(layerInfo->labelsPerParents);
            childrenInfo_s = allocator.allocateArray<NodeInfo>(layerInfo->labelsPerParents);
        }
        __syncthreads();

        Node cNode_r;
        NodeInfo cInfo_r;

        Node const pNode_r = layerInfo->parents[pIdx];
        for (i32 label = layerInfo->minLabel + threadIdx.x; label <= layerInfo->maxLabel; label += blockDim.x)
        {
            if (pNode_r.labels.contains(label))
            {
                // Transition
                auto cState_r = model->stf(pNode_r.state, label);
                if (cState_r.has_value())
                {
                    f64 cBoundSrcToNode = pNode_r.boundSrcToNode + model->scf(pNode_r.state, label);
                    f64 cHeuristicNodeToSink = Model::has_local ? model->local(cState_r.value(), localCtx) : 0;
                    f64 cCost = cBoundSrcToNode + cHeuristicNodeToSink;
                    if (Model::better(cCost, pBound))
                    {
                        // Node
                        cNode_r.state = cState_r.value();
                        cNode_r.boundSrcToNode = cBoundSrcToNode;
                        memcpy(cNode_r.labelsSrcToNode, pNode_r.labelsSrcToNode, sizeof(cNode_r.labelsSrcToNode));
                        cNode_r.labelsSrcToNode[pNode_r.nEdgesSrcToNode] = label;
                        cNode_r.nEdgesSrcToNode = pNode_r.nEdgesSrcToNode + 1;

                        // NodeInfo
                        cInfo_r.hash = Model::has_dom ? Model::domHash(cNode_r.state) : State::hash(cNode_r.state);
                        cInfo_r.boundSrcToNode = cBoundSrcToNode;
                        cInfo_r.idx = -1;
                        cInfo_r.isRepresented = 0;

                        // Write children and info in shared.
                        auto const idxInShared = atomicAdd_block(&nChildrenInShared_s, 1);
                        assert(idxInShared < layerInfo->labelsPerParents);
                        children_s[idxInShared] = cNode_r;
                        childrenInfo_s[idxInShared] = cInfo_r;
                    }
                }
            }
        }
        __syncthreads();

        // Write children in global
        if (threadIdx.x == 0)
        {
            nChildrenInGlobal_s = atomicAdd((ull*)&layerInfo->nChildren, (ull)nChildrenInShared_s);
        }
        __syncthreads();

        for(i32 i = threadIdx.x; i < nChildrenInShared_s; i += blockDim.x)
        {
            i32 const idxInGlobal = nChildrenInGlobal_s + i;
            childrenInfo_s[i].idx = idxInGlobal;
            layerInfo->children[idxInGlobal] = children_s[i];
            childrenInfo[idxInGlobal] = childrenInfo_s[i];
        }
    }
}

template<typename KeyType, typename  KeyDecomposer>
GFL_GLOBAL
void sortKernel(void * tmpMem, std::size_t tmpMemSize, KeyType const * keysIn, KeyType * keysOut, gfl::i64 const * const nKeys)
{
//    printf("Sorting %d keys\n",*nKeys);
//    printf("TMP = %p (%lu) | K_IN = %p | K_OUT = %p | N_KEYS = %p (%d)\n", tmpMem, tmpMemSize, keysIn, keysOut, nKeys, *nKeys);
    cudaStream_t gpuSortQueue;
    cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
    cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, keysIn, keysOut, *nKeys, KeyDecomposer{}, gpuSortQueue);
}

template<typename Model>
GFL_HOST_DEVICE
void checkStatePair(NodeInfo & iInfo, typename Model::State const & iState, NodeInfo & jInfo, typename Model::State const & jState)
{
    using namespace gfl;

    if (Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode))
    {
        if (Model::State::equal(iState, jState))
        {
            jInfo.isRepresented = 1;
        }
    }
    else if (Model::better(jInfo.boundSrcToNode, iInfo.boundSrcToNode))
    {
        if (Model::State::equal(iState, jState))
        {
            iInfo.isRepresented = 1;
        }
    }
    if constexpr (Model::has_dom)
    {
        if (Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode) and Model::dom(iState, jState))
        {
            jInfo.isRepresented = static_cast<i32>(true);
        }
        else if (Model::betterEq(jInfo.boundSrcToNode, iInfo.boundSrcToNode) and Model::dom(jState, iState))
        {
            iInfo.isRepresented = static_cast<i32>(true);
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcReprKernel(LayerInfo<Node> * const layerInfo, NodeInfo * const childrenInfo, Node * const children)
{
    using namespace gfl;

    i32 cBegin,cEnd;
    getBeginEnd(cBegin,cEnd,blockIdx.x,gridDim.x,layerInfo->nChildren);
    for (i32 i = cBegin + threadIdx.x; i < cEnd; i += blockDim.x)
    {
        auto & iInfo = childrenInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < layerInfo->nChildren);
        auto const iChild = children[iInfo.idx].state;
        for (i32 j = i + 1; j < layerInfo->nChildren; j += 1)
        {
            auto & jInfo = childrenInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < layerInfo->nChildren);
            auto const jChild = children[jInfo.idx].state;
            if (iInfo.hash == jInfo.hash)
            {
                checkStatePair<Model>(iInfo, iChild, jInfo, jChild);
            }
            else
            {
                break;
            }
        }
    }
}

template<typename Node>
GFL_GLOBAL
void resetChildrenCount(LayerInfo<Node> * const layerInfo)
{
   layerInfo->tmpInt = layerInfo->nChildren;
   layerInfo->nChildren = 0;
}

GFL_DEVICE inline
void printNodeInfo(NodeInfo const & childInfo)
{
    printf("HASH = %lu, IDX = %ld, IS_REP = %d\n",
           childInfo.hash,
           childInfo.idx,
           childInfo.isRepresented);
}

template<typename Node>
GFL_GLOBAL
void printChildrenInfo(LayerInfo<Node> * layerInfo, NodeInfo * childrenInfo)
{
    printf("--- (%d)\n", layerInfo->nChildren);
    for(auto cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        auto const & childInfo = childrenInfo[cIdx];
        printNodeInfo(childInfo);
    }
}

template<typename Node>
GFL_GLOBAL
void printNotRepCount(LayerInfo<Node> * layerInfo, NodeInfo * childrenInfo)
{
    gfl::i64 count  = 0;
    for(auto cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        count += not childrenInfo[cIdx].isRepresented;
    }
    printf("Rep count = %ld\n", count);
}


