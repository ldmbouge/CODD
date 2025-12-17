#pragma once

#include <cub/cub.cuh>
#include <curand_kernel.h>

#include "Utils.hpp"
#include "LayerInfo.cuh"
#include "StackAllocator.hpp"

enum DDContext : int;
enum LocalContext : int;

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

template <typename T>
GFL_GLOBAL
void swapPtrKernel(T** a, T** b)
{
    swapPtr(a,b);
}

template<typename Node>
GFL_GLOBAL
void prepareForChildrenKernel(LayerInfo<Node> * const layerInfo)
{
    if (layerInfo->nParents != layerInfo->nRepresentatives)
    printf("Parent filtering (%ld): %10ld -> %10ld\n", layerInfo->nParents - layerInfo->nRepresentatives, layerInfo->nParents,layerInfo->nRepresentatives);
    layerInfo->nParents = layerInfo->nRepresentatives;
    layerInfo->nRepresentatives = 0;
}

template<typename Node>
GFL_GLOBAL
void resetLabelsInfo(LayerInfo<Node> * const layerInfo)
{
    layerInfo->labelsInfo.reset();
}

template<typename Model, typename Node>
void calcLabels(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        DDContext const ctx,
        gfl::f64 pBound,
        gfl::f64 dBound)
{
    using namespace gfl;

    Node * const children = layerInfo->children;

    for (i64 cIdx = 0; cIdx < layerInfo->nRepresentatives; cIdx += 1)
    {
        auto & pNode = children[cIdx];
        pNode.labels = model->lgf(pNode.state, ctx, pBound, dBound);
        auto [smallest,largest,count] = pNode.labels.slc();
        layerInfo->labelsInfo.minLabel = std::min(layerInfo->labelsInfo.minLabel,smallest);
        layerInfo->labelsInfo.maxLabel = std::max(layerInfo->labelsInfo.maxLabel,largest);
        layerInfo->labelsInfo.nLabels  = std::max(layerInfo->labelsInfo.nLabels,count);
    }
}

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

    Node * const children = layerInfo->children;

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
    if (cIdx < layerInfo->nRepresentatives)
    {
        auto & pNode = children[cIdx];
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
void calcChildren(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        gfl::f64 pBound,
        LocalContext localCtx)
{
    using State = typename Model::State;
    using namespace gfl;

    Node * const children = layerInfo->children;
    NodeInfo * const childrenInfo = layerInfo->childrenInfo;

    for(i64 pIdx =0; pIdx < layerInfo->nParents; pIdx += 1)
    {
        if (layerInfo->labelsInfo.nLabels > 0)
        {
            Node cNode;
            NodeInfo cInfo;
            Node const pNode = layerInfo->parents[pIdx];
            for (i32 label = layerInfo->labelsInfo.minLabel; label <= layerInfo->labelsInfo.maxLabel; label += 1)
            {
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
                            cInfo.isRepresented = 0;
                            cInfo.idx = layerInfo->nChildren;
                            children[cInfo.idx] = cNode;
                            childrenInfo[cInfo.idx] = cInfo;

                            layerInfo->nChildren += 1;
                        }
                    }
                }
            }
        }
    }
}

template<typename Node>
GFL_GLOBAL
void shuffleRepKernel(
        gfl::i64 const nNodes,
        NodeInfo * const nodesInfo)
{
    using namespace gfl;

    i64 const rIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rIdx < nNodes)
    {
        // Initialize RNG state
        curandState state;
        curand_init(nodesInfo[rIdx].hash, rIdx, 0, &state);

        // Generate random key
        nodesInfo[rIdx].isRepresented = curand(&state) * (1-nodesInfo[rIdx].isRepresented);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcChildrenKernel(
        Model const * const model,
        LayerInfo<Node> * const layerInfo,
        gfl::f64 pBound,
        LocalContext localCtx)
{
    using State = typename Model::State;
    using namespace gfl;

    Node * const children = layerInfo->children;
    NodeInfo * const childrenInfo = layerInfo->childrenInfo;

    assert(blockDim.x == warpSize);

    i64 const pIdx = blockIdx.x;
    if (pIdx < layerInfo->nParents and layerInfo->labelsInfo.nLabels > 0)
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
                        cInfo.isRepresented = 0;
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

template<typename Node>
void copyRep(LayerInfo<Node> * const layerInfo, bool reverse = false)
{
    using namespace gfl;

    for (i64 rIdx  = 0; rIdx < layerInfo->nRepresentatives; rIdx += 1)
    {
        i64 const cIdx = layerInfo->childrenInfo[rIdx].idx;
        i64 const tIdx = not reverse ? rIdx : layerInfo->nRepresentatives - 1 - rIdx;
        layerInfo->children[tIdx] = layerInfo->tmpChildren[cIdx];
    }
}


template<typename Node>
GFL_GLOBAL
void copyRepKernel(
    gfl::i64 const * const nNodes,
    NodeInfo const * const nodesInfo,
    Node const * const src,
    Node * const dst,
    bool reverse = false)
{
    using namespace gfl;

    i64 const rIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rIdx < *nNodes)
    {
        i64 const cIdx = nodesInfo[rIdx].idx;
        i64 const tIdx = not reverse ? rIdx : *nNodes - 1 - rIdx;
        dst[tIdx] = src[cIdx];
    }
}

template<typename T>
GFL_GLOBAL
void copyKernel(
        gfl::i64 const * n,
        T const * const src,
        T * const dst)
{
    using namespace gfl;

    i64 const rIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (rIdx < *n)
    {
        dst[rIdx] = src[rIdx];
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void resetInfoKernel(gfl::i64 const nNodes, Node const * const nodes, NodeInfo * const info)
{
    using namespace gfl;

    i64 cBegin,cEnd;
    getBeginEnd(cBegin, cEnd, blockIdx.x, gridDim.x, nNodes);
    for (i64 i = cBegin + threadIdx.x; i < cEnd; i += blockDim.x) {
        info[i].hash = nodes[i].hash;
        info[i].idx = i;
        info[i].isRepresented = 0;
        info[i].boundSrcToNode = nodes[i].boundSrcToNode;
    }
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
void sortKernel(void * tmpMem, std::size_t tmpMemSize, cub::DoubleBuffer<KeyType> * const doubleBuffer, gfl::i64 const * const nKeys)
{
//    printf("Sorting %d keys\n",*nKeys);
//    printf("TMP = %p (%lu) | K_IN = %p | K_OUT = %p | N_KEYS = %p (%d)\n", tmpMem, tmpMemSize, keysIn, keysOut, nKeys, *nKeys);
    cudaStream_t gpuSortQueue;
    cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
    cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, *doubleBuffer, *nKeys, KeyDecomposer{}, gpuSortQueue);
    cudaStreamDestroy(gpuSortQueue);
}

template<typename Model>
GFL_HOST_DEVICE
void checkStatePair(NodeInfo & iInfo, typename Model::State const & iState, NodeInfo & jInfo, typename Model::State const & jState)
{
    using namespace gfl;

    if (Model::State::equal(iState, jState))
    {
        if (Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode))
        {
            jInfo.isRepresented = 1;
        }
        else
        {
            iInfo.isRepresented = 1;
        }
    }
    if constexpr (Model::has_dom)
    {
        if (Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode) and Model::dom(iState, jState))
        {
            jInfo.isRepresented = 1;
        }
        else if (Model::betterEq(jInfo.boundSrcToNode, iInfo.boundSrcToNode) and Model::dom(jState, iState))
        {
            iInfo.isRepresented = 1;
        }
    }
}

template<typename Model, typename Node>
GFL_HOST_DEVICE
void checkNodesPair(Node & iNode, Node & jNode)
{
    using namespace gfl;

    if (Model::State::equal(iNode.state, jNode.state))
    {
        if (Model::betterEq(iNode.boundSrcToNode, jNode.boundSrcToNode))
        {
            jNode.isRepresented = 1;
        }
        else
        {
            iNode.isRepresented = 1;
        }
    }
    if constexpr (Model::has_dom)
    {
        if (Model::betterEq(iNode.boundSrcToNode, jNode.boundSrcToNode) and Model::dom(iNode.state, jNode.state))
        {
            jNode.isRepresented = 1;
        }
        else if (Model::betterEq(jNode.boundSrcToNode, iNode.boundSrcToNode) and Model::dom(jNode.state, iNode.state))
        {
            iNode.isRepresented = 1;
        }
    }
}

template<typename Model, typename Node>
void calcRep(gfl::i64 const nChildren, Node const * const children,  NodeInfo * const childrenInfo)
{
    using namespace gfl;

    for (i64 i = 0; i < nChildren; i += 1) {
        auto & iInfo = childrenInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < nChildren);
        auto const iChild = children[iInfo.idx].state;
        for (i64 j = i + 1; j < nChildren; j += 1)
        {
            auto & jInfo = childrenInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < nChildren);
            auto const jChild = children[jInfo.idx].state;
            // TODO Check both undominates
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



template<typename Model, typename Node>
GFL_GLOBAL
void calcRepKernel(gfl::i64 const nNodes, Node const * const nodes,  NodeInfo * const nodesInfo)
{
    using namespace gfl;

    i64 cBegin,cEnd;
    getBeginEnd(cBegin,cEnd,blockIdx.x,gridDim.x,nNodes);
    for (i64 i = cBegin + threadIdx.x; i < cEnd; i += blockDim.x) {
        auto & iInfo = nodesInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < nNodes);
        auto const iChild = nodes[iInfo.idx].state;
        for (i64 j = i + 1; j < nNodes; j += 1)
        {
            auto & jInfo = nodesInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < nNodes);
            auto const jChild = nodes[jInfo.idx].state;
            // TODO Check both undominates
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

template<typename Model, typename Node>
GFL_GLOBAL
void calcRepKernel(gfl::i64 const nNodes, Node * const nodes)
{
    using namespace gfl;

    i64 cBegin,cEnd;
    getBeginEnd(cBegin,cEnd,blockIdx.x,gridDim.x,nNodes);
    for (i64 i = cBegin + threadIdx.x; i < cEnd; i += blockDim.x) {
        auto & iChild = nodes[i];
        for (i64 j = i + 1; j < nNodes; j += 1)
        {
            auto & jChild = nodes[j];
            // TODO Check both undominates
            if (iChild.hash == jChild.hash)
            {
                checkNodesPair<Model,Node>(iChild, jChild);
            }
            else
            {
                break;
            }
        }
    }
}


template<typename Node>
void countRep(LayerInfo<Node> * const layerInfo,  NodeInfo const * const childrenInfo)
{
    using namespace gfl;
    for (i64 cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        layerInfo->nRepresentatives += childrenInfo[cIdx].isRepresented == 0;
    }
}

template<typename Node>
GFL_GLOBAL
void countRepKernel(gfl::i64 const nNodes, NodeInfo const * const nodesInfo, gfl::i64 * nRep)

{
    using namespace gfl;

    i64 const cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx < nNodes)
    {
        u32 const maskActiveThreads = __activemask();
        bool isRepresentative = nodesInfo[cIdx].isRepresented == 0;
        u32 const maskRepresentatives = __ballot_sync(maskActiveThreads, isRepresentative);
        i32 const nRepresentatives = popcount(maskRepresentatives);
        if (nRepresentatives > 0)
        {
            if (laneIdx() == 0)
            {
                atomicAdd((ull *) nRep, (ull) nRepresentatives);
            }
        }
    }
}

template<typename Node>
GFL_GLOBAL
void countRepKernel(gfl::i64 const nNodes, Node const * const nodes, gfl::i64 * nRep)

{
    using namespace gfl;

    i64 const cIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cIdx < nNodes)
    {
        u32 const maskActiveThreads = __activemask();
        bool isRepresentative = nodes[cIdx].isRepresented == 0;
        u32 const maskRepresentatives = __ballot_sync(maskActiveThreads, isRepresentative);
        i32 const nRepresentatives = popcount(maskRepresentatives);
        if (nRepresentatives > 0)
        {
            if (laneIdx() == 0)
            {
                atomicAdd((ull *) nRep, (ull) nRepresentatives);
            }
        }
    }
}


GFL_DEVICE inline
void printNodeInfo(NodeInfo const & childInfo)
{
    if(childInfo.isRepresented)
    printf("HASH = %lu, IDX = %ld, IS_REP = %d\n",
           childInfo.hash,
           childInfo.idx,
           childInfo.isRepresented);
}

template<typename Node>
GFL_GLOBAL
void printInfo(gfl::i64 const * const nNodes, NodeInfo const * info)
{
    printf("--- (%llu)\n", *nNodes);
    for(auto cIdx = 0; cIdx < *nNodes; cIdx += 1)
    {
        auto const & childInfo = info[cIdx];
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



template<typename Node>
GFL_GLOBAL
void printDebugInfo(LayerInfo<Node> const * const  layerInfo)
{
    printf("Reprs %ld\n", layerInfo->nRepresentatives);
}
