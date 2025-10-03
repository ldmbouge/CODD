#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "LayerInfo.cuh"
#include "StackAllocator.hpp"

enum DDContext : int;
enum LocalContext : int;

template<typename Model>
GFL_GLOBAL
void calcLabelsKernel(Model const * const model, LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, DDContext const ctx)
{
    using namespace gfl;

    assert(gridDim.x * blockDim.x >= layerInfo->nParents);

    __shared__ i32 minLabel_s;
    __shared__ i32 maxLabel_s;
    __shared__ i32 nLabels_s;

    if (threadIdx.x == 0)
    {
        minLabel_s = numeric_limits<i32>::max();
        maxLabel_s = numeric_limits<i32>::min();
        nLabels_s  = 0;
    }
    __syncthreads();

    int const pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx < layerInfo->nParents)
    {
        //printf("Working on parent %d\n", pIdx);
        auto & parent = layerInfo->parents[pIdx];
        parent.labels = model->lgf(parent.state, ctx);
        auto [smallest,largest,count] = parent.labels.slc();
        atomicMin_block(&minLabel_s, smallest);
        atomicMax_block(&maxLabel_s, largest);
        atomicMax_block(&nLabels_s, count);
    }
    __syncthreads();

    if (pIdx < layerInfo->nParents)
    {
        //printf("Parent %d has %d labels\n", pIdx, nLabels_s);
        atomicMin(&layerInfo->minLabel,minLabel_s);
        atomicMax(&layerInfo->maxLabel,maxLabel_s);
        atomicMax(&layerInfo->labelsPerParents, nLabels_s);
        atomicAdd(&layerInfo->nLables, nLabels_s);
    }
}

template<typename Model>
GFL_GLOBAL
void calcChildrenKernel(
        Model const * const model,
        LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo,
        ChildInfo * const childrenInfo,
        gfl::f64 primalBound,
        LocalContext localCtx)
{
    using State = typename Model::State;
    using Labels = typename Model::Labels;
    using GpuParent = GpuParent<State, Labels>;
    using GpuChild = GpuChild<State>;
    using namespace gfl;

    assert(gridDim.x >= layerInfo->nParents);

    __shared__ i32 nChildrenInShared_s;
    __shared__ i32 nChildrenInGlobal_s;
    __shared__ GpuChild * children_s;
    __shared__ ChildInfo * childrenInfo_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    i32 const pIdx = blockIdx.x;
    if (pIdx < layerInfo->nParents)
    {
        if (threadIdx.x == 0)
        {
            nChildrenInShared_s = 0;
            StackAllocator allocator(shrMem, getSharedMemSize());
            children_s = allocator.allocateArray<GpuChild>(layerInfo->labelsPerParents);
            childrenInfo_s = allocator.allocateArray<ChildInfo>(layerInfo->labelsPerParents);
        }
        __syncthreads();

        GpuChild child_r;
        ChildInfo childInfo_r;

        GpuParent const parent_r = layerInfo->parents[pIdx];
        for (i32 label = layerInfo->minLabel + threadIdx.x; label <= layerInfo->maxLabel; label += blockDim.x)
        {
            if (parent_r.labels.contains(label))
            {
                // Transition
                child_r.parentNode = parent_r.node;
                child_r.label = label;
                auto state_r = model->stf(parent_r.state, label);
                if (state_r.has_value())
                {
                    childInfo_r.boundSrcToNode = parent_r.boundSrcToNode + model->scf(parent_r.state, label);
                    child_r.heuristicNodeToSink = Model::has_local ? model->local(state_r.value(), localCtx) : 0;
                    childInfo_r.cost = childInfo_r.boundSrcToNode + child_r.heuristicNodeToSink;

                    if (Model::better(childInfo_r.cost, primalBound))
                    {
                        child_r.state = state_r.value();
                        childInfo_r.id = pIdx * (layerInfo->maxLabel + 1) + label;
                        childInfo_r.isRepresented = static_cast<i32>(false);
                        childInfo_r.hash = Model::has_dom ? Model::domHash(child_r.state) : State::hash(child_r.state);
                        // Write children and info in shared.
                        auto const idxInShared = atomicAdd_block(&nChildrenInShared_s, 1);
                        assert(idxInShared < layerInfo->labelsPerParents);
                        children_s[idxInShared] = child_r;
                        childrenInfo_s[idxInShared] = childInfo_r;
                    }
                }
            }
        }
        __syncthreads();

        // Write children in global
        if (threadIdx.x == 0)
        {
            nChildrenInGlobal_s = atomicAdd(&layerInfo->nChildren, nChildrenInShared_s);
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
void sortKernel(void * tmpMem, std::size_t tmpMemSize, KeyType const * keysIn, KeyType * keysOut, gfl::i32 const * const nKeys)
{
    //printf("Sorting %d keys\n",*nKeys);
    //printf("TMP = %p (%ul) | K_IN = %p | K_OUT = %p | N_KEYS = %p (%d)\n",tmpMem, tmpMemSize, keysIn, keysOut, nKeys, *nKeys);
    cudaStream_t gpuSortQueue;
    cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
    cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, keysIn, keysOut, *nKeys, KeyDecomposer{}, gpuSortQueue);
}

template<typename Model>
GFL_DEVICE
void checkStatePair(ChildInfo & iInfo, typename Model::State const & iState, ChildInfo & jInfo, typename Model::State const & jState)
{
    using namespace gfl;

    if (iInfo.boundSrcToNode == jInfo.boundSrcToNode)
    {
        if (Model::State::equal(iState, jState))
        {
            jInfo.isRepresented = static_cast<i32>(true);
        }
    }
    if (Model::has_dom and Model::betterEq(jInfo.boundSrcToNode, iInfo.boundSrcToNode))
    {
        if (Model::dom(jState, iState))
        {
            iInfo.isRepresented = static_cast<i32>(true);
        }
    }
    if (Model::has_dom and Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode))
    {
        if (Model::dom(iState, jState))
        {
            jInfo.isRepresented = static_cast<i32>(true);
        }
    }
}

template<typename Model>
GFL_GLOBAL
void calcReprKernel(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo * const childrenInfo)
{
    using namespace gfl;

    i32 cBegin,cEnd;
    getBeginEnd(cBegin,cEnd,blockIdx.x,gridDim.x,layerInfo->nChildren);
    for (i32 i = cBegin + threadIdx.x; i < cEnd; i += blockDim.x)
    {
        auto & iInfo = childrenInfo[i];
        assert(0 <= iInfo.idx);
        assert(iInfo.idx < layerInfo->nChildren);
        auto const iChild = layerInfo->children[iInfo.idx].state;
        for (auto j = i + 1; j < layerInfo->nChildren; j += 1)
        {
            auto & jInfo = childrenInfo[j];
            assert(0 <= jInfo.idx);
            assert(jInfo.idx < layerInfo->nChildren);
            auto const jChild = layerInfo->children[jInfo.idx].state;
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

template<typename Model>
GFL_GLOBAL
void printParents(LayerInfo<typename Model::State, typename Model::Labels> * layerInfo)
{
    using State = typename Model::State;
    using Labels = typename Model::Labels;
    using GpuParent = GpuParent<State, Labels>;
    printf("+++\n");
    for(auto pIdx = 0; pIdx < layerInfo->nParents; pIdx += 1)
    {
        GpuParent const & p = layerInfo->parents[pIdx];
        printf("AN %p\n", p.node);
    }
}

GFL_DEVICE inline
void printChildInfo(ChildInfo const & childInfo)
{
    printf("HASH = %lu, COST = %.1f, ID = %ld, IDX = %d, IS_REP = %d\n",
           childInfo.hash,
           childInfo.cost,
           childInfo.id,
           childInfo.idx,
           childInfo.isRepresented);
}

template<typename Model>
GFL_GLOBAL
void printChildrenInfo(LayerInfo<typename Model::State, typename Model::Labels> * layerInfo, ChildInfo * childrenInfo)
{
    printf("---\n");
    for(auto cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        auto const & childInfo = childrenInfo[cIdx];
        printChildInfo(childInfo);
    }
}


