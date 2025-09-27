#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "LayerInfo.cuh"
#include "StackAllocator.hpp"

enum DDContext : int;
enum LocalContext : int;

template<typename Model>
__global__
void calcLabelsKernel(Model const * const model, LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, DDContext const ctx)
{
    using namespace gfl;

    __shared__ i32 minLabel_s;
    __shared__ i32 maxLabel_s;
    __shared__ i32 nLabels_s;

    if (threadIdx.x == 0)
    {
        minLabel_s = numeric_limits<i32>::max();
        maxLabel_s = numeric_limits<i32>::min();
        nLabels_s  = numeric_limits<i32>::min();
    }
    __syncthreads();

    int const pIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pIdx < layerInfo->nParents)
    {
        //printf("Working on parent %d\n", pIdx);
        auto & parent = layerInfo->parents[pIdx];
        auto labels = model->lgf(parent.state, ctx);
        auto [smallest,largest,count] = labels.slc();
        atomicMin_block(&minLabel_s, smallest);
        atomicMax_block(&maxLabel_s, largest);
        atomicMax_block(&nLabels_s, count);
        parent.labels = labels;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        atomicMin(&layerInfo->minLabel,minLabel_s);
        atomicMax(&layerInfo->maxLabel,maxLabel_s);
        atomicMax(&layerInfo->nLabels,nLabels_s);
    }
}

template<typename Model>
__global__
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

    static_assert(alignof(GpuParent) >= 16);
    static_assert(sizeof(GpuParent) % 16 == 0);

    __shared__ i32 nChildrenInShared_s;
    __shared__ i32 nChildrenInGlobal_s;
    __shared__ GpuChild * children_s;
    __shared__ ChildInfo * childrenInfo_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned
    if (threadIdx.x == 0)
    {
        nChildrenInShared_s = 0;
        StackAllocator allocator(shrMem, getSharedMemSize());
        children_s = allocator.allocateArray<GpuChild>(layerInfo->nLabels);
        childrenInfo_s = allocator.allocateArray<ChildInfo>(layerInfo->nLabels);
    }
    __syncthreads();

    GpuChild child_r;
    ChildInfo childInfo_r;
    i32 const pIdx = blockIdx.x;
    GpuParent const parent_r = layerInfo->parents[pIdx];
    int const label = layerInfo->minLabel + blockIdx.y * blockDim.x + threadIdx.x;
    if (parent_r.labels.contains(label))
    {
        // Transition
        child_r.parentNode = parent_r.node;
        child_r.label = label;
        auto state_r = model->stf(parent_r.state, label);
        if (state_r.has_value())
        {
            childInfo_r.boundSrcToNode = parent_r.boundSrcToNode + model->scf(parent_r.state, label);
            child_r.heuristicNodeToSink = model->has_local ? model->local(state_r.value(), localCtx) : 0;
            childInfo_r.cost = childInfo_r.boundSrcToNode + child_r.heuristicNodeToSink;

                if (Model::better(childInfo_r.cost, primalBound))
                {
                    child_r.state = state_r.value();
                    childInfo_r.id = pIdx * layerInfo->maxLabel + label;
                    childInfo_r.isRepresented = static_cast<i32>(false);
                    childInfo_r.hash = Model::has_dom ? Model::domHash(child_r.state) : State::hash(child_r.state);

                // Write children and info in shared.
                auto const idxInShared = atomicAdd_block(&nChildrenInShared_s,1);
                children_s[idxInShared] = child_r;
                childrenInfo_s[idxInShared] = childInfo_r;
            }
        }
    }
    __syncthreads();

    // Write children in global
    if (threadIdx.x == 0)
    {
        nChildrenInGlobal_s = atomicAdd(&layerInfo->nChildren,nChildrenInShared_s);
    }
    __syncthreads();

    if (threadIdx.x < nChildrenInShared_s)
    {
        i32 const idxInGlobal = nChildrenInGlobal_s + threadIdx.x;
        childrenInfo_s[threadIdx.x].idx = idxInGlobal;
        layerInfo->children[idxInGlobal] = children_s[threadIdx.x];
        childrenInfo[idxInGlobal] = childrenInfo_s[threadIdx.x];
    }
}

template<typename KeyType, typename  KeyDecomposer>
__global__
void sortKernel(void * tmpMem, std::size_t tmpMemSize, KeyType const * keysIn, KeyType * keysOut, gfl::i32 const * const nKeys)
{
    //printf("Sorting %d keys\n",*nKeys);
    //printf("TMP = %p (%ul) | K_IN = %p | K_OUT = %p | N_KEYS = %p (%d)\n",tmpMem, tmpMemSize, keysIn, keysOut, nKeys, *nKeys);
    if (*nKeys > 1)
    {
        cudaStream_t gpuSortQueue;
        cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
        cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, keysIn, keysOut, *nKeys, KeyDecomposer{}, gpuSortQueue);
    }
}

template<typename KeyType>
__global__
void sortKernel(void * tmpMem, std::size_t tmpMemSize, KeyType const * keysIn, KeyType * keysOut, gfl::i32 const * const nKeys)
{
    //printf("Sorting %d keys\n",*nKeys);
    //printf("TMP = %p (%ul) | K_IN = %p | K_OUT = %p | N_KEYS = %p (%d)\n",tmpMem, tmpMemSize, keysIn, keysOut, nKeys, *nKeys);
    if (*nKeys > 1)
    {
        cudaStream_t gpuSortQueue;
        cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
        cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, keysIn, keysOut, *nKeys, 0, sizeof(KeyType) * 8, gpuSortQueue);
    }
}

template<typename Model>
__global__
void calcClassesKernel(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo const * const childrenInfo)
{
    using namespace gfl;

    __shared__ i32 nClassesInShared_s;
    __shared__ i32 nClassesInGlobal_s;
    __shared__ ClassRange *classesRage_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    i32 const currIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (currIdx < layerInfo->nChildren)
    {
        if (threadIdx.x == 0)
        {
            nClassesInShared_s = 0;
            StackAllocator allocator(shrMem, getSharedMemSize());
            classesRage_s = allocator.allocateArray<ClassRange>(blockDim.x);
        }
        __syncthreads();

        u64 prevHash = 0;  // Must be different
        u64 currHash = 1;
        u64 nextHash = 2;
        i32 const currIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (1 <= currIdx and currIdx <= layerInfo->nChildren - 1)
        {
            prevHash = childrenInfo[currIdx - 1].hash;
        }
        assert(0 <= currIdx);
        assert(currIdx <= layerInfo->nChildren - 1);
        currHash = childrenInfo[currIdx].hash;
        if (0 <= currIdx and currIdx <= layerInfo->nChildren - 2)
        {
            nextHash = childrenInfo[currIdx + 1].hash;
        }
        //printf("Considering %d/%d = (%lu,%lu,%lu)\n",currIdx,layerInfo->nChildren,prevHash,currHash,nextHash);
        if (prevHash != currHash and currHash == nextHash) // First of an untrivial class
        {
            i32 endIdx = currIdx + 2;
            for (; endIdx < layerInfo->nChildren; endIdx += 1)
            {
                if (currHash != childrenInfo[endIdx].hash)
                {
                    break;
                }
            }
            auto const idxInShared = atomicAdd_block(&nClassesInShared_s, 1);
            classesRage_s[idxInShared] = {currIdx, endIdx};
        }
        __syncthreads();

        // Write classes in global
        if (threadIdx.x == 0)
        {
            nClassesInGlobal_s = atomicAdd(&layerInfo->nClasses, nClassesInShared_s);
        }
        __syncthreads();

        if (threadIdx.x < nClassesInShared_s)
        {
            i32 const idxInGlobal = nClassesInGlobal_s + threadIdx.x;
            layerInfo->classesRange[idxInGlobal] = classesRage_s[threadIdx.x];
        }
    }
}


template<typename Model>
__global__
void calcReprKernel(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo * const childrenInfo)
{
    using namespace gfl;

    assert(layerInfo->nClasses <= layerInfo->nChildren);

    auto const clIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (clIdx == 0)
    {
        printf("CH = %d | NTC = %d (%.2f)\n", layerInfo->nChildren, layerInfo->nClasses, gfl::div(layerInfo->nClasses, layerInfo->nChildren+1));
    }
    if (clIdx < layerInfo->nClasses)
    {
        ClassRange const clr = layerInfo->classesRange[clIdx];
        for (auto i = clr.begin; i < clr.end - 1; i += 1)
        {
            auto & iInfo = childrenInfo[i];
            assert(iInfo.idx >= 0);
            assert(iInfo.idx < layerInfo->nChildren);
            auto const iChild = layerInfo->children[iInfo.idx].state;
            for (auto j = i + 1; j < clr.end; j += 1)
            {
                auto & jInfo = childrenInfo[j];
                assert(jInfo.idx >= 0);
                assert(jInfo.idx <= layerInfo->nChildren);
                auto const jChild = layerInfo->children[jInfo.idx].state;
                auto const ijEqual = Model::State::equal(iChild, jChild);
                if (ijEqual)
                {
                    if (iInfo.id < jInfo.id)
                    {
                        jInfo.isRepresented = static_cast<i32>(true);
                    }
                    else
                    {
                        iInfo.isRepresented = static_cast<i32>(true);
                    }
                }
                if (Model::has_dom)
                {
                    auto const ijDomEqual = Model::domEq(iChild, jChild);
                    if (ijDomEqual)
                    {
                        auto const iIsDominated = Model::betterEq(jInfo.boundSrcToNode, iInfo.boundSrcToNode) and
                                                  Model::dom(jChild, iChild);
                        auto const jIsDominated = Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode) and
                                                  Model::dom(iChild, jChild);
                        if (iIsDominated)
                        {
                            iInfo.isRepresented = static_cast<i32>(true);
                        }
                        if (jIsDominated)
                        {
                            jInfo.isRepresented = static_cast<i32>(true);
                        }
                    }
                }
            }
        }
    }
}


template<typename Model>
__global__
void printChildInfo(LayerInfo<typename Model::State, typename Model::Labels> * layerInfo, ChildInfo * childrenInfo)
{
    printf("+++\n");
    for(auto cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        auto const & childInfo = childrenInfo[cIdx];
        printf("HASH = %llu, COST = %.1f, ID = %lld, IDX = %d, IS_REP = %d\n",
               childInfo.hash,
               childInfo.boundSrcToNode,
               childInfo.id,
               childInfo.idx,
               childInfo.isRepresented);
    }
}

template<typename Model>
__global__
void printClasses(LayerInfo<typename Model::State, typename Model::Labels> * layerInfo, ClassRange const * const classesRange)
{
    printf("---\n");
    printf("CH = %d | CL = %d\n", layerInfo->nChildren, layerInfo->nClasses);
    for(auto clIdx = 0; clIdx < layerInfo->nClasses; clIdx += 1)
    {
        ClassRange const clr = classesRange[clIdx];
        printf("(%d,%d) ", clr.begin, clr.end);
    }
    printf("\n");
}