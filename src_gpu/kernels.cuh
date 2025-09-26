#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "LayerInfo.cuh"
#include "StackAllocator.hpp"

enum LocalContext : int;

template<typename Model>
__global__
void calcChildrenKernel(
        Model const * const model,
        LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo,
        ChildInfo * const childrenInfo,
        gfl::f64 primalBound,
        LocalContext localCtx)
{
    /* TO OPTIMIZE
     * - Optimize parent load
     * - Rework to parallelize on maxChildren
     */

    using State = typename Model::State;
    using Labels = typename Model::Labels;
    using GpuParent = GpuParent<State, Labels>;
    using GpuChild = GpuChild<State>;
    using namespace gfl;

    static_assert(alignof(GpuParent) >= 16);
    static_assert(sizeof(GpuParent) % 16 == 0);

    __shared__ i32 nChildrenInShared_s;
    __shared__ i32 nChildrenInGlobal_s;
    __shared__ GpuParent * parent_s;
    __shared__ GpuChild * children_s;
    __shared__ ChildInfo * childrenInfo_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    auto const pIdx = blockIdx.x;

    if (threadIdx.x == 0)
    {
        nChildrenInShared_s = 0;
        StackAllocator allocator(shrMem, getSharedMemSize());
        parent_s = allocator.allocate<GpuParent>();
        children_s = allocator.allocateArray<GpuChild>(layerInfo->nLabels);
        childrenInfo_s = allocator.allocateArray<ChildInfo>(layerInfo->nLabels);
        *parent_s = layerInfo->parents[pIdx];
    }
    __syncthreads();

    GpuChild child_r;
    ChildInfo childInfo_r;
    i32 const minLabel = layerInfo->minLabel;
    i32 const maxLabel = layerInfo->maxLabel;
    auto const labels_r = parent_s->labels;
    for(i32 label = minLabel + threadIdx.x; label <= maxLabel; label += blockDim.x)
    {
        if (labels_r.contains(label))
        {
            // Transition
            child_r.parentNode = parent_s->node;
            child_r.label = label;
            auto state_r = model->stf(parent_s->state, label);
            if (state_r.has_value())
            {
                childInfo_r.boundSrcToNode = parent_s->boundSrcToNode + model->scf(parent_s->state, label);
                child_r.heuristicNodeToSink = model->has_local ? model->local(state_r.value(), localCtx) : 0;
                childInfo_r.cost = childInfo_r.boundSrcToNode + child_r.heuristicNodeToSink;

                if (Model::better(childInfo_r.cost, primalBound))
                {
                    child_r.state = state_r.value();
                    childInfo_r.id = pIdx * maxLabel + label;
                    childInfo_r.isRepresented = static_cast<i32>(false);
                    childInfo_r.eqHash = State::hash(child_r.state);
                    childInfo_r.domHash = Model::has_dom ? Model::domHash(child_r.state) : 0;

                    // Write children and info in shared.
                    auto const idxInShared = atomicAdd_block(&nChildrenInShared_s,1);
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
        nChildrenInGlobal_s = atomicAdd(&layerInfo->nChildren,nChildrenInShared_s);
    }
    __syncthreads();
    for (auto idxInShared = threadIdx.x; idxInShared < nChildrenInShared_s; idxInShared += blockDim.x)
    {
        auto const idxInGlobal = nChildrenInGlobal_s + idxInShared;
        childrenInfo_s[idxInShared].idx = idxInGlobal;
        layerInfo->children[idxInGlobal] = children_s[idxInShared];
        childrenInfo[idxInGlobal] = childrenInfo_s[idxInShared];
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
void calcClassesSeqKernel(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo const * const childrenInfo)
{
    /* TO OPTIMIZE
     * - Rework to parallelize
     * - Parallel load
    */

    layerInfo->classesBegin[0] = 0;
    layerInfo->nClasses = 1;
    for (auto cIdx = 1; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        auto const hash1 = Model::has_dom ? childrenInfo[cIdx].domHash   : childrenInfo[cIdx].eqHash;
        auto const hash2 = Model::has_dom ? childrenInfo[cIdx-1].domHash : childrenInfo[cIdx-1].eqHash;
        if (hash1 != hash2)
        {
            layerInfo->classesBegin[layerInfo->nClasses] = cIdx;
            layerInfo->nClasses += 1;
        }
    }
    layerInfo->classesBegin[layerInfo->nClasses] = layerInfo->nChildren;
    //printf("CL = %d | CH = %d\n", layerInfo->nClasses, layerInfo->nChildren);
}


template<typename Model>
__global__
void calcClassesKernel(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo const * const childrenInfo, gfl::i32 * const classesBegin)
{
    using namespace gfl;

    __shared__ i32 nClassesInShared_s;
    __shared__ i32 * classesBegin_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    int chBegin = blockDim.x * blockIdx.x;
    int chEnd = min<int>(chBegin + blockDim.x, layerInfo->nChildren);
    if (chEnd - chBegin > 0)
    {
        if (threadIdx.x == 0)
        {
            nClassesInShared_s = 0;
            StackAllocator allocator(shrMem, getSharedMemSize());
            classesBegin_s = allocator.allocateArray<i32>(blockDim.x);
        }
        __syncthreads();

        auto getHash = [childrenInfo](const int i){ return Model::has_dom ? childrenInfo[i].domHash : childrenInfo[i].eqHash; };
        for (auto i = threadIdx.x; i < blockDim.x; i += blockDim.x)
        {
            int currIdx = chBegin + i;
            int prevIdx = currIdx-1;
            auto prevHash = 0 <= prevIdx and prevIdx < chEnd ? getHash(prevIdx) : 0;
            auto currHash =                  currIdx < chEnd ? getHash(currIdx) : 0;
            if (prevHash != currHash)
            {
                classesBegin_s[i] = currIdx;
                atomicAdd_block(&nClassesInShared_s,currIdx < chEnd);
            }
            else
            {
                classesBegin_s[i] = layerInfo->nChildren;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicAdd(&layerInfo->nClasses,nClassesInShared_s);
        }
        for (auto i = threadIdx.x; i < blockDim.x; i += blockDim.x)
        {
            classesBegin[chBegin + i] = classesBegin_s[i];
        }
    }
}
template<typename Model>
__global__
void calcRepresentatives(LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo, ChildInfo * const childrenInfo)
{
    using namespace gfl;

    assert(layerInfo->nClasses <= layerInfo->nChildren);

    auto const clIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (clIdx < layerInfo->nClasses)
    {
        auto const clBegin = layerInfo->classesBegin[clIdx];
        auto const clEnd = clIdx < layerInfo->nClasses - 1 ? layerInfo->classesBegin[clIdx + 1] : layerInfo->nChildren;
        for (auto i = clBegin; i < clEnd - 1; i += 1)
        {
            auto & iInfo = childrenInfo[i];
            assert(iInfo.idx >= 0);
            assert(iInfo.idx < layerInfo->nChildren);
            auto const & iChild = layerInfo->children[iInfo.idx].state;
            for (auto j = i + 1; j < clEnd; j += 1)
            {
                auto & jInfo = childrenInfo[j];
                assert(jInfo.idx >= 0);
                assert(jInfo.idx <= layerInfo->nChildren);
                auto const & jChild = layerInfo->children[jInfo.idx].state;
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
               Model::has_dom ? childInfo.domHash : childInfo.eqHash,
               childInfo.boundSrcToNode,
               childInfo.id,
               childInfo.idx,
               childInfo.isRepresented);
    }
}

template<typename Model>
__global__
void printClasses(LayerInfo<typename Model::State, typename Model::Labels> * layerInfo, gfl::i32 const * const classesBegin)
{
    printf("---\n");
    printf("CH = %d | CL = %d\n", layerInfo->nChildren, layerInfo->nClasses);
    for(auto clIdx = 0; clIdx < layerInfo->nChildren; clIdx += 1)
    {
        printf("%d ", classesBegin[clIdx]);
    }
    printf("| %d\n",classesBegin[layerInfo->nChildren]);
}