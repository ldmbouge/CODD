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
    __shared__ GpuParent parent_s;
    __shared__ GpuChild * children_s;
    __shared__ ChildInfo * childrenInfo_s;
    extern __shared__ u32 shrMem[]; // 16-byte aligned

    auto const & pIdx = blockIdx.x;

    if (threadIdx.x == 0)
    {
        nChildrenInShared_s = 0;
        StackAllocator allocator(shrMem, getSharedMemSize());
        children_s = allocator.allocateArray<GpuChild>(layerInfo->nLabels);
        childrenInfo_s = allocator.allocateArray<ChildInfo>(layerInfo->nLabels);
        parent_s = layerInfo->parents[pIdx];
    }
    __syncthreads();

    GpuChild child_r;
    ChildInfo childInfo_r;
    i32 const minLabel = layerInfo->minLabel;
    i32 const maxLabel = layerInfo->maxLabel;
    auto const labels_r = parent_s.labels;
    for(i32 label = minLabel + threadIdx.x; label <= maxLabel; label += blockDim.x)
    {
        if (labels_r.contains(label))
        {
            // Transition
            child_r.parentNode = parent_s.node;
            child_r.label = label;
            auto state_r = model->stf(parent_s.state, label);
            if (state_r.has_value())
            {
                childInfo_r.boundSrcToNode = parent_s.boundSrcToNode + model->scf(state_r.value(), label);
                if (model->has_local)
                {
                    child_r.heuristicNodeToSink = model->local(state_r.value(), localCtx);
                }
                else
                {
                    child_r.heuristicNodeToSink = 0;
                }
                childInfo_r.cost = childInfo_r.boundSrcToNode + child_r.heuristicNodeToSink;

                if (Model::better(childInfo_r.cost, primalBound))
                {
                    child_r.state = state_r.value();
                    childInfo_r.id = pIdx * layerInfo->nLabels + label;
                    childInfo_r.isRepresented = static_cast<i32>(false);
                    childInfo_r.eqHash = State::hash(child_r.state);
                    if (Model::has_dom)
                    {
                        childInfo_r.domHash = Model::domHash(child_r.state);
                    }

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
        layerInfo->tmpChildrenInfo[idxInGlobal] = childrenInfo_s[idxInShared];
    }
}

template<typename KeyType, typename  KeyDecomposer>
__global__
void sortKernel(void * tmpMem, std::size_t tmpMemSize, KeyType const * keysIn, KeyType * keysOut, gfl::i32 const * const nKeys)
{
    cudaStream_t gpuSortQueue;
    cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
    cub::DeviceRadixSort::SortKeys(tmpMem, tmpMemSize, keysIn, keysOut, *nKeys, KeyDecomposer{}, gpuSortQueue);
}

//
//
// template<typename Model, typename Compare>
// __global__
// void checkEquivalenceKernel(
//         Model const * const model,
//         LayerInfo<typename Model::State, typename Model::Labels> * const layerInfo)
// {
//     /* TO OPTIMIZE
//      * - Optimize parent load
//      * - Rework to parallelize on maxChildren
//      */
//
//     using State = typename Model::State;
//     using LayerInfoType = LayerInfo<State, typename Model::Labels>;
//     using GpuParent = LayerInfoType::GpuChild;
//     using GpuChild = LayerInfoType::GpuChild;
//     using ChildInfo = LayerInfoType::ChildInfo;
//     using namespace gfl;
//
//     static_assert(alignof(GpuParent) >= 16);
//     static_assert(sizeof(GpuParent) % 16 == 0);
//
//     __shared__ i32 nChildrenInShared_s;
//     __shared__ i64 nChildrenInGlobal_s;
//     __shared__ GpuParent parent_s;
//     __shared__ GpuChild * children_s;
//     __shared__ ChildInfo * childrenInfo_s;
//     __shared__ u128 shrMem[];
//
//     auto const & pIdx = blockIdx.x;
//
//     if (threadIdx.x == 0)
//     {
//         nChildrenInShared_s = 0;
//         StackAllocator allocator(shrMem, getSharedMemSize());
//         children_s = allocator.allocateArray(layerInfo->nLabels);
//         childrenInfo_s = allocator.allocateArray(layerInfo->nLabels);
//         parent_s = layerInfo->parents[pIdx];
//     }
//     __syncthreads();
//
//     GpuChild child_r;
//     ChildInfo childInfo_r;
//     i32 const minLabel = layerInfo->minLabel;
//     i32 const maxLabel = layerInfo->maxLabel;
//     auto const labels_r = parent_s->labels;
//     for(i32 label = minLabel + threadIdx.x; label <= maxLabel; label += blockDim.x)
//     {
//         if (labels_r.contains(label))
//         {
//             // Transition
//             child_r.parentNode = parent_s.node;
//             child_r.label = label;
//             auto state_r = model->stf(parent_s.state, label);
//             if (state_r.has_value())
//             {
//                 childInfo_r.boundSrcToNode = parent_s.boundSrcToNode + model->stc(state_r->value(), label);
//                 if (model->has_local)
//                 {
//                     child_r.heuristicNodeToSink = model->local(child_r.state.value(), localCtx);
//                 }
//                 else
//                 {
//                     child_r.heuristicNodeToSink = 0;
//                 }
//                 childInfo_r.cost = childInfo_r.boundSrcToNode + child_r->heuristicNodeToSink;
//
//                 if (Compare{}.isBetter(childInfo_r.cost, primalBound))
//                 {
//                     child_r.state = state_r;
//                     childInfo_r.nodeId = pIdx * layerInfo->nLabels + label;
//                     childInfo_r.isRepresented = static_cast<i32>(false);
//                     child_r.eqHash = State::hash(child_r.state);
//                     if (Model::hash_dom)
//                     {
//                         child_r.domHash = Model::domHash(child_r.state);
//                     }
//
//                     // Write children and info in shared.
//                     auto const idxInShared = atomicAdd_block(&nChildrenInShared_s,1);
//                     children_s[idxInShared] = child_r;
//                     childrenInfo_s[idxInShared] = childInfo_r;
//                 }
//             }
//         }
//     }
//     __syncthreads();
//
//     // Write children in global
//     if (threadIdx.x == 0)
//     {
//         nChildrenInGlobal_s = atomicAdd(&layerInfo->nChildren,nChildrenInShared_s);
//     }
//     __syncthreads();
//     for (auto idxInShared = threadIdx.x; idxInShared < nChildrenInShared_s; idxInShared += blockDim.x)
//     {
//         auto const idxInGlobal = nChildrenInGlobal_s + idxInShared;
//         childrenInfo_s[idxInShared].idx = idxInGlobal;
//         layerInfo->children[idxInGlobal] = children_s[idxInShared];
//         layerInfo->childrenInfo[idxInGlobal] = childrenInfo_s[idxInShared];
//     }
// }