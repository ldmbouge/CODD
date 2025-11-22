#pragma once

#include <list>

#include "LayerInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <MirrorAllocator.hpp>

#include "kernels.cuh"

template<typename Node>
struct LayerHelper
{
    using LayerInfoType = LayerInfo<Node>;

    gfl::i32 gpuIdx;
    cudaDeviceProp gpuProp;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 gpuMemSize;

    LayerHelper()
    {
        cudaGetDevice(&gpuIdx);
        cudaGetDeviceProperties(&gpuProp,gpuIdx);
        cudaStreamCreateWithFlags(&gpuMainQueue, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&gpuAuxQueue, cudaStreamNonBlocking);
        cudaEventCreate(&childrenOk);
        cudaEventCreate(&infoOk);

        std::size_t gpuFreeMemSize = 0;
        std::size_t gpuTotMemSize = 0;
        cudaMemGetInfo(&gpuFreeMemSize, &gpuTotMemSize);
        gpuMemSize = gpuFreeMemSize - 4lu * 1024lu * 1024lu * 1024lu; // Give 4GB to CUDA runtime
    }

    static
    void clear(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        allocator->clear();
        layerInfo->clear();
    }

    static
    void initParents(gfl::i32 nParents, LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        layerInfo->nParents = nParents;
        layerInfo->parents = allocator->allocateArray<Node>(nParents);
    }

    static
    void initChildren(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        i32 const nChildren = layerInfo->nParents * layerInfo->labelsInfo.nLabels;

        auto children = allocator->allocateArray<Node>(nChildren);
        auto tmpChildren = allocator->allocateArray<Node>(nChildren);
        layerInfo->children = cub::DoubleBuffer<Node>(children, tmpChildren);

        auto childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
        auto tmpChildrenInfo =  allocator->allocateArray<NodeInfo>(nChildren);
        layerInfo->childrenInfo = cub::DoubleBuffer<NodeInfo>(childrenInfo, tmpChildrenInfo);
    }

    static
    void initAux(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        auto const nChildren = layerInfo->nParents * layerInfo->labelsInfo.nLabels;
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->childrenInfo,
                nChildren,
                RepCostDecomposer{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        layerInfo->cubTmpMem = allocator->allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }

    static
    gfl::i64 calcGpuMemSize(gfl::i64 const nParents, gfl::i32 const fanout)
    {
        using namespace gfl;

        i64 gpuMemSize = 0;

        // initParents()
        gpuMemSize += sizeof(Node) * nParents + StackAllocator::DefaultAlign;

        // initChildren()
        i64 const nChildren = nParents * fanout;
        gpuMemSize += 2 * sizeof(Node) * nChildren + StackAllocator::DefaultAlign;
        gpuMemSize += 2 * sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;

        // initAux()
        std::size_t memSizeSort;
        void * dummyTmpMem = nullptr;
        cub::DoubleBuffer<NodeInfo> dummyChildrenInfo(nullptr, nullptr);
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                memSizeSort,
                dummyChildrenInfo,
                nChildren,
                RepCostDecomposer{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        gpuMemSize += memSizeSort;
        return gpuMemSize;
    }

    gfl::i64 getMaxParents(gfl::i32 const branchingFactor)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 ubParents = 1;

        while (calcGpuMemSize(ubParents, branchingFactor) <= gpuMemSize)
        {
            lbParents = ubParents;
            ubParents *= 2;
        }

        while (lbParents < ubParents)
        {
            i64 const mid = lbParents + (ubParents - lbParents + 1) / 2;
            i64 const memSize = calcGpuMemSize(mid, branchingFactor);
            if (memSize <= gpuMemSize) lbParents = mid;  // still fits
            else ubParents = mid - 1; // too big
        }
        return lbParents;
    }

};