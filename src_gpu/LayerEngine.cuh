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
        layerInfo->children = allocator->allocateArray<Node>(nChildren);
        layerInfo->childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
    }

    static
    void initAux(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        auto const nChildren = layerInfo->nParents * layerInfo->labelsInfo.nLabels;
        layerInfo->tmpChildren = allocator->allocateArray<Node>(nChildren);
        layerInfo->tmpChildrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo,
                nChildren,
                DummyDecomposer64{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        layerInfo->cubTmpMem = allocator->allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }

    static
    gfl::i64 calcGpuMemSize(gfl::i64 const nParents, gfl::i32 const fanout)
    {
        using namespace gfl;

        i64 gpuMemSize = 0;
        i64 memSize = 0;

        // initParents()
        memSize = sizeof(Node) * nParents + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;

        // initChildren()
        i64 const nChildren = nParents * fanout;
        memSize = sizeof(Node) * nChildren + StackAllocator::DefaultAlign;
        memSize += sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;

        // initAux()
        memSize = sizeof(Node) * nChildren + StackAllocator::DefaultAlign;
        memSize += sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;
        std::size_t memSizeSort;
        void * dummyTmpMem = nullptr;
        NodeInfo dummyChildInfo[2];
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                memSizeSort,
                &dummyChildInfo[0],
                &dummyChildInfo[1],
                nChildren,
                DummyDecomposer64{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        gpuMemSize += memSizeSort;
        return gpuMemSize;
    }

    gfl::i64 getMaxParents(gfl::i32 const fanout)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 upParents = 1;

        while (calcGpuMemSize(upParents, fanout) <= gpuMemSize) {
            lbParents = upParents;
            upParents *= 2;
        }

        while (lbParents < upParents)
        {
            i64 const mid = lbParents + (upParents - lbParents + 1) / 2;
            i64 const memSize = calcGpuMemSize(mid, fanout);
            if (memSize <= gpuMemSize) lbParents = mid;  // still fits
            else upParents = mid - 1; // too big
        }
        return lbParents;
    }

};