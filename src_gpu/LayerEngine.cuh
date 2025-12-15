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
        layerInfo->tmpChildren = allocator->allocateArray<Node>(nChildren);

        layerInfo->childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
        layerInfo->tmpChildrenInfo =  allocator->allocateArray<NodeInfo>(nChildren);
    }

    static
    void initSim(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        i32 const nChildren = layerInfo->nParents * layerInfo->labelsInfo.nLabels;
        i64 const nPairs = countOrderedPairs(nChildren);

        layerInfo->simInfo = allocator->allocateArray<MergeInfo>(nPairs);
        layerInfo->tmpSimInfo = allocator->allocateArray<MergeInfo>(nPairs);
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
                layerInfo->tmpChildrenInfo,
                nChildren,
                DummyDecomposer64{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        layerInfo->cubTmpMem = allocator->allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }

    static
    gfl::i64 calcMemSize(gfl::i64 const nParents, gfl::i32 const fanout, bool aux, bool sim)
    {
        using namespace gfl;

        i64 memSize = 0;

        // initParents()
        memSize += sizeof(Node) * nParents + StackAllocator::DefaultAlign;

        // initChildren()
        i64 const nChildren = nParents * fanout;
        memSize += 2 * (sizeof(Node) * nChildren + StackAllocator::DefaultAlign);
        memSize += 2 * (sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign);

        if (sim)
        {
            memSize += 2 * (sizeof(MergeInfo) * countOrderedPairs(nChildren) + StackAllocator::DefaultAlign);
        }

        // initAux()
        if (aux)
        {
            std::size_t memSizeSort = 0;
            void *dummyTmpMem = nullptr;
            NodeInfo *dummyChildrenInfo = nullptr;
            cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                    dummyTmpMem,
                    memSizeSort,
                    dummyChildrenInfo,
                    dummyChildrenInfo,
                    nChildren,
                    RepCostDecomposer{}); // Bigger key used
            CHECK_LAST_CUDA_ERROR();
            memSize += memSizeSort;
        }
        return memSize;
    }

    static
    gfl::i64 getMaxParents(gfl::i32 const branchingFactor, gfl::i64 const maxMemSize, bool tempSortBuffer = true)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 ubParents = 1;

        while (calcMemSize(ubParents, branchingFactor, tempSortBuffer) <= maxMemSize)
        {
            lbParents = ubParents;
            ubParents *= 2;
        }

        while (lbParents < ubParents)
        {
            i64 const mid = lbParents + (ubParents - lbParents + 1) / 2;
            i64 const memSize = calcMemSize(mid, branchingFactor, tempSortBuffer);
            if (memSize <= maxMemSize) lbParents = mid;  // still fits
            else ubParents = mid - 1; // too big
        }
        return lbParents;
    }
};