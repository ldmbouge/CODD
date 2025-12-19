#pragma once

#include "BatchInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <BatchFunctions.cuh>

#include "kernels.cuh"

template<typename Node>
struct BatchEngine
{
    using BatchInfoType = BatchInfo<Node>;

    gfl::i32 gpuIdx;
    cudaDeviceProp gpuProp;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 gpuMemSize;

    BatchEngine()
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
    void initBatch(
       BatchInfo<Node> * const batchInfo,
       gfl::StackAllocator * const allocator,
       std::span<Node> const & parents,
       LabelsInfo const & labelsInfo)
    {
        // Init
        batchInfo->reset();
        allocator->clear();

        batchInfo->initParents(parents.size(), allocator);
        memcpy(batchInfo->parents,parents.data(),sizeof(Node) * parents);

        batchInfo->labelsInfo = labelsInfo;
        batchInfo->initChildren(allocator);
    }

    template<typename Model>
    static
    void processBatchExact(
        Model const * const model,
        gfl::f64 pBound,
        gfl::f64 dBound,
        BatchInfo<Node> * const batchInfo)
    {
        calcChildren<Model,Node>(model,pBound,batchInfo);
        filterChildren<Model,Node>(batchInfo);
        batchInfo->labelsInfo.reset();
        calcChildrenLabels<Model,Node>(model,batchInfo,DDExact,pBound,dBound);
    }
};