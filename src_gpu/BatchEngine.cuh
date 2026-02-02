#pragma once

#include "BatchInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <BatchFunctions.cuh>

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

        auto const nParents = parents.size();
        batchInfo->initParents(nParents, allocator);
        memcpy(batchInfo->parents,parents.data(),sizeof(Node) * nParents);

        batchInfo->labelsInfo = labelsInfo;
        auto const nChildren = nParents * labelsInfo.nLabels;
        batchInfo->initChildren(nChildren, allocator);
    }

    static
    void initBatchSwappable(
       BatchInfo<Node> * const batchInfo,
       gfl::StackAllocator * const allocator,
       std::span<Node> const & parents,
       LabelsInfo const & labelsInfo,
       gfl::i64 const maxWidth)
    {
        using namespace gfl;
        // Init
        batchInfo->reset();
        allocator->clear();

        i64 const nParents = parents.size();
        i64 const bufferSize = gfl::max<i64>(maxWidth,nParents) * labelsInfo.nLabels;

        batchInfo->initParents(nParents, bufferSize, allocator);
        memcpy(batchInfo->parents,parents.data(), sizeof(Node) * nParents);
        batchInfo->labelsInfoParents = labelsInfo;
        batchInfo->initCutset(bufferSize, allocator);
        batchInfo->initChildren(bufferSize, allocator);
    }

    template<typename Model>
    static
    void processBatchExact(
        Model const * const model,
        gfl::f64 pBound,
        gfl::f64 dBound,
        BatchInfo<Node> * const batchInfo,
        bool sort)
    {
        calcChildren<Model,Node>(model,pBound,batchInfo);
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
            }
            else
            {
                filterChildren<Model,Node>(batchInfo,sort);
            }
            calcChildrenLabels<Model,Node>(model,batchInfo,DDExact,pBound,dBound);
        }
    }

    template<typename Model>
    static
    void processBatchRelaxed(
          Model const * const model,
          gfl::f64 pBound,
          gfl::f64 dBound,
          gfl::i64 width,
          BatchInfo<Node> * const batchInfo)
    {
        // TODO Manage the case with >= branching factor of the root.

        calcChildren<Model,Node>(model,pBound,batchInfo);
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
            }
            else
            {
                filterChildren<Model,Node>(batchInfo,true);
                if (batchInfo->nChildren > width)
                {
                    calcMergePartition<Model,Node>(width,batchInfo);
                    saveCutset<Model,Node>(width,batchInfo);
                    mergeChildren<Model,Node>(width,batchInfo);
                }
                calcChildrenLabels<Model,Node>(model,batchInfo,DDRelaxed,pBound,dBound);
            }

            // printf("After (%d)\n", batchInfo->nChildren);
            // printNodes(batchInfo->nChildren, batchInfo->children);
        }
    }
};