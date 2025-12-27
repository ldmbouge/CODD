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

        batchInfo->initParents(parents.size(), allocator);
        memcpy(batchInfo->parents,parents.data(),sizeof(Node) * parents.size());

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
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
            }
            else
            {
                filterChildren<Model,Node>(batchInfo);
                batchInfo->labelsInfo.reset();
                calcChildrenLabels<Model,Node>(model,batchInfo,DDExact,pBound,dBound);
            }
        }
    }

   template<typename Model>
   static
   void processBatchExactWithBound(
       Model const * const model,
       gfl::f64 pBound,
       gfl::f64 dBound,
       BatchInfo<Node> * const batchInfo,
       gfl::f64 bound
       )
    {
        processBatchExact(model,pBound,dBound,batchInfo);
        updateNodesDual<Model>(&batchInfo->nChildren, batchInfo->children, bound);
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

        // printf("Parents (%d)\n", batchInfo->nParents);
        // printNodes(batchInfo->nParents, batchInfo->parents);

        calcChildren<Model,Node>(model,pBound,batchInfo);

        // printf("Children (%d)\n", batchInfo->nChildren);
        // printNodes(batchInfo->nChildren, batchInfo->children);



        if (batchInfo->nChildren > 0)
        {
            // printf("Before (%d)\n", batchInfo->nChildren);
            // printNodes(batchInfo->nChildren, batchInfo->children);

            if (model->isTarget(batchInfo->children[0].state))
            {
                // printf("Before Reduce (%d)\n", batchInfo->nChildren);
                // printNodes(batchInfo->nChildren, batchInfo->children);

                keepOnlyBestChild<Model,Node>(batchInfo);

                // printf("After Reduce (%d)\n", batchInfo->nChildren);
                // printNodes(batchInfo->nChildren, batchInfo->children);
            }
            else
            {
                filterChildren<Model,Node>(batchInfo);
                if (batchInfo->nChildren > width)
                {
                    mergeChildren<Model,Node>(width,batchInfo);
                }
                batchInfo->labelsInfo.reset();
                calcChildrenLabels<Model,Node>(model,batchInfo,DDRelaxed,pBound,dBound);
            }

            // printf("After (%d)\n", batchInfo->nChildren);
            // printNodes(batchInfo->nChildren, batchInfo->children);
        }
    }
};