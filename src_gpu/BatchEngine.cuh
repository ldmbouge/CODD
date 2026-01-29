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
        BatchInfo<Node> * const batchInfo,
        bool sort)
    {
        //processBatchExactWithBound(model,pBound,dBound,bestValue<Model>(),batchInfo,sort);
        calcChildren<Model,Node>(model,pBound,batchInfo);
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
            }
            else
            {
                //printNodes(batchInfo->nChildren, batchInfo->children);
                int nBPrefix = 0;
                for (int i = 0; i < batchInfo->nChildren; i += 1)
                {
                    auto & const tmptmp = batchInfo->children[i];
                    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                    bool isPrefix = true;
                    for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                    {
                        isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                    }
                    if (isPrefix)
                    {
                        nBPrefix+= 1;

                    }
                }
                filterChildren<Model,Node>(batchInfo,sort);

                int nAPrefix = 0;
                for (int i = 0; i < batchInfo->nChildren; i += 1)
                {
                    auto & const tmptmp = batchInfo->children[i];
                    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                    bool isPrefix = true;
                    for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                    {
                        isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                    }
                    if (isPrefix)
                    {
                        nAPrefix+= 1;

                    }
                }
                if (nBPrefix > nAPrefix)
                {
                    printf("OPT ANCESTOR FILTERING: %d -> %d\n",nBPrefix, nAPrefix);
                    fflush(stdout);
                }


                batchInfo->labelsInfo.reset();
                calcChildrenLabels<Model,Node>(model,batchInfo,DDExact,pBound,dBound);
                //printNodes(batchInfo->nChildren, batchInfo->children);
            }
        }
    }

   template<typename Model>
   static
   void processBatchExactWithBound(
       Model const * const model,
       gfl::f64 pBound,
       gfl::f64 dBound,
       gfl::f64 hBound,
       BatchInfo<Node> * const batchInfo,
       bool sort
       )
    {
        calcChildren<Model,Node>(model,pBound,hBound,batchInfo);
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
            }
            else
            {
                //printNodes(batchInfo->nChildren, batchInfo->children);

                int nBPrefix = 0;
                for (int i = 0; i < batchInfo->nChildren; i += 1)
                {
                    auto & const tmptmp = batchInfo->children[i];
                    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                    bool isPrefix = true;
                    for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                    {
                        isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                    }
                    if (isPrefix)
                    {
                        nBPrefix+= 1;

                    }
                }
                filterChildren<Model,Node>(batchInfo,sort);

                int nAPrefix = 0;
                for (int i = 0; i < batchInfo->nChildren; i += 1)
                {
                    auto & const tmptmp = batchInfo->children[i];
                    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                    bool isPrefix = true;
                    for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                    {
                        isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                    }
                    if (isPrefix)
                    {
                        nAPrefix+= 1;

                    }
                }
                if (nBPrefix > 0)
                {
                    printf("OPT ANCESTOR FILTERING: %d -> %d\n",nBPrefix, nAPrefix);
                    fflush(stdout);
                }


                batchInfo->labelsInfo.reset();
                calcChildrenLabels<Model,Node>(model,batchInfo,DDExact,pBound,dBound);
                //printNodes(batchInfo->nChildren, batchInfo->children);
            }
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

        // printf("Parents (%d)\n", batchInfo->nParents);
        // printNodes(batchInfo->nParents, batchInfo->parents);

        for(int nIdx = 0; nIdx < batchInfo->nParents; nIdx += 1)
        {
            Node const * n = &batchInfo->parents[nIdx];
            assert(n->sumEdgesSrcToNode <= n->heuristicBound);
        }

        calcChildren<Model,Node>(model,pBound,batchInfo);

        for(int nIdx = 0; nIdx < batchInfo->nChildren; nIdx += 1)
        {
            Node const * n = &batchInfo->children[nIdx];
            assert(n->sumEdgesSrcToNode <= n->heuristicBound);
        }


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
                filterChildren<Model,Node>(batchInfo,false);

                for(int nIdx = 0; nIdx < batchInfo->nChildren; nIdx += 1)
                {
                    assert(batchInfo->children[nIdx].sumEdgesSrcToNode <= batchInfo->children[nIdx].heuristicBound);
                }


                if (batchInfo->nChildren > width)
                {
                    mergeChildren<Model,Node>(width,batchInfo);

                    for(int nIdx = 0; nIdx < batchInfo->nChildren; nIdx += 1)
                    {
                        assert(batchInfo->children[nIdx].sumEdgesSrcToNode <= batchInfo->children[nIdx].heuristicBound);
                    }
                }
                batchInfo->labelsInfo.reset();
                calcChildrenLabels<Model,Node>(model,batchInfo,DDRelaxed,pBound,dBound);
            }

            // printf("After (%d)\n", batchInfo->nChildren);
            // printNodes(batchInfo->nChildren, batchInfo->children);
        }
    }
};