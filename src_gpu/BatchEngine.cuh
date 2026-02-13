#pragma once

#include "BatchInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <BatchFunctions.cuh>
#include <Debug.cuh>

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
       gfl::i64 const maxWidth,
       gfl::i64 const maxDepth)
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
        batchInfo->initCutset(maxWidth*maxDepth, allocator);
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

        using namespace gfl;
        // TODO Manage the case with >= branching factor of the root.
#ifdef G_DEBUG
        std::vector<int> const opt = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};

        auto checkNode = [&](Node const & pNode)
        {
            if (pNode.isAncestorOf(opt))
            {
                printf("Ancestor of OPT found!\n");
                assert(pNode.fValue >= 17);
            }
            return true;
        };

        auto checkNodes = [&] (Node const * nodes, gfl::i64 nNodes)
        {
            bool ancFound = false;
            for (gfl::i64 i = 0; i < nNodes; i++)
            {
                auto const & node = nodes[i];
                if (node.isAncestorOf(opt))
                {
                    ancFound = true;
                    assert(node.fValue >= 17);
                }
            }
            return ancFound;
        };
#endif

        calcChildren<Model,Node>(model,pBound,batchInfo);

#ifdef G_DEBUG
        bool ancFound = checkNodes(batchInfo->children,batchInfo->nChildren);
#endif
        if (batchInfo->nChildren > 0)
        {
            if (model->isTarget(batchInfo->children[0].state))
            {
                keepOnlyBestChild<Model,Node>(batchInfo);
#ifdef G_DEBUG
                if (ancFound)
                {
                    assert(batchInfo->children[0].isAncestorOf(opt));
                    assert(batchInfo->children[0].fValue >= 17);
                }
#endif
            }
            else
            {
#ifdef G_DEBUG
                assert(not ancFound or checkNodes(batchInfo->children,batchInfo->nChildren));
#endif
                filterChildren<Model,Node>(batchInfo,false);
#ifdef G_DEBUG
                assert(not ancFound or checkNodes(batchInfo->children,batchInfo->nChildren));
#endif

                // BatchInfo<Node> & bi = *batchInfo;
                // if(batchInfo->nChildren <= width)
                // {
                //     printf("BEFORE MERGING\n");
                //     for (i64 i = 0; i < bi.nChildren; i += 1)
                //     {
                //         NodeInfo & iInfo = bi.nodesInfo[i];
                //         i64 const iIdx = iInfo.idx;
                //         assert(0 <= iIdx);
                //         assert(iIdx <  bi.nChildren);
                //         Node const & iNode = bi.children[iIdx];
                //         Node::print(iNode);
                //     }
                // }

                if (batchInfo->nChildren > width)
                {

                    if (batchInfo->cutsetSize == 0)
                    {
                        calcChildrenLabels<Model,Node>(model,batchInfo,DDRelaxed,pBound,dBound);
                    }
                    mergeChildren<Model,Node>(width,batchInfo);


                    //saveCutset<Model,Node>(width,batchInfo);
#ifdef G_DEBUG
                    if(ancFound)
                    {
                        bool inChildren = checkNodes(batchInfo->children,batchInfo->nChildren);
                        bool inCutset = checkNodes(batchInfo->cutset,batchInfo->cutsetSize);
                        assert(inChildren or inCutset);
                    }
#endif
                }
                if (batchInfo->nChildren <= width or batchInfo->cutsetSize  != 0)
                {
                    calcChildrenLabels<Model,Node>(model,batchInfo,DDRelaxed,pBound,dBound);
                }

                // printf("AFTER MERGING\n");
                // for (i64 i = 0; i < bi.nChildren; i += 1)
                // {
                //     NodeInfo & iInfo = bi.nodesInfo[i];
                //     i64 const iIdx = iInfo.idx;
                //     assert(0 <= iIdx);
                //     assert(iIdx <  bi.nChildren);
                //     Node const & iNode = bi.children[iIdx];
                //     Node::print(iNode);
                // }


            }

            // printf("After (%d)\n", batchInfo->nChildren);
            // printNodes(batchInfo->nChildren, batchInfo->children);
        }
    }
};