#pragma once

#include <list>

#include "LayerInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <MirrorAllocator.hpp>

#include "kernels.cuh"

template<typename State, typename Labels>
struct LayerHelper
{
    using Node = LightNode<State, Labels>;
    using LayerInfoType = LayerInfo<State, Labels>;

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

        i32 const nChildren = layerInfo->nParents * layerInfo->labelsPerParents;
        layerInfo->children = allocator->allocateArray<Node>(nChildren);
        layerInfo->childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
    }

    static
    void initAux(LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        // CUB
        auto const nChildren = layerInfo->nParents * layerInfo->labelsPerParents;
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
    gfl::i64 calcGpuMemSize(gfl::i32 const nParents, gfl::i32 const fanout)
    {
        using namespace gfl;

        i64 gpuMemSize = 0;
        std::size_t memSize = 0;

        // initParents()
        memSize = sizeof(Node) * nParents + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;

        // initChildren()
        auto const nChildren = nParents * fanout;
        memSize = sizeof(Node) * nChildren + StackAllocator::DefaultAlign;
        memSize += sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;

        // initAux()
        memSize = sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        gpuMemSize += memSize;
        void * dummyTmpMem = nullptr;
        NodeInfo dummyChildInfo[2];
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                memSize,
                &dummyChildInfo[0],
                &dummyChildInfo[1],
                nChildren,
                DummyDecomposer64{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        gpuMemSize += memSize;

        return gpuMemSize;
    }

    gfl::i64 getMaxParents(gfl::i64 const nParents, gfl::i32 const fanout)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 upParents = nParents;
        while (lbParents < upParents)
        {
            i64 mid = lbParents + (upParents - lbParents + 1) / 2;
            if (calcGpuMemSize(mid, fanout) <= gpuMemSize) lbParents = mid;  // still fits
            else upParents = mid - 1; // too big
        }
        return lbParents;
    }

};



/*
public:
    void clear() noexcept
    {
        using namespace gfl;

        mirrAllocator->clear();
        layerInfo = mirrAllocator->allocate<LayerInfoType>();
        new (layerInfo.h) LayerInfoType();
    }

    void initParents(std::vector<LNode> const & layer) noexcept
    {
        using namespace gfl;

        layerInfo->nParents = layer.size();
        layerInfo->parents = mirrAllocator->allocateArray<LNode>(layerInfo->nParents);
        memcpy(layerInfo->parents.h, layer.data(), sizeof(LNode) * layerInfo->nParents);
    }


    void offloadComputation(
            std::vector<LNode> const & layer,
            Model const * const model,
            double const primalBound,
            DDContext const ddCtx,
            LocalContext const localCtx)
    {
        using namespace gfl;

        clear();

        // Parents
        initParents(layer);
        auto const ioMem = mirrAllocator->getMem();
        cudaMemcpyAsync(ioMem.d, ioMem.h, mirrAllocator->h.calcUsedMemSize(), cudaMemcpyHostToDevice, gpuMainQueue);
        CHECK_LAST_CUDA_ERROR();

        // Labels
        i32 blockSize = 128;
        dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
        calcLabelsKernel<Model><<<gridSize, blockSize, 0, gpuMainQueue>>>(model, layerInfo.d, ddCtx);
        CHECK_LAST_CUDA_ERROR();
        cudaMemcpyAsync(layerInfo.h, layerInfo.d, sizeof(LayerInfoType), cudaMemcpyDeviceToHost, gpuMainQueue);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(gpuMainQueue);
        CHECK_LAST_CUDA_ERROR();

        if (layerInfo->labelsPerParents > 0)
        {
            initChildren();
            initAux();

            cudaMemcpyAsync(layerInfo.d, layerInfo.h, sizeof(LayerInfoType), cudaMemcpyHostToDevice, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();

            blockSize = roundUpToMultiple<i32>(layerInfo->labelsPerParents, 32);
            gridSize = layerInfo->nParents;
            i32 shrMemSize = sizeof(LNode) * layerInfo->labelsPerParents + StackAllocator::DefaultAlign +
                             sizeof(NodeInfo) * layerInfo->labelsPerParents;
            calcChildrenKernel<Model><<<gridSize, blockSize, shrMemSize, gpuMainQueue>>>(
                    model,
                    layerInfo.d,
                    layerInfo->childrenInfo.d,
                    primalBound,
                    localCtx);
            CHECK_LAST_CUDA_ERROR();
            cudaEventRecord(childrenOk, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();

//            printf("After calcChildrenKernel\n");
//            printChildrenInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
//            CHECK_LAST_CUDA_ERROR();
//            cudaStreamSynchronize(gpuMainQueue);
//            CHECK_LAST_CUDA_ERROR();

            // Representatives
            sortKernel<NodeInfo, HashDecomposer><<<1, 1, 0, gpuMainQueue>>>(
                    layerInfo->cubTmpMem,
                    layerInfo->cubTmpMemSize,
                    layerInfo->childrenInfo.d,
                    layerInfo->tmpChildrenInfo,
                    &layerInfo.d->nChildren);
            CHECK_LAST_CUDA_ERROR();

//        printf("After sortByHash\n");
//        printChildrenInfo<Model><<<1, 1, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);
//        CHECK_LAST_CUDA_ERROR();
//        cudaStreamSynchronize(gpuMainQueue);
//        CHECK_LAST_CUDA_ERROR();

            blockSize = 128;
            gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->labelsPerParents, blockSize);
            calcReprKernel<Model><<<gridSize, blockSize, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);
            CHECK_LAST_CUDA_ERROR();

            sortKernel<NodeInfo, RepBoundDecomposer><<<1, 1, 0, gpuMainQueue>>>(
                    layerInfo->cubTmpMem,
                    layerInfo->cubTmpMemSize,
                    layerInfo->tmpChildrenInfo,
                    layerInfo->childrenInfo.d,
                    &layerInfo.d->nChildren);
            CHECK_LAST_CUDA_ERROR();
        }
    }

    void retrieveNodes(std::vector<LNode> & tmpLayer,  std::vector<NodeInfo> & nodeInfoNext)
    {
        using namespace gfl;

        cudaEventSynchronize(childrenOk);
        CHECK_LAST_CUDA_ERROR();
        cudaMemcpyAsync(&layerInfo.h->nChildren, &layerInfo.d->nChildren, sizeof(i32), cudaMemcpyDeviceToHost, gpuAuxQueue);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(gpuAuxQueue);
        CHECK_LAST_CUDA_ERROR();

        tmpLayer.clear();
        nodeInfoNext.clear();
        tmpLayer.resize(layerInfo.h->nChildren);
        nodeInfoNext.resize(layerInfo.h->nChildren);
        if (layerInfo->nChildren > 0)
        {
            cudaMemcpyAsync(tmpLayer.data(), layerInfo->children.d, sizeof(LNode) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuAuxQueue);
            CHECK_LAST_CUDA_ERROR();
            cudaMemcpyAsync(nodeInfoNext.data(), layerInfo->childrenInfo.d, sizeof(NodeInfo) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();
            cudaDeviceSynchronize();
            CHECK_LAST_CUDA_ERROR();
        }
    }

//    bool getChild(GpuChild &child)
//    {
//        if (nChildProcessed < layerInfo->nChildren)
//        {
//            auto const &childInfo = layerInfo->childrenInfo[nChildProcessed];
//            if (not childInfo.isRepresented)
//            {
//                child = layerInfo->children[childInfo.idx];
//                nChildProcessed += 1;
//                return true;
//            } else
//            {
//                return false;
//            }
//        } else
//        {
//            return false;
//        }
//    }

    gfl::i64 getNodesPerBatch(gfl::i64 const nParents, gfl::i32 const branchingFactor)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 upParents = nParents;
        while (lbParents < upParents)
        {
            i64 mid = lbParents + (upParents - lbParents + 1) / 2;
            auto [hMemSize, dMemSize] = calcMemSize(mid, branchingFactor);
            if (hMemSize <= cpuMemSize and dMemSize <= gpuMemSize) lbParents = mid;  // still fits
            else upParents = mid - 1; // too big
        }
        return lbParents;
    }

};
 */