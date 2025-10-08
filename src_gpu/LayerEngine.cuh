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

    cudaMemLocation memLocCpu;
    cudaMemLocation memLocGpu;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 gpuMemSize;

    LayerHelper()
    {
        memLocCpu.type = cudaMemLocationTypeHost;
        memLocGpu.type = cudaMemLocationTypeDevice;
        cudaGetDevice(&memLocGpu.id);
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
    void clear(LayerInfoType ** layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        allocator->clear();
        *layerInfo = allocator->allocate<LayerInfoType>();
        new (*layerInfo) LayerInfoType();
    }

    static
    void initParents(std::vector<Node> const & layer, LayerInfoType * layerInfo, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        layerInfo->nParents = layer.size();
        layerInfo->parents = allocator->allocateArray<Node>(layerInfo->nParents);
        memcpy(layerInfo->parents, layer.data(), sizeof(Node) * layerInfo->nParents);
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
                DummyDecomposer96{}); // Bigger key used
        layerInfo->cubTmpMem = allocator->allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
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


    gfl::tuple<gfl::i64, gfl::i64> calcMemSize(gfl::i32 const nParents, gfl::i32 const branchingFactor)
    {
        using namespace gfl;

        i64 hMemSize = 0;
        i64 dMemSize = 0;
        std::size_t memSize = 0;

        // clear()
        memSize = sizeof(LayerInfoType) + StackAllocator::DefaultAlign;
        hMemSize += memSize;
        dMemSize += memSize;

        // initParents()
        memSize = sizeof(LNode) * nParents + StackAllocator::DefaultAlign;
        hMemSize += memSize;
        dMemSize += memSize;

        // initChildren()
        auto const nChildren = nParents * branchingFactor;
        memSize = sizeof(LNode) * nChildren + StackAllocator::DefaultAlign;
        memSize += sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        hMemSize += memSize;
        dMemSize += memSize;

        // initAux()
        memSize = sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign;
        dMemSize += memSize;
        void * dummyTmpMem = nullptr;
        NodeInfo dummyChildInfo[2];
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                memSize,
                &dummyChildInfo[0],
                &dummyChildInfo[1],
                nChildren,
                DummyDecomposer96{}); // Bigger key used
        dMemSize += memSize;

        return {hMemSize, dMemSize};
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