#pragma once

#include <list>

#include "LayerInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>
#include <MirrorAllocator.hpp>

#include "kernels.cuh"

enum DDContext : int;
enum LocalContext : int;

template<typename Model>
class LayerEngine
{
    using State = Model::State;
    using Labels = Model::Labels;
    using LayerInfoType = LayerInfo<State, Labels>;
    using GpuParent = GpuParent<State, Labels>;
    using GpuChild = GpuChild<State>;

protected:
    gfl::MirrorAllocator * mirrAllocator;
    gfl::MirrorPtr<LayerInfoType> layerInfo;
    gfl::i32 gpuDeviceId;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 nChildProcessed;
    gfl::i64 cpuMemSize;
    gfl::i64 gpuMemSize;

    void clear() noexcept
    {
        using namespace gfl;
        mirrAllocator->clear();
        layerInfo = mirrAllocator->allocate<LayerInfoType>();
        new(layerInfo.h) LayerInfoType();
    }

    void initParents(std::list<ANode::Ptr> const &layer) noexcept
    {
        layerInfo->nParents = layer.size();
        layerInfo->parents = mirrAllocator->allocateArray<GpuParent>(layerInfo->nParents);

        auto pIdx = 0;
        for (auto const &pANode: layer)
        {
            // Parents
            auto *const pNode = static_cast<Node<State> *>(pANode.operator->()); // Retrieve non-const pointer
            GpuParent &gParent = layerInfo->parents[pIdx];
            gParent.state = pNode->get();
            gParent.boundSrcToNode = pNode->getBound();
            gParent.node = pANode.operator->(); // Retrieve non-const pointer
            pIdx += 1;
        }
    }

    void initChildren() noexcept
    {
        // Children
        auto const nChildren = layerInfo->nParents * layerInfo->labelsPerParents;
        layerInfo->children = mirrAllocator->allocateArray<GpuChild>(nChildren);
        layerInfo->childrenInfo = mirrAllocator->allocateArray<ChildInfo>(nChildren);
    }

    void initAux() noexcept
    {
        // CUB
        auto const nChildren = layerInfo->nParents * layerInfo->labelsPerParents;
        layerInfo->tmpChildrenInfo = mirrAllocator->d.allocateArray<ChildInfo>(nChildren);
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                layerInfo.h->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo.h,
                nChildren,
                DummyDecomposer96{}); // Bigger key used
        layerInfo->cubTmpMem = mirrAllocator->d.allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
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
        memSize = sizeof(GpuParent) * nParents + StackAllocator::DefaultAlign;
        hMemSize += memSize;
        dMemSize += memSize;

        // initChildren()
        auto const nChildren = nParents * branchingFactor;
        memSize = sizeof(GpuChild) * nChildren + StackAllocator::DefaultAlign;
        memSize += sizeof(ChildInfo) * nChildren + StackAllocator::DefaultAlign;
        hMemSize += memSize;
        dMemSize += memSize;

        // initAux()
        memSize = sizeof(ChildInfo) * nChildren + StackAllocator::DefaultAlign;
        dMemSize += memSize;
        void * dummyTmpMem = nullptr;
        ChildInfo dummyChildInfo[2];
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

public:
    LayerEngine() :
            mirrAllocator(nullptr),
            layerInfo(nullptr)
    {
        using namespace gfl;

        cudaGetDevice(&gpuDeviceId);
        cudaStreamCreateWithFlags(&gpuMainQueue, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&gpuAuxQueue, cudaStreamNonBlocking);
        cudaEventCreate(&childrenOk);
        cudaEventCreate(&infoOk);

        // Automatic batch size
        std::size_t gpuFreeMemSize = 0;
        std::size_t gpuTotMemSize = 0;
        cudaMemGetInfo(&gpuFreeMemSize, &gpuTotMemSize);
        gpuMemSize = gpuFreeMemSize - 4lu * 1024lu * 1024lu * 1024lu; // Give 4GB to CUDA runtime
        cpuMemSize = gpuMemSize / 2; // Rough estimation
        mirrAllocator = new MirrorAllocator(cpuMemSize, gpuMemSize);
    }

    void offloadComputation(
            std::list<ANode::Ptr> const &layer,
            Model const *const model,
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

            // Children
//        printf("P = %d | ", layerInfo->nParents);
//        fflush(stdout);

            initChildren();
            initAux();

//        printf("IO = ");
//        gfl::printMemSize(mirrAllocator.h.calcUsedMemSize());
//        printf(" /");
//        gfl::printMemSize(mirrAllocator.h.calcTotalMemSize());
//        printf(" (%4.1f%%) | ", gfl::div(100 * mirrAllocator.h.calcUsedMemSize(), mirrAllocator.h.calcTotalMemSize()));
//        printf("IO + TMP = ");
//        gfl::printMemSize(mirrAllocator.d.calcUsedMemSize());
//        printf(" /");
//        gfl::printMemSize(mirrAllocator.d.calcTotalMemSize());
//        printf(" (%4.1f%%)\n", gfl::div(100 * mirrAllocator.d.calcUsedMemSize(), mirrAllocator.d.calcTotalMemSize()));
//        fflush(stdout);

            cudaMemcpyAsync(layerInfo.d, layerInfo.h, sizeof(LayerInfoType), cudaMemcpyHostToDevice, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();

            blockSize = roundUpToMultiple<i32>(layerInfo->labelsPerParents, 32);
            gridSize = layerInfo->nParents;
            i32 shrMemSize = sizeof(GpuChild) * layerInfo->labelsPerParents + StackAllocator::DefaultAlign +
                             sizeof(ChildInfo) * layerInfo->labelsPerParents;
            calcChildrenKernel<Model><<<gridSize, blockSize, shrMemSize, gpuMainQueue>>>(
                    model,
                    layerInfo.d,
                    layerInfo->childrenInfo.d,
                    primalBound,
                    localCtx);
            CHECK_LAST_CUDA_ERROR();
            cudaEventRecord(childrenOk, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();

//        printf("After calcChildrenKernel\n");
//        printChildrenInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
//        CHECK_LAST_CUDA_ERROR();
//        cudaStreamSynchronize(gpuMainQueue);
//        CHECK_LAST_CUDA_ERROR();

            // Representatives
            sortKernel<ChildInfo, HashDecomposer><<<1, 1, 0, gpuMainQueue>>>(
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

//        printf("After calcReprKernel\n");
//        printChildrenInfo<Model><<<1, 1, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);
//        CHECK_LAST_CUDA_ERROR();
//        cudaStreamSynchronize(gpuMainQueue);
//        CHECK_LAST_CUDA_ERROR();

            sortKernel<ChildInfo, RepCostDecomposer><<<1, 1, 0, gpuMainQueue>>>(
                    layerInfo->cubTmpMem,
                    layerInfo->cubTmpMemSize,
                    layerInfo->tmpChildrenInfo,
                    layerInfo->childrenInfo.d,
                    &layerInfo.d->nChildren);
            CHECK_LAST_CUDA_ERROR();

//        printf("After sortByRepCost\n");
//        printChildrenInfo<Model><<<1, 1, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
//        CHECK_LAST_CUDA_ERROR();
//        cudaStreamSynchronize(gpuMainQueue);
        }

    }

    void retrieveNodes()
    {
        using namespace gfl;

        cudaEventSynchronize(childrenOk);
        CHECK_LAST_CUDA_ERROR();
        cudaMemcpyAsync(&layerInfo.h->nChildren, &layerInfo.d->nChildren, sizeof(i32), cudaMemcpyDeviceToHost, gpuAuxQueue);
        CHECK_LAST_CUDA_ERROR();
        cudaStreamSynchronize(gpuAuxQueue);
        CHECK_LAST_CUDA_ERROR();
        //printf("Nodes to retrieve %d\n",layerInfo->nChildren);
        if (layerInfo->nChildren > 0)
        {
            cudaMemcpyAsync(layerInfo->children.h, layerInfo->children.d, sizeof(GpuChild) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuAuxQueue);
            CHECK_LAST_CUDA_ERROR();
            cudaMemcpyAsync(layerInfo->childrenInfo.h, layerInfo->childrenInfo.d, sizeof(ChildInfo) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();
            cudaDeviceSynchronize();
            CHECK_LAST_CUDA_ERROR();
        }
        nChildProcessed = 0;
    }

    bool getChild(GpuChild &child)
    {
        if (nChildProcessed < layerInfo->nChildren)
        {
            auto const &childInfo = layerInfo->childrenInfo[nChildProcessed];
            if (not childInfo.isRepresented)
            {
                child = layerInfo->children[childInfo.idx];
                nChildProcessed += 1;
                return true;
            } else
            {
                return false;
            }
        } else
        {
            return false;
        }
    }

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