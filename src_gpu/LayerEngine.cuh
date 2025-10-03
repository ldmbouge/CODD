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
    using LayerInfoType = LayerInfo<State,Labels>;
    using GpuParent = GpuParent<State,Labels>;
    using GpuChild = GpuChild<State>;

protected:
    gfl::MirrorAllocator mirrAllocator;
    gfl::MirrorPtr<LayerInfoType> layerInfo;
    gfl::i32 gpuDeviceId;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 nChildProcessed;

    constexpr static auto gpuIoMemSize  = 22ul * 1024ul * 1024ul * 1024ul;
    constexpr static auto gpuTmpMemSize = 22ul * 1024ul * 1024ul * 1024ul;

public:
    LayerEngine() :
        mirrAllocator(gpuIoMemSize, gpuIoMemSize + gpuTmpMemSize),
        layerInfo(nullptr)
    {
        cudaGetDevice(&gpuDeviceId);
        cudaStreamCreateWithFlags(&gpuMainQueue, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&gpuAuxQueue, cudaStreamNonBlocking);
        cudaEventCreate(&childrenOk);
        cudaEventCreate(&infoOk);
    }

    void offloadComputation(
        std::list<ANode::Ptr> const & layer,
        Model const * const model,
        double const primalBound,
        DDContext const ddCtx,
        LocalContext const localCtx)
    {
        using namespace gfl;

        clear();

        // Parents
        initParents(layer);
        auto const ioMem = mirrAllocator.getMem();
        cudaMemcpyAsync(ioMem.d, ioMem.h, mirrAllocator.h.calcUsedMemSize(), cudaMemcpyHostToDevice, gpuMainQueue);

        // Labels
        i32 blockSize = 128;
        dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
        calcLabelsKernel<Model><<<gridSize, blockSize, 0, gpuMainQueue>>>(model,layerInfo.d,ddCtx);
        cudaMemcpyAsync(layerInfo.h, layerInfo.d, sizeof(LayerInfoType), cudaMemcpyDeviceToHost, gpuMainQueue);
        cudaStreamSynchronize(gpuMainQueue);

        // Children
        printf("P = %d | ", layerInfo->nParents);
        printf("C = %d | ", layerInfo->nLables);
        fflush(stdout);

        initChildren();
        initAux();

        printf("IO = ");
        gfl::printMemSize(mirrAllocator.h.calcUsedMemSize());
        printf(" /");
        gfl::printMemSize(mirrAllocator.h.calcTotalMemSize());
        printf(" (%4.1f%%) | ", gfl::div(100 * mirrAllocator.h.calcUsedMemSize(), mirrAllocator.h.calcTotalMemSize()));
        printf("IO + TMP = ");
        gfl::printMemSize(mirrAllocator.d.calcUsedMemSize());
        printf(" /");
        gfl::printMemSize(mirrAllocator.d.calcTotalMemSize());
        printf(" (%4.1f%%)\n", gfl::div(100 * mirrAllocator.d.calcUsedMemSize(), mirrAllocator.d.calcTotalMemSize()));
        fflush(stdout);

        cudaMemcpyAsync(layerInfo.d, layerInfo.h, sizeof(LayerInfoType), cudaMemcpyHostToDevice,gpuMainQueue);
        blockSize = roundUpToMultiple<i32>(layerInfo->labelsPerParents, 32);
        gridSize = layerInfo->nParents;
        i32 shrMemSize = sizeof(GpuChild) * layerInfo->labelsPerParents + StackAllocator::DefaultAlign +
                         sizeof(ChildInfo) * layerInfo->labelsPerParents;
        calcChildrenKernel<Model><<<gridSize,blockSize,shrMemSize,gpuMainQueue>>>(
                model,
                layerInfo.d,
                layerInfo->childrenInfo.d,
                primalBound,
                localCtx);
        //printChildrenInfo<Model><<<1, 1, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
        cudaEventRecord(childrenOk, gpuMainQueue);

        // Representatives
        sortKernel<ChildInfo,HashDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->childrenInfo.d,
                layerInfo->tmpChildrenInfo,
                &layerInfo.d->nChildren);
        blockSize = 128;
        gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->labelsPerParents, blockSize);
        calcReprKernel<Model><<<gridSize,blockSize,0,gpuMainQueue>>>(layerInfo.d,layerInfo->tmpChildrenInfo);
        sortKernel<ChildInfo,RepCostDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo.d,
                &layerInfo.d->nChildren);
        //printChildrenInfo<Model><<<1, 1, 0, gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
    }

    void retrieveNodes()
    {
        using namespace gfl;

        cudaEventSynchronize(childrenOk);
        cudaMemcpyAsync(&layerInfo.h->nChildren, &layerInfo.d->nChildren, sizeof(i32), cudaMemcpyDeviceToHost, gpuAuxQueue);
        cudaStreamSynchronize(gpuAuxQueue);
        //printf("Nodes to retrieve %d\n",layerInfo->nChildren);
        if (layerInfo->nChildren > 0)
        {
            cudaMemcpyAsync(layerInfo->children.h, layerInfo->children.d, sizeof(GpuChild) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuAuxQueue);
            cudaMemcpyAsync(layerInfo->childrenInfo.h, layerInfo->childrenInfo.d, sizeof(ChildInfo) * layerInfo->nChildren, cudaMemcpyDeviceToHost, gpuMainQueue);
            cudaDeviceSynchronize();
        }
        nChildProcessed = 0;
    }

    bool getChild(GpuChild & child)
    {
        if (nChildProcessed < layerInfo->nChildren)
        {
            auto const & childInfo = layerInfo->childrenInfo[nChildProcessed];
            if (not childInfo.isRepresented)
            {
                child = layerInfo->children[childInfo.idx];
                nChildProcessed += 1;
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return false;
        }
    }

protected:
    void clear() noexcept
    {
        using namespace gfl;
        mirrAllocator.clear();
        layerInfo = mirrAllocator.allocate<LayerInfoType>();
        new (layerInfo.h) LayerInfoType();
    }

    void initParents(std::list<ANode::Ptr> const & layer) noexcept
    {
        layerInfo->nParents = layer.size();
        layerInfo->parents = mirrAllocator.allocateArray<GpuParent>(layerInfo->nParents);

        auto pIdx = 0;
        for (auto const & pANode: layer)
        {
            // Parents
            auto * const pNode = static_cast<Node<State>*>(pANode.operator->()); // Retrieve non-const pointer
            GpuParent & gParent = layerInfo->parents[pIdx];
            gParent.state = pNode->get();
            gParent.boundSrcToNode = pNode->getBound();
            gParent.node = pANode.operator->(); // Retrieve non-const pointer
            pIdx += 1;
        }
    }

    void initChildren() noexcept
    {
        // Children
        layerInfo->children = mirrAllocator.allocateArray<GpuChild>(layerInfo->nLables);
        layerInfo->childrenInfo = mirrAllocator.allocateArray<ChildInfo>(layerInfo->nLables);
    }

    void initAux() noexcept
    {
        // CUB
        layerInfo->tmpChildrenInfo = mirrAllocator.d.allocateArray<ChildInfo>(layerInfo->nLables);
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
            layerInfo.h->cubTmpMem,
            layerInfo->cubTmpMemSize,
            layerInfo->tmpChildrenInfo,
            layerInfo->childrenInfo.h,
            layerInfo->nLables,
            DummyDecomposer96{}); // Bigger key used
        layerInfo->cubTmpMem = mirrAllocator.d.allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }
};
