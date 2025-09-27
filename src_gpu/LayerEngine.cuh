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
    gfl::MirrorAllocator ioAllocator;
    gfl::StackAllocator tmpAllocator;
    gfl::MirrorPtr<LayerInfoType> layerInfo;
    gfl::i32 gpuDeviceId;
    cudaStream_t gpuMainQueue;
    cudaStream_t gpuAuxQueue;
    cudaEvent_t childrenOk;
    cudaEvent_t infoOk;
    gfl::i64 nChildProcessed;

    constexpr static auto gpuIoMemSize = 8ul * 1024ul * 1024ul * 1024ul; // 8GB
    constexpr static auto gpuTmpMemSize = 8ul * 1024ul * 1024ul * 1024ul; // 8GB

public:
    LayerEngine() :
        ioAllocator(gpuIoMemSize),
        tmpAllocator(gfl::mallocDevice<void>(gpuTmpMemSize), gpuTmpMemSize),
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

        // Parents
        clear();
        initParents(layer);
        auto const ioMem = ioAllocator.getMem();
        cudaMemcpyAsync(ioMem.d, ioMem.h, ioAllocator.calcUsedMemSize(), cudaMemcpyHostToDevice, gpuMainQueue);

        // Labels
        i32 blockSize = 128;
        dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
        calcLabelsKernel<Model><<<gridSize, blockSize, 0, gpuMainQueue>>>(model,layerInfo.d,ddCtx);
        cudaMemcpyAsync(layerInfo.h, layerInfo.d, sizeof(LayerInfoType), cudaMemcpyDeviceToHost,gpuMainQueue);

        // Children
        cudaStreamSynchronize(gpuMainQueue);
        initChildren();
        initAux();
        cudaMemcpyAsync(layerInfo.d, layerInfo.h, sizeof(LayerInfoType), cudaMemcpyHostToDevice,gpuMainQueue);
        blockSize = 64;
        i32 shrMemSize = sizeof(GpuChild) * layerInfo->nLabels + StackAllocator::DefaultAlign +
                         sizeof(ChildInfo) * layerInfo->nLabels;
        assert(blockSize < 1024);       // We assume that all the children fits in a block
        assert(shrMemSize < 48 * 1024); // We assume that all the children fits in shared
        calcChildrenKernel<Model><<<layerInfo->nParents,blockSize,shrMemSize,gpuMainQueue>>>(
                model,
                layerInfo.d,
                layerInfo->childrenInfo.d,
                primalBound,
                localCtx);
        cudaEventRecord(childrenOk, gpuMainQueue);

        // Classes
        sortKernel<ChildInfo,HashDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->childrenInfo.d,
                layerInfo->tmpChildrenInfo,
                &layerInfo.d->nChildren);
        blockSize = 128;
        gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->nLabels, blockSize);
        shrMemSize = sizeof(ClassRange) * blockSize + StackAllocator::DefaultAlign;
        calcClassesKernel<Model><<<gridSize,blockSize,shrMemSize,gpuMainQueue>>>(layerInfo.d,layerInfo->tmpChildrenInfo);

        // Representatives
        blockSize = 128;
        gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->nLabels, blockSize);
        calcReprKernel<Model><<<gridSize,blockSize,0,gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);
        sortKernel<ChildInfo,RepCostDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo.d,
                &layerInfo.d->nChildren);
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
        ioAllocator.clear();
        tmpAllocator.clear();
        layerInfo = ioAllocator.allocate<LayerInfoType>();
        new (layerInfo.h) LayerInfoType();
    }

    void initParents(std::list<ANode::Ptr> const & layer) noexcept
    {
        layerInfo->nParents = layer.size();
        layerInfo->parents = ioAllocator.allocateArray<GpuParent>(layerInfo->nParents);

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
        auto const nMaxChildren = layerInfo->nParents * layerInfo->nLabels;
        layerInfo->children = ioAllocator.allocateArray<GpuChild>(nMaxChildren);
        layerInfo->childrenInfo = ioAllocator.allocateArray<ChildInfo>(nMaxChildren);
    }

    void initAux() noexcept
    {
        // Auxiliary Information
        auto const nMaxChildren = layerInfo->nParents * layerInfo->nLabels;
        layerInfo->classesRange = tmpAllocator.allocateArray<ClassRange>(nMaxChildren);

        // CUB
        layerInfo->tmpChildrenInfo = tmpAllocator.allocateArray<ChildInfo>(nMaxChildren);
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
            layerInfo.h->cubTmpMem,
            layerInfo->cubTmpMemSize,
            layerInfo->tmpChildrenInfo,
            layerInfo->childrenInfo.h,
            nMaxChildren,
            DummyDecomposer128{});
        layerInfo->cubTmpMem = tmpAllocator.allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }
};
