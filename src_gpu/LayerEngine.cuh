#pragma once

#include <list>

#include "LayerInfo.cuh"
#include <Malloc.hpp>
#include <StackAllocator.hpp>

#include "kernels.cuh"

enum DDContext : int;
enum LocalContext : int;

template<typename Model>
class LayerEngine
{
    using State = Model::State;
    using Labels = Model::Labels;
    using LayerInfoType =  LayerInfo<State,Labels>;
    using GpuParent = GpuParent<State,Labels>;
    using GpuChild = GpuChild<State>;

protected:
    gfl::StackAllocator ioAllocator;
    gfl::StackAllocator tmpAllocator;
    LayerInfoType * layerInfo;
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
        ioAllocator(gfl::mallocManaged<void>(gpuIoMemSize), gpuIoMemSize),
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
        init(layer, model, ddCtx);
        cudaMemPrefetchAsync(ioAllocator.getMem(), ioAllocator.calcUsedMemSize(), gpuDeviceId, gpuMainQueue);

        auto blockSize = gfl::roundUpToMultiple<gfl::i32>(layerInfo->nLabels,32);
        assert(blockSize > 0 and blockSize <= 128); // We assume up to 128 children per parent
        auto const shrMemSize =
            sizeof(GpuParent) + gfl::StackAllocator::DefaultAlign +
            sizeof(GpuChild) * layerInfo->nLabels + gfl::StackAllocator::DefaultAlign +
            sizeof(ChildInfo) * layerInfo->nLabels;
        assert(shrMemSize > 0 and shrMemSize < 48 * 1024); // We assume that all the children fits in default shared memory size
        calcChildrenKernel<Model><<<layerInfo->nParents, blockSize, shrMemSize, gpuMainQueue>>>(
                model,
                layerInfo,
                layerInfo->childrenInfo,
                primalBound,
                localCtx);
        cudaEventRecord(childrenOk, gpuMainQueue);

        using HashDecomposer = std::conditional_t<Model::has_dom, DomHashDecomposer, EqHashDecomposer>;
        sortKernel<ChildInfo,HashDecomposer><<<1, 1, 0, gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->childrenInfo,
                layerInfo->tmpChildrenInfo,
                &layerInfo->nChildren);

//
//        printChildInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
//
        calcClassesBeginKernel<Model><<<1,1,0,gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
//
//        //printClasses<Model><<<1,1,0,gpuMainQueue>>>(layerInfo);
//
        blockSize = 128;
        auto gridSize = gfl::roundUpDivPosInt<gfl::i32>(layerInfo->nParents * layerInfo->nLabels, blockSize);
        calcRepresentatives<Model><<<gridSize, blockSize,0,gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);

        sortKernel<ChildInfo,RIDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo,
                &layerInfo->nChildren);

//        printChildInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo, layerInfo->childrenInfo);
        cudaEventRecord(infoOk, gpuMainQueue);
    }

    void retrieveNodes()
    {
        cudaEventSynchronize(childrenOk);
        cudaMemPrefetchAsync(&layerInfo->nChildren, sizeof(layerInfo->nChildren), cudaCpuDeviceId, gpuAuxQueue);
        cudaStreamSynchronize(gpuAuxQueue);
        //printf("Retrieved %d nodes\n",layerInfo->nChildren);
        if (layerInfo->nChildren > 0)
        {
            cudaMemPrefetchAsync(layerInfo->children, sizeof(GpuChild) * layerInfo->nChildren, cudaCpuDeviceId, gpuAuxQueue);
            cudaEventSynchronize(infoOk);
            cudaMemPrefetchAsync(layerInfo->childrenInfo, sizeof(ChildInfo) * layerInfo->nChildren, cudaCpuDeviceId, gpuAuxQueue);
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
    void init(std::list<ANode::Ptr> const & layer, Model const * const model, DDContext const ctx) noexcept
    {
        ioAllocator.clear();
        tmpAllocator.clear();
        layerInfo = ioAllocator.allocate<LayerInfoType>();

        layerInfo->nParents = layer.size();
        layerInfo->parents = ioAllocator.allocateArray<GpuParent>(layerInfo->nParents);

        layerInfo->minLabel = std::numeric_limits<gfl::i32>::max();
        layerInfo->maxLabel = std::numeric_limits<gfl::i32>::min();
        layerInfo->nLabels = std::numeric_limits<gfl::i32>::min();

        auto pIdx = 0;
        for (auto const & p : layer)
        {
            // Parents
            auto * const pNode = static_cast<Node<State>*>(p.operator->()); // Retrieve non-const pointer
            GpuParent & gParent = layerInfo->parents[pIdx];
            gParent.state = pNode->get();
            gParent.labels = model->lgf(pNode->get(), ctx);
            gParent.boundSrcToNode = pNode->getBound();
            gParent.node = p.operator->(); // Retrieve non-const pointer
            pIdx += 1;

            // Labels
            auto [smallest,largest,count] = gParent.labels.slc();
            layerInfo->minLabel = std::min(layerInfo->minLabel,smallest);
            layerInfo->maxLabel = std::max(layerInfo->maxLabel,largest);
            layerInfo->nLabels = std::max(layerInfo->nLabels,count);
        }

        // Children
        auto const maxChildren = layerInfo->nParents * layerInfo->nLabels;
        layerInfo->nChildren = 0;
        layerInfo->children = ioAllocator.allocateArray<GpuChild>(maxChildren);
        layerInfo->nRepresentatives = 0;
        layerInfo->childrenInfo = ioAllocator.allocateArray<ChildInfo>(maxChildren);

        // Auxiliary Information
        layerInfo->nClasses = 0;
        layerInfo->classesBegin = tmpAllocator.allocateArray<gfl::i32>(maxChildren + 1); // The + 1 is for loops in case nClasses == nChildren

        // CUB
        layerInfo->tmpChildrenInfo = tmpAllocator.allocateArray<ChildInfo>(maxChildren);
        layerInfo->cubTmpMem = nullptr;
        layerInfo->cubTmpMemSize = 0;
        cub::DeviceRadixSort::SortKeys(
            layerInfo->cubTmpMem,
            layerInfo->cubTmpMemSize,
            layerInfo->tmpChildrenInfo,
            layerInfo->childrenInfo,
            maxChildren,
            dummyDecomposer{}); // Determine temporary memory size
        layerInfo->cubTmpMem = tmpAllocator.allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }
};
