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
    using LayerInfoType =  LayerInfo<State,Labels>;
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

        initParents(layer);
        auto const ioMem = ioAllocator.getMem();
        cudaMemcpyAsync(ioMem.d, ioMem.h, ioAllocator.calcUsedMemSize(), cudaMemcpyHostToDevice, gpuMainQueue);

        i32 blockSize = 128;
        dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
        calcLabelsKernel<Model><<<gridSize, blockSize, 0, gpuMainQueue>>>(model,layerInfo.d,ddCtx);
        cudaMemcpyAsync(layerInfo.h, layerInfo.d, sizeof(LayerInfoType), cudaMemcpyDeviceToHost,gpuMainQueue);
        cudaStreamSynchronize(gpuMainQueue);
        initChildren();
        cudaMemcpyAsync(layerInfo.d, layerInfo.h, sizeof(LayerInfoType), cudaMemcpyHostToDevice,gpuMainQueue);

        blockSize = 32;
        i32 const blocksPerParent = roundUpDivPosInt<i32>(layerInfo->maxLabel - layerInfo->minLabel + 1, blockSize);
        gridSize = dim3(layerInfo->nParents, blocksPerParent, 1);
        i32 shrMemSize = sizeof(GpuChild) * blockSize + StackAllocator::DefaultAlign +
                         sizeof(ChildInfo) * blockSize;
        assert(shrMemSize > 0 and shrMemSize < 48 * 1024); // We assume that all the children fits in default shared memory size
        calcChildrenKernel<Model><<<gridSize, blockSize, shrMemSize, gpuMainQueue>>>(
                model,
                layerInfo.d,
                layerInfo->childrenInfo.d,
                primalBound,
                localCtx);
        cudaEventRecord(childrenOk, gpuMainQueue);

        //printChildInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);

        using HashDecomposer = std::conditional_t<Model::has_dom, DomHashDecomposer, EqHashDecomposer>;
        sortKernel<ChildInfo,HashDecomposer><<<1, 1, 0, gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->childrenInfo.d,
                layerInfo->tmpChildrenInfo,
                &layerInfo.d->nChildren);
        //printChildInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);

        blockSize = 128;
        gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->nLabels, blockSize);
        shrMemSize = sizeof(i32) * blockSize + gfl::StackAllocator::DefaultAlign;
        calcClassesKernel<Model><<<gridSize,blockSize,shrMemSize,gpuMainQueue>>>(
                layerInfo.d,
                layerInfo->tmpChildrenInfo,
                layerInfo->tmpClassesBegin);
        sortKernel<<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpClassesBegin,
                layerInfo->classesBegin,
                &layerInfo.d->nChildren);
        //calcClassesSeqKernel<Model><<<1,1,0,gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
        //printClasses<Model><<<1,1,0,gpuMainQueue>>>(layerInfo, layerInfo->classesBegin);

        calcRepresentatives<Model><<<gridSize, blockSize,0,gpuMainQueue>>>(layerInfo.d, layerInfo->tmpChildrenInfo);

        sortKernel<ChildInfo,RepIdDecomposer><<<1,1,0,gpuMainQueue>>>(
                layerInfo->cubTmpMem,
                layerInfo->cubTmpMemSize,
                layerInfo->tmpChildrenInfo,
                layerInfo->childrenInfo.d,
                &layerInfo.d->nChildren);

        //printChildInfo<Model><<<1,1,0,gpuMainQueue>>>(layerInfo.d, layerInfo->childrenInfo.d);
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
    void initParents(std::list<ANode::Ptr> const & layer) noexcept
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
        for (auto const &p: layer)
        {
            // Parents
            auto *const pNode = static_cast<Node<State> *>(p.operator->()); // Retrieve non-const pointer
            GpuParent &gParent = layerInfo->parents[pIdx];
            gParent.state = pNode->get();
            gParent.boundSrcToNode = pNode->getBound();
            gParent.node = p.operator->(); // Retrieve non-const pointer
            pIdx += 1;
        }
    }

    void initChildren() noexcept
    {
        // Children
        auto const maxChildren = layerInfo->nParents * layerInfo->nLabels;
        layerInfo->nChildren = 0;
        layerInfo->children = ioAllocator.allocateArray<GpuChild>(maxChildren);
        layerInfo->childrenInfo = ioAllocator.allocateArray<ChildInfo>(maxChildren);

        // Auxiliary Information
        layerInfo->nClasses = 0;
        layerInfo->classesBegin = tmpAllocator.allocateArray<gfl::i32>(maxChildren);

        // CUB
        layerInfo->tmpChildrenInfo = tmpAllocator.allocateArray<ChildInfo>(maxChildren);
        layerInfo->tmpClassesBegin = tmpAllocator.allocateArray<gfl::i32>(maxChildren + 1);
        layerInfo->cubTmpMem = nullptr;
        layerInfo->cubTmpMemSize = 0;
        cub::DeviceRadixSort::SortKeys(
            layerInfo.h->cubTmpMem,
            layerInfo->cubTmpMemSize,
            layerInfo->tmpChildrenInfo,
            layerInfo->childrenInfo.h,
            maxChildren,
            DummyDecomposer{}); // Determine temporary memory size
        layerInfo->cubTmpMem = tmpAllocator.allocate<gfl::u8>(layerInfo->cubTmpMemSize, 16);
    }
};
