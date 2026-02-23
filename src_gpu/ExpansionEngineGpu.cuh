#pragma once

#include "Contexts.hpp"
#include "GFL.hpp"

#include "ExpansionData.hpp"
#include "Node.hpp"
#include "ExpansionEngine.hpp"
#include "ExpansionKernels.cuh"

#ifdef __CUDACC__
#include "Sort.cuh"
#include <cub/cub.cuh>
#endif

template<typename Model,typename Node>
class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
{
    using ExpansionEngine = ExpansionEngine<Model,Node>;
    using ExpansionEngine::expData;
    using ExpansionEngine::cutData;
    using ExpansionEngine::nFlagged;
    using ExpansionEngine::width_;

    gfl::ArrayView<gfl::u8> cubAuxMem;
    bool targetFound;

    static
    gfl::i64 cubAuxMemSize(gfl::i64 const nNodes)
    {
        std::size_t memSize = 0;
        void * dummyTmpMem = nullptr;
        NodeInfo * dummyNodeInfo = nullptr;
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                memSize,
                dummyNodeInfo,
                dummyNodeInfo,
                nNodes,
                DummyDecomposer64{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        return memSize;
    }

    void initCubAuxMem(gfl::i32 const maxNodes, gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        i64 const memSize = cubAuxMemSize(maxNodes);
        cubAuxMem = ArrayView<u8>(memSize,alloc);
    }

    void swapParentsAndChildren()
    { expData.swapParentsAndChildren(); }

    void expandParents(
        Model const * const model,
        gfl::f64 const primal)
    {
        using namespace gfl;
        auto & parents = expData.parents;

        i32 blockSize = 128;
        i32 gridSize = parents.size();
        expandParentsKernel<<<gridSize,blockSize>>>(model, &expData, primal);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    template<typename KeyDecomposer>
    void sortKernelDispatcher(
        gfl::ArrayView<NodeInfo> * const in,
        gfl::ArrayView<NodeInfo> * const out,
        gfl::ArrayView<gfl::u8> const * const tmp,
        bool reverse = false)
    {
        using namespace gfl;

        assert(tmp != nullptr);
        assert(in != nullptr);
        assert(out != nullptr);
        assert(in->size() <= out->size());

        auto * inBuffer = in->data();
        auto * outBuffer = out->data();
        i32 const nElements = in->size();
        auto * tmpBuffer = tmp->data();
        size_t tmpBufferSize = scast<size_t>(tmp->dataMemSize());
        if (reverse)
            cub::DeviceRadixSort::SortKeysDescending(
                tmpBuffer,
                tmpBufferSize,
                inBuffer,
                outBuffer,
                nElements,
                KeyDecomposer{});
        else
            cub::DeviceRadixSort::SortKeys(
                tmpBuffer,
                tmpBufferSize,
                inBuffer,
                outBuffer,
                nElements,
                KeyDecomposer{});
        CHECK_LAST_CUDA_ERROR();
    }


    void filterRepresentedChildren()
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;
        i64 const RepresentedFlag = 1;
        i64 const RepresentativeFlag = 0;

        // Init
        nFlagged = 0;
        tmpInfo.resizeTo(childrenInfo.size());

        // Sort by hash

        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(children.size(), blockSize);
        calcHashKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernelDispatcher<NodeInfo::HashDecomposer>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        // Find representatives
        setFlagKernel<<<gridSize,blockSize>>>(RepresentativeFlag, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        flagRepresentedChildrenKernel<Model><<<gridSize,blockSize>>>(RepresentedFlag,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernelDispatcher<NodeInfo::FlagDecomposer>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        countFlaggedKernel<<<gridSize,blockSize>>>(RepresentativeFlag,&nFlagged,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        resizeToKernel<<<1,1>>>(&childrenInfo,&nFlagged);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        resizeToKernel<<<1,1>>>(&tmpNodes,&nFlagged);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        swapKernel<<<1,1>>>(&tmpNodes, &children);
        CHECK_LAST_CUDA_ERROR();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void sortChildrenByG()
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;

        // Init
        tmpNodes.resizeTo(children.size());

        // Processing
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(children.size(), blockSize);
        setScoreGKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernelDispatcher<NodeInfo::ScoreDecomposer>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo,children.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpNodes, &children);
        CHECK_LAST_CUDA_ERROR();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void saveCutset()
    {
        using namespace gfl;
        auto const & parents = expData.parents;
        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & parentsInfo = expData.parentInfo;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;
        auto & cutset = cutData;
        i64 const ParentToNotSaveFlag = 1;
        i64 const ParentToSaveFlag = 0;

        // Init
        nFlagged = 0;
        parentsInfo.resizeTo(parents.size());

        // Find parents to save
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(expData.children.size(),blockSize);
        resetInfoKernel<<<gridSize,blockSize>>>(&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        setFlagKernel<<<gridSize,blockSize>>>(ParentToNotSaveFlag,&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        flagParentsToSaveKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&parentsInfo,width_,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();

        // Update children ancestor flag
        updateAncestorKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&parentsInfo,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();

        // Save parents in cutset by *reverse* f value
        resizeToKernel<<<1,1>>>(&tmpInfo,parentsInfo.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        sortKernelDispatcher<NodeInfo::FlagDecomposer>(&parentsInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&nFlagged,&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        setScoreFKernel<Model><<<gridSize,blockSize>>>(&parents,&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernelDispatcher<NodeInfo::ScoreDecomposer>(&parentsInfo,&tmpInfo,&cubAuxMem,true);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
        resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
        CHECK_LAST_CUDA_ERROR();

        // printKernel<<<1,1>>>(4,cutset.lastSegmentPtr());
        // printKernel<<<1,1>>>(5,&parents);
        // printKernel<<<1,1>>>(6,&parentsInfo);
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        copyByInfoKernel<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(),&parents, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();

        // printKernel<<<1,1>>>(7,cutset.lastSegmentPtr());
        // printKernel<<<1,1>>>(8,&parents);
        // printKernel<<<1,1>>>(9,&parentsInfo);
        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void mergeChildren()
    {
        using namespace gfl;

        auto const & parents = expData.parents;
        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & childrenPrefix = expData.tmpView;

        // Init
        tmpNodes.resizeTo(children.size());
        initChildrenPrefix(width_,&children,&childrenPrefix);

        // Merge the last children - (width - 1) nodes
        i32 const blockSize       = 32;
        i32 const nodesPerThread  = 32;
        i32 const reductionFactor = blockSize * nodesPerThread;  // 1024
        i32 const nNodes          = childrenPrefix.size(); // real item count
        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);  // gridDim for pass1, real count for pass2
        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);  // gridDim for pass2, real count for pass3
        // pass1: childrenPrefix → tmpNodes, count = nNodes
        reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&childrenPrefix,&tmpNodes,nNodes);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // pass2: tmpNodes → childrenPrefix, count = nBlocks1
        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
        reductionSeqKernel<Model,Node><<<1,1>>>(&childrenPrefix, nBlocks2);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        resizeToKernel<<<1,1>>>(&children,width_);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo,width_);
        CHECK_LAST_CUDA_ERROR();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void calcOutLabels(
        Model const * const model,
        gfl::VectorView<Node> * const nodes,
        gfl::i64 const primal,
        gfl::i64 const dual)
    {
        using namespace gfl;

        if (not nodes->empty())
        {
            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(nodes->size(),blockSize);
            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,nodes,primal,dual,DDRelaxed);
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    void checkForTarget(
        Model const * const model,
        gfl::VectorView<Node> * const nodes)
    {
        using namespace gfl;

        targetFound = false;
        if (not nodes->empty())
        {
            checkForTargetKernel<<<1,1>>>(&targetFound,model,nodes);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    void keepOnlyBestChild()
    {
        using namespace gfl;

        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;

        if (not children.empty())
        {
            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(children.size(), blockSize);
            setScoreGKernel<Model><<<gridSize, blockSize>>>(&children, &childrenInfo);
            CHECK_LAST_CUDA_ERROR();
            sortKernelDispatcher<NodeInfo::ScoreDecomposer>(&childrenInfo,&tmpInfo,&cubAuxMem);
            CHECK_LAST_CUDA_ERROR();
            swapKernel<<<1,1>>>(&tmpInfo, &childrenInfo);
            CHECK_LAST_CUDA_ERROR();
            resizeToKernel<<<1,1>>>(&childrenInfo,1);
            CHECK_LAST_CUDA_ERROR();
            resizeToKernel<<<1,1>>>(&tmpNodes,1);
            CHECK_LAST_CUDA_ERROR();
            copyByInfoKernel<<<gridSize, blockSize>>>(&tmpNodes, &children, &childrenInfo);
            CHECK_LAST_CUDA_ERROR();
            swapKernel<<<1,1>>>(&tmpNodes, &children);
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    void finalizeCutset(
        Model const * const model,
        gfl::f64 const primal,
        gfl::f64 const dual)
    {
        using namespace gfl;

        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;
        auto & cutset = cutData;

        if (not children.empty() and not cutset.nodes().empty())
        {
            // printf("CUTSET SIZE = %d\n", cutset.nodes().size());
            // fflush(stdout);
            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(cutset.nodes().size(), blockSize);
            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,cutset.nodesPtr(),primal,dual,DDRelaxed);
            CHECK_LAST_CUDA_ERROR();
            setHKernel<Model><<<gridSize,blockSize>>>(&children,cutset.nodesPtr());
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    void expandLayerRelaxed(
        Model const * model,
        gfl::f64 const primal,
        gfl::f64 const dual)
    {
        using namespace gfl;
        auto & children = expData.children;
        expandParents(model,primal);
        if (children.size() > 0)
        {
            filterRepresentedChildren();
            if (children.size() > width_)
            {
                sortChildrenByG();
                saveCutset();
                mergeChildren();
            }
            calcOutLabels(model,&children,primal,dual);
            checkForTarget(model,&children);
        }
    }

public:

    void initRelaxedExpansion(
     gfl::i32 const width,
     gfl::i32 const branchFactor,
     gfl::i32 const depth,
     gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        ExpansionEngine::initRelaxedExpansion(width, branchFactor, depth, alloc);
        i32 const maxNodes = width * branchFactor;
        initCubAuxMem(maxNodes,alloc);
    }

    void expandRelaxed(
           Model const * model,
           Node const & node,
           gfl::f64 const primal,
           gfl::f64 const dual)
    {
        using namespace gfl;
        auto & children = expData.children;

        expData.clear();
        cutData.clear();
        expData.parents.pushBackGpu(node);
        expandLayerRelaxed(model, primal, dual);
        while (not children.empty() and not targetFound)
        {
            swapParentsAndChildren();
            expandLayerRelaxed(model,primal,dual);
        }
        keepOnlyBestChild();
        finalizeCutset(model,primal,dual);
    }


    bool hasTarget() const noexcept {return expData.hasTarget();}
    Node const & getTarget() const noexcept {return expData.getTarget();}

    Node const getTargetFromGpu() const noexcept
    {
        Node trg;
        cudaMemcpy(&trg,&expData.children.front(),sizeof(Node),cudaMemcpyDeviceToHost);
        return trg;
    }



// #ifdef __CUDACC__
//     static
//     gfl::i64 cubAuxMemSize(gfl::i64 const nNodes)
//     {
//         std::size_t memSize = 0;
//         void * dummyTmpMem = nullptr;
//         NodeInfo * dummyNodeInfo = nullptr;
//         cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
//                 dummyTmpMem,
//                 memSize,
//                 dummyNodeInfo,
//                 dummyNodeInfo,
//                 nNodes,
//                 DummyDecomposer64{}); // Bigger key used
//         CHECK_LAST_CUDA_ERROR();
//         return memSize;
//     }
//
//     static
// #ifdef __CUDACC__
//     gfl::i64 dataMemSize(gfl::i64 const nParents, gfl::i32 const maxBranchFactor, bool cubAuxMem = false)
// #else
//     gfl::i64 dataMemSize(gfl::i64 const nParents, gfl::i32 const maxBranchFactor)
// #endif
//     {
//         using namespace gfl;
//
//         i64 const nChildren = nParents * maxBranchFactor;
//
//         i64 memSize = 0;
//         memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // parents
//         memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // children
//         memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
//         memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
// #ifdef __CUDACC__
//         if (cubAuxMem) memSize += cubAuxMemSize(nChildren) + DefaultAlign; // auxMem for GPU sort
// #endif
//         memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // tmpNodes
//         memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // tmpNodesInfo
//
//         return memSize;
//     }
//
//     template<typename Fn>
//     static
//     gfl::i64 calcMaxParents(Fn calcMemSize, gfl::i64 const maxMemSize)
//     {
//         using namespace gfl;
//
//         // Binary search on the number of parents
//         i64 lbParents = 0;
//         i64 ubParents = 1;
//         while (calcMemSize(ubParents) <= maxMemSize)
//         {
//             lbParents = ubParents;
//             ubParents *= 2;
//         }
//         while (lbParents < ubParents)
//         {
//             i64 const midParents = lbParents + (ubParents - lbParents + 1) / 2;
//             i64 const memSize = calcMemSize(midParents);
//             if (memSize <= maxMemSize) lbParents = midParents;  // still fits
//             else ubParents = midParents - 1; // too big
//         }
//         return lbParents;
//     }
};
