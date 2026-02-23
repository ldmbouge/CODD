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

    void swapParentsAndChildren(ExpansionData<Node> const * const expData)
    { expData->swapParentsAndChildren(); }

    void expandParents(
        Model const * const model,
        gfl::f64 const primal)
    {
        using namespace gfl;
        auto & parents = expData->parents;

        i32 blockSize = 128;
        i32 gridSize = ceil(parents.size(), blockSize);
        expandParentsKernel<<<gridSize,blockSize>>>(model, &expData, primal);
        cudaDeviceSynchronize();
    }

    void filterRepresentedChildren()
    {
        using namespace gfl;
        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & childrenInfo = expData->childrenInfo;
        auto & tmpInfo = expData->tmpNodesInfo;
        i64 const RepresentedFlag = 1;
        i64 const RepresentativeFlag = 0;

        // Init
        nFlagged = 0;
        tmpInfo.resizeTo(childrenInfo.size());

        // Sort by hash
        i32 const blockSize = 128;
        i32 const gridSize = ceil(children.size(), blockSize);
        calcHashKernel<<<gridSize,blockSize>>>(&children, &childrenInfo);
        sortKernel<NodeInfo::HashDecomposer><<<1,1>>>(cubAuxMem,&childrenInfo, &tmpInfo);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);

        // Find representatives
        setFlagKernel(RepresentativeFlag, &childrenInfo);
        flagRepresentedChildrenKernel<<<gridSize,blockSize>>>(RepresentedFlag,&children,&childrenInfo);
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,cubAuxMem);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        countFlaggedKernel<<<gridSize,blockSize>>>(RepresentativeFlag,&nFlagged,&childrenInfo);
        resizeToKernel(&childrenInfo,&nFlagged);
        resizeToKernel(&tmpNodes,&nFlagged);
        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo,&nFlagged);
        swapKernel<<<1,1>>>(&tmpNodes, &children);

        cudaDeviceSynchronize();
    }

    void sortChildrenByG(ExpansionData<Node> * const expData)
    {
        using namespace gfl;
        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & childrenInfo = expData->childrenInfo;
        auto & tmpInfo = expData->tmpNodesInfo;

        // Init
        tmpNodes.resizeTo(children.size());

        // Processing
        i32 const blockSize = 128;
        i32 const gridSize = ceil(children.size(), blockSize);
        setScoreGKernel<<<gridSize,blockSize>>>(&children, &childrenInfo);
        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&childrenInfo, &tmpInfo,cubAuxMem);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children, &childrenInfo);
        swapKernel<<<1,1>>>(&tmpNodes, &children);

        cudaDeviceSynchronize();
    }

    void saveCutset()
    {
        using namespace gfl;
        auto const & parents = expData->parents;
        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & parentsInfo = expData->parentInfo;
        auto & childrenInfo = expData->childrenInfo;
        auto & tmpInfo = expData->tmpInfo;
        auto & cutset = cutData;
        i64 const ParentToNotSaveFlag = 1;
        i64 const ParentToSaveFlag = 0;

        // Init
        nFlagged = 0;
        parentsInfo.resizeTo(parents.size());

        // Find parents to save
        i32 const blockSize = 128;
        i32 const gridSize = ceil(expData->children.size(),blockSize);
        resetInfoKernel<<<gridSize,blockSize>>>(&parentsInfo);
        setFlagKernel<<<gridSize,blockSize>>>(ParentToNotSaveFlag,&parentsInfo);
        flagParentsToSaveKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&parentsInfo,width_,&children,&childrenInfo);

        // Update children ancestor flag
        updateAncestorKernel(ParentToSaveFlag,&parentsInfo,&children,&childrenInfo);

        // Save parents in cutset by *reverse* f value
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo,&tmpInfo,cubAuxMem);
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        countFlaggedKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&nFlagged,&parentsInfo);
        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        setScoreFKernel<<<gridSize,blockSize>>>(&parents,parentsInfo);
        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&parentsInfo, &tmpInfo, cubAuxMem,true);
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
        copyByInfoKernel(cutset->lastSegmentPtr(), &parents, &parentsInfo);

        cudaDeviceSynchronize();
    }

    void mergeChildren()
    {
        using namespace gfl;

        auto const & parents = expData->parents;
        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & childrenInfo = expData->nodesInfo;
        auto & childrenPrefix = expData->tmpView;

        // Init
        tmpNodes.resizeTo(children.size());

        // Merge the last children - (width - 1) nodes
        i32 const blockSize       = 32;
        i32 const nodesPerThread  = 32;
        i32 const reductionFactor = blockSize * nodesPerThread;  // 1024
        i32 const nNodes          = children.size(); // real item count
        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);  // gridDim for pass1, real count for pass2
        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);  // gridDim for pass2, real count for pass3
        initChildrenPrefix<Model>(width_, &children, &childrenPrefix);
        // pass1: childrenPrefix → tmpNodes, count = nNodes
        reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&childrenPrefix,&tmpNodes,nNodes);
        // pass2: tmpNodes → childrenPrefix, count = nBlocks1
        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
        // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
        reductionSeqKernel<Model,Node><<<1,1>>>(&childrenPrefix, nBlocks2);
        resizeToKernel<<<1,1>>>(&children,width_);
        resizeToKernel<<<1,1>>>(&childrenInfo,width_);

        cudaDeviceSynchronize();
    }

    void calcOutLabels(
        Model const * const model,
        gfl::ArrayView<Node> * const nodes,
        gfl::i64 const primal,
        gfl::i64 const dual)
    {
        using namespace gfl;

        i32 const blockSize = 128;
        i32 const gridSize = ceil(nodes->size(),blockSize);
        calcOutLabelsKernel<<<gridSize,blockSize>>>(model,nodes,primal,dual,DDRelaxed);

        cudaDeviceSynchronize();
    }

    void keepOnlyBestChild()
    {
        using namespace gfl;

        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & childrenInfo = expData->nodesInfo;
        auto & tmpInfo = expData->tmpInfo;

        if (not children.empty())
        {
            i32 const blockSize = 128;
            i32 const gridSize = ceil(children.size(), blockSize);
            setScoreGKernel<<<gridSize, blockSize>>>(&children, &childrenInfo);
            sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,cubAuxMem);
            swapKernel(&tmpInfo, &childrenInfo);
            resizeToKernel<<<1,1>>>(&childrenInfo,1);
            copyByInfoKernel(&tmpNodes, &children, &childrenInfo);
            swapKernel(&tmpNodes, &children);
            resizeToKernel<<<1,1>>>(&children,1);
        }
    }

    void finalizeCutset(
        Model const * const model,
        gfl::f64 const primal,
        gfl::f64 const dual)
    {
        using namespace gfl;

        auto & children = expData->children;
        auto & tmpNodes = expData->tmpNodes;
        auto & childrenInfo = expData->nodesInfo;
        auto & tmpInfo = expData->tmpInfo;
        auto & cutset = cutData;

        if (not children.empty())
        {
            i64 const f = expData->getTarget().f();
            i32 const blockSize = 128;
            i32 const gridSize = ceil(cutset->nodes().size(), blockSize);
            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,&cutset->nodes(),primal,dual,DDRelaxed);
            setHKernel<<<gridSize,blockSize>>>(f,&cutset->nodes());
        }
    }

    void expandLayerRelaxed(
        Model const * model,
        gfl::f64 const primal,
        gfl::f64 const dual)
    {
        using namespace gfl;
        auto & children = expData->children;

        expandParents(model,primal);
        filterRepresentedChildren();
        if (children.size() > width_)
        {
            sortChildrenByG();
            saveCutset();
            mergeChildren();
        }
        calcOutLabels(model,&children,primal,dual);
    }

public:
    void expandRelaxed(
           Model const * model,
           Node const & node,
           gfl::f64 const primal,
           gfl::f64 const dual)
    {
        using namespace gfl;
        auto & children = expData->children;

        expData->clear();
        cutData->clear();
        expData->parents.pushBack(node);
        expandLayerRelaxed(model, primal, dual);
        while (not children.empty() and not children.front().isTarget(model))
        {
            swapParentsAndChildren();
            expandLayerRelaxed(model,primal,dual);
        }
        keepOnlyBestChild();
        finializeCutset(model,primal,dual);
    }


    bool hasTarget() const noexcept {return expData.hasTarget();}
    Node const & getTarget() const noexcept {return expData.getTarget();}

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
