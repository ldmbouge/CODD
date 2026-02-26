#pragma once

#include "Contexts.hpp"
#include "GFL.hpp"

#include "Node.hpp"
#include "ExpansionEngine.hpp"
#include "ExpansionKernels.cuh"
#include "ExpansionMacroKernel.cuh"

#ifdef __CUDACC__
#include "Sort.cuh"
#include <cub/cub.cuh>
#endif

template<typename Model,typename Node>
class ExpansionEngineGpu : public ExpansionEngine<Model,Node>
{
public:
    using ExpansionEngine = ExpansionEngine<Model,Node>;
    using ExpansionEngine::expData;
    using ExpansionEngine::cutData;
    using ExpansionEngine::nFlagged;
    using ExpansionEngine::width_;

    gfl::ArrayView<gfl::u8> cubAuxMem;
    gfl::i32 recLvl;

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
                DummyDecomposer96{}); // Bigger key used
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
    {
        expData.swapParentsAndChildren();
    }

    void expandParents(
        Model const * const model,
        gfl::f64 const primal,
        gfl::i32 const brachFactor)
    {
        using namespace gfl;
        auto & parents = expData.parents;

        i32 blockSize = 32;
        i32 gridSize = ceil<i32>(parents.size() *brachFactor,blockSize) ;
        expandParentsKernel<<<gridSize,blockSize>>>(model, &expData, primal, brachFactor);
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
        sortKernel<NodeInfo::HashDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();

        // Find representatives
        setFlagKernel<<<gridSize,blockSize>>>(RepresentativeFlag, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        flagRepresentedChildrenKernel<Model><<<gridSize,blockSize>>>(RepresentedFlag,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSize,blockSize>>>(RepresentativeFlag,&nFlagged,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo,&nFlagged);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpNodes,&nFlagged);
        CHECK_LAST_CUDA_ERROR();
        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpNodes, &children);
        CHECK_LAST_CUDA_ERROR();

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void sortChildrenByG(Model const * const model)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & tmpNodes = expData.tmpNodes;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;

        // Init
        tmpNodes.resizeTo(children.size());
        tmpInfo.resizeTo(childrenInfo.size());

        // Processing
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(children.size(), blockSize);
        setScoreMergeKernel<Model><<<gridSize,blockSize>>>(model, &children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
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

        // Save parents in cutset
        resizeToKernel<<<1,1>>>(&tmpInfo,parentsInfo.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&nFlagged,&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
        setScoreFKernel<Model><<<gridSize,blockSize>>>(&parents,&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&parentsInfo,&tmpInfo,&cubAuxMem,true);
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

        copyByInfoKernel<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(),&parents, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();

        // printKernel<<<1,1>>>(7,cutset.lastSegmentPtr());
        // printKernel<<<1,1>>>(8,&parents);
        // printKernel<<<1,1>>>(9,&parentsInfo);

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
        // pass2: tmpNodes → childrenPrefix, count = nBlocks1
        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
        CHECK_LAST_CUDA_ERROR();
        // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
        reductionSeqKernel<Model,Node><<<1,1>>>(&childrenPrefix, nBlocks2);
        CHECK_LAST_CUDA_ERROR();
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
        gfl::optional<Node> * target,
        gfl::VectorView<Node> * const nodes)
    {
        using namespace gfl;

        target->reset();
        if (not nodes->empty())
        {
            checkForTargetKernel<<<1,1>>>(target,model,nodes);
        }
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void onlyBestTargets()
    {
        using namespace gfl;
        auto & targets = expData.children;
        auto & targetsInfo = expData.childrenInfo;
        auto & tmpInfo = expData.tmpInfo;

        if (not targets.empty())
        {
            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(targets.size(), blockSize);
            setScoreGKernel<Model><<<gridSize, blockSize>>>(&targets, &targetsInfo);
            CHECK_LAST_CUDA_ERROR();
            resizeToKernel<<<1,1>>>(&tmpInfo,targetsInfo.sizePtr());
            CHECK_LAST_CUDA_ERROR();
            sortKernel<NodeInfo::ScoreFlagDecomposer><<<1,1>>>(&targetsInfo,&tmpInfo,&cubAuxMem);
            CHECK_LAST_CUDA_ERROR();
            swapKernel<<<1,1>>>(&tmpInfo, &targetsInfo);
            CHECK_LAST_CUDA_ERROR();
            copyBestTargetsKernel<<<1,1>>>(
                &expData.bestTargetNode,
                &expData.bestExactTargetNode,
                &targets,
                &targetsInfo);
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

        if (expData.bestTargetNode.has_value() and not cutset.nodes().empty())
        {
            // printf("CUTSET SIZE = %d\n", cutset.nodes().size());
            // fflush(stdout);
            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(cutset.nodes().size(), blockSize);
            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,cutset.nodesPtr(),primal,dual,DDRelaxed);
            CHECK_LAST_CUDA_ERROR();
            setHKernel<Model><<<gridSize,blockSize>>>(&expData.bestTargetNode,cutset.nodesPtr());
            CHECK_LAST_CUDA_ERROR();

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    void expandLayerRelaxed(
        Model const * model,
        gfl::f64 const primal,
        gfl::f64 const dual,
        gfl::i32 const brachFactor)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & parents = expData.parents;
        auto & childrenInfo = expData.childrenInfo;
        auto & parentsInfo = expData.parentInfo;

        i32 blockSize = 32;
        i32 gridSize = ceil<i32>(parentsInfo.size()*brachFactor,blockSize) ;
        assert(parentsInfo.size() <= width_);
        expandParentsNewKernel<<<gridSize,blockSize>>>(model, &expData, primal, brachFactor);
        filterRepresentedKernel<<<1,1>>>(this);
        sortByGKernel<<<1,1>>>(model,this);
        //saveCutsetLELKernel<<<1,1>>>(this);
        //saveCutsetKernel<<<1,1>>>(this);
        saveCutsetNewKernel<<<1,1>>>(this);
        mergeChildrenNewKernel<<<1,1>>>(this);
        calcOutLabelsNewKernel<<<gridSize,blockSize>>>(model,&children, &childrenInfo, primal,dual,DDRelaxed);
        checkForTargetNewKernel<<<1,1>>>(model,&expData.bestTargetNode,&children,&childrenInfo);
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
           Node const * node,
           gfl::f64 const primal,
           gfl::f64 const dual,
           gfl::i32 const brachFactor)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;

        expData.clear();
        cutData.clear();
        expData.children.pushBackGpu(node); // Not on parents! It will swap internally.
        //recLvl = 0;
        //expandRelaxedRecKernel<<<1,1>>>(model, this, primal, dual, brachFactor);
        //cudaDeviceSynchronize();
        initRootInfoKernel<<<1,1>>>(&expData.childrenInfo);
        cudaDeviceSynchronize();
        while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        {
            swapParentsAndChildrenKernel<<<1,1>>>(this);
            cudaDeviceSynchronize();
            //expandLayerRelaxedKernel<<<1,1>>>(model, this, primal, dual, brachFactor);
            expandLayerRelaxed(model, primal, dual, brachFactor);
            cudaDeviceSynchronize();
        }
        onlyBestTargets();
        finalizeCutset(model,primal,dual);
    }

    void expandRelaxed(
        Model const * model,
        std::vector<Node> const & parents,
        gfl::f64 const primal,
        gfl::f64 const dual,
        gfl::i32 const brachFactor)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;

        expData.clear();
        cutData.clear();
        expData.children.pushBackGpu(parents.data(), parents.size()); // Not on parents! It will swap internally.
        i32 const blockSize = 128;
        i32 const gridSizde = ceil<i32>(parents.size(),blockSize);
        resizeToKernel<<<1,1>>>(&expData.childrenInfo, parents.size());
        resetInfoIdxKernel<<<gridSizde,blockSize>>>(&expData.childrenInfo);
        //recLvl = 0;
        //expandRelaxedRecKernel<<<1,1>>>(model, this, primal, dual, brachFactor);
        //cudaDeviceSynchronize();
        //initRootInfoKernel<<<1,1>>>(&expData.childrenInfo);
        cudaDeviceSynchronize();
        while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        {
            swapParentsAndChildrenKernel<<<1,1>>>(this);
            cudaDeviceSynchronize();
            //expandLayerRelaxedKernel<<<1,1>>>(model, this, primal, dual, brachFactor);
            expandLayerRelaxed(model, primal, dual, brachFactor);
            cudaDeviceSynchronize();
        }
        onlyBestTargets();
        finalizeCutset(model,primal,dual);
    }
};



