#pragma once

#include "Contexts.hpp"
#include "GFL.hpp"

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
public:
    using ExpEng = ExpansionEngine<Model,Node>;
    using ExpEng::expData;
    using ExpEng::cutData;
    using ExpEng::nFlagged;
    using ExpEng::nNodes;
    using ExpEng::width_;
    using ExpEng::exact;
    using ExpEng::branchFactor_;

    bool toSave;
    gfl::ArrayView<gfl::u8> cubAuxMem;

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
    void expandParents(
    Model const * model,
    gfl::f64 const primal)
    {
        using namespace gfl;

        auto & parents     = expData.parents;
        auto & parentsInfo  = expData.parentInfo;
        auto & children    = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpInfo      = expData.tmpInfo;
        constexpr u8 IsNotChildren = 1;
        constexpr u8 IsChildren    = 0;

        i32 const maxParents  = width_;
        i32 const maxChildren = width_ * branchFactor_;
        i32 const blockSize   = 256;
        i32 const gridSizeParents  = maxParents;                        // one block per parent
        i32 const gridSizeChildren = ceil<i32>(maxChildren, blockSize); // over childrenInfo

        resizeToKernel<<<1,1>>>(&children,maxChildren);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo,maxChildren);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpInfo,maxChildren);
        CHECK_LAST_CUDA_ERROR();

        setFlagKernel<<<gridSizeChildren, blockSize>>>(IsNotChildren, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        expandParentsKernel<<<gridSizeParents, blockSize>>>(model, &expData, primal, IsChildren, branchFactor_);
        CHECK_LAST_CUDA_ERROR();
        setValueKernel<<<1,1>>>(&nFlagged, scast<i64>(0));
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSizeChildren, blockSize>>>(IsChildren, &nFlagged, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();

        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&childrenInfo, &tmpInfo, &cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
    }

    void filterRepresentedChildren()
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpNodes = expData.tmpNodes;
        auto & tmpInfo = expData.tmpInfo;
        constexpr u8 Represented = 1;
        constexpr u8 NotRepresented = 0;

        i32 const maxChildren = width_ * branchFactor_;
        i32 const blockSize    = 256;
        i32 const gridSize     = ceil<i32>(maxChildren, blockSize);


        resizeToKernel<<<1,1>>>(&tmpInfo,childrenInfo.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        calcHashKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::HashDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();

        setFlagKernel<<<gridSize,blockSize>>>(NotRepresented, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        flagRepresentedKernel<Model><<<gridSize,blockSize>>>(Represented,&children,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        setValueKernel<<<1,1>>>(&nFlagged,scast<i64>(0));
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSize,blockSize>>>(NotRepresented,&nFlagged,&childrenInfo);
        CHECK_LAST_CUDA_ERROR();

        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&childrenInfo,&nFlagged);
        CHECK_LAST_CUDA_ERROR();
    }

    void sortChildrenByG(gfl::f64 const lambda = 1.0)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpNodes = expData.tmpNodes;
        auto & tmpInfo = expData.tmpInfo;

        i32 const maxChildren = width_ * branchFactor_;
        i32 const blockSize = 256;
        i32 const gridSize = ceil<i32>(maxChildren, blockSize);

        resizeToKernel<<<1,1>>>(&tmpNodes,children.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpInfo,childrenInfo.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        setScoreAsGKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo, lambda);
        //setScoreAsFKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo, lambda);
        CHECK_LAST_CUDA_ERROR();

        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        CHECK_LAST_CUDA_ERROR();
        copyByInfoIdxKernel<<<gridSize,blockSize>>>(&tmpNodes, &children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpNodes, &children);
        CHECK_LAST_CUDA_ERROR();
    }


    void saveCutsetLEL()
    {
        using namespace gfl;
        auto const & parents = expData.parents;
        auto & parentsInfo = expData.parentInfo;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpNodes = expData.tmpNodes;
        auto & tmpInfo = expData.tmpInfo;
        auto & cutset = cutData;
        constexpr u8 ToNotSave = 1;
        constexpr u8 ToSave = 0;

        i32 const maxChildren = width_ * branchFactor_;
        i32 const blockSize = 256;
        i32 const gridSize = ceil<i32>(maxChildren, blockSize);

        markAndResizeByKernel<<<1,1>>>(&cutset, width_, &parentsInfo, &childrenInfo);       // ← updates markOffset_/markSize_
        CHECK_LAST_CUDA_ERROR();
        saveByCutsetMarkKernel<<<gridSize,blockSize>>>(&cutset, &parents, &parentsInfo); // ← mark() called device-side
        CHECK_LAST_CUDA_ERROR();

    }

    void saveCutset(bool saveLayer = false)
    {
        using namespace gfl;
        auto const & parents = expData.parents;
        auto & parentsInfo = expData.parentInfo;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & tmpNodes = expData.tmpNodes;
        auto & tmpInfo = expData.tmpInfo;
        auto & cutset = cutData;
        constexpr u8 ToNotSave = 1;
        constexpr u8 ToSave = 0;

        i32 const maxChildren = width_ * branchFactor_;
        i32 const blockSize = 256;
        i32 const gridSize = ceil<i32>(maxChildren, blockSize);

        resetInfoIdxKernel<<<gridSize,blockSize>>>(&parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        setFlagKernel<<<gridSize,blockSize>>>(ToNotSave, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        flagToSaveKernel<<<gridSize,blockSize>>>(width_,ToSave, &parentsInfo, &children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        updateAncInCutKernel<<<gridSize,blockSize>>>(ToSave, &parentsInfo, &children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpInfo, parentsInfo.sizePtr());
        CHECK_LAST_CUDA_ERROR();
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo, &tmpInfo, &cubAuxMem);
        CHECK_LAST_CUDA_ERROR();
        swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        setValueKernel<<<1,1>>>(&nFlagged, scast<i64>(0));
        CHECK_LAST_CUDA_ERROR();
        countFlaggedKernel<<<gridSize,blockSize>>>(ToSave, &nFlagged, &parentsInfo);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
        CHECK_LAST_CUDA_ERROR();
        markAndResizeByKernel<<<1,1>>>(&cutset, &nFlagged);       // ← updates markOffset_/markSize_
        CHECK_LAST_CUDA_ERROR();
        copyByCutsetMarkKernel<<<gridSize,blockSize>>>(&cutset, &parents, &parentsInfo); // ← mark() called device-side
        CHECK_LAST_CUDA_ERROR();

    }

    void trimChildren()
    {
        using namespace gfl;

        auto & childrenInfo       = expData.childrenInfo;

        // Shrink childrenInfo to width_ — merged node already sits at correct idx
        shrinkToKernel<<<1,1>>>(&childrenInfo, width_);
        CHECK_LAST_CUDA_ERROR();
    }

    void mergeChildren()
    {
        using namespace gfl;

        auto & children           = expData.children;
        auto & childrenInfo       = expData.childrenInfo;
        auto & tmpInfo            = expData.tmpInfo;
        auto & childrenInfoSuffix = expData.tmpInfoView;

        i32 const maxChildren     = width_ * branchFactor_;
        i32 const blockSize       = 32;
        i32 const nodesPerThread  = 32;
        i32 const reductionFactor = blockSize * nodesPerThread;
        i32 const suffixMaxSize   = max<i32>(0,maxChildren - (width_ - 1));
        i32 const nBlocks1        = ceil<i32>(suffixMaxSize, reductionFactor); // max blocks for pass1
        i32 const nBlocks2        = ceil<i32>(nBlocks1,      reductionFactor); // max blocks for pass2

        // Compute suffix = childrenInfo[width_-1 .. end]
        initSuffix<<<1,1>>>(width_, &childrenInfo, &childrenInfoSuffix);
        CHECK_LAST_CUDA_ERROR();

        copyValueKernel<<<1,1>>>(&nNodes, childrenInfoSuffix.sizePtr());
        CHECK_LAST_CUDA_ERROR();

        // Resize tmpInfo to max possible pass1 output count
        resizeToKernel<<<1,1>>>(&tmpInfo, nBlocks1);
        CHECK_LAST_CUDA_ERROR();

        // pass1: childrenInfoSuffix → tmpInfo, one result per block
        reduceByInfoKernel<Model,Node><<<nBlocks1, blockSize>>>(
            &children,&childrenInfoSuffix, &tmpInfo, &nNodes);
        CHECK_LAST_CUDA_ERROR();


        // Compute actual number of nodes produced by pass1
        ceilKernel<<<1,1>>>(&nNodes, reductionFactor);
        CHECK_LAST_CUDA_ERROR();

        // Resize tmpInfo to max possible pass1 output count
        resizeToKernel<<<1,1>>>(&tmpInfo, &nNodes);
        CHECK_LAST_CUDA_ERROR();

        // pass2: tmpInfo → childrenInfoSuffix, one result per block
        reduceByInfoKernel<Model,Node><<<nBlocks2, blockSize>>>(
            &children, &tmpInfo, &childrenInfoSuffix, &nNodes);
        CHECK_LAST_CUDA_ERROR();

        // Compute actual number of nodes produced by pass2
        ceilKernel<<<1,1>>>(&nNodes, reductionFactor);
        CHECK_LAST_CUDA_ERROR();

        // pass3: sequential final reduction, result at children[childrenInfoSuffix[0].idx]
        reduceByInfoSeqKernel<Model,Node><<<1,1>>>(&children, &childrenInfoSuffix, &nNodes);
        CHECK_LAST_CUDA_ERROR();

        // Shrink childrenInfo to width_ — merged node already sits at correct idx
        shrinkToKernel<<<1,1>>>(&childrenInfo, width_);
        CHECK_LAST_CUDA_ERROR();
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

        if (expData.bestTargetNode.has_value() and not cutset.nodes()->empty())
        {
            i32 const blockSize = 256;
            i32 const gridSize = ceil<i32>(cutset.nodes()->size(), blockSize);
            calcOutLabelsKernel<<<gridSize,blockSize>>>(model,cutset.nodesPtr(),primal,dual,DDRelaxed);
            CHECK_LAST_CUDA_ERROR();
            setHKernel<Model><<<gridSize,blockSize>>>(&expData.bestTargetNode,cutset.nodesPtr());
            CHECK_LAST_CUDA_ERROR();
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

        ExpEng::initRelaxedExpansion(width, branchFactor, depth, alloc);
        i32 const maxNodes = width_ * branchFactor_;
        initCubAuxMem(maxNodes,alloc);
    }

    void initRestrictedExpansion(
     gfl::i32 const width,
     gfl::i32 const branchFactor,
     gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        ExpEng::initRestrictedExpansion(width, branchFactor, alloc);
        i32 const maxNodes = width_ * branchFactor_;
        initCubAuxMem(maxNodes,alloc);
    }

    void expandRestricted(
        Model const * model,
        std::vector<Node> const & parents,
        gfl::f64 const primal,
        gfl::f64 const dual
        )
    {

        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & bestTrgt = expData.bestTargetNode;
        auto & bestExactTrgt = expData.bestExactTargetNode;

        expData.clear();
        cutData.clear();
        expData.children.pushBackGpuAsync(parents.data(), parents.size());
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&expData.childrenInfo, parents.size());
        CHECK_LAST_CUDA_ERROR();
        i32 const blockSize = 256;
        i32 const gridSize = ceil<i32>(width_ * branchFactor_, blockSize);
        resetInfoIdxKernel<<<gridSize, blockSize>>>(&expData.childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // i32 const initialDepth = parents.data()->depth();
        // i32 const relativeDepth = 2;//initialDepth;
        // i32 currentDepth = initialDepth;
        while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        {
            expData.swapParentsAndChildren();
            expandParents(model, primal);
            filterRepresentedChildren();
            sortChildrenByG();
            trimChildren();
            calcOutLabelsKernel<<<gridSize, blockSize>>>(model, &children, &childrenInfo, primal, dual, DDRelaxed);
            CHECK_LAST_CUDA_ERROR();
            checkForTargetKernel<<<1,1>>>(model, &expData.bestTargetNode, &children, &childrenInfo);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        copyBestTargetsKernel<Model,Node><<<1,1>>>(&bestTrgt,&bestExactTrgt,&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    void expandRelaxed(
        Model const * model,
        std::vector<Node> const & parents,
        gfl::f64 const primal,
        gfl::f64 const dual,
        gfl::f64 const lambda
        )
    {

        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & bestTrgt = expData.bestTargetNode;
        auto & bestExactTrgt = expData.bestExactTargetNode;

        expData.clear();
        cutData.clear();
        expData.children.pushBackGpuAsync(parents.data(), parents.size());
        CHECK_LAST_CUDA_ERROR();
        resizeToKernel<<<1,1>>>(&expData.childrenInfo, parents.size());
        CHECK_LAST_CUDA_ERROR();
        i32 const blockSize = 256;
        i32 const gridSize = ceil<i32>(width_ * branchFactor_, blockSize);
        resetInfoIdxKernel<<<gridSize, blockSize>>>(&expData.childrenInfo);
        CHECK_LAST_CUDA_ERROR();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // i32 const initialDepth = parents.data()->depth();
        // i32 const relativeDepth = 2;//initialDepth;
        // i32 currentDepth = initialDepth;
        while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        {
            expData.swapParentsAndChildren();
            expandParents(model, primal);
            filterRepresentedChildren();
            sortChildrenByG(lambda);
            saveCutset();
            mergeChildren();
            calcOutLabelsKernel<<<gridSize, blockSize>>>(model, &children, &childrenInfo, primal, dual, DDRelaxed);
            CHECK_LAST_CUDA_ERROR();
            checkForTargetKernel<<<1,1>>>(model, &expData.bestTargetNode, &children, &childrenInfo);
            CHECK_LAST_CUDA_ERROR();
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            if (cutData.nodes()->size() + width_ > cutData.nodes()->capacity())
            {
                //printf("[DBG] Flushing GPU cutset buffer of size %lld\n", cutData.nodes()->size());
                cutData.saveFragmentFromGpu();
            }

            // printf("nChildren %llu\n", childrenInfo.size());
            // fflush(stdout);
        }

        copyBestTargetsKernel<Model,Node><<<1,1>>>(&bestTrgt,&bestExactTrgt,&children, &childrenInfo);
        CHECK_LAST_CUDA_ERROR();

        if (not cutData.nodes()->empty())
        {
            //printf("[DBG] Flushing GPU cutset buffer of size %lld\n", cutData.nodes()->size());
            cutData.saveFragmentFromGpu();
        }
        // printf("---\n", childrenInfo.size());
        // fflush(stdout);
        // finalizeCutset(model, primal, dual);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
};



