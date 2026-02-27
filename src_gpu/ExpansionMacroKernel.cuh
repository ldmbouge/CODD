#pragma once
#include "ExpansionKernels.cuh"

//Forward declaration
template<typename Model, typename Node>
class ExpansionEngineGpu;

template<typename Model, typename Node>
GFL_GLOBAL
void filterRepresentedKernel(ExpansionEngineGpu<Model,Node> * const expEng)
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto & children = expData.children;
    auto & tmpNodes = expData.tmpNodes;
    auto & childrenInfo = expData.childrenInfo;
    auto & tmpInfo = expData.tmpInfo;
    auto & nFlagged = expEng->nFlagged;
    auto & cubAuxMem = expEng->cubAuxMem;
    constexpr i64 RepresentedFlag = 1;
    constexpr i64 RepresentativeFlag = 0;

    if (not children.empty())
    {
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(childrenInfo.size(), blockSize);

        // Init
        nFlagged = 0;
        assert(childrenInfo.size() <= tmpInfo.capacity());
        assert(children.size() == childrenInfo.size()); // ← add this
        tmpInfo.resizeTo(childrenInfo.size());

        // Sort by hash
        calcHashKernel<Model><<<gridSize,blockSize>>>(&children, &childrenInfo);
        sortKernel<NodeInfo::HashDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);

        // Find representatives
        setFlagKernel<<<gridSize,blockSize>>>(RepresentativeFlag, &childrenInfo);
        flagRepresentedChildrenKernel<Model><<<gridSize,blockSize>>>(RepresentedFlag,&children,&childrenInfo);
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        countFlaggedKernel<<<gridSize,blockSize>>>(RepresentativeFlag,&nFlagged,&childrenInfo);
        resizeToKernel<<<1,1>>>(&childrenInfo,&nFlagged);
        //resizeToKernel<<<1,1>>>(&tmpNodes,&nFlagged);
        //copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes,&children,&childrenInfo);
        //swapKernel<<<1,1>>>(&tmpNodes, &children);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void sortByGKernel(
    Model const * const model,
    ExpansionEngineGpu<Model,Node> * const expEng)
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto & children = expData.children;
    auto & tmpNodes = expData.tmpNodes;
    auto & childrenInfo = expData.childrenInfo;
    auto & tmpInfo = expData.tmpInfo;
    auto & cubAuxMem = expEng->cubAuxMem;

    if (children.size() > expEng->width_)
    {
        // Init
        tmpInfo.resizeTo(childrenInfo.size());
        tmpNodes.resizeTo(childrenInfo.size());

        // Processing
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(childrenInfo.size(), blockSize);

        setScoreMergeKernel<Model><<<gridSize,blockSize>>>(model, &children, &childrenInfo);
        sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&childrenInfo,&tmpInfo,&cubAuxMem);
        swapKernel<<<1,1>>>(&childrenInfo, &tmpInfo);
        copyByInfoKernel<<<gridSize,blockSize>>>(&tmpNodes, &children, &childrenInfo);
        swapKernel<<<1,1>>>(&tmpNodes, &children);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void saveCutsetNewKernel(ExpansionEngineGpu<Model,Node> * const expEng )
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto const & parents = expData.parents;
    auto & children = expData.children;
    auto & tmpNodes = expData.tmpNodes;
    auto & parentsInfo = expData.parentInfo;
    auto & childrenInfo = expData.childrenInfo;
    auto & tmpInfo = expData.tmpInfo;
    auto & nFlagged = expEng->nFlagged;
    auto & cutset = expEng->cutData;
    auto & cubAuxMem = expEng->cubAuxMem;
    auto & suffixChildrenInfo = expData.tmpInfoView;
    auto & width = expEng->width_;
    constexpr i64 NotSaveFlag = 1;
    constexpr i64 SaveFlag = 0;

    if (childrenInfo.size() > width)
    {
        // Find children to save
        nFlagged = 0;
        i32 const prefixSize = width - 1;
        suffixChildrenInfo = childrenInfo.slice(prefixSize,childrenInfo.size());
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(suffixChildrenInfo.size(),blockSize);
        setFlagKernel<<<gridSize,blockSize>>>(NotSaveFlag,&suffixChildrenInfo);
        flagChildrenToSaveKernel<<<gridSize,blockSize>>>(SaveFlag,&suffixChildrenInfo,&children);
        resizeToKernel<<<1,1>>>(&tmpInfo,suffixChildrenInfo.sizePtr());
        sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(
            &suffixChildrenInfo,
            scast<ArrayView<NodeInfo>*>(&tmpInfo),
            &cubAuxMem);
        countFlaggedKernel<<<gridSize,blockSize>>>(SaveFlag, &nFlagged, &tmpInfo);
        resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
        resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
        //assertCopyPrecondKernel<<<1,1>>>(&children, &tmpInfo, &cutset);
        copyByInfoKernel<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(), &children, &tmpInfo, false);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void saveCutsetKernel(ExpansionEngineGpu<Model,Node> * const expEng )
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto const & parents = expData.parents;
    auto & children = expData.children;
    auto & tmpNodes = expData.tmpNodes;
    auto & parentsInfo = expData.parentInfo;
    auto & childrenInfo = expData.childrenInfo;
    auto & tmpInfo = expData.tmpInfo;
    auto & nFlagged = expEng->nFlagged;
    auto & cutset = expEng->cutData;
    auto & cubAuxMem = expEng->cubAuxMem;
    constexpr i64 ParentToNotSaveFlag = 1;
    constexpr i64 ParentToSaveFlag = 0;

    if (children.size() > expEng->width_)
    {

        // // Init
        // nFlagged = 0;
        // parentsInfo.resizeTo(parents.size());
        //
        // // Find parents to save
        // i32 const blockSize = 128;
        // i32 const gridSize = ceil<i32>(expData.children.size(),blockSize);
        // resetInfoKernel<<<gridSize,blockSize>>>(&parentsInfo);
        // setFlagKernel<<<gridSize,blockSize>>>(ParentToNotSaveFlag,&parentsInfo);
        // flagParentsToSaveKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&parentsInfo,expEng->width_,&children,&childrenInfo);
        //
        // // Update children ancestor flag
        // updateAncestorKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&parentsInfo,&children,&childrenInfo);
        //
        // // Save parents in cutset
        // resizeToKernel<<<1,1>>>(&tmpInfo,parentsInfo.sizePtr());
        // sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo,&tmpInfo,&cubAuxMem);
        // swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        // countFlaggedKernel<<<gridSize,blockSize>>>(ParentToSaveFlag,&nFlagged,&parentsInfo);
        // resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        // resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
        // setScoreFKernel<Model><<<gridSize,blockSize>>>(&parents,&parentsInfo);
        // sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&parentsInfo,&tmpInfo,&cubAuxMem,true);
        // swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
        // resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
        // resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
        //
        // // printKernel<<<1,1>>>(4,cutset.lastSegmentPtr());
        // // printKernel<<<1,1>>>(5,&parents);
        // // printKernel<<<1,1>>>(6,&parentsInfo);
        //
        // copyByInfoKernel<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(),&parents, &parentsInfo);
        // // printKernel<<<1,1>>>(7,cutset.lastSegmentPtr());
        // // printKernel<<<1,1>>>(8,&parents);
        // // printKernel<<<1,1>>>(9,&parentsInfo);
        if (childrenInfo.size() > expEng->width_)  // ← childrenInfo
        {
            nFlagged = 0;
            parentsInfo.resizeTo(parents.size());

            i32 const blockSize = 128;
            i32 const gridSize = ceil<i32>(childrenInfo.size(), blockSize);  // ← childrenInfo
            resetInfoIdxKernel<<<gridSize,blockSize>>>(&parentsInfo);
            setFlagKernel<<<gridSize,blockSize>>>(ParentToNotSaveFlag, &parentsInfo);
            flagParentsToSaveKernel<<<gridSize,blockSize>>>(ParentToSaveFlag, &parentsInfo, expEng->width_, &children, &childrenInfo);

            updateAncestorKernel<<<gridSize,blockSize>>>(ParentToSaveFlag, &parentsInfo, &children, &childrenInfo);

            resizeToKernel<<<1,1>>>(&tmpInfo, parentsInfo.sizePtr());
            sortKernel<NodeInfo::FlagDecomposer><<<1,1>>>(&parentsInfo, &tmpInfo, &cubAuxMem);
            swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
            countFlaggedKernel<<<gridSize,blockSize>>>(ParentToSaveFlag, &nFlagged, &parentsInfo);
            resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
            resizeToKernel<<<1,1>>>(&tmpInfo, &nFlagged);
            setScoreFKernel<Model><<<gridSize,blockSize>>>(&parents, &parentsInfo);
            sortKernel<NodeInfo::ScoreDecomposer><<<1,1>>>(&parentsInfo, &tmpInfo, &cubAuxMem, true);
            swapKernel<<<1,1>>>(&tmpInfo, &parentsInfo);
            resizeToKernel<<<1,1>>>(&parentsInfo, &nFlagged);
            resizeByKernel<<<1,1>>>(&cutset, &nFlagged);
            copyByInfoKernel<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(), &parents, &parentsInfo);
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void saveCutsetLELKernel(ExpansionEngineGpu<Model,Node> * const expEng )
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto const & parents = expData.parents;
    auto & children = expData.children;
    auto & cutset = expEng->cutData;
    auto & cubAuxMem = expEng->cubAuxMem;

    if (children.size() > expEng->width_ and cutset.nodes().empty())
    {
        i32 const blockSize = 128;
        i32 const gridSize = ceil<i32>(expData.children.size(),blockSize);
        cutset.addSegment(children.size());
        copyAll<<<gridSize,blockSize>>>(cutset.lastSegmentPtr(), &children);
    }
}


template<typename Model, typename Node>
GFL_GLOBAL
void mergeChildrenNewKernel(ExpansionEngine<Model,Node> * const expEng)
{
    using namespace gfl;

    auto & expData           = expEng->expData;
    auto & children          = expData.children;
    auto & childrenInfo      = expData.childrenInfo;
    auto & tmpInfo           = expData.tmpInfo;
    auto & childrenInfoSuffix           = expData.tmpInfoView;
    auto & width             = expEng->width_;

    if (childrenInfo.size() > width)
    {
        // suffix of childrenInfo — nodes to be merged
        i32 const prefixSize = width - 1;
        childrenInfoSuffix = childrenInfo.slice(prefixSize,childrenInfo.size());

        i32 const nNodes = childrenInfoSuffix.size();
        tmpInfo.resizeTo(nNodes);
        i32 const blockSize       = 32;
        i32 const nodesPerThread  = 32;
        i32 const reductionFactor = blockSize * nodesPerThread;
        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);
        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);

        // pass1: childrenInfoSuffix → tmpInfo
        reduceByInfoKernel<Model,Node><<<nBlocks1,blockSize>>>(&children, &childrenInfoSuffix, &tmpInfo, nNodes);
        // pass2: tmpInfo → childrenInfoSuffix
        reduceByInfoKernel<Model,Node><<<nBlocks2,blockSize>>>(&children, &tmpInfo, &childrenInfoSuffix, nBlocks1);
        // pass3: final sequential reduction, result at children[childrenInfoSuffix[0].idx]
        reduceByInfoSeqKernel<Model,Node><<<1,1>>>(&children, &childrenInfoSuffix, nBlocks2);

        // children array untouched in size — just shrink childrenInfo
        // the merged node already sits in children[childrenInfo[prefixSize].idx]
        resizeToKernel<<<1,1>>>(&childrenInfo, width);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void mergeChildrenKernel(ExpansionEngine<Model,Node> * const expEng )
{
    using namespace gfl;

    auto & expData = expEng->expData;
    auto const & parents = expData.parents;
    auto & children = expData.children;
    auto & tmpNodes = expData.tmpNodes;
    auto & childrenInfo = expData.childrenInfo;
    auto & childrenPrefix = expData.tmpView;
    auto & width = expEng->width_;

    if (children.size() > width)
    {
        // Init
        tmpNodes.resizeTo(children.size());
        initChildrenPrefix(width,&children,&childrenPrefix);

        // Merge the last children - (width - 1) nodes
        i32 const blockSize       = 32;
        i32 const nodesPerThread  = 32;
        i32 const reductionFactor = blockSize * nodesPerThread;  // 1024
        i32 const nNodes          = childrenPrefix.size(); // real item count
        i32 const nBlocks1        = ceil<i32>(nNodes,   reductionFactor);  // gridDim for pass1, real count for pass2
        i32 const nBlocks2        = ceil<i32>(nBlocks1, reductionFactor);  // gridDim for pass2, real count for pass3
        // pass1: childrenPrefix → tmpNodes, count = nNodes
        reductionKernel<Model,Node><<<nBlocks1, blockSize>>>(&childrenPrefix,&tmpNodes,nNodes);
        // pass2: tmpNodes → childrenPrefix, count = nBlocks1
        reductionKernel<Model,Node><<<nBlocks2, blockSize>>>(&tmpNodes,&childrenPrefix,nBlocks1);
        // pass3: childrenPrefix → childrenPrefix[0], count = nBlocks2
        resizeToKernel<<<1,1>>>(&children,width);
        resizeToKernel<<<1,1>>>(&childrenInfo,width);
    }
}



template<typename Model, typename Node>
GFL_GLOBAL
void PLayerRelaxedKernel(
        Model const * model,
        ExpansionEngineGpu<Model,Node> * const expEng,
        gfl::f64 const primal,
        gfl::f64 const dual,
        gfl::i32 const brachFactor)
{
    using namespace gfl;
    auto & expData = expEng->expData;
    auto & children = expData.children;
    auto & parents = expData.parents;

    i32 blockSize = 32;
    i32 gridSize = ceil<i32>(parents.size() *brachFactor,blockSize) ;
    expandParentsKernel<<<gridSize,blockSize>>>(model, &expData, primal, brachFactor);
    filterRepresentedKernel<<<1,1>>>(expEng);
    sortByGKernel<<<1,1>>>(model,expEng);
    saveCutsetLELKernel<<<1,1>>>(expEng);
    mergeChildrenKernel<<<1,1>>>(expEng);
    calcOutLabelsKernel<<<gridSize,blockSize>>>(model,&children,primal,dual,DDRelaxed);
    checkForTargetKernel<<<1,1>>>(model,&expData.bestTargetNode,&children);
}

template<typename Model, typename Node>
GFL_GLOBAL
void expandRelaxedRecKernel(
       Model const * model,
       ExpansionEngineGpu<Model,Node> * const expEng,
       gfl::f64 const primal,
       gfl::f64 const dual,
       gfl::i32 const brachFactor)
{
    using namespace gfl;
    auto & expData = expEng->expData;
    //printf("Entering LVL %d | C = %d | BT = %d\n", expEng->recLvl++, expEng->expData.children.size(), expData.bestTargetNode.has_value());
    if (not expData.children.empty() and not expData.bestTargetNode.has_value())
    {
        expData.swapParentsAndChildren();
        //printf("In LVL %d | P = %d\n", expEng->recLvl++, expEng->expData.parents.size());
        expandLayerRelaxedKernel<<<1,1>>>(model, expEng, primal, dual, brachFactor);
        expandRelaxedRecKernel<<<1,1,0,cudaStreamTailLaunch>>>(model, expEng, primal, dual, brachFactor);
    }
}




