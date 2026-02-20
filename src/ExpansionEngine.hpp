#pragma once

#include "GFL.hpp"

#include "ExpansionData.hpp"
#include "CutsetData.hpp"
#include "ExpansionFunctions.hpp"

template<typename Model,typename Node>
class ExpansionEngine
{
private:
    using ExpansionData = ExpansionData<Node>;
    using CutsetData = CutsetData<Node>;

    gfl::i32 width_{0};
    ExpansionData * expData{nullptr};
    CutsetData * cutData{nullptr};
#ifdef __CUDACC__
    gfl::ArrayView<gfl::u8> cubAuxMem;
#endif

public:
    void initFullRelaxedExpansion(gfl::i32 const width, gfl::i32 const branchFactor, gfl::i32 const depth, gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        width_ = width;
        expData = new (alloc) ExpansionData();
        cutData = new (alloc) CutsetData();
        i32 const maxNodes = width * branchFactor;
        expData->init(maxNodes, alloc);
        cutData->init(width, branchFactor, depth, alloc);
    }

    void fullyExpandRelaxed(
        Model const * model,
        Node const & node,
        gfl::f64 const primal,
        gfl::f64 const dual)
    {
        using namespace gfl;
        expData->clear();
        cutData->clear();
        expData->parents.pushBack(node);
        expandLayerRelaxed(model, primal, dual);
        while (not expData->children.empty() and not expData->children.front().isTarget(model))
        {
            expData->swapParentsAndChildren();
            expandLayerRelaxed(model, primal, dual);
        }
        onlyBestTarget(model,expData->children, expData->nodesInfo);
        finializeCutset<Model,Node>(model,expData,cutData,primal,dual,DDRelaxed);
        // printf("CUTSET:\n");
        // for(auto const & c : cutData->nodes()) {Node::print(c);printf("\n");}
        // printf("\n");
    }

    bool hasTarget() const noexcept {return expData->hasTarget();}
    Node const & getTarget() const noexcept {return expData->getTarget();}
    gfl::ArrayView<gfl::ArrayView<Node>> cutset() const noexcept {return cutData->segments();}

private:
    void expandLayerRelaxed(Model const * const model, gfl::f64 const pBound, gfl::f64 const dBound)
    {

        expandParents(model, expData, pBound);
        filterChildren<Model,Node>(expData);
        // printf("BEFORE MERGE:\n");
        // for(auto const & c : expData->children) {Node::print(c);printf("\n");}
        // printf("\n");

        if (expData->children.size() > width_)
        {
            mergeChildren<Model,Node>(width_,expData,cutData);
        }
        // printf("AFTER MERGE:\n");
        // for(auto const & c : expData->children) {Node::print(c);printf("\n");}
        // printf("\n");
        calcOutLabels<Model,Node>(model,expData->children,pBound,dBound,DDRelaxed);

    }

#ifdef __CUDACC__
    static
    gfl::i64 cubAuxMemSize(gfl::i64 const nNodes)
    {
        std::size_t memSize = 0;
        void * dummyTmpMem = nullptr;
        NodeInfo * dummyNodeInfo = nullptr;
        cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                dummyTmpMem,
                dummyMemSize,
                dummyNodeInfo,
                dummyNodeInfo,
                nNodes,
                DummyDecomposer128{}); // Bigger key used
        CHECK_LAST_CUDA_ERROR();
        return memSize;
    }
#endif

    static
#ifdef __CUDACC__
    gfl::i64 dataMemSize(gfl::i64 const nParents, gfl::i32 const maxBranchFactor, bool cubAuxMem = false)
#else
    gfl::i64 dataMemSize(gfl::i64 const nParents, gfl::i32 const maxBranchFactor)
#endif
    {
        using namespace gfl;

        i64 const nChildren = nParents * maxBranchFactor;

        i64 memSize = 0;
        memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // parents
        memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // children
        memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
        memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
#ifdef __CUDACC__
        if (cubAuxMem) memSize += cubAuxMemSize(nChildren) + DefaultAlign; // auxMem for GPU sort
#endif
        memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // tmpNodes
        memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // tmpNodesInfo

        return memSize;
    }

    template<typename Fn>
    static
    gfl::i64 calcMaxParents(Fn calcMemSize, gfl::i64 const maxMemSize)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 ubParents = 1;
        while (calcMemSize(ubParents) <= maxMemSize)
        {
            lbParents = ubParents;
            ubParents *= 2;
        }
        while (lbParents < ubParents)
        {
            i64 const midParents = lbParents + (ubParents - lbParents + 1) / 2;
            i64 const memSize = calcMemSize(midParents);
            if (memSize <= maxMemSize) lbParents = midParents;  // still fits
            else ubParents = midParents - 1; // too big
        }
        return lbParents;
    }
};