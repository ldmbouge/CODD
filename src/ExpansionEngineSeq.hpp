#pragma once

#include "GFL.hpp"

#include "ExpansionData.hpp"
#include "CutsetData.hpp"
#include "ExpansionEngine.hpp"
#include "ExpansionFunctions.hpp"
#include "Sort.cuh"


template<typename Model,typename Node>
class ExpansionEngineSeq : public ExpansionEngine<Model,Node>
{
    using ExpansionEngine<Model,Node>::expData;
    using ExpansionEngine<Model,Node>::width_;
public:
    using ExpansionEngine<Model,Node>::cutData;

    void fullyExpandRelaxed(
            Model const * model,
            std::vector<Node> const & nodes,
            gfl::f64 const primal,
            gfl::f64 const dual)
   {
      using namespace gfl;
      expData.clear();
      cutData.clear();
      expData.parents.pushBack(nodes);
      expandLayerRelaxed(model, primal, dual);
      while (not expData.children.empty() and not expData.children.front().isTarget(model))
         {
            expData.swapParentsAndChildren();
            expandLayerRelaxed(model, primal, dual);
         }
      onlyBestTargets(model,&expData);
      finializeCutset<Model,Node>(model,&expData,&cutData,primal,dual,DDRelaxed);
   }

private:
    void expandLayerRelaxed(Model const * const model, gfl::f64 const pBound, gfl::f64 const dBound)
    {
        // printf("BEFORE EXP (PARENTS):\n");
        // for(auto const & c : expData.parents) {Node::print(c);printf("\n");}
        // printf("\n");
       //std::cout << "PARENTS ARE:" << expData->parents << "\n";
       expandParents(model, &expData, pBound);
       //std::cout << "RAW MEAT:" << expData->children << "\n";

       // std::cout << "----------------------------------------------------------------------" << "\n";
       // {
       //    int i=0;
       //    for(const auto& c : expData->children) {
       //       std::cout << "KID[" << i<< "]= ";
       //       Node::print(c); std::cout << "\n";
       //       i++;
       //    }
       // }
       // std::cout << "----------------------------------------------------------------------" << "\n";
        // printf("BEFORE FLT:\n");
        // for(auto const & c : expData.children) {Node::print(c);printf("\n");}
        // printf("\n");

       filterChildren<Model,Node>(&expData);
        // printf("BEFORE MRG:\n");
        // for(auto const & c : expData.children) {Node::print(c);printf("\n");}
        // printf("\n");
       if (expData.children.size() > width_)
          {
             mergeChildren<Model,Node>(width_,&expData,&cutData);
          }
       // if  (expData->children[0].depth() >= 41) {
       //    std::cout << "We are deep! " << "\n";
       //    {
       //       int i=0;
       //       for(const auto& c : expData->children) {
       //          std::cout << "KID[" << i<< "]= ";
       //          Node::print(c); std::cout << "\n";
       //          i++;
       //       }
       //    }
       // }
       calcOutLabels<Model,Node>(model,&expData.children,pBound,dBound,DDRelaxed);
        // printf("AFTER MRG:\n");
        // for(auto const & c : expData.children) {Node::print(c);printf("\n");}
        // printf("\n");
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
                memSize,
                dummyNodeInfo,
                dummyNodeInfo,
                nNodes,
                DummyDecomposer64{}); // Bigger key used
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
