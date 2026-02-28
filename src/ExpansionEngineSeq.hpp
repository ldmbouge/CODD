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
    using ExpEng = ExpansionEngine<Model,Node>;
    using ExpEng::expData;
    using ExpEng::width_;

    void expandLayerRelaxed(Model const * const model, gfl::f64 const pBound, gfl::f64 const dBound)
    {
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

    void expandLayerRestricted(Model const * const model, gfl::f64 const pBound, gfl::f64 const dBound)
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


public:
    using ExpansionEngine<Model,Node>::cutData;
    using ExpEng::exact;

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

    void expandRestricted(
       Model const * model,
       std::vector<Node> const & parents,
       gfl::f64 const primal,
       gfl::f64 const dual)
    {
        using namespace gfl;
        auto & children = expData.children;
        auto & childrenInfo = expData.childrenInfo;
        auto & bestTrgt = expData.bestTargetNode;

        expData.clear();
        cutData.clear();

        expData.children.pushBack(parents);
        childrenInfo.resizeTo(parents.size());
        resetInfoIdx(childrenInfo);

        exact = true;
        while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        {
            expData.swapParentsAndChildren();
            expandParents(model,expData,primal);
            filterRepresentedChildren<Model,Node>(expData);
            sortChildrenByF<Model,Node>(expData);
            exact = exact and (childrenInfo.size() <= width_);
            childrenInfo.resizeTo(min<i64>(width_, childrenInfo.size()));
            calcOutLabels<Model,Node>(model, children, childrenInfo, primal, dual, DDRestricted);
            checkForTarget(model, bestTrgt, children, childrenInfo);
        }


        // while (not childrenInfo.empty() and not expData.bestTargetNode.has_value())
        // {
        //     expData.swapParentsAndChildren();
        //     expandParents(model,expData,primal);
        //     filterRepresentedChildren<Model,Node>(expData);
        //     //sortChildrenByG<Model,Node>(expData);
        //     sortChildrenByF<Model,Node>(expData);
        //     exact = exact and (childrenInfo.size() <= width_);
        //     childrenInfo.resizeTo(min<i64>(width_, childrenInfo.size()));
        //     calcOutLabels<Model,Node>(model, children, childrenInfo, primal, dual, DDRestricted);
        //     checkForTarget(model, bestTrgt, children, childrenInfo);
        //     printf("RS: %llu (&d)\n", childrenInfo.size());
        // }
        // printf("===");
    }
};
