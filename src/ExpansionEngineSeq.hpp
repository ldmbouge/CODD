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

public:
    using ExpansionEngine<Model,Node>::cutData;
    using ExpEng::exact;
    using ExpEng::completed;

    void expandRelaxed(
            Model const * model,
            std::vector<Node> const & parents,
            gfl::f64 const primal,
            gfl::f64 const dual,
            gfl::f64 const lambda,
            bool saveCut = true)
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
        while (not childrenInfo.empty()
               and not expData.bestTargetNode.has_value())
        {
            //printf("Iteration %d\n", i++); fflush(stdout);
            expData.swapParentsAndChildren();
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            expandParents(model,expData,primal);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            filterRepresentedChildren<Model,Node>(expData);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            sortChildrenByG<Model,Node>(expData, lambda);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            exact = exact and (childrenInfo.size() <= width_);
            if (expData.childrenInfo.size() > width_)
            {
                if (saveCut)
                {
                    saveCutset(width_,expData,cutData);
                }
                mergeChildren<Model,Node>(width_,expData);
                childrenInfo.resizeTo(min<i64>(width_, childrenInfo.size()));
            }
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            calcOutLabels<Model,Node>(model, children, childrenInfo, primal, dual, DDRestricted);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            checkForTarget(model, expData);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            if (cutData.nodes()->size() + width_ > cutData.nodes()->capacity())
            {
                //printf("[DBG] Flushing GPU cutset buffer of size %lld\n", cutData.nodes()->size());
                cutData.saveFragment();
            }
        }
        copyBestTargets<Model>(expData);
        if (not cutData.nodes()->empty())
        {
            //printf("[DBG] Flushing GPU cutset buffer of size %lld\n", cutData.nodes()->size());
            cutData.saveFragment();
        }
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

        expData.children.pushBack(parents);
        childrenInfo.resizeTo(parents.size());
        resetInfoIdx(childrenInfo);

        exact = true;
        while (not childrenInfo.empty()
               and not expData.bestTargetNode.has_value())
        {
            //printf("Iteration %d\n", i++); fflush(stdout);
            expData.swapParentsAndChildren();
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            expandParents(model,expData,primal);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            filterRepresentedChildren<Model,Node>(expData);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            sortChildrenByG<Model,Node>(expData);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            exact = exact and (childrenInfo.size() <= width_);
            childrenInfo.resizeTo(min<i64>(width_, childrenInfo.size()));
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            calcOutLabels<Model,Node>(model, children, childrenInfo, primal, dual, DDRestricted);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            checkForTarget(model,expData);
            //for (auto const & i : childrenInfo) {children[i.idx].print(); printf("\n");}
            //printf("\n"); fflush(stdout);
            if (not childrenInfo.empty() and isWorse<Model>(children[childrenInfo[0].idx].g(), primal))
            {
                completed = false;
                break;
            }
        }
    }
};
