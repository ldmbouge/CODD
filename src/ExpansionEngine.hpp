#pragma once

#include "GFL.hpp"

#include "ExpansionData.hpp"
#include "CutsetData.hpp"
#include "ExpansionFunctions.hpp"
#
#ifdef __CUDACC__
#include <Sort.cuh>
#include <cub/cub.cuh>
#endif

template<typename Model,typename Node>
class ExpansionEngine
{
public:
    using ExpansionData = ExpansionData<Node>;
    using CutsetData = CutsetData<Node>;

    ExpansionData expData;
    CutsetData cutData;
    gfl::i64 nFlagged;
    gfl::i32 width_;
    gfl::i32 branchFactor_;
    gfl::i32 depth_;
    gfl::i64 nNodes;
    bool exact;

public:

    void initRestrictedExpansion(gfl::i32 const width, gfl::i32 const branchFactor, gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        width_ = width;
        branchFactor_ = branchFactor;
        expData.init(width_ * branchFactor_, alloc);
    }

    void initRelaxedExpansion(gfl::i32 const width, gfl::i32 const branchFactor, gfl::i32 const depth, gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        width_ = width;
        branchFactor_ = branchFactor;
        depth_ = depth;

        printf("Relaxed working memory: ");
        printMemSize(alloc.usedSize());
        printf("\n");

        expData.init( width_, branchFactor_, alloc);

        printf("Relaxed working memory: ");
        printMemSize(alloc.usedSize());
        printf("\n");


        cutData.init( width_, branchFactor, depth, alloc);

        printf("Relaxed working memory: ");
        printMemSize(alloc.usedSize());
        printf("\n");

    }

    gfl::tuple<gfl::optional<Node> const &,gfl::optional<Node> const &> getTargets() const noexcept
    {
        return {expData.bestTargetNode, expData.bestExactTargetNode};
    }
};
