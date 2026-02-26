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
    gfl::i32 nFlagged;
    gfl::i32 width_;

public:

    void initRelaxedExpansion(gfl::i32 const width, gfl::i32 const branchFactor, gfl::i32 const depth, gfl::ArenaAllocator & alloc)
    {
        using namespace gfl;

        width_ = width;
        nFlagged = 0;

        i32 const maxNodes = width_ * branchFactor;
        expData.init(maxNodes, alloc);
        cutData.init(width, branchFactor, depth, alloc);

    }

    gfl::tuple<gfl::ArrayView<Node>, gfl::ArrayView<gfl::i32>> cutset() const noexcept
    { return {cutData.nodes(),cutData.offsets()};}

    gfl::tuple<gfl::optional<Node> const &,gfl::optional<Node> const &> getTargets() const noexcept
    {return {expData.bestTargetNode, expData.bestExactTargetNode};}
};
