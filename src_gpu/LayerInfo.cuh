#pragma once

#include "node.hpp"
#include "MirrorAllocator.hpp"

#include <Types.hpp>

template<typename State, typename Labels, int N>
struct alignas(16) LightNode
{
    State state;
    Labels labels;
    gfl::f64 boundSrcToNode;
    gfl::u8 nEdgesSrcToNode;
    gfl::u8 labelsSrcToNode[gfl::roundUpToMultiple<int>(N,64)];

    GFL_HOST_DEVICE
    LightNode() noexcept {};

    GFL_HOST_DEVICE
    LightNode(LightNode const & other) noexcept {memcpy(this,&other,sizeof(LightNode));};
};

struct NodeInfo
{
    gfl::u64 hash;
    gfl::f64 boundSrcToNode;
    gfl::i64 idx;
    gfl::u32 isRepresented;

    GFL_HOST_DEVICE
    NodeInfo() noexcept {};
};

struct LabelsInfo
{
    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    LabelsInfo():
        minLabel(gfl::numeric_limits<gfl::i32>::max()),
        maxLabel(gfl::numeric_limits<gfl::i32>::min()),
        nLabels(0)
    {};

    void reset()
    {
        minLabel = gfl::numeric_limits<gfl::i32>::max();
        maxLabel = gfl::numeric_limits<gfl::i32>::min();
        nLabels = 0;
    }

    void update(LabelsInfo const & other)
    {
        minLabel = gfl::min<int>(minLabel, other.minLabel);
        maxLabel = gfl::max<int>(maxLabel, other.maxLabel);
        nLabels = gfl::max<int>(nLabels, other.nLabels);
    }

    void update(gfl::backend::tuple<gfl::i32, gfl::i32,gfl::i32> slc)
    {
        minLabel = gfl::min<int>(minLabel, gfl::backend::get<0>(slc));
        maxLabel = gfl::max<int>(maxLabel, gfl::backend::get<1>(slc));
        nLabels = gfl::max<int>(nLabels, gfl::backend::get<2>(slc));
    }
};

template<typename Node>
struct LayerInfo
{
    // Parents
    gfl::i64 nParents;
    Node * parents;

    LabelsInfo labelsInfo;

    // Children
    gfl::i64 nChildren;
    Node * children;
    NodeInfo * childrenInfo;

    // CUB
    NodeInfo * tmpChildrenInfo;
    std::size_t cubTmpMemSize;
    void * cubTmpMem;

    void clear()
    {
        nParents = 0;
        parents = nullptr;
        labelsInfo = LabelsInfo();
        nChildren = 0;
        children = nullptr;
        childrenInfo = nullptr;
        tmpChildrenInfo = nullptr;
        cubTmpMemSize = 0;
        cubTmpMem = nullptr;
    }
};

struct DummyDecomposer64
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&, gfl::u32&> operator()(NodeInfo &) const
    {
        gfl::u32 tmp32 = 0;
        return {tmp32,tmp32};
    }
};

struct HashDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.hash};
    }
};

struct IdxDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::i64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.idx};
    }
};

struct SelectNotRep
{
    GFL_HOST_DEVICE
    bool operator()(NodeInfo const & nodeInfo) const
    {
        return nodeInfo.isRepresented != 1;
    }
};