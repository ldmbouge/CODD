#pragma once

#include <cub/cub.cuh>
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
    gfl::u8 labelsSrcToNode[N];

    GFL_HOST_DEVICE
    LightNode() noexcept {};

    GFL_HOST_DEVICE
    LightNode(State const & s, Labels const & l) noexcept :
            state(s),
            labels(l),
            boundSrcToNode(0),
            nEdgesSrcToNode(0)
        {};
};

struct alignas(16) NodeInfo
{
    gfl::u64 hash;
    gfl::i64 idx;
    gfl::u32 isRepresented;
    gfl::f64 boundSrcToNode;

    GFL_HOST_DEVICE
    NodeInfo() noexcept {};
};

struct alignas(16) MergeInfo
{
    gfl::i32 aIdx;
    gfl::i32 bIdx;
    gfl::f32 score;
    gfl::u32 isRepresented;

    GFL_HOST_DEVICE
    MergeInfo() noexcept {};
    MergeInfo(gfl::i32 const i, gfl::i32 const j, gfl::f32 const s) noexcept :
        aIdx(i),
        bIdx(j),
        score(s),
        isRepresented(0)
        {};
};

struct LabelsInfo
{
    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    LabelsInfo() noexcept :
        minLabel(gfl::numeric_limits<gfl::i32>::max()),
        maxLabel(gfl::numeric_limits<gfl::i32>::min()),
        nLabels(0)
    {};

    LabelsInfo(gfl::backend::tuple<gfl::i32, gfl::i32,gfl::i32> const & t) noexcept:
            minLabel(gfl::backend::get<0>(t)),
            maxLabel(gfl::backend::get<1>(t)),
            nLabels(gfl::backend::get<2>(t))
    {};

    GFL_HOST_DEVICE
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
    Node * tmpParents;

    gfl::f64 dBound;
    LabelsInfo labelsInfo;

    // Children
    gfl::i64 nChildren;
    gfl::i64 nRepresentatives;
    Node * children;
    Node * tmpChildren;
    // TODO Make union NodeInfo - SimInfo to save space
    NodeInfo * childrenInfo;
    NodeInfo * tmpChildrenInfo;
    MergeInfo * mergeInfo;
    MergeInfo * tmpMergeInfo;

    // Aux
    std::size_t cubTmpMemSize;
    void * cubTmpMem;

    void clear()
    {
        nParents = 0;
        parents = nullptr;
        tmpParents = nullptr;
        dBound = 0;
        labelsInfo = LabelsInfo();
        nChildren = 0;
        nRepresentatives = 0;
        children = nullptr;
        tmpChildren = nullptr;
        childrenInfo = nullptr;
        tmpChildrenInfo = nullptr;
        cubTmpMemSize = 0;
        cubTmpMem = nullptr;
    }
};


struct DummyDecomposer64
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&> operator()(NodeInfo &) const
    {
        gfl::u64 tmp = 0;
        return {tmp};
    }
};

struct DummyDecomposer128
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::f64&, gfl::f64&> operator()(NodeInfo &) const
    {
        gfl::f64 tmp = 0;
        return {tmp,tmp};
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

struct CmpNodeByHash
{
    template <typename Node>
    GFL_HOST_DEVICE
    bool operator()(const Node &lhs, const Node &rhs)
    {
        return lhs.hash < rhs.hash;
    }
};

struct CmpNodeByRep
{
    template <typename Node>
    GFL_HOST_DEVICE
    bool operator()(const Node &lhs, const Node &rhs)
    {
        return lhs.isRepresented < rhs.isRepresented;
    }
};


struct RepDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.isRepresented};
    }
};

struct RepCostDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&,gfl::f64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.isRepresented,nodeInfo.boundSrcToNode};
    }
};

struct RepHashDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&, gfl::u64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.isRepresented, nodeInfo.hash};
    }
};

struct RepCostHashDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&,gfl::f64&,gfl::u64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.isRepresented,nodeInfo.boundSrcToNode,nodeInfo.hash};
    }
};
