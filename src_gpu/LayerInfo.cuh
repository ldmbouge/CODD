#pragma once

#include "node.hpp"
#include "MirrorAllocator.hpp"

#include <Types.hpp>

template<typename State, typename Labels>
struct LightNode
{
    State state;
    Labels labels;
    gfl::f64 boundSrcToNode;
    gfl::u8 nEdgesSrcToNode;
    gfl::u8 labelsSrcToNode[128];

    GFL_HOST_DEVICE
    LightNode() noexcept {};
};

struct NodeInfo
{
    gfl::u64 hash;
    gfl::f64 boundSrcToNode;
    gfl::i32 idx;
    gfl::u32 isRepresented;

    GFL_HOST_DEVICE
    NodeInfo() noexcept {};
};

template<typename State, typename Labels>
struct LayerInfo
{
    using Node = LightNode<State, Labels>;

    // Parents
    gfl::i32 nParents;
    Node * parents;

    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 labelsPerParents;

    // Children
    gfl::i32 nChildren;
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
        minLabel = gfl::numeric_limits<gfl::i32>::max();
        maxLabel = gfl::numeric_limits<gfl::i32>::min();
        labelsPerParents = 0;
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
    GFL_DEVICE
    gfl::tuple<gfl::u64&> operator()(NodeInfo & nodeInfo) const
    {
        return {nodeInfo.hash};
    }
};