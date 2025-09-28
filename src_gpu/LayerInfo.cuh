#pragma once

#include "node.hpp"
#include "MirrorAllocator.hpp"

#include <Types.hpp>

template<typename State, typename Labels>
struct alignas(16) GpuParent
{
    State state;
    Labels labels;
    gfl::f64 boundSrcToNode;
    ANode * node;
};

template<typename State>
struct alignas(16) GpuChild
{
    State state;
    gfl::i32 label;
    gfl::f64 heuristicNodeToSink;
    ANode * parentNode;
};

struct alignas(16) ChildInfo
{
    gfl::u64 hash;
    gfl::f64 boundSrcToNode;
    gfl::f64 cost;
    gfl::i64 id;
    gfl::i32 idx;
    gfl::u32 isRepresented;
};

struct alignas(8) ClassRange
{
    gfl::i32 begin;
    gfl::i32 end;
};

template<typename State, typename Labels>
struct LayerInfo
{
    using GpuParent = GpuParent<State, Labels>;
    using GpuChild = GpuChild<State>;

    // Parents
    gfl::i32 nParents;
    gfl::MirrorPtr<GpuParent> parents;

    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    // Children
    gfl::i32 nChildren;
    gfl::MirrorPtr<GpuChild> children;
    gfl::MirrorPtr<ChildInfo> childrenInfo;

    // Auxiliary Information
    gfl::i32 nClasses;
    ClassRange * classes;

    // CUB
    ChildInfo * tmpChildrenInfo;
    std::size_t cubTmpMemSize;
    void * cubTmpMem;

    LayerInfo() :
            nParents(0),
            parents(nullptr, nullptr),
            minLabel(gfl::numeric_limits<gfl::i32>::max()),
            maxLabel(gfl::numeric_limits<gfl::i32>::min()),
            nLabels(gfl::numeric_limits<gfl::i32>::min()),
            nChildren(0),
            children(nullptr, nullptr),
            childrenInfo(nullptr, nullptr),
            nClasses(0),
            classes(nullptr),
            tmpChildrenInfo(nullptr),
            cubTmpMemSize(0),
            cubTmpMem(nullptr)
    {}
};

struct DummyDecomposer96
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u32&, gfl::u64&> operator()(ChildInfo &) const
    {
        gfl::u32 tmp32 = 0;
        gfl::u64 tmp64 = 0;
        return{tmp32,tmp64};
    }
};

struct HashDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::u64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.hash};
    }
};

struct RepCostDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::u32&,gfl::f64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.isRepresented,childInfo.cost};
    }
};

