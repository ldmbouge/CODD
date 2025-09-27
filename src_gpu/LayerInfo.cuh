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
    gfl::u32 isRepresented; // No atomic operations on bools in CUDA
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
    GpuParent * parents;

    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    // Children
    gfl::i32 nChildren;
    GpuChild * children;
    ChildInfo * childrenInfo;

    // Auxiliary Information
    gfl::i32 nClasses;
    ClassRange * classes;

    // CUB
    ChildInfo * tmpChildrenInfo;
    std::size_t cubTmpMemSize;
    void * cubTmpMem;

    LayerInfo() :
            nParents(0),
            parents(nullptr),
            minLabel(gfl::numeric_limits<gfl::i32>::max()),
            maxLabel(gfl::numeric_limits<gfl::i32>::min()),
            nLabels(gfl::numeric_limits<gfl::i32>::min()),
            nChildren(0),
            children(nullptr),
            childrenInfo(nullptr),
            nClasses(0),
            classes(nullptr),
            tmpChildrenInfo(nullptr),
            cubTmpMemSize(0),
            cubTmpMem(nullptr)
    {}
};

struct DummyDecomposer128
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&, gfl::u64&> operator()(ChildInfo &) const
    {
        gfl::u64 tmp = 0;
        return{tmp,tmp};
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

struct RepIdDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::u32&,gfl::i64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.isRepresented,childInfo.id};
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

