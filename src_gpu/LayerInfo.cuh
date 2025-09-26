#pragma once

#include "node.hpp"

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
    gfl::u64 eqHash;
    gfl::u64 domHash;
    gfl::f64 boundSrcToNode;
    gfl::f64 cost;
    gfl::i64 id;
    gfl::i32 idx;
    gfl::u32 isRepresented; // No atomic operations on bools in CUDA
};

template<typename State, typename Labels>
struct LayerInfo
{
    // Parents
    gfl::i32 nParents;
    GpuParent<State,Labels> * parents;

    // Labels
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    // Children
    gfl::i32 nChildren;
    GpuChild<State> * children;
    ChildInfo * childrenInfo;

    // Auxiliary Information
    gfl::i32 nClasses;
    gfl::i32 * classesBegin;

    // CUB
    ChildInfo * tmpChildrenInfo;
    gfl::i32 * tmpClassesBegin;
    std::size_t cubTmpMemSize;
    void * cubTmpMem;
};

struct dummyDecomposer
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&, gfl::u64&, gfl::u64&> operator()(ChildInfo &) const
    {
        gfl::u64 tmp = 0;
        return{tmp,tmp,tmp};
    }
};

struct CostDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::f64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.cost};
    }
};

struct EqHashDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::u64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.eqHash};
    }
};

struct DomHashDecomposer
{
    GFL_DEVICE
    gfl::tuple<gfl::u64&> operator()(ChildInfo & childInfo) const
    {
        return {childInfo.domHash};
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

template<typename T>
struct NoDecomposer
{
    GFL_DEVICE
    gfl::tuple<T&> operator()(T & t) const
    {
        return {t};
    }
};

