#pragma once

#include "Utils.hpp"

struct NodeInfo
{
    gfl::u64 hash;
    gfl::i64 idx;
    gfl::u32 flag;
    gfl::f64 score;

    GFL_HOST_DEVICE
    NodeInfo() noexcept {};
};

struct LabelsInfo
{
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    LabelsInfo(gfl::backend::tuple<gfl::i32,gfl::i32,gfl::i32> const & t) noexcept
    {
        reset();
        update(t);
    };

    GFL_HOST_DEVICE
    void reset()
    {
        minLabel = gfl::numeric_limits<gfl::i32>::max();
        maxLabel = gfl::numeric_limits<gfl::i32>::min();
        nLabels = 0;
    }

    void update(LabelsInfo const & li)
    {
        update(li.minLabel,li.maxLabel, li.nLabels);
    }

    void update(gfl::backend::tuple<gfl::i32, gfl::i32,gfl::i32> t)
    {
        update(gfl::backend::get<0>(t),gfl::backend::get<1>(t), gfl::backend::get<2>(t));
    }

    void update(gfl::i32 const minl, gfl::i32 const maxl, gfl::i32 const nl)
    {
        minLabel = gfl::min<gfl::i32>(minLabel, minl);
        maxLabel = gfl::max<gfl::i32>(maxLabel, maxl);
        nLabels = gfl::max<gfl::i32>(nLabels, nl);
    }

    LabelsInfo() noexcept {reset();}
};

template<typename Node>
struct BatchInfo
{
    gfl::f64 dBound;
    LabelsInfo labelsInfo;

    gfl::i64 nParents;
    Node * parents;
    Node * tmpParents;

    gfl::i64 nChildren;
    Node * children;
    Node * tmpChildren;
    gfl::i64 nFlagged;
    NodeInfo * childrenInfo;
    NodeInfo * tmpChildrenInfo;

    std::size_t cubTmpMemSize;
    void * cubTmpMem;

    void reset()
    {
        dBound = 0;
        labelsInfo.reset();

        nParents = 0;
        parents = nullptr;
        tmpParents = nullptr;

        nChildren = 0;
        children = nullptr;
        tmpChildren = nullptr;
        nFlagged = 0;
        childrenInfo = nullptr;
        tmpChildrenInfo = nullptr;

        cubTmpMemSize = 0;
        cubTmpMem = nullptr;
    }

    BatchInfo() noexcept {reset();}
};
