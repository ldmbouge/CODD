#pragma once

#include <GFL.hpp>

template<typename Node>
class CutsetData
{
    gfl::VectorView<Node> nodes_;
    gfl::i64 markOffset_ = 0;  // offset into nodes_ where last segment starts
    gfl::i64 markSize_   = 0;  // size of last segment

public:
    void init(gfl::i64 nNodes, gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;
        nodes_ = VectorView<Node>(nNodes, alloc);
    }

    GFL_HOST_DEVICE
    void clear() noexcept
    {
        nodes_.clear();
        markOffset_ = 0;
        markSize_   = 0;
    }

    GFL_HOST_DEVICE
    void markAndResizeBy(gfl::i64 const count) noexcept
    {
        markOffset_ = nodes_.resizeBy(count);  // returns old size = start of new segment
        markSize_   = count;
    }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> mark() const noexcept
    {
        return nodes_.slice(markOffset_, markOffset_ + markSize_);
    }

    GFL_HOST_DEVICE
    gfl::VectorView<Node> const * nodes() const noexcept { return &nodes_; }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> const * nodesPtr() const noexcept { return &nodes_; }
};