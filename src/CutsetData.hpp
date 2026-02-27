#pragma once

#include <GFL.hpp>


template <typename Node>
class CutsetData
{
    gfl::VectorView<Node> nodes_;
    gfl::ArrayView<Node> mark_;

    static
    gfl::i64 size(
        gfl::i32 const width,
        gfl::i32 const branchFactor,
        gfl::i32 const depth) noexcept
    {
        return branchFactor * depth * (2 * width - depth + 1) / 2;
    }

public:
    void init(
        gfl::i32 const width,
        gfl::i32 const branchFactor,
        gfl::i32 const depth,
        gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;
        nodes_  = VectorView<Node>(size(width,branchFactor,depth), alloc);
    }

    GFL_HOST_DEVICE
    static
    gfl::i64 dataMemSize (
        gfl::i32 const width,
        gfl::i32 const branchFactor,
        gfl::i32 const depth) noexcept
    {
        using namespace gfl;

        i64 const memSize = VectorView<Node>::dataMemSize(size(width,branchFactor,depth)) + DefaultAlign;
        return memSize;
    }

    GFL_HOST_DEVICE
    void clear() noexcept { nodes_.clear();}

    GFL_HOST_DEVICE
    gfl::VectorView<Node> const * nodes() const noexcept { return &nodes_;}

    GFL_HOST_DEVICE
    void markAndResizeBy(gfl::i32 const count) noexcept
    {
        gfl::u64 const oldSize = nodes_.resizeBy(count);
        mark_ = nodes_.slice(oldSize, nodes_.size());
    }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> const * nodesPtr() const noexcept { return &nodes_; }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> const * mark() const noexcept { return &mark_;}
};
