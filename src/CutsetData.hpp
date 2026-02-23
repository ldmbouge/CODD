#pragma once

#include <GFL.hpp>

template <typename Node>
class CutsetData
{

    gfl::VectorView<Node> nodes_{};
    gfl::VectorView<gfl::ArrayView<Node>> segments_{};
    gfl::ArrayView<Node> lastSegment_;

public:
    void init(
            gfl::i32 const width,
            gfl::i32 const branch_factor,
            gfl::i32 const depth,
            gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;


        nodes_ = VectorView<Node>(width * branch_factor * depth,alloc),
        segments_ = VectorView<ArrayView<Node>>(depth, alloc);
    }

    GFL_HOST_DEVICE
    static
    gfl::i32 data_mem_size(gfl::i32 const maxWidth, gfl::i32 const maxBranchFactor, gfl::i32 const maxDepth) noexcept
    {
        using namespace gfl;

        i64 memSize = 0;
        memSize += VectorView<Node>::dataMemSize(maxWidth * maxBranchFactor * maxDepth) + DefaultAlign + // nodes_
        memSize += VectorView<i32>::dataMemSize(maxDepth) + DefaultAlign; // offsets_
        return memSize;
    }

    GFL_HOST_DEVICE
    void clear() noexcept { nodes_.clear(); segments_.clear(); }

    GFL_HOST_DEVICE
   gfl::ArrayView<Node> nodes() const noexcept {return nodes_;}
   gfl::ArrayView<Node> const * nodesPtr() const noexcept {return &nodes_;}

    GFL_HOST_DEVICE
    gfl::ArrayView<gfl::ArrayView<Node>> segments() const noexcept {return segments_;}

    GFL_HOST_DEVICE
    void addSegment(gfl::i32 const size) noexcept
    {
        using namespace gfl;

        i32 const oldSize = nodes_.resizeBy(size);
        lastSegment_ = nodes_.slice(oldSize, nodes_.size());
        segments_.pushBack(lastSegment_);
    }

    gfl::ArrayView<Node> const * lastSegmentPtr() const noexcept { return &lastSegment_;}
};