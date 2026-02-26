#pragma once

#include <GFL.hpp>


template <typename Node>
class CutsetData
{

    gfl::VectorView<Node> nodes_{};
    gfl::VectorView<gfl::i32> offsets_{};
    gfl::ArrayView<Node> lastSegment_;

public:
    void init(
            gfl::i32 const width,
            gfl::i32 const branch_factor,
            gfl::i32 const depth,
            gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;
        nodes_   = VectorView<Node>(width * branch_factor * depth, alloc);
        //nodes_   = VectorView<Node>(branch_factor * depth * (2 * width - depth + 1) / 2, alloc);

        offsets_ = VectorView<i32>(depth, alloc);
    }

    GFL_HOST_DEVICE
    static
    gfl::i64 dataMemSize(gfl::i32 const maxWidth, gfl::i32 const maxBranchFactor, gfl::i32 const maxDepth) noexcept
    {
        using namespace gfl;
        i64 memSize = 0;
        memSize += VectorView<Node>::dataMemSize(maxWidth * maxBranchFactor * maxDepth) + DefaultAlign;
        memSize += VectorView<i32>::dataMemSize(maxDepth) + DefaultAlign;
        return memSize;
    }

    GFL_HOST_DEVICE
    void clear() noexcept { nodes_.clear(); offsets_.clear(); }

    GFL_HOST_DEVICE
   gfl::ArrayView<Node> nodes() const noexcept {return nodes_;}
   gfl::ArrayView<Node> const * nodesPtr() const noexcept {return &nodes_;}

    GFL_HOST_DEVICE
    gfl::ArrayView<gfl::i32> offsets() const noexcept {return offsets_;}

    GFL_HOST_DEVICE
    gfl::i32 numSegments() const noexcept { return offsets_.size(); }

    GFL_HOST_DEVICE
    static
    gfl::ArrayView<Node> segment(
        gfl::ArrayView<gfl::i32> const offsets,
        gfl::ArrayView<Node> const nodes,
        gfl::i32 const i) noexcept
    {
        using namespace gfl;
        i32 const begin = offsets.at(i);
        i32 const end   = (i + 1 < offsets.size()) ? offsets.at(i + 1) : nodes.size();
        return nodes.slice(begin, end);
    }

    GFL_HOST_DEVICE
     void addSegment(gfl::i32 const size) noexcept
    {
        using namespace gfl;

        assert(nodes_.size() + size <= nodes_.capacity());
        assert(offsets_.size() < offsets_.capacity());

        i32 const begin = nodes_.resizeBy(size);
        offsets_.pushBack(&begin);
        lastSegment_ = nodes_.slice(begin, nodes_.size());
    }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> const * lastSegmentPtr() const noexcept { return &lastSegment_; }
};
