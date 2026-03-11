#pragma once

#include <GFL.hpp>

template<typename Node>
class CutsetData
{
    gfl::VectorView<Node> nodes_;
    gfl::i64 markOffset_ = 0;  // offset into nodes_ where last segment starts
    gfl::i64 markSize_   = 0;  // size of last segment
    std::vector<gfl::ArrayView<Node>> fragments_;
    bool toSave_;
    bool saved_;
    Pool memPool;

    /*
     * CUTSET SIZE BOUNDS FOR LIMITED-WIDTH TREES
     *
     * A cutset collects the parents of merged nodes when a layer exceeds width w.
     * The cutset forms an antichain: no node in the cutset is an ancestor of another,
     * because once a node is in the cutset all its descendants are blocked from it.
     *
     * Merging only starts at layer l* = ceil(log_bf(w)), before that the tree
     * has fewer than w nodes and no merging occurs.
     *
     * FIXED BRANCHING FACTOR bf:
     *   At each layer, w*bf children are merged down to w, requiring w*(bf-1) merges.
     *   Lex merging is optimal: each cutset node contributes all bf children as merges,
     *   leaving no surviving children to waste future slots.
     *   This gives exactly w*(bf-1)/bf cutset nodes per layer, over (d - l*) layers:
     *
     *       cutset <= w * (d - l*) * (bf-1)/bf
     *
     * DECREASING BRANCHING FACTOR bf, bf-1, ..., 1:
     *   Same argument applies layer by layer, but branching factor at layer l is (bf-l).
     *   The first l* layers have no merging, so we skip them in the sum.
     *   Summing contributions: w * sum_{k=2}^{bf-l*} (k-1)/k
     *   = w * (bf - l* - H_{bf-l*}) where H_n = sum_{k=1}^{n} 1/k.
     *   Note: depth d = bf is fixed by the branching structure, not a free parameter.
     *
     *       cutset <= w * (bf - l* - H_{bf-l*})
     */

    // l* = ceil(log_bf(w)): first layer where merging starts
    gfl::i32 firstMergeLayer(gfl::f64 const w, gfl::f64 const bf)
    {
        using namespace gfl;
        return scast<i32>(std::floor(std::log(w) / std::log(bf))) + 1;
    }

    gfl::f64 harmonic(gfl::i64 const n)
    {
        using namespace gfl;
        f64 h = 0.0;
        for (i32 k = 1; k <= n; k++)
        {
            h += 1.0 / scast<f64>(k);
        }
        return h;
    }

    gfl::i64 cutsetFixed(gfl::i32 const w, gfl::i32 const d, gfl::i32 const bf)
    {
        using namespace gfl;
        i64 const lstar = firstMergeLayer(w, bf);
        i64 const effective_depth = max<i64>(0, d - lstar);
        return ceil<i64>(w * effective_depth * (bf - 1), bf);
    }

    gfl::i64 cutsetDecreasing(gfl::i32 const w, gfl::i32 const bf, gfl::i32 const depth)
    {
        using namespace gfl;
        i32 const lstar = firstMergeLayer(w, bf);
        i32 const effective_bf = bf - lstar;
        if (effective_bf <= 1)
        {
            //Never reaches width w
            return 0;
        }
        return scast<i64>(ceil(w * (effective_bf - harmonic(effective_bf))));
    }

public:
    void init(gfl::i32 const width, gfl::i32 const branchFactor, gfl::i32 const depth, gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;
        nodes_ = VectorView<Node>(width * 4, alloc);
    }

    void clear() noexcept
    {
        nodes_.clear();
        fragments_.clear();
        toSave_ = false;
        saved_ = false;
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
    std::vector<gfl::ArrayView<Node>>  const & fragments() const noexcept { return fragments_; }

    GFL_HOST_DEVICE
    gfl::ArrayView<Node> const * nodesPtr() const noexcept { return &nodes_; }

    GFL_HOST_DEVICE
    bool saved() const noexcept { return saved_; }

    GFL_HOST_DEVICE
    void saved(bool value) noexcept { saved_ = value; }

    GFL_HOST_DEVICE
    bool toSave() const noexcept { return toSave_; }

    GFL_HOST_DEVICE
    void toSave(bool value) noexcept { toSave_ = value; }

    void saveFragmentFromGpu()
    {
        Node * const fragment = new (&memPool) Node[nodes_.size()];
        CHECK_CUDA_ERROR(cudaMemcpy(
            fragment,
            nodes_.data(),
            nodes_.dataMemSize(),
            cudaMemcpyDeviceToHost));
        fragments_.push_back(gfl::ArrayView<Node>(nodes_.size(), fragment));
        nodes_.clear();
        markOffset_ = 0;
        markSize_   = 0;
    }

};