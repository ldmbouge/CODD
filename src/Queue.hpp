#pragma once

#include <algorithm>
#include <GFL.hpp>
#include "BoundsUtils.hpp"
#include "heap.hpp"
#include "store.hpp"

template<typename Model, typename Node>
class Queue
{
protected:
    static constexpr  gfl::i32 DefaultSegmentSize = 16 * 1024 * 1024; // 16MB
    static constexpr gfl::i32 InitialQueueSize = 256 * 1024;

    static constexpr auto betterF = [](Node const * const n1, Node const * const n2) noexcept {
        return isBetter<Model>(n1->f(), n2->f());
    };

    Pool memPool_;
    Heap<Node const *,decltype(betterF)> heap_;
    gfl::i64 pulled_;
    gfl::i64 pushed_;

public:

    Queue() noexcept:
        memPool_(DefaultSegmentSize),
        heap_(&memPool_, InitialQueueSize, betterF),
        pulled_(0),
        pushed_(0)
    {}

    void push(Node const * const node) noexcept
    {
        using namespace gfl;
        heap_.insertHeap(node);
        pushed_++;
    }

    void push(gfl::ArrayView<Node> const & nodes) noexcept
    {
        using namespace gfl;

        if (not nodes.empty())
        {
            for (auto const & n : nodes) push(&n);
        }
    }

    void push(gfl::tuple<gfl::ArrayView<Node>, gfl::ArrayView<gfl::i32>> const & cutset)
    {
        using namespace gfl;

        auto [nodesTmp, offsets] = cutset;
        ArrayView<Node> nodes(nodesTmp.size(), new (&memPool_) Node[nodesTmp.size()]);
        std::memcpy(nodes.data(), nodesTmp.data(), nodesTmp.dataMemSize());
        for (i32 i = 0; i < offsets.size(); ++i)
        {
            i32 const begin = offsets.at(i);
            i32 const end   = (i + 1 < offsets.size()) ? offsets.at(i + 1) : nodes.size();
            i32 const size = end - begin;
            ArrayView slice(size, nodes.data() + begin);
            push(slice);
        }
    }

    gfl::i64 pushed() const noexcept {return pushed_;}

    gfl::i64 pulled() const noexcept {return pulled_;}

    gfl::i64 size() const noexcept {return pushed_ - pulled_;}

    bool empty() const noexcept
    {
        return pulled_ == pushed_;
    }

    Node const * pullBest() noexcept
    {
        using namespace gfl;

        assert(not empty());

        Node const * const bestNode = heap_.extractMax();
        pulled_++;
        return bestNode;
    }
};
