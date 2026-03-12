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
    static constexpr  gfl::i32 DefaultSegmentSize = 1024 * 1024 * 1024; // 1GB
    static constexpr gfl::i32 InitialQueueSize = 256 * 1024;

    static constexpr auto betterF = [](Node const * const n1, Node const * const n2) noexcept {
        using namespace gfl;
        f64 const f1 =  n1->f();
        f64 const f2 =  n2->f();
        f64 const g1 =  n1->g();
        f64 const g2 =  n2->g();
        i32 const & d1 = n1->depth();
        i32 const & d2 = n2->depth();
        return isBetter<Model>(f1,f2) or
               (f1 == f2 and d1 > d2) or
               (f1 == f2 and d1 == d2 and isBetter<Model>(g1,g2)); // or
               //(f1 == f2 and d1 == d2 and Model::srf(n1->state()) < Model::srf(n2->state()));
    };

    static constexpr auto deepestG = [](Node const * const n1, Node const * const n2) noexcept {
        using namespace gfl;
        f64 const g1 =  n1->g();
        f64 const g2 =  n2->g();
        f64 const f1 =  n1->f();
        f64 const f2 =  n2->f();
        i32 const & d1 = n1->depth();
        i32 const & d2 = n2->depth();
        //return isBetter<Model>(f1,f2) or (f1 == f2 and d1 > d2);
        return  d1 > d2 or (d1 == d2 and isBetter<Model>(g1,g2));
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

    void push(Node const * const node, gfl::f64 const primal) noexcept
    {
        using namespace gfl;
        if (isBetterEq<Model>(node->f(),primal))
        {
            //printf("[DBG] Pushing node with f %.2f\n",node->f());
            heap_.insertHeap(node);
            pushed_++;
        }
    }

    void pushBatch(gfl::ArrayView<Node> const & nodes, gfl::f64 const f, gfl::f64 const primal) noexcept
    {
        using namespace gfl;
        for (auto & node : nodes)
        {
            node.h(worse<Model>(f - node.g(), node.h()));
            if (isBetterEq<Model>(node.f(),primal))
            {
                heap_.insert(&node);
                pushed_++;
            }
        }
        heap_.buildHeap();
    }

    void push(gfl::ArrayView<Node> const & nodes,  gfl::f64 const primal) noexcept
    {
        using namespace gfl;

        if (not nodes.empty())
        {
            for (auto const & n : nodes) push(&n, primal);
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

    Node const * peekBest() noexcept
    {
        using namespace gfl;

        assert(not empty());

        Node const * const bestNode = heap_.peekMax();
        return bestNode;
    }
};
