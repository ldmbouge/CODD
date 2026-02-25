#pragma once

#include <algorithm>
#include <GFL.hpp>
#include "BoundsUtils.hpp"

template<typename Model, typename Node>
class Queue
{
    gfl::i64 pulled_{0};
    gfl::i64 pushed_{0};

    static constexpr gfl::i32 DefaultLayerSize  = 4096;
    static constexpr gfl::i32 DefaultLayerCount = 1024;
    using Layer = gfl::Vector<Node>;

    gfl::Vector<Layer> layers{DefaultLayerCount};

    Layer & layer(gfl::i32 const lIdx) noexcept
    {
        using namespace gfl;
        if (lIdx >= layers.size())
        {
            i32 const oldSize = layers.resizeTo(lIdx + 1);
            ArrayView<Layer> newLayers = layers.slice(oldSize, layers.size());
            for (auto & l : newLayers) { new (&l) Layer(DefaultLayerSize); }
        }
        return layers[lIdx];
    }

public:


    void push(Node const & node) noexcept
    {
        using namespace gfl;
        i32 const depth = node.depth();
        layer(depth);
        layers[depth].pushBack(node);
        pushed_++;
    }

    void push(gfl::ArrayView<Node> const & nodes) noexcept
    {
        using namespace gfl;

        if (nodes.size() > 0)
        {
            i32 const depth = nodes[0].depth();
            auto const sameDepth = [depth](Node const & n) { return n.depth() == depth; };
            auto const revDual   = [](Node const & n1, Node const & n2)
            { return
                isWorse<Model>(n1.f(), n2.f()) or
                n1.f() == n2.f() and isWorse<Model>(n1.g(), n2.g());
            };

            // printf("CUTSET (%d) SEG:\n", nodes.size());
            // for(auto const & n : nodes)
            // {
            //     n.print();
            //     printf("\n");
            // }
            // printf("\n");
            // fflush(stdout);
            assert(std::all_of(nodes.begin(), nodes.end(), sameDepth));
            //assert(std::is_sorted(nodes.begin(), nodes.end(), revDual));
            std::sort(nodes.begin(), nodes.end(), revDual);

            Layer & l = layer(depth);
            i32 const oldSize = l.resizeBy(nodes.size());
            std::memcpy(&l[oldSize], nodes.data(), nodes.dataMemSize());
            std::inplace_merge(l.begin(), l.begin() + oldSize, l.end(), revDual);

            pushed_ += nodes.size();
        }
    }

    void push(gfl::tuple<gfl::ArrayView<Node>, gfl::ArrayView<gfl::i32>> const & cutset)
    {
        using namespace gfl;

        auto [nodes, offsets] = cutset;
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

    gfl::f64 bestDual() const noexcept
    {
        using namespace gfl;
        f64 best = worst<Model>();
        for (i32 i = 0; i < layers.size(); ++i)
            if (not layers[i].empty())
                best = better<Model>(best, layers[i].back().f());
        return best;
    }

    Node pullBest() noexcept
    {
        using namespace gfl;

        f64 best = worst<Model>();
        i32 lIdx  = 0;
        for (i32 i = 0; i < layers.size(); ++i)
        {
            if (not layers[i].empty())
            {
                f64 const dual = layers[i].back().f();
                if (isBetter<Model>(dual, best))
                {
                    best = dual;
                    lIdx = i;
                }
            }
        }
        Node const bestNode = layers[lIdx].back();
        layers[lIdx].popBack();
        pulled_++;
        return bestNode;
    }
};
