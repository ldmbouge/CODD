#pragma once

#include <algorithm>

#include <GFL.hpp>

#include "BoundsUtils.hpp"
#include "Contexts.hpp"
#include "CutsetData.hpp"

template<typename Model, typename Node>
class Queue
{
    static constexpr gfl::i32 DefaultLayerSize = 4096;
    using Layer = gfl::Vector<Node>;

    gfl::Vector<Layer> layers{DefaultLayerSize};

    Layer & layer(gfl::i32 const lIdx) noexcept
    {
        using namespace gfl;
        if (lIdx >= layers.size())
        {
            i32 const oldSize = layers.resizeTo(lIdx+1);
            ArrayView<Layer> newLayers = layers.slice(oldSize, layers.size());
            for (auto & l : newLayers) { new (&l) Layer(DefaultLayerSize); }
        }
        return layers[lIdx];
    }

public:
    Queue() noexcept = default;

    void push(Node const & node) noexcept
    {
        using namespace gfl;

        i32 const depth = node.depth();
        layer(depth).pushBack(node);
    }

    void push(gfl::ArrayView<Node> const & nodes) noexcept
    {
        using namespace gfl;

        if (nodes.size() > 0)
        {
            i32 const depth = nodes[0].depth();
            auto const same_depth = [depth](Node const & n) { return n.depth() == depth; };
            auto const revDual = [](Node const & n1, Node const & n2)
                { return isWorse<Model>(n1.dual(),n2.dual()); };
            assert(std::all_of(nodes.begin(), nodes.end(), same_depth));
            assert(std::is_sorted(nodes.begin(), nodes.end(), revDual));

            Layer & l = layer(depth);
            i32 const old_size = layers.size();
            l.resizeBy(nodes.size());
            std::memcpy(&layers[old_size], nodes.data(), nodes.dataMemSize());
            std::inplace_merge(l.begin(), &l[old_size], l.end(), revDual);
        }
    }

    void push(gfl::ArrayView<gfl::ArrayView<Node>> const & cutset)
    {
        using namespace gfl;
        for (auto const & segment : cutset) { push(segment); }
    }

    bool empty() const noexcept
    {
        bool empty = true;
        for (auto const & l : layers){empty = empty and l.empty();}
        return empty;
    }

    gfl::f64 bestDual() const noexcept
    {
        // TODO Rework as heap.

        using namespace gfl;

        f64 bestDual = worst<Model>();
        for (auto const & l : layers){ better<Model>( l.back().dual(), bestDual); }
        return bestDual;
    }

    Node const & pullBest() noexcept
    {
        // TODO Rework as heap.

        using namespace gfl;

        f64 bestDual = worst<Model>();
        i32 lIdx = 0;
        for (i32 i = 0; i < layers.size(); ++i)
        {
            f64 const dual = layers[i].back().dual();
            if (isBetter<Model>(dual, bestDual))
            {
                bestDual = dual;
                lIdx = i;
            }
        }
        Node const & bestNode = layers[lIdx].popBack();
        return bestNode;
    }
};
