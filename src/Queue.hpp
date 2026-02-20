#pragma once

#include <algorithm>

#include <GFL.hpp>

#include "BoundsUtils.hpp"
#include "Contexts.hpp"
#include "CutsetData.hpp"

template<typename Model, typename Node>
class Queue
{
    using PushListener = std::function<void(gfl::i32)>;
    using PullListener = std::function<void(gfl::i32)>;
    std::vector<PushListener> pushListeners;
    std::vector<PullListener> pullListeners;

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


    void onPush(PushListener l) { pushListeners.emplace_back(std::move(l));}
    void onPull(PullListener l) { pullListeners.emplace_back(std::move(l));}

    void
    notifyPush(gfl::i32 const n)
    { for(auto const & l : pushListeners) { l(n); }}

    void
    notifyPull(gfl::i32 const n)
    { for(auto const & l : pullListeners) { l(n); }}

    void push(Node const & node) noexcept
    {
        using namespace gfl;

        i32 const depth = node.depth();
        layer(depth).pushBack(node);
        notifyPush(1);
    }

    void push(gfl::ArrayView<Node> const & nodes) noexcept
    {
        using namespace gfl;

        if (nodes.size() > 0)
        {
            i32 const depth = nodes[0].depth();
            auto const sameDepth = [depth](Node const & n) { return n.depth() == depth; };
            auto const revDual = [](Node const & n1, Node const & n2)
                { return isWorse<Model>(n1.f(),n2.f()); };

            assert(std::all_of(nodes.begin(), nodes.end(), sameDepth));
            assert(std::is_sorted(nodes.begin(), nodes.end(), revDual));

            Layer & l = layer(depth);
            i32 const oldSize = l.resizeBy(nodes.size());
            std::memcpy(&l[oldSize], nodes.data(), nodes.dataMemSize());
            std::inplace_merge(l.begin(), &l[oldSize], l.end(), revDual);
            notifyPush(nodes.size());
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
        for (auto const & l : layers)
        {
            if (not l.empty())
            {
                Node const & node = l.back();
                bestDual = better<Model>(bestDual, node.f());
            }
        }
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
            if (not layers[i].empty())
            {
                f64 const dual = layers[i].back().f();
                if (isBetter<Model>(dual, bestDual))
                {
                    bestDual = dual;
                    lIdx = i;
                }
            }
        }
        Node const & bestNode = layers[lIdx].popBack();
        notifyPull(1);

        // printf("EXTRACTED:\n");
        // Node::print(bestNode);
        // printf("\n");
        return bestNode;
    }
};
