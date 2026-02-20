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

    static constexpr gfl::i32 DefaultLayerSize  = 4096;
    static constexpr gfl::i32 DefaultLayerCount = 1024;
    using Layer = std::vector<Node>;

    std::vector<Layer> layers{DefaultLayerCount};

    Layer & layer(gfl::i32 const lIdx) noexcept
    {
        using namespace gfl;
        if (lIdx >= layers.size())
        {
           //i32 const oldSize = layers.size();
            layers.resize(lIdx + 1);
            // ArrayView<Layer> newLayers = layers.slice(oldSize, layers.size());
            // for (auto & l : newLayers) { new (&l) Layer(DefaultLayerSize); }
        }
        return layers[lIdx];
    }

public:

    void onPush(PushListener l) { pushListeners.emplace_back(std::move(l)); }
    void onPull(PullListener l) { pullListeners.emplace_back(std::move(l)); }

    void notifyPush(gfl::i32 const n)
    { for (auto const & l : pushListeners) { l(n); } }

    void notifyPull(gfl::i32 const n)
    { for (auto const & l : pullListeners) { l(n); } }

    void push(Node const & node) noexcept
    {
        using namespace gfl;
        i32 const depth = node.depth();
        layer(depth);
        layers[depth].push_back(node);
        notifyPush(1);
    }

    void push(gfl::ArrayView<Node> const & nodes) noexcept
    {
        using namespace gfl;

        if (nodes.size() > 0)
        {
            i32 const depth = nodes[0].depth();
            auto const sameDepth = [depth](Node const & n) { return n.depth() == depth; };
            auto const revDual   = [](Node const & n1, Node const & n2)
                { return isWorse<Model>(n1.f(), n2.f()); };

            assert(std::all_of(nodes.begin(), nodes.end(), sameDepth));
            assert(std::is_sorted(nodes.begin(), nodes.end(), revDual));

            layer(depth);
            Layer & l         = layers[depth];
            i32 const oldSize =l.size();
                l.resize( l.size() + nodes.size());
            std::memcpy(&l[oldSize], nodes.data(), nodes.dataMemSize());
            std::inplace_merge(l.begin(), l.begin() + oldSize, l.end(), revDual);
            notifyPush(nodes.size());
        }
    }

    void push(gfl::ArrayView<gfl::ArrayView<Node>> const & cutset)
    {
        for (auto const & segment : cutset) { push(segment); }
    }

    bool empty() const noexcept
    {
        for (gfl::i32 i = 0; i <= layers.size(); ++i)
            if (not layers[i].empty()) return false;
        return true;
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

        f64  best = worst<Model>();
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
        layers[lIdx].pop_back();
        notifyPull(1);
        return bestNode;
    }
};
