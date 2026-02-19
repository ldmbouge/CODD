#pragma once
#include <vector>

#include "ExpansionInfo.cuh"
#include "BoundsHelpers.cuh"
#include "Types.hpp"

template<typename Node, typename Model>
struct LayersHelper
{
    using LayerType = std::vector<Node>;

    std::vector<LayerType>  layers;
    std::vector<LabelsInfo> labelsInfo;
    std::vector<double>     dBounds;

    LayersHelper() noexcept {};

    void clear() noexcept
    {
        for (auto & l : layers)
        {
           l.clear();
        }

        for (auto & li : labelsInfo)
        {
           li.reset();
        }

        for (auto & b : dBounds)
        {
           b = bestValue<Model>();
        }
    }

    LayerType & getLayer(gfl::i32 const lIdx) noexcept
    {
        if (layers.empty() or layers.size() <= lIdx)
        {
            layers.resize(lIdx + 1);
        }
        return layers[lIdx];
    }

    LabelsInfo & getLabelsInfo(gfl::i32 const lIdx) noexcept
    {
        if (labelsInfo.empty() or labelsInfo.size() <= lIdx)
        {
            labelsInfo.resize(lIdx + 1);
        }
        return labelsInfo[lIdx];
    }

    double & getBound(gfl::i32 const lIdx) noexcept
    {
        if (dBounds.empty() or dBounds.size() <= lIdx)
        {
            dBounds.resize(lIdx + 1, worstValue<Model>());
        }
        return dBounds[lIdx];
    }

    gfl::i32 calcDeepestNotEmpty() const noexcept
    {
        using namespace gfl;

        i32 lIdx = -1;
        for (i32 i = 0; i < layers.size(); i += 1)
        {
            if (not layers[i].empty())
            {
                lIdx = i;
            }
        }
        return lIdx;
    }

    gfl::i32 calcDeepestMostPromising() const noexcept
    {
        using namespace gfl;

        double bound = worstValue<Model>();
        i32 lIdx = -1;
        for (i32 i = 0; i < layers.size(); i += 1)
        {
            if (isBetterEq<Model>(dBounds[i], bound))
            {
                bound = dBounds[i];
                lIdx = i;
            }
        }

        // Unbounded root
        if (lIdx < 0)
        {
            return calcDeepestNotEmpty();
        }

        assert(lIdx >= 0);
        return lIdx;
    }

    bool allLayersEmpty() const noexcept
    {
        bool allEmpty = true;
        for (auto const & l : layers)
        {
            allEmpty = allEmpty and l.empty();
        }

        return allEmpty;
    }

    gfl::i64 countAllNodes() const noexcept
    {
        using namespace gfl;
        i64 nNodes = 0;
        for (auto const & l : layers)
        {
            nNodes += l.size();
        }
        return nNodes;
    }

    void forEachNode(std::function<void(Node)> const & f)
    {
        for (auto const & l : layers)
        {
           for (auto const & n : l)
           {
               f(n);
           }
        }
    }

    void addToLayer(gfl::i32 const lIdx, std::span<Node> const & nodes, LabelsInfo const & li) noexcept
    {
        using namespace gfl;

        // Append nodes
        auto & layer = getLayer(lIdx);
        i64 const layerOldSize = layer.size();
        layer.resize(layerOldSize + nodes.size());
        std::span<Node> const oldLayer(layer.data(), layerOldSize);
        std::span<Node> const extension(layer.data() + layerOldSize, nodes.size());
        memcpy(extension.data(),nodes.data(),sizeof(Node) * nodes.size());

        // Update label info
        getLabelsInfo(lIdx).update(li);

        // Keep reverse sorted since we pop from the tail
        std::sort(extension.begin(), extension.end(), isWorst<Model>);
        std::inplace_merge(layer.data(), layer.data() + layerOldSize, layer.data() + layer.size(), isWorst<Model>);

        // Update dual bound
        getBound(lIdx) = layer.back().fValue;
    }

    void addToLayer(gfl::i32 const lIdx, Node const & node) noexcept
    {
        std::span<Node> const nodes(&node,1);
        LabelsInfo const li(node.slc());
        addToLayer(lIdx, nodes, li);
    }

    void removeSuffixFromLayer(gfl::i32 const lIdx, gfl::i32 const size) noexcept
    {
        auto & layer = getLayer(lIdx);
        layer.resize(layer.size() - size);

        // No need to update label info

        // Update dual bound
        getBound(lIdx) = layer.back().fValue;
    }
};
