#pragma once
#include <vector>

#include "BatchInfo.cuh"
#include "Types.hpp"

template<typename Node>
struct LayersHelper
{
    using LayerType = std::vector<Node>;

    std::vector<LayerType>  layers;
    std::vector<LabelsInfo> labelsInfo;

    LayersHelper() noexcept {};

    void clear() noexcept
    {
        bool allEmpty = true;
        for (auto & l : layers)
        {
           l.clear();
        }

        for (auto & li : labelsInfo)
        {
           li.reset();
        }
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

    gfl::i32 calcDeepestNotEmpty() const noexcept
    {
        using namespace gfl;
        for (i32 i = layers.size() - 1; i >= 0; i -= 1)
        {
            if (not layers[i].empty())
            {
                return i;
            }
        }
        return -1;
    }

    LayerType & getLayer(gfl::i32 const lIdx) noexcept
    {
        if (layers.size() <= lIdx)
        {
            layers.resize(lIdx + 1);
        }
        return layers[lIdx];
    }

    LabelsInfo & getLabelsInfo(gfl::i32 const lIdx) noexcept
    {
        if (labelsInfo.size() <= lIdx)
        {
            labelsInfo.resize(lIdx + 1);
        }
        return labelsInfo[lIdx];
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
};
