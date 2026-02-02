#pragma once
#include <vector>

#include "BatchInfo.cuh"
#include "BoundsHelpers.cuh"
#include "Types.hpp"

template<typename Node, typename Model>
struct LayersHelper
{
    using LayerType = std::vector<Node>;

    std::vector<LayerType>  layers;
    std::vector<LabelsInfo> labelsInfo;
    std::vector<double>     bounds;

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

    gfl::i32 calcDeepestMostPromising() const noexcept
    {
        using namespace gfl;
        double bound = worstValue<Model>();
        i32 lIdx = -1;
        for (i32 i = layers.size() - 1; i >= 0; i -= 1)
        {
            if (isBetter<Model>(bounds[i], bound))
            {
                bound = bounds[i];
                lIdx = i;
            }
        }
        if (lIdx < 0)
        {
            return calcDeepestNotEmpty();
        }
        assert(lIdx >= 0) ;


        // printf("Selecting layer %d with hBound %.1f\n", lIdx,bound);
        // fflush(stdout);

        return lIdx;
    }

    gfl::i32 calcShallowMostPromising() const noexcept
    {
        using namespace gfl;
        double bound = worstValue<Model>();
        i32 lIdx = -1;
        for (i32 i = layers.size()-1; i >= 0; i -= 1)
        {
            if (isBetterEq<Model>(bounds[i], bound))
            {
                bound = bounds[i];
                lIdx = i;
            }
        }
        return lIdx;
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
        if (bounds.empty() or bounds.size() <= lIdx)
        {
            bounds.resize(lIdx + 1, worstValue<Model>());
        }
        return bounds[lIdx];
    }

    void updateBound(gfl::i32 const lIdx) noexcept
    {
        double oldBound = getBound(lIdx);
        double tBound = worstValue<Model>();
        for (auto const & n : getLayer(lIdx))
        {
            double const nBound = n.fValue;
            tBound = calcBetter<Model>(tBound,nBound);
        }
        getBound(lIdx) = tBound;
        // printf("Layer %d bound: %.1f -> %.1f\n", lIdx,oldBound, tBound);
        // fflush(stdout);
    }

    double calcBestBound() const noexcept
    {
        double bound = worstValue<Model>();
        for (auto const & b : bounds)
        {
            bound = calcBetter<Model>(bound,b);
        }

        // gfl::Array<double>::print(bounds.data(), bounds.data() + bounds.size(), "%.1f");
        // printf("\n");

        return bound;
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
};
