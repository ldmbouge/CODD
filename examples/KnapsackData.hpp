#pragma once

#include <iostream>
#include <fstream>

#include <GFL.hpp>

class KnapsackData
{
    protected:
    gfl::i32 items;
    gfl::i32 capacity;
    gfl::ArrayView<gfl::i32> weights;
    gfl::ArrayView<gfl::i32> profits;

public:
    inline
    void init(std::string const& instancePath, gfl::ArenaAllocator& alloc)
    {
        using namespace gfl;

        std::ifstream file(instancePath);
        assert(file.is_open());

        file >> items;
        file >> capacity;
        profits = ArrayView<i32>(items, alloc);
        weights = ArrayView<i32>(items, alloc);
        for (i32 i = 0; i < items; ++i)
        {
            file >> profits[i];
            file >> weights[i];
        }
        auto const toKey = [](i32 const & p, i32 const & w) { return -scast<f32>(p) / scast<f32>(w);};
        sortByKeyFn(toKey, profits, weights);
    }
};