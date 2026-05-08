#pragma once

#include <fstream>
#include <iostream>

#include <GFL.hpp>

template<gfl::i32 Cities>
class TsptwData
{
public:
    struct TimeWindow
    {
        gfl::i32  b, e;

        TimeWindow(gfl::i32 const b, gfl::i32 const e) noexcept : b(b), e(e) {}

        TimeWindow() noexcept :
            b(gfl::numeric_limits<gfl::i32>::max()),
            e(gfl::numeric_limits<gfl::i32>::min())
        {}
    };

protected:
    constexpr static int Depot = 0;
    gfl::i32 n{0};
    gfl::ArrayView<gfl::ArrayView<gfl::i32>> d;
    gfl::ArrayView<TimeWindow> tw;
    gfl::ArrayView<gfl::i32> dInNS;
    gfl::ArrayView<gfl::i32> dIn;
    gfl::ArrayView<gfl::i32> dOut;
    gfl::ArrayView<gfl::i32> permIn;
    gfl::ArrayView<gfl::i32> permOut;

public:
    void init(std::string const& instancePath, gfl::ArenaAllocator& alloc)
    {
        using namespace gfl;

        std::ifstream file(instancePath);
        assert(file.is_open());

        file >> n;
        d = ArrayView<ArrayView<i32>>(n,alloc);
        for (auto i = 0; i < n; i++)
        {
            d[i] = ArrayView<i32>(n,alloc);
            for (auto j = 0; j < n; j++)
            {
                file >> d[i][j];
            }
        }
        tw = ArrayView<TimeWindow>(n,alloc);
        for (auto i = 0; i < n; i++)
        {
            i32 b, e;
            file >> b >> e;
            tw[i] = TimeWindow(b, e);
        }
        file.close();

        dInNS = ArrayView<i32>(n,alloc);
        dIn = ArrayView<i32>(n,alloc);
        dOut = ArrayView<i32>(n,alloc);
        permIn = ArrayView<i32>(n,alloc);
        permOut = ArrayView<i32>(n,alloc);

        auto allCities = gfl::BitSet<BitSet<>::num_words(Cities)>(0,n-1);
        auto [firstCity,lastCity,_] = allCities.summary();
        for (i32 j = firstCity; j <= lastCity; j++)
        {
            if (allCities.contains(j))
            {
                auto [e1, minIn] = argmin(allCities - j, [j,this](i32 const k) { return d[k][j]; });
                auto [e2, minOut] = argmin(allCities - j, [j,this](i32 const k) { return d[j][k]; });
                dIn[j] = minIn;
                dInNS[j] = minIn;
                dOut[j] = minOut;
                permIn[j] = j;
                permOut[j] = j;
            }
        }

        auto const cmp =  [](double const a, double const b) { return a < b; };
        sortByKeyFn(cmp, dOut, permOut);
        sortByKeyFn(cmp, dIn, permIn);
    }
};