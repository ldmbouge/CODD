#pragma once

#include "ModelSpecs.hpp"

#include <GFL.hpp>

#include <iostream>

#include "KnapsackData.hpp"

template<gfl::i32 BranchFactor, gfl::i32 Items>
class Knapsack : public KnapsackData
{
    using KnapsackData::items;
    using KnapsackData::capacity;
    using KnapsackData::profits;
    using KnapsackData::weights;

public:

    constexpr static bool is_maximization = true;
    using OutLabels = gfl::BitSet<gfl::BitSet<>::num_words(BranchFactor)>;

    class State
    {
        gfl::i32 n;          // variable index
        gfl::i32 c;          // remaining capacity

    public:
        GFL_HOST_DEVICE
        State() = default;

        GFL_HOST_DEVICE
        State(gfl::i32 const n, gfl::i32 const c) : n(n), c(c) {}

        GFL_HOST_DEVICE static
        bool equal(State const & s1, State const & s2) noexcept
        {
            return
                s1.n == s2.n and
                s1.c == s2.c;
        }

        GFL_HOST_DEVICE
        static
        gfl::u64 hash(State const & s) noexcept
        {
            using namespace gfl;
            u64 seed = 0;
            hashCombine(seed, s.n);
            hashCombine(seed, s.c);
            return seed;
        }

        GFL_HOST_DEVICE
        void print() const
        {
            printf("N %d | C %d", n, c);
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return os << "<" <<
                   "N=" << s.n << "," <<
                   "C=" << s.c << ">";
        }

        friend  class Knapsack;
    };

    State initial() const noexcept
    {
        return State(0, capacity);
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return State(items,0);
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        return s.n == items;
    }

    GFL_HOST_DEVICE
    OutLabels lgf(State const & s, double pBound, double dBound, DDContext ddCtx) const noexcept
    {
        using namespace gfl;
        if (s.n < items)
        {
            return OutLabels(0,s.c >= weights[s.n]);
        }
        else
        {
            return OutLabels();
        }
    }

    GFL_HOST_DEVICE
     gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        using namespace gfl;

        if (s.n < items)
        {
            return State(s.n+1,s.c - l * weights[s.n]);
        }
        else
            return nullopt;
    }

    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        assert(l == 0 or l == 1);
        return profits[s.n] * l;
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE static
    State smf(State const & s1, State const & s2) noexcept
    {
        using namespace gfl;
        assert(s1.n == s2.n);
        return State(s1.n,max<i32>(s1.c,s2.c));
    }

    constexpr static bool has_heur = true;
    GFL_HOST_DEVICE
    gfl::f64 h(State const & s, HContext hCtx) const noexcept
    {
        using namespace gfl;
        double nn = 0;
        int    rc = s.c;
        for(auto i= s.n; i < items; ++i)
        {
            if (weights[i] <= rc) {
                rc -= weights[i];
                nn += profits[i];
            } else {
                nn += (rc / weights[i]) * profits[i];
                break;
            }
        }
        return nn;
    }

    constexpr static bool has_dom = false;
};
