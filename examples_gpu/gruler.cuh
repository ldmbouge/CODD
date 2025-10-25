#pragma once

#include "model.hpp"
#include "gruler_base.hpp"
#include "util.hpp"
#include <Common.hpp>
#include <Backend.hpp>

#include <iostream>
#include <cuda_runtime.h>

template<int N = 256>
struct GRuler : GRulerBase
{
    // Model
    using Set = NatSet<gfl::roundUpDivPosInt<unsigned short>(N, 64)>;
    using Labels = Set;

    struct State
    {
        Set m; // set of marks
        Set d; // set of distances
        int k;   // number of marks made
        int e;   // last mark
        int sm;  // smallest unused distance

        GFL_HOST_DEVICE
        static bool equal(State const & s1, State const & s2) noexcept
        {
            return
                s1.k == s2.k and
                s1.e == s2.e and
                s1.m == s2.m;
        }

        GFL_HOST_DEVICE
        static std::size_t hash(State const & s) noexcept
        {
            std::size_t seed = 0;
            hash_combine(seed, s.m.hash());
            hash_combine(seed, s.d.hash());
            hash_combine(seed, static_cast<std::size_t>(s.k));
            hash_combine(seed, static_cast<std::size_t>(s.e));
            hash_combine(seed, static_cast<std::size_t>(s.sm));
            return seed;
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return os << "<" <<
                   "MRKS=" << s.m << "," <<
                   "DIST=" << s.d << "," << ">";
        }
    };

    State initial() const noexcept
    {
        return {Set{0},Set{},1,0,1};
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return State{Set{},Set{},n,0,L+1};  // smallest unused distance should be set explicitly to L+1
    }

    bool isTarget(State const & s) const noexcept
    {
        return s.k == n;
    }

    constexpr static bool has_simple_lgf = false;
    GFL_HOST_DEVICE
    Labels lgf(State const & s, DDContext ctx, double pBound, double dBound) const noexcept
    {
        using namespace gfl;

        int ub = L+1;
        if (s.k < n/2)
            ub = gfl::min<int>(pBound,L+1)/2 - OPT[(n/2)-s.k];
        else
            ub = gfl::min<int>(pBound-1,L+1) - OPT[n-s.k];
        auto lb = s.e+s.sm;
        if (s.k<n-1)
            lb = gfl::max<int>(lb,roundUpDivPosInt<int>(s.k*(s.k-1),2),OPT[s.k+1]);
        else
            lb = gfl::max<int>(lb,roundUpDivPosInt<int>(s.k*(s.k-1),2),OPT[s.k]+1);
        Set vr;
        for(int label = lb;label <= ub;label++)
        {
            Set leg(s.d);
            bool legal = (leg.interWith(label - s.m).empty());
            if (legal)
            {
                vr.insert(label);
            }
        }
        return vr;
    }

    GFL_HOST_DEVICE
     gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        if (s.k == n-1)  // this moves goes to the sink
        {
            return State{Set{}, Set{}, n, 0, L + 1};
        }
        else
        {
            Set d_new = (l - s.m) | s.d;
            int smallest_dist = s.sm;
            while (d_new.contains(smallest_dist)) smallest_dist += 1;
            State rv { s.m | Set{l}, d_new, s.k + 1, l, smallest_dist };
            return rv;
        }
    }

    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return l - s.e;
    }

    GFL_HOST_DEVICE constexpr static
    bool better(double const & c1, double const & c2) noexcept { return c1 < c2; }
    GFL_HOST_DEVICE
    constexpr static bool betterEq(double const & c1, double const & c2)  noexcept { return c1 <= c2; }
    constexpr static double bestValue() noexcept { return std::numeric_limits<double>::lowest(); }
    constexpr static double worstValue() noexcept { return std::numeric_limits<double>::max(); } // lowest() instead of min() because double

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE
    gfl::optional<State> smf(State const & s1, State const & s2) const noexcept
    {
        if (s1.k == s2.k and s1.e == s2.e)
        {
            return GRuler{s1.m & s2.m, s1.d & s2.d, s1.k, min<int>(s1.e,s2.e), min<int>(s1.sm,s2.sm)};
        }
        else
        {
            return std::nullopt; // return  the empty optional
        }
    }

    constexpr static bool has_local = false;
    constexpr static bool has_dom = false;
};


static_assert(IsModel<GRuler<>>);