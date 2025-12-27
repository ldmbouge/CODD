#pragma once

#include "model.hpp"
#include "misp_base.hpp"
#include "util.hpp"
#include <Common.hpp>
#include <Backend.hpp>

#include <iostream>

template<int N = 256>
struct Misp : MispBase<N>
{
    constexpr static bool is_maximization = true;

    // Model
    using Set = NatSet<gfl::roundUpDivPosInt<unsigned short>(N, 64)>;
    using Labels = Set;

    struct State
    {
        Set sel;
        int   n;

        GFL_HOST_DEVICE
        static bool equal(State const & s1, State const & s2) noexcept
        {
            return s1.n == s2.n and s1.sel == s2.sel;
        }

        GFL_HOST_DEVICE
        static std::size_t hash(State const & s) noexcept
        {
            std::size_t seed = 0;
            hash_combine(seed, s.sel.hash());
            hash_combine(seed, static_cast<std::size_t>(s.n));
            return seed;
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return os << "<SEL=" << s.sel << ">";
        }
    };

    State initial() const noexcept
    {
        return State{Set(0,MispBase<N>::nNodes-1),0};
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return State{Set(),MispBase<N>::nNodes};
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        return s.n >= MispBase<N>::nNodes;
    }

    GFL_HOST_DEVICE
    Labels lgf(State const & s, DDContext ctx, double pBound, double dBound) const noexcept
    {
        return Set(0,1);
    }

    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        if (s.n >= MispBase<N>::nNodes)
        {
            return gfl::nullopt;
        }
        else
        {
            if (l and not s.sel.contains(s.n))
                return gfl::nullopt; // we cannot take n (label==1) if not legal.
            Set out = s.sel;
            out.remove(s.n);   // remove n from state
            if (l) out.diffWith(MispBase<N>::adj[s.n]); // remove neighbors of n from state (when taking n -- label==1 -- )
           // return State{out, out.empty() ? MispBase<N>::nNodes : s.n + 1}; // build state accordingly
            return State{out, s.n + 1}; // build state accordingly
        }
    }

    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return static_cast<double>(l);
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE
    static
    State smf(State const & s1, State const & s2) noexcept
    {
        assert(s1.n == s2.n);
        return State{s1.sel | s2.sel,min(s1.n,s2.n)};
    }
    // State similarity function
    GFL_HOST_DEVICE
    static
    gfl::f32 ssf(State const & s1, State const & s2) noexcept
    {
        gfl::f32 n = 0.0;
        gfl::f32 mean = 0.0;
        sim_combine(mean,n,s1.sel.iou(s2.sel));
        return mean;
    };


    constexpr static bool has_local = true;
    GFL_HOST_DEVICE
    double local(State const & s, LocalContext ctx) const noexcept
    {
        int minAdj = gfl::numeric_limits<int>::max();
        for (int i = 0; i < MispBase<N>::nNodes; i += 1)
        {
            if(s.sel.contains(i))
                minAdj = gfl::min<int>(minAdj, (s.sel | MispBase<N>::adj[i]).size());
        }
        return s.sel.size() - MispBase<N>::minAdj;
    }

    constexpr static bool has_dom = true;
    GFL_HOST_DEVICE
    static bool dom(State const & s1, State const & s2) noexcept
    {
        //return  (aU == bU) && (ae == be) && at < bt;
        return
                s1.n == s2.n and
                s2.sel <= s1.sel;
        //return ae==be && at < bt;
    }
    GFL_HOST_DEVICE
    static std::size_t domHash(State const & s) noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, static_cast<std::size_t>(s.n));
        //hash_combine(seed, gfl::roundUpToMultiple<std::size_t>(s.sel.size(),2));
        return seed;
    }
    GFL_HOST_DEVICE
    static bool domEq(State const & s1, State const & s2) noexcept
    {
        return s1.n == s2.n;
    }
};


static_assert(IsModel<Misp<>>);