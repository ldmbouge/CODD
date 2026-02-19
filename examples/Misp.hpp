#pragma once

#include "MispData.hpp"
#include "ModelSpecs.hpp"
#include <iostream>

#include <GFL.hpp>
#include <Utils.hpp>

template<gfl::i32 BranchFactor>
class Misp : public MispData<BranchFactor>
{
    using MispData<BranchFactor>::nodes;
    using MispData<BranchFactor>::edges;
    using MispData<BranchFactor>::adj;
    using ItemSet = gfl::BitSet<gfl::BitSet<>::num_words(BranchFactor)>;

public:
    constexpr static bool is_maximization = true;

    using OutLabels = gfl::BitSet<gfl::BitSet<>::num_words(BranchFactor)>;

    class State
    {
        ItemSet sel{};
        gfl::i32  n{0};

    public:

        State(ItemSet const & sel, gfl::i32 const n) noexcept : sel(sel), n(n) {}

        GFL_HOST_DEVICE
        static
        bool equal(State const & s1, State const & s2) noexcept
        {
            return s1.n == s2.n and s1.sel == s2.sel;
        }

        GFL_HOST_DEVICE
        static
        gfl::u64 hash(State const & s) noexcept
        {
            using namespace gfl;
            u64 seed = 0;
            hashCombine(seed, s.sel.hash());
            return seed;
        }

        GFL_HOST_DEVICE
        static
        void print(State const & s)
        {
            printf("<SEL=");
            s.sel.print();
            printf(">");
        }

        friend class Misp;
    };

    State initial() const noexcept
    {
        return State(ItemSet(0,nodes-1),0);
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return State(ItemSet(), nodes);
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        assert(s.n <= nodes);
        return s.n == nodes;
    }

    GFL_HOST_DEVICE
    OutLabels lgf(State const & s, double pBound, double dBound, DDContext ddCtx) const noexcept
    {
        return OutLabels(0,1);
    }

    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        if (s.n == nodes)
        {
            return gfl::nullopt;
        }
        else
        {
            if (l and (not s.sel.contains(s.n))) return gfl::nullopt; // we cannot take n (label==1) if not legal.
            ItemSet out = s.sel;
            out.remove(s.n);   // remove n from state
            if (l) out.diffWith(adj[s.n]); // remove neighbors of n from state (when taking n -- label==1 -- )
            return State(out, s.n + 1); // build state accordingly
        }
    }

    GFL_HOST_DEVICE
    gfl::f64 scf(State const & s, int const l) const noexcept
    {
        using namespace gfl;
        return scast<f64>(l);
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE
    static
    State smf(State const & s1, State const & s2) noexcept
    {
        assert(s1.n == s2.n);
        return State(s1.sel | s2.sel,s1.n);
    }

    constexpr static bool has_heur = true;
    GFL_HOST_DEVICE
    gfl::f64 h(State const & s, HContext hCtx) const noexcept
    {
        using namespace gfl;
        return scast<f64>(s.sel.size());
    }

    constexpr static bool has_dom = false;
};
