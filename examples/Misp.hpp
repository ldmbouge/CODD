#pragma once

#include "MispData.hpp"
#include "ModelSpecs.hpp"
#include <iostream>

#include <GFL.hpp>
#include <Utils.hpp>

template<gfl::i32 BranchFactor, gfl::i32 Items>
class Misp : public MispData<Items>
{
    using MispData<Items>::nodes;
    using MispData<Items>::edges;
    using MispData<Items>::adj;
    using ItemSet = gfl::BitSet<gfl::BitSet<>::num_words(Items)>;

public:
    constexpr static bool is_maximization = true;

    using OutLabels = gfl::BitSet<gfl::BitSet<>::num_words(BranchFactor)>;

    class State
    {
        ItemSet sel{};
        gfl::i32  n{0};

    public:
        State() = default;

        GFL_HOST_DEVICE
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
            printf("<N=%d,SEL=",s.n);
            s.sel.print();
            printf(">");
        }

        GFL_HOST_DEVICE
        void print() const noexcept{ print(*this); }

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
        return OutLabels(0,s.sel.contains(s.n));
    }

    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        ItemSet out = s.sel;
        out.remove(s.n);
        if (l)  out.diffWith(adj[s.n]);
        return State(out, s.n + 1); // build state accordingly
    }

    GFL_HOST_DEVICE
    gfl::f64 scf(State const & s, int const l) const noexcept
    {
        return l;
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE
    static
    State smf(State const & s1, State const & s2) noexcept
    {
        assert(s1.n == s2.n);
        return State(s1.sel | s2.sel,s1.n);
    }

    constexpr static bool has_rank = true;
    // State ranking  function
    GFL_HOST_DEVICE
    static
    gfl::f32 srf(State const & s) noexcept
    {
        using namespace gfl;
        f32 score = scast<f32>(s.sel.capacity()) - scast<f32>(s.sel.size());
        score /= scast<f32>(s.sel.capacity());
        //score += 1.0f;
        //score *= 10.0;
        //score +=  scast<f32>(s.sel.smallest()) / scast<f32>(s.sel.capacity());
        return score;
    };

    constexpr static bool has_heur = true;
    GFL_HOST_DEVICE
    gfl::f64 h(State const & s, HContext hCtx) const noexcept
    {
        using namespace gfl;
        return scast<f64>(s.sel.size());
    }

    constexpr static bool has_dom = false;
};
