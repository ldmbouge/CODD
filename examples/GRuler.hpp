#pragma once

#include "GRulerData.hpp"
#include "ModelSpecs.hpp"
#include <GFL.hpp>
#include <iostream>

template<gfl::i32 BranchFactor, gfl::i32 Items>
class GRuler : public GRulerData
{
    using GRulerData::n;
    using GRulerData::L;
    using GRulerData::OPT;
    using MarksSet = gfl::BitSet<gfl::BitSet<>::num_words(Items)>;

public:
    constexpr static bool is_maximization = false;
    using OutLabels = MarksSet;

    class State
    {
        MarksSet m;   // set of mark positions placed so far
        MarksSet d;   // set of distances already used (always contains 0)
        gfl::i32 k;   // number of marks placed (= layer in the DD)
        gfl::i32 e;   // position of the last (rightmost) mark
        gfl::i32 sm;  // smallest distance NOT yet used (always >= 1)

    public:
        GFL_HOST_DEVICE
        State() = default;

        GFL_HOST_DEVICE
        State(MarksSet const & m, MarksSet const & d, gfl::i32 k, gfl::i32 e, gfl::i32 sm) noexcept :
            m(m), d(d), k(k), e(e), sm(sm) {}

        GFL_HOST_DEVICE static
        bool equal(State const & s1, State const & s2) noexcept
        {
            return s1.k == s2.k and s1.e == s2.e and s1.m == s2.m;
        }

        GFL_HOST_DEVICE static
        gfl::u64 hash(State const & s) noexcept
        {
            using namespace gfl;
            u64 seed = 0;
            hashCombine(seed, s.m.hash());
            hashCombine(seed, s.d.hash());
            hashCombine(seed, s.k);
            hashCombine(seed, s.e);
            hashCombine(seed, s.sm);
            return seed;
        }

        GFL_HOST_DEVICE
        void print() const
        {
            printf("K %d | E %d | SM %d | ", k, e, sm);
            printf("M "); m.print();
            printf(" | D "); d.print(); printf("\n");
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return os << "<"
                   "MRKS=" << s.m << ","
                   "DIST=" << s.d << "," << ">";
        }

        friend class GRuler;
    };

    // Initial state: no marks placed yet, force first transition at label=0 via lgf.
    // d={0} establishes the invariant "distance 0 is always used" which propagates
    // automatically through stf (via inheritance) and smf (via intersection).
    // This guarantees sm >= 1 everywhere without any special casing.
    State initial() const noexcept
    {
        MarksSet d;
        d.insert(0);
        return State(MarksSet(), MarksSet(0), 0, 0, 1);
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return State(MarksSet(), MarksSet(), n, 0, L + 1);
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        return s.k == n;
    }

    GFL_HOST_DEVICE
    OutLabels lgf(State const & s, double pBound, double dBound, DDContext ddCtx) const noexcept
    {
        using namespace gfl;

        if (s.k != 0)
        {
            i32 ub = L + 1;
            if (s.k < n / 2)
                ub = min<i32>(pBound, L + 1) / 2 - OPT[(n / 2) - s.k];
            else
                ub = min<i32>(pBound - 1, L + 1) - OPT[n - s.k];

            // sm >= 1 always (guaranteed by d containing 0 from initial state)
            // so lb = e + sm > e always, loop never starts AT last mark
            auto lb = s.e + s.sm;
            if (s.k < n - 1)
                lb = max<i32>(lb, ceil<i32>(s.k * (s.k - 1), 2), OPT[s.k + 1]);
            else
                lb = max<i32>(lb, ceil<i32>(s.k * (s.k - 1), 2), OPT[s.k] + 1);

            MarksSet vr;
            for (i32 label = lb; label <= ub; label++)
            {
                MarksSet leg(s.d);
                // printf("Leg = "); MarksSet::print(leg);  printf("\n");
                // printf("s.m = "); MarksSet::print(s.m);  printf("\n");
                // printf("label = &d\n---\n");
                // fflush(stdout);
                bool legal = (leg.interWith(label - s.m).empty());
                if (legal)
                    vr.insert(label);
            }
            return vr;
        }
        else
        {
            // k=0: force placement of mark 0 as the only valid first transition
            return OutLabels(0);
        }
    }

    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        using namespace gfl;

        // d always contains 0 (inherited from s.d via | operation)
        // so smallest_dist starts at s.sm >= 1 and advances correctly
        MarksSet d_new = (l - s.m) | s.d;
        i32 smallest_dist = s.sm;
        while (d_new.contains(smallest_dist)) smallest_dist += 1;

        return State(s.m | MarksSet(l), d_new, s.k + 1, l, smallest_dist);
    }

    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return l - s.e;
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE static
    State smf(State const & s1, State const & s2) noexcept
    {
        using namespace gfl;
        assert(s1.k == s2.k);

        // Intersection: fewer marks/distances = more labels valid = correct relaxation
        // d always contains 0 in both states → intersection always contains 0 → sm >= 1
        MarksSet const m_new  = s1.m & s2.m;
        MarksSet const d_new  = s1.d & s2.d;

        // k is the layer index, not the size of m after intersection
        i32 const k_new  = s1.k;

        // e = largest mark in merged m: always in m_new by definition,
        // no inconsistency possible (unlike min(e1,e2) which may not be in m_new)
        i32 const e_new  = m_new.largest();

        // sm = smallest distance not in d_new, computed directly from d_new.
        // Always >= 1 because d_new contains 0 (invariant preserved by intersection)
        i32 const sm_new = (~d_new).smallest();

        return State(m_new, d_new, k_new, min<i32>(s1.e,s2.e), min<i32>(s1.sm,s2.sm));
    }

    GFL_HOST_DEVICE static
    gfl::f32 ssf(State const & s1, State const & s2) noexcept
    {
        using namespace gfl;
        gfl::f32 n    = 0.0;
        gfl::f32 mean = 0.0;
        gfl::sim_combine(mean, n, s1.m.iou(s2.m));
        return mean;
    }

    constexpr static bool has_heur = false;
    constexpr static bool has_dom  = false;
};