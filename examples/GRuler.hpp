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
    using OutLabels = gfl::BitSet<gfl::BitSet<>::num_words(Items)>;

    class State
    {
        MarksSet m;   // set of mark positions placed so far
        MarksSet d;   // set of distances already used
        gfl::i32 k;   // number of marks placed
        gfl::i32 e;   // position of the last (rightmost) mark
        gfl::i32 sm;  // smallest distance NOT yet used

    public:
        GFL_HOST_DEVICE
        State() = default;

        GFL_HOST_DEVICE
        State(MarksSet const & m, MarksSet const & d, gfl::i32 k, gfl::i32 e, gfl::i32 sm) noexcept :
            m(m), d(d), k(k), e(e), sm(sm) {}

        GFL_HOST_DEVICE static
        bool equal(State const & s1, State const & s2) noexcept
        {
            return
                s1.k == s2.k and
                s1.e == s2.e and
                s1.m == s2.m;
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
            printf("M "); m.printAsInts();
            printf(" | D "); d.printAsInts(); printf("\n");
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return os << "<" <<
                   "MRKS=" << s.m << "," <<
                   "DIST=" << s.d << "," << ">";
        }

        friend class GRuler;
    };

    State initial() const noexcept
    {
        return State(MarksSet(0),MarksSet(),1,0,1);
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

        i32 ub = L+1;
        if (s.k < n/2)
            ub = min<i32>(ub, pBound)/2 - OPT[(n/2)-s.k];
        else
            ub = min<i32>(ub, pBound-1) - OPT[n-s.k];
        i32 lb = max<i32>(s.e + s.sm,ceil<i32>(s.k*(s.k-1),2));
        lb = max<i32>(lb ,(s.k < n - 1) ? OPT[s.k+1] : OPT[s.k]+1);
        MarksSet vr;
        for(i32 label = lb;label <= ub;label++)
        {
            MarksSet leg(s.d);
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
        using namespace gfl;

        MarksSet d_new = (l - s.m) | s.d;
        i32 smallest_dist = s.sm;
        while (d_new.contains(smallest_dist)) smallest_dist += 1;
        State rv(s.m | MarksSet(l), d_new, s.k + 1, l, smallest_dist);
        return rv;
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
        return State(
            s1.m & s2.m,
            s1.d & s2.d,
            s1.k,
            min<i32>(s1.e,s2.e),
            min<i32>(s1.sm,s2.sm));
    }

    constexpr static bool has_heur = false;
    constexpr static bool has_dom  = false;
};