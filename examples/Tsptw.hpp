#pragma once

#include "TsptwData.hpp"

#pragma once

#include "MispData.hpp"
#include "ModelSpecs.hpp"
#include <iostream>

#include <GFL.hpp>
#include <Utils.hpp>

template<gfl::i32 BranchFactor, gfl::i32 Cities>
class Tsptw : public TsptwData<Cities>
{
    using TsptwData<Cities>::Depot;
    using TsptwData<Cities>::n;
    using TsptwData<Cities>::d;
    using TsptwData<Cities>::tw;
    using TsptwData<Cities>::dInNS;
    using TsptwData<Cities>::dIn;
    using TsptwData<Cities>::dOut;
    using TsptwData<Cities>::permIn;
    using TsptwData<Cities>::permOut;

    using CitiesSet = gfl::BitSet<gfl::BitSet<>::num_words(Cities)>;

public:
    constexpr static bool is_maximization = false;
    using OutLabels = gfl::BitSet<gfl::BitSet<>::num_words(BranchFactor)>;

    class State
    {
        CitiesSet pos;  // current city (set for merged nodes, singleton otherwise)
        CitiesSet must; // cities that are unvisited in all nodes in the prefix of this state
        CitiesSet may;  // cities that are visited in some, but not all, nodes in the prefix
        int hops; // number of visited cities
        int eat;   // earliest and latest times at pos (a and b are same for exact nodes)
        int ldt;

    public:
        State() = default;

        GFL_HOST_DEVICE
        State(CitiesSet const & pos, CitiesSet const & must, CitiesSet const & may, gfl::i32 const hops, gfl::i32 const eat, gfl::i32 const ldt) noexcept :
            pos(pos), must(must), may(may), hops(hops), eat(eat), ldt(ldt) {}

        GFL_HOST_DEVICE static
        bool equal(State const & s1, State const & s2) noexcept
        {
            return
                    s1.pos  == s2.pos  and
                    s1.hops == s2.hops and
                    s1.must == s2.must and
                    s1.may  == s2.may and
                    s1.eat  == s2.eat and
                    s1.ldt  == s2.ldt;
        }

        GFL_HOST_DEVICE static
        gfl::u64 hash(State const & s) noexcept
        {
            using namespace gfl;

            u64 seed = 0;
            hashCombine(seed, s.pos.hash());
            hashCombine(seed, s.must.hash());
            hashCombine(seed, s.may.hash());
            hashCombine(seed, scast<u64>(s.hops));
            hashCombine(seed, scast<u64>(s.eat));
            hashCombine(seed, scast<u64>(s.ldt));
            return seed;
        }

        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return
                os << "<" <<
                "POS=" << s.pos << "," <<
                "HOPS=" << s.hops << "," <<
                "TIME=[" << s.eat << "," << s.ldt << "]," <<
                "MUST=" << s.must << "," <<
                "MAY=" << s.may << ">";
        }

        friend class Tsptw;
    };

    State initial() const noexcept
    {
        auto const s = State(CitiesSet(Depot), CitiesSet(Depot+1,n-1), CitiesSet(),0, 0, 0);
        return s;
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        auto const c1 = s.pos.contains(Depot);
        auto const c2 = s.hops == n;
        return c1 and c2;
    }

    GFL_HOST_DEVICE
    OutLabels lgf(State const & s, double pBound, double dBound, DDContext ddCtx) const noexcept
    {
        using namespace gfl;

        if (s.hops >= n - 1)
        {
            i32 const eat = s.eat + min(s.pos, [this](i32 const p) { return d[p][Depot]; });
            return eat <= tw[Depot].e ? CitiesSet(Depot) : CitiesSet();
        }
        else
        {
            auto const f = [this,&s](auto const u) {
                i32 const eat = s.eat + min(s.pos - u, [this,u](const int p) { return d[p][u]; });
                return eat <= tw[u].e and (s.pos.size() > 1 || !s.pos.contains(u));
            };
            const auto rv = filter(s.must | s.may, f);
            return rv;
        }
    }

    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        using namespace gfl;

        if (l == Depot)
        {
            return State( CitiesSet(Depot), CitiesSet(), CitiesSet(),n, 0, 0);;
        }
        else
        {
            i32 const eat = gfl::max<i32>(s.eat + min(s.pos - l, [this,l](const int p) {return d[p][l]; }),
                                    tw[l].b);
            CitiesSet newMust = s.must - l;
            if (any(newMust, [this,l,eat](int u) { return eat + d[l][u] > tw[u].e; }))
                return nullopt; // at least one in the next must violates its time window ub.
            i32 const ldt = (s.eat == s.ldt)
                               ? eat
                               : gfl::min<int>(s.ldt + max(s.pos - l, [this,l](const int p) { return d[p][l]; }),
                                          tw[l].e);
            return State(CitiesSet(l), newMust, s.may - l, s.hops + 1, eat, ldt);
        }
    }

    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return min(s.pos - l,[this,l](int p) { return d[p][l]; });
    }

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE static
    State smf(State const & s1, State const & s2) noexcept
    {
        const auto newMust = s1.must & s2.must;
        return State {
            s1.pos | s2.pos,
            newMust,
            (s1.must | s1.may | s2.must | s2.may) - newMust,
            s1.hops,
            gfl::min<int>(s1.eat, s2.eat),
            gfl::max<int>(s1.ldt, s2.ldt)
         };
    }

    constexpr static bool has_heur= true;
    GFL_HOST_DEVICE
    double h(State const & s, HContext ctx) const noexcept
    {
        using namespace gfl;

        i32 const inf = numeric_limits<i32>::max();
        auto const violatesTW = [s,this](int p) { return s.eat + dInNS[p] > tw[p].e; };

        if (any(s.must, violatesTW))
        {
            // Nasty bug here. Sorting was wrong. (flipped >)
            //std::cout << "INF1\n";
            return inf; // but for violatesTW to be correct, since it says "If There exist a city in must such that ..."
        } // the index "p" to the closure is a city name. Can't used dIn. Must use the non-sorted version.

        i32 const completeTour = (n - 1) - s.hops - s.must.size();
        i32 mandatoryIn = 0, mandatoryOut = 0;
        // mandatory part
        for (i32 i = 0; i < n; i++)
        {
            mandatoryIn += dIn[i] * s.must.contains(permIn[i]);
            mandatoryOut += dOut[i] * s.must.contains(permOut[i]);
        }
        if (s.may.size() > 0)
        {
            //std::cout << completeTour << "=" << sz << "-" << s.hops << "-" << s.must.size()<< " " << s.must << "\n";
            //std::cout << s.may << " " << s.may.size() <<"-"<< count(s.may, violatesTW) <<"<"<< completeTour << "\n";
            if (s.may.size() - count(s.may, violatesTW) < completeTour)
            {
                //std::cout << "INF2\n";
                return inf;
            }
            i32 nIn = 0;
            i32 nOut = 0;
            for (i32 i = 0; i < n && nIn < completeTour; i++)
            {
                bool const use = s.may.contains(permIn[i]);
                mandatoryIn += dIn[i] * use;
                nIn += use;
            }
            for (i32 i = 0; i < n and nOut < completeTour; i++)
            {
                bool const use = s.may.contains(permOut[i]);
                mandatoryOut += dOut[i] * use;
                nOut += use;
            }
        }
        i32 const returnToDepot = (mandatoryIn == 0)
                                      ? min(s.must | s.may | s.pos, [this](int p) { return d[p][Depot]; })
                                      : min(s.must | s.may, [this](int p) { return d[p][Depot]; });

        if (s.eat + mandatoryIn + returnToDepot > tw[Depot].e)
        {
            //std::cout << "INF3\n";
            return inf;
        }
        return gfl::max<double>(mandatoryIn, mandatoryOut) + returnToDepot;
    }

    constexpr static bool has_dom = true;
    GFL_HOST_DEVICE static
    bool dom(State const & s1, State const & s2) noexcept
    {
        return
            s1.pos == s2.pos and
            s1.must <= s2.must and
            s1.hops == s2.hops and
            s1.eat < s2.eat;
    }
    GFL_HOST_DEVICE static
    gfl::u64 domHash(State const & s) noexcept
    {
        using namespace gfl;
        u64 seed = 0;
        hashCombine(seed, s.pos.hash());
        hashCombine(seed, s.must.hash());
        hashCombine(seed, s.hops);
        return seed;
    }

    GFL_HOST_DEVICE static
    bool domEq(State const & s1, State const & s2) noexcept
    {
        return
            s1.pos == s2.pos and
            s1.must == s2.must and
            s1.may == s2.may and
            s1.hops == s2.hops;
    }
};