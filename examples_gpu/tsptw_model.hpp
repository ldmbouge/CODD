#pragma once

#include "model.hpp"
#include "util.hpp"

struct TSPTW
{
   // Instance data
    struct TimeWindow
    {
        int a,b;

        TimeWindow() noexcept : a(std::numeric_limits<int>::max()), b(std::numeric_limits<int>::min()) {}
        TimeWindow(int const a,int const b) noexcept : a(a),b(b) {}
        friend std::ostream & operator<<(std::ostream & os, TimeWindow const & t)
        {
            return os << "[" << t.a << "," << t.b << "]";
        }

    };

    constexpr static int depot = 0;
    int n;
    Matrix<int,2> d;
    FArray<TimeWindow> tw;
    FArray<int> dInNS;
    FArray<int> dIn;
    FArray<int> dOut;
    FArray<int> permIn;
    FArray<int> permOut;

    // Model
    using Labels = NatSet<2>;
    struct State
    {
        using Set = NatSet<2>;

        Set pos;  // current city (set for merged nodes, singleton otherwise)
        Set must; // cities that are unvisited in all nodes in the prefix of this state
        Set may;  // cities that are visited in some, but not all, nodes in the prefix
        int hops; // number of visited cities
        int ta;   // earliest and latest times at pos (a and b are same for exact nodes)
        int tb;

        GFL_HOST_DEVICE
        State() {}
        GFL_HOST_DEVICE
        State(Set const & pos,Set const & must,Set const & may,int const hops,int const ta,int const tb) noexcept : pos(pos),must(must),may(may),hops(hops),ta(ta),tb(tb) {}
        GFL_HOST_DEVICE
        State(Set && pos,Set && must,Set && may, int const hops, int const ta,int const tb) noexcept : pos(pos),must(must),may(may),hops(hops),ta(ta),tb(tb) {}

        GFL_HOST_DEVICE
        static bool equal(State const & s1, State const & s2) noexcept
        {
            return
                    s1.pos  == s2.pos  and
                    s1.hops == s2.hops and
                    s1.must == s2.must and
                    s1.may  == s2.may  and
                    s1.ta   == s2.ta   and
                    s1.tb   == s2.tb;
        }
        GFL_HOST_DEVICE
        static std::size_t hash(State const & s) noexcept
        {
            std::size_t seed = 0;
            hash_combine(seed, s.pos.hash());
            hash_combine(seed, s.must.hash());
            hash_combine(seed, s.may.hash());
            hash_combine(seed, static_cast<std::size_t>(s.hops));
            hash_combine(seed, static_cast<std::size_t>(s.ta));
            hash_combine(seed, static_cast<std::size_t>(s.tb));
            return seed;
        }
        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return
                os << "<" <<
                "POS=" << s.pos << "," <<
                "HOPS=" << s.hops << "," <<
                "TIME=[" << s.ta << "," << s.tb << "]," <<
                "MUST=" << s.must << "," <<
                "MAY=" << s.may << ">";
        }
    };

    State initial() const noexcept
    {
        using Set = State::Set;
        auto s = State( Set({depot}), Set(depot+1,n-1), Set(),0, 0, 0);
        return s;
    }
    GFL_HOST_DEVICE
    State target() const noexcept
    {
        using Set = State::Set;
        auto s = State( Set({depot}), Set(), Set(),n, 0, 0);
        return s;
    }
    bool isTarget(State const & s) const noexcept
    {
        auto const c1 = s.pos.contains(depot);
        auto const c2 = s.hops == n;
        return c1 and c2;
    }

    GFL_HOST_DEVICE
    Labels lgf(State const & s, DDContext ctx) const noexcept
    {
        if (s.hops >= n - 1)
        {
            /*s.must.empty() && s.may.empty()*/
            const int a = s.ta + min(s.pos, [this](const int p) { return d[p][depot]; });
            return (a <= tw[depot].b) ? State::Set{depot} : State::Set{};
        }
        else
        {
            // std::cout << (s) << "\n";
            // std::cout << "pre-filter : " << (s.must|s.may) << "\n";
            const auto f = filter(
                s.must | s.may,
                [this,&s](const auto u)
                {
                    const int a = s.ta + min(s.pos - u, [this,u](const int p) { return d[p][u]; });
                    return a <= tw[u].b && (s.pos.size() > 1 || !s.pos.contains(u));
                }
            );
            //std::cout << "state: " << s << "\nnext: " << f << "\n";
            return f;
        }
    }
    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        if (l == depot)
        {
            return target();
        }
        else
        {
            const int ta = gfl::max<int>(s.ta + min(s.pos - l, [this,l](const int p) {return d[p][l]; }),
                                    tw[l].a);
            auto newMust = s.must - l;
            if (any(newMust, [this,l,ta](int u) { return ta + d[l][u] > tw[u].b; }))
                return gfl::nullopt; // at least one in the next must violates its time window ub.
            const int tb = (s.ta == s.tb)
                               ? ta
                               : gfl::min<int>(s.tb + max(s.pos - l, [this,l](const int p) { return d[p][l]; }),
                                          tw[l].b);
            return State{State::Set{l}, newMust, s.may - l, s.hops + 1, ta, tb};
        }
    }
    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return min(s.pos - l,[this,l](int p) { return d[p][l]; });
    }

    GFL_HOST_DEVICE constexpr static
    bool better(double const & c1, double const & c2) noexcept { return c1 < c2; }
    GFL_HOST_DEVICE
    constexpr static bool betterEq(double const & c1, double const & c2)  noexcept { return c1 <= c2; }
    constexpr static double bestValue() noexcept { return std::numeric_limits<double>::max(); }
    constexpr static double worstValue() noexcept { return std::numeric_limits<double>::lowest(); } // lowest() instead of min() because double

    constexpr static bool has_merge = true;
    GFL_HOST_DEVICE
    gfl::optional<State> smf(State const & s1, State const & s2) const noexcept
    {
        if (s1.hops != s2.hops) return gfl::nullopt;

        const auto newMust = s1.must & s2.must;
        return State {
            s1.pos | s2.pos,
            newMust,
            (s1.must | s1.may | s2.must | s2.may) - newMust,
            s1.hops,
            gfl::min<int>(s1.ta, s2.ta),
            gfl::max<int>(s1.tb, s2.tb)
         };
    }

    constexpr static bool has_local = true;
    GFL_HOST_DEVICE
    double local(State const & s, LocalContext ctx) const noexcept
    {
        const auto inf = gfl::numeric_limits<int>::max();
        const auto violatesTW = [ta=s.ta,this](int p) { return ta + dInNS[p] > tw[p].b; };

        if (any(s.must, violatesTW))
        {
            // Nasty bug here. Sorting was wrong. (flipped >)
            //std::cout << "INF1\n";
            return inf; // but for violatesTW to be correct, since it says "If There exist a city in must such that ..."
        } // the index "p" to the closure is a city name. Can't used dIn. Must use the non-sorted version.

        const int completeTour = (n - 1) - s.hops - s.must.size();
        int mandatoryIn = 0, mandatoryOut = 0;
        // mandatory part
        for (int i = 0; i < n; i++)
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
            int nIn = 0, nOut = 0;
            for (int i = 0; i < n && nIn < completeTour; i++)
            {
                const bool use = s.may.contains(permIn[i]);
                mandatoryIn += dIn[i] * use;
                nIn += use;
            }
            for (int i = 0; i < n && nOut < completeTour; i++)
            {
                const bool use = s.may.contains(permOut[i]);
                mandatoryOut += dOut[i] * use;
                nOut += use;
            }
        }
        const int returnToDepot = (mandatoryIn == 0)
                                      ? min(s.must | s.may | s.pos, [this](int p) { return d[p][depot]; })
                                      : min(s.must | s.may, [this](int p) { return d[p][depot]; });

        if (s.ta + mandatoryIn + returnToDepot > tw[depot].b)
        {
            //std::cout << "INF3\n";
            return inf;
        }
        return gfl::max<double>(mandatoryIn, mandatoryOut) + returnToDepot;
    }

    constexpr static bool has_dom = true;
    GFL_HOST_DEVICE
    static bool dom(State const & s1, State const & s2) noexcept
    {
        return
            s1.must <= s2.must and
            s1.ta < s2.ta and
            s1.hops == s2.hops and
            s1.pos == s2.pos;
    }
    GFL_HOST_DEVICE
    static std::size_t domHash(State const & s) noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, s.pos.hash());
        hash_combine(seed, s.must.hash());
        hash_combine(seed, static_cast<std::size_t>(s.hops));
        return seed;
    }
    GFL_HOST_DEVICE
    static bool domEq(State const & s1, State const & s2) noexcept
    {
        return
            s1.pos == s2.pos and
            s1.must == s2.must and
            s1.may == s2.may and
            s1.hops == s2.hops;
    }
};

static_assert(IsModel<TSPTW>);

namespace std
{

    template <>
    struct hash<TSPTW::State> {
        std::size_t operator()(const TSPTW::State& s) const noexcept {
            return TSPTW::State::hash(s);
        }
    };

    template <>
    struct equal_to<TSPTW::State> {
        bool operator()(const TSPTW::State& a, const TSPTW::State& b) const noexcept {
            return TSPTW::State::equal(a, b);
        }
    };
}