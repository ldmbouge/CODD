#pragma once

#include "ModelSpec.hpp"
#include "tsptw_base.hpp"
#include "util.hpp"
#include <FunctionQualifiers.hpp>
#include <Backend.hpp>

template<int N = 64>
struct TSPTW1 : TSPTWBase
{
    // Model
    using Set = NatSet<gfl::roundUpDivPosInt<unsigned  short>(N,64)>;
    using Labels = Set;
    struct State
    {
        Set U;
        short e;
        short t;
        short hops;

        GFL_HOST_DEVICE
        State() : U(), e(0), t(0), hops(0)
        {}

        GFL_HOST_DEVICE
        State(Set const & U, int e, int t, int hops)
                : U(U)
                , e(static_cast<short>(e))
                , t(static_cast<short>(t))
                , hops(static_cast<short>(hops))
        {}

        GFL_HOST_DEVICE
        static bool equal(State const & s1, State const & s2) noexcept
        {
            return
                    s1.U  == s2.U  and
                    s1.e == s2.e and
                    s1.t == s2.t and
                    s1.hops  == s2.hops;
        }
        GFL_HOST_DEVICE
        static std::size_t hash(State const & s) noexcept
        {
            std::size_t seed = 0;
            hash_combine(seed, s.U.hash());
            hash_combine(seed, static_cast<std::size_t>(s.e));
            hash_combine(seed, static_cast<std::size_t>(s.t));
            hash_combine(seed, static_cast<std::size_t>(s.hops));
            return seed;
        }
        friend std::ostream & operator<<(std::ostream & os, State const & s)
        {
            return
                    os << "<" <<
                       "E=" << s.e << "," <<
                       "U=" << s.U << "," <<
                       "T=" << s.t << "," <<
                       "HOPS=" << s.hops << ">";
        }
    };

    State initial() const noexcept
    {
        return {Set(depot+1,n-1), depot, 0,  0 };
    }

    GFL_HOST_DEVICE
    State target() const noexcept
    {
        return {Set(), depot, 0, n };
    }

    GFL_HOST_DEVICE
    bool isTarget(State const & s) const noexcept
    {
        return s.e == depot && s.hops == n;
    }

    GFL_HOST_DEVICE
    Labels lgf(State const & s, DDContext ctx, double pBound, double dBound) const noexcept
    {
        if (s.hops >= n-1)
            return (s.t + d[s.e][depot] <= tw[depot].b) and s.e != depot ? Set{depot} : Set{}; // that's the only way to return the depot
        else
            return filter(s.U, [this,s](auto& u){ return s.t + d[s.e][u] <= tw[u].b; }); // neither s.e nor depot IN s.U
    }
    GFL_HOST_DEVICE
    gfl::optional<State> stf(State const & s, int l) const noexcept
    {
        if (l==depot) {
            return State{Set(),depot,0,n};
        } else {
            const auto& [U,e,t,hops] = s;
            const int nextT = gfl::max<int>(t + d[e][l], tw[l].a);
            Set nextU = U - l;
            for(auto u : nextU)
                if(nextT + d[l][u] > tw[u].b)
                    return gfl::nullopt;
            return State{nextU, l, nextT, hops + 1};
        }
    }
    GFL_HOST_DEVICE
    double scf(State const & s, int l) const noexcept
    {
        return d[s.e][l];
    }

    GFL_HOST_DEVICE constexpr static
    bool better(double const & c1, double const & c2) noexcept { return c1 < c2; }
    GFL_HOST_DEVICE
    constexpr static bool betterEq(double const & c1, double const & c2)  noexcept { return c1 <= c2; }
    constexpr static double bestValue() noexcept { return std::numeric_limits<double>::lowest(); }
    constexpr static double worstValue() noexcept { return std::numeric_limits<double>::max(); } // lowest() instead of min() because double

    constexpr static bool has_merge = false;

    constexpr static bool has_local = true;
    GFL_HOST_DEVICE
    double local(State const & s, LocalContext ctx) const noexcept
    {
        int sumIn = 0,sumOut = 0,n1=0,n2=0;
        for(int i = 0; (i < n) && (n1 < n-s.hops); i++) {
            if( s.U.contains(permIn[i]) || permIn[i] == depot) {
                sumIn += dIn[i];
                n1++;
            }
        }
        for(int i = 0; (i < n) && (n2 < n-s.hops); i++) {
            if( s.U.contains(permOut[i]) || permOut[i] == s.e) {
                sumOut += dOut[i];
                n2++;
            }
        }
        return gfl::max<double>(sumIn,sumOut);
    }

    constexpr static bool has_dom = true;
    GFL_HOST_DEVICE
    static bool dom(State const & s1, State const & s2) noexcept
    {
        //return  (aU == bU) && (ae == be) && at < bt;
        return
            s1.U == s2.U and
            s1.e == s2.e and
            s1.hops == s2.hops and
            s1.t < s2.t;
        //return ae==be && at < bt;
    }
    GFL_HOST_DEVICE
    static std::size_t domHash(State const & s) noexcept
    {
        std::size_t seed = 0;
        hash_combine(seed, s.U.hash());
        hash_combine(seed, static_cast<std::size_t>(s.e));
        hash_combine(seed, static_cast<std::size_t>(s.hops));
        return seed;
    }
};

static_assert(IsModel<TSPTW1<>>);