#pragma once

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl
{
    // References
    // - https://softwareengineering.stackexchange.com/a/402543
    // - https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
    GFL_HOST_DEVICE inline
    void hashCombine(u64 & seed, u64 const h)
    {  seed ^= h + 0x9e3779b97f4a7c15ull + (seed<<6) + (seed>>2); }

    GFL_HOST_DEVICE inline
    u32 hash64to32(u64 const x)
    {
        return scast<u32>(x ^ (x >> 32));
    }

    GFL_HOST_DEVICE inline
    void sim_combine(float & mean, float & n, float const val)
    {
        n += 1.0;
        mean += (val - mean) / n;
    }

}
