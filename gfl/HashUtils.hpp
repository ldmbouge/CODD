#pragma once

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl
{
    // https://stackoverflow.com/a/50978188
    GFL_HOST_DEVICE inline
    void hashCombine(u64 & seed, u64 h)
    {
        h ^= h >> 32;
        h *= 0x5555555555555555ull;
        h ^= h >> 32;
        h *= 17316035218449499591ull;
        h ^= h >> 32;
        seed = ((seed << 21) | (seed >> 43)) ^ h;
    }
}
