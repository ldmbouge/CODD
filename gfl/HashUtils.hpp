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
    { seed ^= h + 0x9e3779b9 + (seed<<6) + (seed>>2); }
}
