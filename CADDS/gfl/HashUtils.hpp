#pragma once

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl {

GFL_HOST_DEVICE inline
void hashCombine(u64 & seed, u64 h) {
   seed ^= h + 0x9e3779b97f4a7c15ull + (seed <<  6) + (seed >>  2);
}
}
