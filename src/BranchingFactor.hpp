#pragma once

#include <GFL.hpp>

template<gfl::i32 V, bool D>
struct BranchingFactor
{
    static constexpr gfl::i32 value  = V;
    static constexpr bool     isDecr = D;
};

template<gfl::i32 V>
using FixedBranching = BranchingFactor<V, false>;

template<gfl::i32 V>
using DecrBranching  = BranchingFactor<V, true>;