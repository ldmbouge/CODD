#pragma once

#include <GFL.hpp>
#include "Node.hpp"

struct DummyDecomposer64
{

    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&> operator()(NodeInfo &) const
    {
        gfl::u64 tmp = 0;
        return {tmp};
    }
};

struct DummyDecomposer96
{

    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64&,gfl::u32&> operator()(NodeInfo &) const
    {
        gfl::u64 tmp64 = 0;
        gfl::u32 tmp32 = 0;
        return {tmp64,tmp32};
    }
};

