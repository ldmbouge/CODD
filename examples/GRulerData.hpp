#pragma once

#include <iostream>
#include <fstream>

#include <GFL.hpp>

class GRulerData
{
    protected:
    gfl::i32 n;
    gfl::i32 L;
    gfl::i32 const OPT[20] = {0,0,1,3,6,11,17,25,34,44,55,72,85,106,127,151,177,199,216,246};

public:
    inline
    void init(std::string const& instancePath, gfl::ArenaAllocator& alloc)
    {
        using namespace gfl;

        std::ifstream file(instancePath);
        assert(file.is_open());

        file >> n;
        file >> L;
    }
};