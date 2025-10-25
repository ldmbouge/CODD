#pragma once

#include <iostream>
#include <fstream>

struct GRulerBase
{
    int n;
    int L;
    const int OPT[20] = {0,0,1,3,6,11,17,25,34,44,55,72,85,106,127,151,177,199,216,246};

    static inline
    void parseFile(GRulerBase * const model, std::string const & instance, gfl::StackAllocator & allocator)
    {
        auto m = model;

        std::ifstream file(instance);
        if (not file.good())
        {
            std::cerr << "File does not exist or could not be opened: " << instance << std::endl;
            exit(EXIT_FAILURE);
        }
        file >> m->n;
        file >> m->L;
        file.close();
    }
};