#pragma once

#include <iostream>
#include <fstream>
#include <Utils.hpp>

template<int N = 256>
struct MispBase
{
    using Set = NatSet<gfl::roundUpDivPosInt<unsigned short>(N,64)>;

    int nNodes;
    int nEdges;
    FArray<Set> adj;
    int minAdj;

    static inline
    void parseFile(MispBase * const model, std::string const & instance, gfl::StackAllocator & allocator)
    {
        auto m = model;

        std::ifstream file(instance);
        if (not file.good())
        {
            std::cerr << "File does not exist or could not be opened: " << instance << std::endl;
            exit(EXIT_FAILURE);
        }
        char c;
        file >> c;
        assert(c == 'p');
        std::string s;
        file >> s;
        assert(s == "edge");
        file >> m->nNodes;
        file >> m->nEdges;
        new (&m->adj) FArray<Set>(m->nNodes,allocator); // The = operator gives issues whith destruction
        for (int i = 0; i < m->nNodes; i += 1)
        {
            new (&m->adj[i]) Set();
        }
        for (int i = 0; i < m->nEdges; i += 1)
        {
            int a,b;
            file >> c;
            assert(c == 'e');
            file >> a;
            file >> b;
            a -= 1; // Make it 0-based
            b -= 1; // Make it 0-based
            m->adj[a].insert(b);
            m->adj[b].insert(a);
        }
        file.close();

    }
};











