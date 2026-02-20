#pragma once

#include <fstream>
#include <iostream>

#include <GFL.hpp>

template<gfl::i32 Items>
class MispData
{
protected:
    using ItemSet = gfl::BitSet<gfl::BitSet<>::num_words(Items)>;

    gfl::i32               nodes{0};
    gfl::i32               edges{0};
    gfl::ArrayView<ItemSet> adj{};

public:
    void init(std::string const& instancePath, gfl::ArenaAllocator& alloc)
    {
        using namespace gfl;

        // DIMACS format (1-indexed)
        // p edge <nodes> <edges>
        // e <u> <v>
        // ...

        std::ifstream file(instancePath);
        assert(file.is_open());

        // Problem line
        char        c;
        std::string tag;
        file >> c   ; assert(c   == 'p');
        file >> tag ; assert(tag == "edge");
        file >> nodes;
        file >> edges;

        // Adjacency lists
        adj = ArrayView<ItemSet>(nodes, alloc);
        for(auto & is : adj){ is = ItemSet(); }

        // Edge lines
        for (i32 i = 0; i < edges; ++i)
        {
            i32 u, v;
            file >> c ; assert(c == 'e');
            file >> u ;
            file >> v ;
            u -= 1; // convert to 0-based
            v -= 1; // convert to 0-based
            adj[u].insert(v);
            adj[v].insert(u);
        }
    }
};