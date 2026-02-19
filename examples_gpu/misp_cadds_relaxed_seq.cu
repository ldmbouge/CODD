#include "cadds_relaxed_seq.cuh"
#include "../examples/misp.hpp"

int main(int argc,char* argv[])
{
    constexpr int MaxBranchinFactor = 256;
    constexpr int MaxDepth = 256;
    using Model = Misp<MaxBranchinFactor,MaxDepth>;
    using Node = LightNode<Model::State, Model::Labels, maxBranchingFactor>;

    return run_cadds_relaxed_seq<Model,Node>(argc, argv, maxDepth, maxBranchingFactor);
}