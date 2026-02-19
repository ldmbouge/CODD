#include "cadds_relaxed.cuh"
#include "../examples/misp.hpp"

int main(int argc,char* argv[])
{
    constexpr int N = 256;
    using Model = Misp<N>;
    using Node = LightNode<Model::State, Model::Labels, N>;

    return run_cadds_relaxed<Model,Node>(argc, argv);
}