#include "cadds_relaxed_seq.cuh"
#include "misp.cuh"

int main(int argc,char* argv[])
{
    constexpr int N = 200;
    using Model = Misp<N>;
    using Node = LightNode<Model::State, Model::Labels, N>;

    return run_cadds_relaxed_seq<Model,Node>(argc, argv);
}