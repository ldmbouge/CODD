#include "bro_seq_base.cuh"
#include "misp.cuh"

int main(int argc,char* argv[])
{
    constexpr int N = 256;
    using Model = Misp<N>;
    using Node = LightNode<Model::State, Model::Labels, N>;

    return run_bro_seq<Model,Node>(argc, argv);
}