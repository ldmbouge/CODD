#include "bro_base.cuh"
#include "tsptw_1.cuh"

int main(int argc,char* argv[])
{
    constexpr int N = 64*1;
    using Model = TSPTW1<N>;
    using Node = LightNode<Model::State, Model::Labels,N>;

    return run_bro<Model,Node>(argc, argv);
}