#include "bro_base.cuh"
#include "tsptw_1.cuh"

int main(int argc,char* argv[])
{
    using Model = TSPTW1<64>;
    using Node = LightNode<Model::State, Model::Labels, 64>;

    return run_bro<Model,Node>(argc, argv);
}