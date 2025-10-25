#include "bro_base.cuh"
#include "gruler.cuh"

int main(int argc,char* argv[])
{
    using Model = GRuler<256>;
    using Node = LightNode<Model::State, Model::Labels, 32>;

    return run_bro<Model,Node>(argc, argv);
}