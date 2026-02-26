
#include "../examples/GRuler.hpp"
#include "CoddRelaxedGpu.cuh"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 128;
    constexpr int Items = 128;
    constexpr int Depth = 15;

    using Model = GRuler<BranchFactor,Items>;
    using Node = Node<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runRelaxedGpu<Model,Node,BranchFactor,Depth>(argc, argv);
}