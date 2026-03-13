
#include "../examples/GRuler.hpp"
#include "CoddHybrid.cuh"

int main(int argc,char* argv[])
{
    // Based on 16
    constexpr int BranchFactor = 180;
    constexpr int Items = 256;
    constexpr int Depth = 16;

    using Model = GRuler<BranchFactor,Items>;
    using Node = LNode<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runHybrid<Model,Node,BranchFactor,Depth>(argc, argv);
}