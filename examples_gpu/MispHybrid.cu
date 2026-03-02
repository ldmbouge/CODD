
#include "../examples/Misp.hpp"
#include "CoddHybrid.cuh"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 2;
    constexpr int Items = 200;
    constexpr int Depth = 200;

    using Model = Misp<BranchFactor,Items>;
    using Node = SNode<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runHybrid<Model,Node,BranchFactor,Depth>(argc, argv);
}