
#include "../examples/Knapsack.hpp"
#include "CoddHybrid.cuh"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 2;
    constexpr int Items = 5010;
    constexpr int Depth = 5010;

    using Model = Knapsack<BranchFactor,Items>;
    using Node = SNode<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runHybrid<Model,Node,BranchFactor,Depth>(argc, argv);
}
