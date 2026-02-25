
#include "Knapsack.hpp"
#include "CoddRelaxedSeq.hpp"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 2;
    constexpr int Items = 0;
    constexpr int Depth = 2000;

    using Model = Knapsack<BranchFactor,Items>;
    using Node = Node<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runRelaxedSeq<Model,Node,BranchFactor,Depth>(argc, argv);
}