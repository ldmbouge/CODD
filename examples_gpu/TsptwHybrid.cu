
#include "../examples/Tsptw.hpp"
#include "CoddHybrid.cuh"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 233;
    constexpr int Cities = 233;
    constexpr int Depth = 233;

    using Model = Tsptw<BranchFactor,Cities>;
    using Node = LNode<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runHybrid<Model,Node,BranchFactor,Depth>(argc, argv);
}