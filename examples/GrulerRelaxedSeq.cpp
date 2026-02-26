
#include "GRuler.hpp"
#include "CoddRelaxedSeq.hpp"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 128;
    constexpr int Marks = 128;
    constexpr int Depth = 15;

    using Model = GRuler<BranchFactor,Marks>;
    using Node = Node<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runRelaxedSeq<Model,Node,BranchFactor,Depth>(argc, argv);
}