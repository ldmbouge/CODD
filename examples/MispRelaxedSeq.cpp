
#include "Misp.hpp"
#include "CoddRelaxedSeq.hpp"

int main(int argc,char* argv[])
{
    constexpr int BranchFactor = 2;
    constexpr int Depth = 200;

    using Model = Misp<BranchFactor>;
    using Node = Node<Model::State,Model::OutLabels,Depth>;

    static_assert(IsModel<Model>);
    static_assert(std::is_trivially_copyable_v<Node>);

    return runRelaxedSeq<Model,Node,BranchFactor,Depth>(argc, argv);
}