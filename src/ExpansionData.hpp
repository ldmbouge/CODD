#pragma once

#include "GFL.hpp"
#include <Node.hpp>

template<typename Node>
struct ExpansionData
{
    gfl::VectorView<Node> parents{};
    gfl::VectorView<Node> children{};
    gfl::VectorView<NodeInfo> nodesInfo{};

    gfl::VectorView<Node> tmpNodes{};
    gfl::VectorView<NodeInfo> tmpNodesInfo{};


    void init(gfl::i32 const nNodes, gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;

        parents = VectorView<Node>(nNodes, alloc);
        children = VectorView<Node>(nNodes, alloc);
        nodesInfo = VectorView<NodeInfo>(nNodes, alloc);

        tmpNodes = VectorView<Node>(nNodes, alloc);
        tmpNodesInfo = VectorView<NodeInfo>(nNodes, alloc);
    }

    void init(gfl::ArenaAllocator & alloc, gfl::i32 const nParents, gfl::i32 const maxBranchFactor) noexcept
    {
        using namespace gfl;

        i32 const nChildren = nParents * maxBranchFactor;

        parents = VectorView<Node>(nParents,alloc);
        children = VectorView<Node>(nChildren, alloc);
        nodesInfo = VectorView<NodeInfo>(nChildren,alloc);

        tmpNodes = VectorView<Node>(nChildren,alloc);
        tmpNodesInfo = VectorView<NodeInfo>(nChildren,alloc);
    }

    void clear() noexcept
    {
        parents.clear();
        children.clear();
        nodesInfo.clear();

        tmpNodes.clear();
        tmpNodesInfo.clear();
    }

    void addToParents(Node const & node) noexcept
    {
        parents.pushBack(node);
    }

    void swapParentsAndChildren()
    {
        using namespace gfl;

        VectorView<Node>::swap(parents,children);
        nodesInfo.clear();
        tmpNodes.clear();
        tmpNodesInfo.clear();
    }

    bool hasTarget() const noexcept
    {
        return children.size() == 1;
    }

    Node const & getTarget() const noexcept
    {
        return children.front();
    }

    static
    gfl::i64 dataMemSize(gfl::i32 const nParents, gfl::i32 const maxBranchFactor)
    {
        using namespace gfl;

        i32 const nChildren = nParents * maxBranchFactor;

        i64 memSize = 0;
        memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // parents
        memSize += VectorView<Node>::dataMemSize(nParents) + DefaultAlign; // children
        memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
        memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // auxNodesInfo
        memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // tmpNodes
        memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // tmpNodesInfo

        return memSize;
    }
};
