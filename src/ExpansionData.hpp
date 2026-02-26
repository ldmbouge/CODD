#pragma once

#include "GFL.hpp"
#include <Node.hpp>

template<typename Node>
struct ExpansionData
{

    gfl::optional<Node> bestTargetNode;
    gfl::optional<Node> bestExactTargetNode;

    gfl::VectorView<Node> parents{};
    gfl::VectorView<Node> children{};

    gfl::VectorView<NodeInfo> childrenInfo{};
    gfl::VectorView<NodeInfo> parentInfo{};

    gfl::ArrayView<Node> tmpView;
    gfl::ArrayView<NodeInfo> tmpInfoView;
    gfl::VectorView<Node> tmpNodes{};
    gfl::VectorView<NodeInfo> tmpInfo{};

    void init(gfl::i32 const nNodes, gfl::ArenaAllocator & alloc) noexcept
    {
        using namespace gfl;

        parents = VectorView<Node>(nNodes, alloc);
        parentInfo = VectorView<NodeInfo>(nNodes, alloc);

        children = VectorView<Node>(nNodes, alloc);
        childrenInfo = VectorView<NodeInfo>(nNodes, alloc);

        tmpNodes = VectorView<Node>(nNodes, alloc);
        tmpInfo = VectorView<NodeInfo>(nNodes, alloc);
    }

    void init(gfl::ArenaAllocator & alloc, gfl::i32 const nParents, gfl::i32 const maxBranchFactor) noexcept
    {
        using namespace gfl;

        i32 const nChildren = nParents * maxBranchFactor;

        parents = VectorView<Node>(nParents,alloc);
        parentInfo = VectorView<NodeInfo>(nParents, alloc);

        children = VectorView<Node>(nChildren, alloc);
        childrenInfo = VectorView<NodeInfo>(nChildren,alloc);

        tmpNodes = VectorView<Node>(nChildren,alloc);
        tmpInfo = VectorView<NodeInfo>(nChildren,alloc);
    }

    void clear() noexcept
    {
        bestTargetNode.reset();
        bestExactTargetNode.reset();
        
        parents.clear();
        parentInfo.clear();

        children.clear();
        childrenInfo.clear();

        tmpNodes.clear();
        tmpInfo.clear();
    }

    void addToParents(Node const & node) noexcept
    {
        parents.pushBack(node);
    }

    GFL_HOST_DEVICE
    void swapParentsAndChildren()
    {
        using namespace gfl;

        VectorView<Node>::swap(parents,children);
        VectorView<NodeInfo>::swap(parentInfo,childrenInfo);
        children.clear();
        childrenInfo.clear();
        tmpNodes.clear();
        tmpInfo.clear();
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
        memSize += VectorView<NodeInfo>::dataMemSize(nParents) + DefaultAlign; // parentsInfo
        memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // children
        memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // childrenInfo
        memSize += VectorView<Node>::dataMemSize(nChildren) + DefaultAlign; // tmpNodes
        memSize += VectorView<NodeInfo>::dataMemSize(nChildren) + DefaultAlign; // tmpInfo

        return memSize;
    }
};
