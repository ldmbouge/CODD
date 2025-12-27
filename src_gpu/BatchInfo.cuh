#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "Array.hpp"
template<typename State, typename Labels, int N>
struct alignas(16) LightNode
{
    State state;
    Labels labels;
    gfl::f64 boundSrcToNode;
    gfl::f64 dualBound;
    gfl::u8 isNotExact;
    gfl::u8 nEdgesSrcToNode;
    gfl::u8 labelsSrcToNode[N];

    GFL_HOST_DEVICE
    LightNode() noexcept {};

    GFL_HOST_DEVICE
    LightNode(State const & s, Labels const & l) noexcept :
            state(s),
            labels(l),
            isNotExact(0),
            boundSrcToNode(0),
            nEdgesSrcToNode(0)
    {}

    GFL_HOST_DEVICE
    void reset() noexcept {memset(this,0,sizeof(LightNode));}

    template<typename Node>
    GFL_HOST_DEVICE static
    void print(Node const & node)
    {
        using namespace gfl;
        printf("S2N: %.1f | ", node.boundSrcToNode);
        printf("DBND: %.1f | ",node.dualBound);
        printf("EXCT: %d | ", 1-node.isNotExact);
        printf("EDGS: %d", node.nEdgesSrcToNode);
    }


    template<typename Node>
    GFL_HOST_DEVICE static
    void printLabels(Node const & node)
    {
        using namespace gfl;
        Array<u8>::print(node.labelsSrcToNode, node.labelsSrcToNode + node.nEdgesSrcToNode);
    }
};

struct NodeInfo
{
    gfl::u64 hash;
    gfl::i64 idx;
    gfl::u32 flag;
    gfl::f64 score;

    GFL_HOST_DEVICE
    NodeInfo() noexcept {};
};

struct DummyDecomposer128
{
    GFL_HOST_DEVICE
    gfl::tuple<gfl::f64&, gfl::f64&> operator()(NodeInfo &) const
    {
        gfl::f64 tmp = 0;
        return {tmp,tmp};
    }
};

struct LabelsInfo
{
    gfl::i32 minLabel;
    gfl::i32 maxLabel;
    gfl::i32 nLabels;

    LabelsInfo(gfl::backend::tuple<gfl::i32,gfl::i32,gfl::i32> const & t) noexcept
    {
        reset();
        update(t);
    };

    GFL_HOST_DEVICE
    void reset()
    {
        minLabel = gfl::numeric_limits<gfl::i32>::max();
        maxLabel = gfl::numeric_limits<gfl::i32>::min();
        nLabels = 0;
    }

    void update(LabelsInfo const & li)
    {
        update(li.minLabel,li.maxLabel, li.nLabels);
    }

    void update(gfl::backend::tuple<gfl::i32, gfl::i32,gfl::i32> t)
    {
        update(gfl::backend::get<0>(t),gfl::backend::get<1>(t), gfl::backend::get<2>(t));
    }

    void update(gfl::i32 const minl, gfl::i32 const maxl, gfl::i32 const nl)
    {
        minLabel = gfl::min<gfl::i32>(minLabel, minl);
        maxLabel = gfl::max<gfl::i32>(maxLabel, maxl);
        nLabels = gfl::max<gfl::i32>(nLabels, nl);
    }

    LabelsInfo() noexcept {reset();}
};

template<typename Node>
struct BatchInfo
{
    LabelsInfo labelsInfo;

    gfl::i64 nParents;
    Node * parents;
    Node * tmpParents;

    gfl::i64 nChildren;
    Node * children;
    Node * tmpChildren;
    gfl::i64 nFlagged;
    NodeInfo * childrenInfo;
    NodeInfo * tmpChildrenInfo;

    std::size_t auxTmpMemSize;
    void * auxTmpMem;

    void reset()
    {
        labelsInfo.reset();

        nParents = 0;
        parents = nullptr;
        tmpParents = nullptr;

        nChildren = 0;
        children = nullptr;
        tmpChildren = nullptr;
        nFlagged = 0;
        childrenInfo = nullptr;
        tmpChildrenInfo = nullptr;

        auxTmpMemSize = 0;
        auxTmpMem = nullptr;
    }

    BatchInfo() noexcept {reset();}

    void initParents(gfl::i64 nParents, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        this->nParents = nParents;
        parents = allocator->allocateArray<Node>(nParents);
    }

    void useChildrenAsParents(gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        i64 const nChildren = nParents * labelsInfo.nLabels;

        children = allocator->allocateArray<Node>(nChildren);
        tmpChildren = allocator->allocateArray<Node>(nChildren);

        childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
        tmpChildrenInfo =  allocator->allocateArray<NodeInfo>(nChildren);
    }

    void initChildren(gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        i64 const nChildren = nParents * labelsInfo.nLabels;

        children = allocator->allocateArray<Node>(nChildren);
        tmpChildren = allocator->allocateArray<Node>(nChildren);

        childrenInfo = allocator->allocateArray<NodeInfo>(nChildren);
        tmpChildrenInfo =  allocator->allocateArray<NodeInfo>(nChildren);
    }

    void initAux(gfl::i64 const memSize, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        auxTmpMemSize = memSize;
        auxTmpMem = allocator->allocate<u8>(memSize, 16);
    }

    static
    gfl::i64 calcMemSize(gfl::i64 const nParents, gfl::i32 const branchingFactor, bool auxBuffers)
    {
        using namespace gfl;

        i64 memSize = 0;

        // initParents()
        memSize += sizeof(Node) * nParents + StackAllocator::DefaultAlign;

        // initChildren()
        i64 const nChildren = nParents * branchingFactor;
        memSize += 2 * (sizeof(Node) * nChildren + StackAllocator::DefaultAlign);
        memSize += 2 * (sizeof(NodeInfo) * nChildren + StackAllocator::DefaultAlign);

        // initAux()
        if (auxBuffers)
        {


            std::size_t dummyMemSize = 0;
            void * dummyTmpMem = nullptr;
            NodeInfo * dummyChildrenInfo = nullptr;
            cub::DeviceRadixSort::SortKeys( // Initialize cubTmpMemSize
                    dummyTmpMem,
                    dummyMemSize,
                    dummyChildrenInfo,
                    dummyChildrenInfo,
                    nChildren,
                    DummyDecomposer128{}); // Bigger key used
            CHECK_LAST_CUDA_ERROR();
            memSize += dummyMemSize;
        }
        return memSize;
    }

    static
    gfl::i64 calcMaxParents(gfl::i32 const branchingFactor, gfl::i64 const maxMemSize, bool auxBuffers = true)
    {
        using namespace gfl;

        // Binary search on the number of parents
        i64 lbParents = 0;
        i64 ubParents = 1;

        while (calcMemSize(ubParents, branchingFactor, auxBuffers) <= maxMemSize)
        {
            lbParents = ubParents;
            ubParents *= 2;
        }

        while (lbParents < ubParents)
        {
            i64 const mid = lbParents + (ubParents - lbParents + 1) / 2;
            i64 const memSize = calcMemSize(mid, branchingFactor, auxBuffers);
            if (memSize <= maxMemSize) lbParents = mid;  // still fits
            else ubParents = mid - 1; // too big
        }
        return lbParents;
    }
};
