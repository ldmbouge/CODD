#pragma once

#include <cub/cub.cuh>

#include "Utils.hpp"
#include "Array.hpp"

template<typename State, typename Labels, int N>
struct alignas(16) LightNode
{
    State state;
    Labels labels;
    gfl::f64 gValue;
    gfl::f64 fValue;
    gfl::u8 isApproximated;
    gfl::u8 hasAncestorInCutset;
    gfl::i16 nEdgesSrcToNode;
    gfl::i16 labelsSrcToNode[N];

    GFL_HOST_DEVICE
    LightNode() noexcept {};

    GFL_HOST_DEVICE
    LightNode(State const & s, Labels const & l, gfl::f64 const fValue) noexcept :
            state(s),
            labels(l),
            fValue(fValue),
            isApproximated(0),
            hasAncestorInCutset(0),
            gValue(0),
            nEdgesSrcToNode(0)
    {}

    GFL_HOST_DEVICE
    void reset() noexcept {memset(this,0,sizeof(LightNode));}

    GFL_HOST_DEVICE static
   void print(LightNode const & node)
    {
        using namespace gfl;
        printMeta(node);
        printf(" | LBS: ");
        printLabels(node);
        printf("\n");
    }

    GFL_HOST static
    void printMeta(LightNode const & node)
    {
        using namespace gfl;
        std::cout << node.state << " | ";
        printf("F: %.1f | ", node.fValue);
        printf("G: %.1f | ", node.gValue);
        printf("EXT: %d | ", 1 - node.isApproximated);
        printf("AIC: %d | ", node.hasAncestorInCutset);
        printf("EDS: %d", node.nEdgesSrcToNode);
    }

    GFL_HOST_DEVICE static
    void printLabels(LightNode const & node)
    {
        using namespace gfl;
        Array<i16>::print(node.labelsSrcToNode, node.labelsSrcToNode + node.nEdgesSrcToNode);
    }

    template<typename Cmp>
    GFL_HOST_DEVICE
    constexpr
    static bool cmpByF(LightNode const & n1, LightNode const & n2)
    {
        return Cmp(n1.fValue,n2.fValue);
    }

    GFL_HOST_DEVICE
    constexpr
    bool isAncestorOf(std::vector<int> const & sol) const
    {
        bool isAncestor = true;
        int const l = gfl::min<int>(nEdgesSrcToNode, sol.size());
        for (int i = 0; i < l and isAncestor; i += 1)
        {
            isAncestor = isAncestor and sol[i] == labelsSrcToNode[i];
        }
        return isAncestor;
    }
};

struct NodeInfo
{
    union
    {
        gfl::u64 hash;
        gfl::u32 flag;
        gfl::f64 score;
    };
    gfl::i64 idx;
    gfl::i64 pIdx;


    GFL_HOST_DEVICE
    NodeInfo() noexcept {};

    GFL_HOST_DEVICE static
     void print(NodeInfo const & ni)
    {
        using namespace gfl;
        printf("H: %7llu | ", ni.hash % 10000000);
        printf("IDX: %7lld | ", ni.idx);
        printf("FLGS: %8d | ", ni.flag);
        printf("SCR: %7.5f", ni.score);
    }

    GFL_HOST_DEVICE
    constexpr
    static bool cmpByHash(NodeInfo const & n1, NodeInfo const & n2)
    {
        return n1.hash < n2.hash;
    }

    GFL_HOST_DEVICE
    constexpr
    static bool cmpByFlag(NodeInfo const & n1, NodeInfo const & n2)
    {
        return n1.flag < n2.flag;
    }

    GFL_HOST_DEVICE
    constexpr
    static bool cmpByScore(NodeInfo const & n1, NodeInfo const & n2)
    {
        return n1.score < n2.score;
    }
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
    LabelsInfo labelsInfoParents;
    LabelsInfo labelsInfoChildren;

    gfl::i64 nParents;
    Node * parents;

    gfl::i64 nChildren;
    Node * children;
    Node * tmpChildren;

    gfl::i64 nFlagged;
    NodeInfo * nodesInfo;
    NodeInfo * tmpNodesInfo;

    gfl::i64 cutsetSize;
    Node * cutset;

    gfl::i64 nChildrenToCopyPrefix;
    gfl::i64 nChildrenToCopySuffix;
    gfl::i64 nChildrenToMerge;

    std::size_t auxTmpMemSize;
    void * auxTmpMem;

    void reset()
    {
        labelsInfoParents.reset();
        labelsInfoChildren.reset();

        nParents = 0;
        parents = nullptr;

        nChildren = 0;

        children = nullptr;
        tmpChildren = nullptr;

        nFlagged = 0;
        nodesInfo = nullptr;
        tmpNodesInfo = nullptr;

        cutsetSize = 0;
        cutset = nullptr;

        auxTmpMemSize = 0;
        auxTmpMem = nullptr;
    }

    void swapParentsAndChildren()
    {
        nParents = nChildren;
        nChildren = 0;
        Node * tmpPtr = parents;
        parents = children;
        children = tmpPtr;
        nFlagged = 0;
        labelsInfoParents = labelsInfoChildren;
        labelsInfoChildren.reset();

    }

    BatchInfo() noexcept {reset();}

    void initParents(gfl::i64 nParents, gfl::StackAllocator * allocator) noexcept
    {
        initParents(nParents,nParents,allocator);
    }

    void initParents(gfl::i64 nParents, gfl::i64 bufferSize, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        assert(bufferSize >= nParents);

        this->nParents = nParents;
        parents = allocator->allocateArray<Node>(bufferSize);
    }

    void initChildren(gfl::i64 bufferSize, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        this->nChildren = 0;
        children = allocator->allocateArray<Node>(bufferSize);
        tmpChildren = allocator->allocateArray<Node>(bufferSize);

        nodesInfo = allocator->allocateArray<NodeInfo>(bufferSize);
        tmpNodesInfo =  allocator->allocateArray<NodeInfo>(bufferSize);
    }

    void initCutset(gfl::i64 bufferSize, gfl::StackAllocator * allocator) noexcept
    {
        using namespace gfl;

        this->cutsetSize = 0;
        cutset = allocator->allocateArray<Node>(bufferSize);
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
        memSize += 3 * (sizeof(Node) * nChildren + StackAllocator::DefaultAlign);
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
