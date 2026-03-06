#pragma once

#include <GFL.hpp>
#include <type_traits>
#include <cmath>

#include "ArenaAllocator.hpp"
#include "BoundsUtils.hpp"

template<typename State, typename OutLabels>
class alignas(gfl::DefaultAlign) Node
{
    State state_;
    OutLabels outLabels_;
    gfl::f64 g_;
    gfl::f64 h_;
    gfl::u8 approximated_;
    gfl::u8 ancestorInCutset_;
    gfl::i16 depth_;

public:
    GFL_HOST_DEVICE
    Node() noexcept {}

    GFL_HOST_DEVICE
    Node(Node const &) noexcept = default;

    GFL_HOST_DEVICE
    Node& operator=(Node const&) noexcept = default;

    GFL_HOST_DEVICE
    Node(State const & s,
        gfl::f64 const g, gfl::f64 const h,
        gfl::u8 const approximated = false,
        gfl::u8 const ancestorInCutset = false,
        gfl::i16 const depth = 0) noexcept :
            state_(s),
            g_(g), h_(h),
            approximated_(approximated),
            ancestorInCutset_(ancestorInCutset),
            depth_(depth)
    {}

    template<typename Model>
    static
    Node * makeRoot(Model const * const model)
    {
        using namespace gfl;

        auto const state = model->initial();
        f64 h = best<Model>();
        if constexpr (Model::has_heur) h = model->h(state, DDInit);
        Node * const n = new Node(state, 0, h);
        n->outLabels_= model->lgf(state, worst<Model>(), best<Model>(), DDExact);
        return n;
    }

    GFL_HOST_DEVICE
    State const & state() const noexcept { return state_; }
    GFL_HOST_DEVICE
    void state(State const & state) noexcept {state_ = state; }

    GFL_HOST_DEVICE
    gfl::f64 f() const noexcept { return g_ + h_; }

    GFL_HOST_DEVICE
    gfl::f64 g() const noexcept { return g_; }
    GFL_HOST_DEVICE
    void g(gfl::f64 const g) noexcept { g_ = g; }

    GFL_HOST_DEVICE
    gfl::f64 h() const noexcept { return h_; }
    GFL_HOST_DEVICE
    void h(gfl::f64 const h) noexcept { h_ = h; }

    GFL_HOST_DEVICE
    gfl::i16 depth() const noexcept { return depth_; }

    GFL_HOST_DEVICE
    bool approximated() const noexcept { return approximated_; }
    GFL_HOST_DEVICE
    void approximated(bool const approximated) noexcept { approximated_ = approximated; }

    GFL_HOST_DEVICE
    bool ancestorInCutset() const noexcept { return ancestorInCutset_; }
    GFL_HOST_DEVICE
    void ancestorInCutset(bool const ancestorInCutset)  noexcept { ancestorInCutset_ = ancestorInCutset; }

    GFL_HOST_DEVICE
    OutLabels const & outLabels() const noexcept { return outLabels_; }
    GFL_HOST_DEVICE
    void outLabels(OutLabels const & outLabels) noexcept { outLabels_ = outLabels; }

    template<typename Model>
    GFL_HOST_DEVICE
    bool isTarget(Model const * const model) const
    {
        assert(model != nullptr);
        bool target = model->isTarget(state_);
        return target;
    }

    GFL_HOST_DEVICE
    static
    void print(Node const & node)
    {
        using namespace gfl;
        printf("F: %.1f", node.f());
        printf(" | "); printf("G: %.1f", node.g());
        printf(" | "); printf("H: %.1f", node.h());
        printf(" | "); printf("EXT: %d", 1 - node.approximated());
        printf(" | "); printf("AIC: %d", node.ancestorInCutset());
        printf(" | "); printf("DPT: %d", node.depth());
        printf(" | "); node.state().print();

    }

    GFL_HOST_DEVICE
    void print() const
    { print(*this); }
};

template<typename State, typename OutLabels, int MaxDepth>
class LNode : public Node<State, OutLabels>
{
     gfl::u8 prefixLabels[MaxDepth];

    GFL_HOST_DEVICE
    LNode(Node<State, OutLabels> const & n) noexcept : Node<State, OutLabels>(n)
    {}

public:
    using BaseNode = Node<State, OutLabels>;

    GFL_HOST_DEVICE
    LNode() noexcept {}

    GFL_HOST_DEVICE
    LNode(State const& s,
          gfl::f64 const g, gfl::f64 const h,
          gfl::i32 const label,
          LNode const& pNode) noexcept :
        Node<State, OutLabels>(s, g, h, pNode.approximated(), pNode.ancestorInCutset(), pNode.depth()+1)
    {
        for (gfl::i16 i = 0; i < pNode.depth(); ++i) prefixLabels[i] = pNode.prefixLabels[i];
        prefixLabels[pNode.depth()] = label;
    }

    template<typename Model>
    static
    LNode * makeRoot(Model const * const model)
    {
        using namespace gfl;
        Node<State, OutLabels> const * n = Node<State, OutLabels>::makeRoot(model);
        return new LNode(*n);
    }

    void printSolution() const noexcept
    {
        using namespace gfl;
        ArrayView<u8>::print(prefixLabels, prefixLabels+ this->depth());
    }
};

template<typename State, typename OutLabels, int MaxChooses>
class SNode : public Node<State, OutLabels>
{
    gfl::BitSet<gfl::BitSet<>::num_words(MaxChooses)> chooses_;

    GFL_HOST_DEVICE
    SNode(Node<State, OutLabels> const & n) noexcept : Node<State, OutLabels>(n)
    {
        chooses_.clear();
    }

public:
    using BaseNode = Node<State, OutLabels>;

    SNode() noexcept {}

    GFL_HOST_DEVICE
    SNode(State const& s,
          gfl::f64 const g, gfl::f64 const h,
          gfl::i32 const label,
          SNode const& pNode) noexcept :
        Node<State, OutLabels>(s, g, h, pNode.approximated(), pNode.ancestorInCutset(), pNode.depth()+1)
    {
        using namespace gfl;

        assert(label == 0 or label == 1);
        chooses_ = pNode.chooses_;
        if (label == 1)
        {
            i32 const label_ = this->depth();
            assert(not chooses_.contains(label_));
            chooses_.insert(label_);
        }
    }

    template<typename Model>
    static
    SNode * makeRoot(Model const * const model)
    {
        using namespace gfl;
        Node<State, OutLabels> const * n = Node<State, OutLabels>::makeRoot(model);
        return new SNode(*n);
    }

    void printSolution() const noexcept
    {
        using namespace gfl;
       chooses_.printAsInts( this->depth());
    }
};

struct NodeInfo
{
    // Hold tight! This is brittle
    gfl::i64 idx{0};
    gfl::u8 flag;
    union
    {
        gfl::u64 hash;
        gfl::f64 score;
    };
    gfl::i64 pIdx;


    NodeInfo() = default;

    GFL_HOST_DEVICE
    NodeInfo(gfl::i32 const idx, gfl::i32 const pIdx, gfl::i8 const flag = 0) noexcept :
        idx(idx), pIdx(pIdx), flag(flag)
    {}

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByHash(NodeInfo const & n1, NodeInfo const & n2) {return n1.hash < n2.hash;}

#ifdef __CUDACC__
    struct HashDecomposer
    {
        GFL_HOST_DEVICE
        gfl::tuple<gfl::u64&> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.hash};}
    };
#endif

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByFlag(NodeInfo const & n1, NodeInfo const & n2) {return n1.flag < n2.flag;}

#ifdef __CUDACC__
    struct FlagDecomposer
    {
        GFL_HOST_DEVICE
        gfl::tuple<gfl::u8&> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.flag};}
    };
#endif

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByScore(NodeInfo const & n1, NodeInfo const & n2) {return n1.score < n2.score;}

#ifdef __CUDACC__
    struct ScoreDecomposer
    {
        GFL_HOST_DEVICE
        gfl::tuple<gfl::f64&> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.score};}
    };
#endif

#ifdef __CUDACC__
    struct ScoreFlagDecomposer
    {
        GFL_HOST_DEVICE
        gfl::tuple<gfl::f64&,gfl::u8&> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.score, nodeInfo.flag};}
    };
#endif

    GFL_HOST_DEVICE
    static
    void print(NodeInfo const & ni)
    {
        using namespace gfl;
        printf("IDX: %lld", scast<lld>(ni.idx));
        printf(" | "); printf("PIDX: %lld", scast<lld>(ni.pIdx));
        //printf(" | "); printf("INFO: (%7d,%7llu,%7.2f)", ni.flag, scast<llu>(ni.hash) % 10000000ll, fmod(ni.score,10000000.0));
    }

    GFL_HOST_DEVICE
    void print() const
    { print(*this);}
};



static_assert(std::is_trivially_copyable_v<NodeInfo>);
