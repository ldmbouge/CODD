#pragma once

#include <GFL.hpp>
#include <type_traits>
#include <cmath>

#include "ArenaAllocator.hpp"
#include "BoundsUtils.hpp"

template<typename State, typename OutLabels,  int Depth>
class alignas(gfl::DefaultAlign) Node
{
    State state_;
    OutLabels outLabels{};
    gfl::f64 g_{0};
    gfl::f64 h_{0};
    gfl::u8 approximated_;
    gfl::u8 ancestorInCutset_{0};
    gfl::i16 pathLen{0};
    gfl::i16 prefixPath[Depth]{};

public:

    Node() = default;

    GFL_HOST_DEVICE
    Node(State const & s, gfl::f64 const g, gfl::f64 const h, gfl::i32 const label, Node const & pNode) noexcept :
       state_(s), g_(g), h_(h),
       approximated_(pNode.approximated_),
       ancestorInCutset_(pNode.ancestorInCutset_),
       pathLen(pNode.pathLen+1)
    {
        for (int i = 0; i < pNode.pathLen; i += 1)  prefixPath[i] = pNode.prefixPath[i];
        prefixPath[pNode.pathLen] = label;
    }

    GFL_HOST_DEVICE
    Node(State const & s, OutLabels const & outLabels, gfl::f64 const h) noexcept :
       state_(s), outLabels(outLabels), h_(h),approximated_(0)
    {}

    template<typename Model>
    static
    Node makeRoot(Model const * const model)
    {
        using namespace gfl;

        auto const state = model->initial();
        auto const labels = model->lgf(state, worst<Model>(), best<Model>(), DDExact);
        f64 const h =  Model::has_heur ? model->h(state, DDInit) : best<Model>();
        return Node(state, labels, h);
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
    gfl::i16 depth() const noexcept { return pathLen; }

    GFL_HOST_DEVICE
    bool approximated() const noexcept { return approximated_; }
    GFL_HOST_DEVICE
    void approximated(bool const approximated) noexcept { approximated_ = approximated; }

    GFL_HOST_DEVICE
    bool ancestorInCutset() const noexcept { return ancestorInCutset_; }
    GFL_HOST_DEVICE
    void ancestorInCutset(bool const ancestorInCutset)  noexcept { ancestorInCutset_ = ancestorInCutset; }

    GFL_HOST_DEVICE
    OutLabels const & labels() const noexcept { return outLabels; }
    GFL_HOST_DEVICE
    void labels(OutLabels const & labels) noexcept { outLabels = labels; }

    template<typename Model>
    GFL_HOST_DEVICE
    bool isTarget(Model const * const model) const
    {
        assert(model != nullptr);
        bool target = model->isTarget(state_);
        assert(not target or f() == g());
        return target;
    }


    gfl::ArrayView<gfl::i16 const> path() const noexcept
    {
        using namespace gfl;
        ArrayView<i16 const> const path(pathLen, prefixPath);
        return path;
    }
   friend std::ostream& operator<<(std::ostream& os,const Node& n) {
      return os << (n.approximated_ ? "1" : "0");
   }
    GFL_HOST
    static
    void print(Node const & node)
    {
        using namespace gfl;
        printf("F: %.1f", node.f());
        printf(" | "); printf("G: %.1f", node.g());
        printf(" | "); printf("H: %.1f", node.h());
        printf(" | "); printf("EXT: %d", 1 - node.approximated());
        printf(" | "); printf("CUT: %d", node.ancestorInCutset());
        printf(" | "); printf("DPT: %d", node.pathLen);
        printf(" | "); printf("LBS: "); ArrayView<i16>::print(node.prefixPath, node.pathLen);  // This is correct
        //printf(" | "); printf("ST: "); State::print(node.state_);
    }
};

struct NodeInfo
{
    union
    {
        gfl::i64 flag{0};
        gfl::u64 hash;
        gfl::f64 score;
    };
    gfl::i32 idx{0};
    gfl::i32 pIdx{0};

    NodeInfo() = default;

    GFL_HOST_DEVICE
    NodeInfo(gfl::i32 const idx, gfl::i32 const pIdx) noexcept :
        idx(idx),
        pIdx(pIdx)
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
        gfl::tuple<gfl::i64&> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.flag};}
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

    GFL_HOST_DEVICE
    static
    void print(NodeInfo const & ni)
    {
        using namespace gfl;
        printf("INFO: (%7lu,%7llu,%7.2f)", ni.flag, scast<llu>(ni.hash) % 10000000ll, std::fmod(ni.score,10000000.0));
        printf(" | "); printf("IDX: %lld", scast<lld>(ni.idx));
        printf(" | "); printf("PIDX: %lld", scast<lld>(ni.pIdx));
    }
};

static_assert(std::is_trivially_copyable_v<NodeInfo>);
