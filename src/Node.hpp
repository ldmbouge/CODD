#pragma once

#include <GFL.hpp>
#include <type_traits>
#include <cmath>

#include "ArenaAllocator.hpp"
#include "BoundsUtils.hpp"

template<typename State, typename OutLabels,  int Depth>
class alignas(gfl::DefaultAlign) Node
{
    State state_{};
    OutLabels outLabels{};
    gfl::f64 g_{0};
    gfl::f64 h_{0};
    gfl::u8 approximated_{0};
    gfl::u8 ancestorInCutset_{0};
    gfl::i16 pathLen{0};
    gfl::i16 prefixPath[Depth]{};

    Node() = delete;

public:
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
        state_(s), outLabels(outLabels), h_(h)
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

    State const & state() const noexcept { return state_; }
    void state(State const & state) noexcept {state_ = state; }

    gfl::f64 f() const noexcept { return g_ + h_; }

    gfl::f64 g() const noexcept { return g_; }
    void g(gfl::f64 const g) noexcept { g_ = g; }

    gfl::f64 h() const noexcept { return h_; }
    void h(gfl::f64 const h) noexcept { h_ = h; }

    gfl::i16 depth() const noexcept { return pathLen; }

    bool approximated() const noexcept { return approximated_; }
    void approximated(bool const approximated) noexcept { approximated_ = approximated; }

    bool ancestorInCutset() const noexcept { return ancestorInCutset_; }
    void ancestorInCutset(bool const ancestorInCutset)  noexcept { ancestorInCutset_ = ancestorInCutset; }

    OutLabels const & labels() const noexcept { return outLabels; }
    void labels(OutLabels const & labels) noexcept { outLabels = labels; }

    template<typename Model>
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
        printf(" | "); printf("ST: "); State::print(node.state_);
    }
};

struct NodeInfo
{
    union
    {
        gfl::u64 flag{0};
        gfl::u64 hash;
        gfl::f64 score;
    };
    gfl::i32 idx{0};
    gfl::i32 pIdx{0};

    NodeInfo() = default;

    NodeInfo(gfl::i32 const idx, gfl::i32 const pIdx) noexcept :
        idx(idx),
        pIdx(pIdx)
    {}

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByHash(NodeInfo const & n1, NodeInfo const & n2) {return n1.hash < n2.hash;}

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByFlag(NodeInfo const & n1, NodeInfo const & n2) {return n1.flag < n2.flag;}

    GFL_HOST_DEVICE
    constexpr static
    bool cmpByScore(NodeInfo const & n1, NodeInfo const & n2) {return n1.score < n2.score;}

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