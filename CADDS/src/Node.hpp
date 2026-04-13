#pragma once

#include <GFL.hpp>

#include <BoundsUtils.hpp>
#include <Contexts.hpp>
#include <type_traits>

template<typename State, typename OutLabels>
class alignas(gfl::DefaultAlign) Node {
  State     _state;
  OutLabels _outLabels;
  gfl::f64  _g;
  gfl::f64  _h;
  gfl::i16  _depth;

public:
  Node() noexcept = default;

  GFL_HOST_DEVICE
  Node(State const & s, gfl::f64 const g, gfl::f64 const h, gfl::i16 const depth = 0) noexcept
      : _state(s), _g(g), _h(h), _depth(depth)
  {}

  template<typename Model>
  static Node * makeRoot(Model const * const model) {
    using namespace gfl;
    auto const state = model->initial();
    f64 h = best<Model>();
    if constexpr (Model::has_heur)
      h = model->h(state, DDInit);
    Node * const n = new Node(state, 0, h);
    n->_outLabels = model->lgf(state, worst<Model>(), best<Model>(), DDExact);
    return n;
  }

  GFL_HOST_DEVICE
  State const & state() const noexcept { return _state; }

  GFL_HOST_DEVICE
  void state(State const & state) noexcept { _state = state; }

  GFL_HOST_DEVICE
  gfl::f64 f() const noexcept { return _g + _h; }

  GFL_HOST_DEVICE
  gfl::f64 g() const noexcept { return _g; }

  GFL_HOST_DEVICE
  void g(gfl::f64 const g) noexcept { _g = g; }

  GFL_HOST_DEVICE
  gfl::f64 h() const noexcept { return _h; }

  GFL_HOST_DEVICE
  void h(gfl::f64 const h) noexcept { _h = h; }

  GFL_HOST_DEVICE
  gfl::i16 depth() const noexcept { return _depth; }

  GFL_HOST_DEVICE
  OutLabels const & outLabels() const noexcept { return _outLabels; }

  GFL_HOST_DEVICE
  void outLabels(OutLabels const & outLabels) noexcept { _outLabels = outLabels; }

  GFL_HOST_DEVICE static
  void print(Node const & node) {
    using namespace gfl;
    printf("G: %.1f", node.g());
    printf(" | ");
    printf("H: %.1f", node.h());
    printf(" | ");
    printf("DPT: %d", node.depth());
    printf(" | ");
    node.state().print();
  }

  GFL_HOST_DEVICE void print() const { print(*this); }
};

template<typename State, typename OutLabels, int MaxDepth>
class LNode : public Node<State, OutLabels> {
  gfl::u8 _prefixLabels[MaxDepth];

  GFL_HOST_DEVICE
  LNode(Node<State, OutLabels> const & n) noexcept : Node<State, OutLabels>(n) {}

public:
  using BaseNode = Node<State, OutLabels>;

  GFL_HOST_DEVICE
  LNode() noexcept = default;

  GFL_HOST_DEVICE
  LNode(State const & s, gfl::f64 const g, gfl::f64 const h, gfl::i32 const label, LNode const & pNode) noexcept
      : Node<State, OutLabels>(s, g, h, pNode.depth() + 1) {
    for (gfl::i16 i = 0; i < pNode.depth(); ++i) {
      _prefixLabels[i] = pNode._prefixLabels[i];
    }
    _prefixLabels[pNode.depth()] = label;
  }

  template<typename Model>
  static
  LNode * makeRoot(Model const * const model) {
    using namespace gfl;
    Node<State, OutLabels> const * n = Node<State, OutLabels>::makeRoot(model);
    return new LNode(*n);
  }

  gfl::ArrayView<gfl::u8 const> chooses() const noexcept {
    using namespace gfl;
    ArrayView<u8 const> const pLabels(&_prefixLabels[0], this->depth());
    return pLabels;
  }
};

struct NodeInfo {
  union {
    gfl::u64 hash;
    gfl::f64 score;
  };
  gfl::i32 idx;
  gfl::u8 flag;

  NodeInfo() noexcept = default;

  GFL_HOST_DEVICE
  NodeInfo(gfl::i32 const idx, gfl::i8 const flag = 0) noexcept : hash(0), idx(idx), flag(flag) {}

  GFL_HOST_DEVICE constexpr
  static bool cmpByHash(NodeInfo const & n1, NodeInfo const & n2) {
    return n1.hash < n2.hash;
  }

#ifdef __CUDACC__
  struct HashDecomposer {
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u64 &> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.hash}; }
  };
#endif

  GFL_HOST_DEVICE constexpr static
  bool cmpByFlag(NodeInfo const & n1, NodeInfo const & n2) {
    return n1.flag < n2.flag;
  }

#ifdef __CUDACC__
  struct FlagDecomposer {
    GFL_HOST_DEVICE
    gfl::tuple<gfl::u8 &> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.flag}; }
  };
#endif

  GFL_HOST_DEVICE constexpr static
  bool cmpByScore(NodeInfo const & n1, NodeInfo const & n2) {
    return n1.score < n2.score;
  }

#ifdef __CUDACC__
  struct ScoreDecomposer {
    GFL_HOST_DEVICE gfl::tuple<gfl::f64 &> operator()(NodeInfo & nodeInfo) const { return {nodeInfo.score}; }
  };
#endif
};

static_assert(std::is_trivially_copyable_v<NodeInfo>);