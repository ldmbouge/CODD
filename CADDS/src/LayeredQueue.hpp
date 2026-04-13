#pragma once

#include <GFL.hpp>

template<typename Node>
class LayeredQueue {
  using LayerType = std::vector<Node>;
  std::vector<LayerType> _layers;
  std::vector<gfl::i32> _branchingFactors;
  gfl::i64 _pushed;
  gfl::i64 _pulled;

  LayerType & getLayer(gfl::i32 const lIdx) noexcept {
    using namespace gfl;
    assert(lIdx >= 0);
    while (lIdx >= scast<i32>(_layers.size())) {
      _layers.emplace_back();
      _branchingFactors.push_back(0);
    }
    return _layers[lIdx];
  }

public:
  LayeredQueue() noexcept : _layers(), _branchingFactors(), _pushed(0), _pulled(0) {}

  gfl::i64 pushed() const noexcept { return _pushed; }
  gfl::i64 pulled() const noexcept { return _pulled; }
  gfl::i64 size() const noexcept { return _pushed - _pulled; }
  bool empty() const noexcept { return _pulled == _pushed; }

  gfl::i32 deepestNotEmpty() const noexcept {
    using namespace gfl;
    for (i32 i = scast<i32>(_layers.size() - 1); i >= 0; i -= 1) {
      if (not _layers[i].empty())
        return i;
    }
    assert(false);
    return -1;
  }

  gfl::i32 & getBranchingFactor(gfl::i32 const lIdx) noexcept {
    using namespace gfl;
    assert(lIdx >= 0);
    while (lIdx >= scast<i32>(_layers.size())) {
      _layers.emplace_back();
      _branchingFactors.push_back(0);
    }
    return _branchingFactors[lIdx];
  }

  gfl::ArrayView<Node> pullUpTo(gfl::i32 const lIdx, gfl::i32 const count) {
    using namespace gfl;
    assert(count >= 0);
    auto & layer = getLayer(lIdx);
    i32 const toPull = min<i32>(count, layer.size());
    ArrayView<Node> nodes(layer.data() + layer.size() - toPull, toPull);
    layer.resize(layer.size() - toPull);
    _pulled += toPull;
    return nodes;
  }

  void push(gfl::i32 const lIdx, gfl::i32 const branchingFactor, gfl::ArrayView<Node> const & nodes) noexcept {
    using namespace gfl;
    auto & bf = getBranchingFactor(lIdx);
    bf = max<i32>(bf, branchingFactor);
    auto & layer = getLayer(lIdx);
    i32 const oldSize = layer.size();
    layer.resize(layer.size() + nodes.size());
    std::memcpy(layer.data() + oldSize, nodes.data(), nodes.dataMemSize());
    _pushed += nodes.size();
  }

  void push(gfl::i32 const lIdx, gfl::i32 const branchingFactor, Node const * const node) noexcept {
    using namespace gfl;
    ArrayView<Node> nodes(const_cast<Node*>(node), 1);
    push(lIdx, branchingFactor, nodes);
  }

#ifdef __CUDACC__
  void pushFromGpu(gfl::i32 const lIdx, gfl::i32 const branchingFactor, gfl::ArrayView<Node> const & nodes) noexcept {
    using namespace gfl;
    auto & bf = getBranchingFactor(lIdx);
    bf = max<i32>(bf, branchingFactor);
    auto & layer = getLayer(lIdx);
    i32 const oldSize = layer.size();
    layer.resize(layer.size() + nodes.size());
    cudaMemcpy(layer.data() + oldSize, nodes.data(), nodes.dataMemSize(), cudaMemcpyDeviceToHost);
    _pushed += nodes.size();
  }
#endif
};