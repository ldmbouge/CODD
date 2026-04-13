#pragma once

#include "ExpansionEngine.hpp"
#include "GFL.hpp"

#include "ExpansionFunctionsGpu.cuh"
#include "Sort.hpp"

template<typename Model, typename Node>
class ExpansionEngineGpu : public ExpansionEngine<Node> {
  using ExpansionEngine<Node>::_data;
  gfl::i32 _flagged = 0;
  gfl::ArrayView<gfl::u8> _cubTmpBuffer;

  static
  gfl::i64 cubTmpMemSize(gfl::i64 const nNodes) {
    using namespace gfl;
    usize memSize = 0;
    void * dummyTmpMem = nullptr;
    NodeInfo * dummyNodeInfo = nullptr;
    cub::DeviceRadixSort::SortKeys(
      dummyTmpMem,
      memSize,
      dummyNodeInfo,
      dummyNodeInfo,
      nNodes,
      gfl::DummyDecomposer64<NodeInfo>{});
    CHECK_LAST_CUDA_ERROR();
    return scast<i64>(memSize);
  }

public:
  void allocData(gfl::ArenaAllocator & alloc, gfl::i32 const nodes, gfl::i32 const maxBranchFactor) {
    using namespace gfl;
    i32 const nParents = nodes;
    i32 const nChildren = nParents * maxBranchFactor;
    _data.init(nParents, nChildren, alloc);
    _cubTmpBuffer = gfl::ArrayView<u8>(cubTmpMemSize(nChildren), alloc);
  }

  static
  gfl::u64 getBatchSize(gfl::i64 const maxMemSize, gfl::i32 const branchingFactor) {
    using namespace gfl;
    i64 lbNodes = 0;
    i64 ubNodes = 1;

    auto const calcMemSize = [branchingFactor](i32 const nodes) {
      i32 const nParents = nodes;
      i32 const nChildren = nParents * branchingFactor;
      i64 const dms = ExpansionData<Node>::dataMemSize(nParents, nChildren) + DefaultAlign;
      i64 const ctms = cubTmpMemSize(nChildren) + DefaultAlign;
      return dms + ctms;
    };

    while (calcMemSize(ubNodes) <= maxMemSize) {
      lbNodes = ubNodes;
      ubNodes *= 2;
    }

    while (lbNodes < ubNodes) {
      i64 const mid = lbNodes + (ubNodes - lbNodes + 1) / 2;
      i64 const memSize = calcMemSize(mid);
      if (memSize <= maxMemSize)
        lbNodes = mid;
      else
        ubNodes = mid - 1;
    }
    return lbNodes;
  }

  void expand(
    Model const * model,
    gfl::ArrayView<Node> const & parents,
    gfl::i32 branchingFactor,
    gfl::f64 const primal) {
    using namespace gfl;
    i32 const blockSize = 256;
    i32 const cGridSize = ceilDiv<i32>(_data.children.capacity(), blockSize);

    _data.clear();
    _data.branchingFactor = branchingFactor;
    _data.parents.pushBackToGpuAsync(parents);

    expandParents(model, primal, &_data, &_flagged, &_cubTmpBuffer);
    filterRepresentedChildren<Model>(&_data, &_flagged, &_cubTmpBuffer);
    resizeToKernel<<<1, 1>>>(&_data.tmpNodes, &_data.childrenInfo);
    CHECK_LAST_CUDA_ERROR();
    copyByInfoIdxKernel<<<cGridSize, blockSize>>>(&_data.tmpNodes, &_data.children, &_data.childrenInfo);
    CHECK_LAST_CUDA_ERROR();
    swapKernel<<<1, 1>>>(&_data.tmpNodes, &_data.children);
    CHECK_LAST_CUDA_ERROR();
    resizeToKernel<<<1, 1>>>(&_data.children, &_data.childrenInfo);
    CHECK_LAST_CUDA_ERROR();
    setValueKernel<<<1, 1>>>(&_data.branchingFactor, 0);
    CHECK_LAST_CUDA_ERROR();
    calcOutLabelsKernel<<<cGridSize, blockSize>>>(
      model,
      &_data.children,
      &_data.branchingFactor,
      primal,
      best<Model>(),
      DDRestricted);
    CHECK_LAST_CUDA_ERROR();
    copyBestTargetNodeKernel<Model,Node><<<1, 1>>>(model, &_data);
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  }
};