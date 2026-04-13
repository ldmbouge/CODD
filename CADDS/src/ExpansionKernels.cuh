#pragma once

#include <GFL.hpp>

#include <cub/cub.cuh>

#include "Contexts.hpp"
#include "ExpansionFunctionsSeq.hpp"

template<typename T>
GFL_GLOBAL
void resizeToKernel(gfl::VectorView<T> * const v, gfl::i64 const size) {
  v->resizeTo(size);
}

template<typename T, typename I>
GFL_GLOBAL
void resizeToKernel(gfl::VectorView<T> * const v, I const * const size) {
  v->resizeTo(*size);
}

template<typename T, typename U>
GFL_GLOBAL
void resizeToKernel(gfl::VectorView<T> * const trg, gfl::VectorView<U> * const src) {
  trg->resizeTo(src->size());
}

GFL_GLOBAL
void resetInfoIdxKernel(gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;

  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo & info = nodesInfo->at(i);
    info.idx = scast<i32>(i);
  }
}

GFL_GLOBAL
void setFlagKernel(gfl::u8 const flag, gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;

  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo & info = nodesInfo->at(i);
    info.flag = flag;
  }
}

template<typename Model, typename Node>
GFL_GLOBAL
void expandParentsKernel(Model const * const model,
                         ExpansionData<Node> * const data,
                         gfl::f64 const primal,
                         gfl::u8 const flag) {
  using namespace gfl;

  auto const & parents = data->parents;
  auto & children = data->children;
  auto & childrenInfo = data->childrenInfo;
  i32 const branchingFactor = data->branchingFactor;

  i32 const pIdx = blockIdx.x;
  if (pIdx < parents.size()) {
    Node const & pNode = parents[pIdx];
    auto const & pOutLabels = pNode.outLabels();
    auto [minl, maxl, nlabels] = pOutLabels.summary();
    assert(nlabels <= branchingFactor);
    for (i32 label = minl + threadIdx.x; label <= maxl; label += blockDim.x) {
      if (pOutLabels.contains(label)) {
        auto const cState = model->stf(pNode.state(), label);
        if (cState.has_value()) {
          f64 const tCost = model->scf(pNode.state(), label);
          f64 const cG = pNode.g() + tCost;
          f64 cH = pNode.f() - cG;
          if constexpr (Model::has_heur) {
            f64 const h = model->h(cState.value(), BBCtx);
            cH = worse<Model>(cH, h);
          }
          if (isBetter<Model>(cG + cH, primal)) {
            i64 const offset = pIdx * branchingFactor + pOutLabels.rank(label);
            children[offset] = Node(cState.value(), cG, cH, label, pNode);
            childrenInfo[offset] = NodeInfo(offset, flag);
          }
        }
      }
    }
  }
}

template<typename T>
GFL_GLOBAL
void setValueKernel(T * const t, T const v) {
  *t = v;
}

GFL_GLOBAL
void countFlaggedKernel(gfl::u8 const flag, gfl::i32 * const count, gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;
  __shared__ i32 count_s;

  if (threadIdx.x == 0) {
    count_s = 0;
  }
  __syncthreads();
  i32 count_r = 0;
  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo const & info = nodesInfo->at(i);
    count_r += info.flag == flag;
  }
  if (count_r > 0) {
    atomicAdd_block(&count_s, count_r);
  }
  __syncthreads();

  if (threadIdx.x == 0 and count_s > 0) {
    atomicAdd(count, count_s);
  }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcHashKernel(gfl::ArrayView<Node> const * const nodes, gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;

  assert(nodes->size() >= nodesInfo->size());
  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo & info = nodesInfo->at(i);
    Node const & node = nodes->at(info.idx);
    if constexpr (Model::has_dom)
      info.hash = Model::domHash(node.state());
    else
      info.hash = Model::State::hash(node.state());
  }
}

template<typename KeyDecomposer, typename Buffer>
GFL_GLOBAL
void sortKernel(Buffer * const inBuffer,
                Buffer * const outBuffer,
                gfl::ArrayView<gfl::u8> const * const cubTmpBuffer,
                bool const reverse = false) {
  using namespace gfl;

  assert(cubTmpBuffer != nullptr);
  assert(inBuffer != nullptr);
  assert(outBuffer != nullptr);
  assert(inBuffer->size() <= outBuffer->size());

  auto tmpBufferMemSize = scast<usize>(cubTmpBuffer->dataMemSize());

  if (reverse)
    cub::DeviceRadixSort::SortKeysDescending(cubTmpBuffer->data(),
                                             tmpBufferMemSize,
                                             inBuffer->data(),
                                             outBuffer->data(),
                                             inBuffer->size(),
                                             KeyDecomposer{});
  else
    cub::DeviceRadixSort::SortKeys(cubTmpBuffer->data(),
                                   tmpBufferMemSize,
                                   inBuffer->data(),
                                   outBuffer->data(),
                                   inBuffer->size(),
                                   KeyDecomposer{});
}

template<typename T>
GFL_GLOBAL
void swapKernel(T * const a, T * const b) {
  T::swap(*a, *b);
}

template<typename Model, typename Node>
GFL_GLOBAL
void flagRepresentedKernel(gfl::i64 const flag,
                           gfl::ArrayView<Node> const * const nodes,
                           gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;
  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo & iInfo = nodesInfo->at(i);
    Node const & iNode = nodes->at(iInfo.idx);
    for (i64 j = i + 1; j < nodesInfo->size(); j += 1) {
      NodeInfo & jInfo = nodesInfo->at(j);
      Node const & jNode = nodes->at(jInfo.idx);
      if (iInfo.hash == jInfo.hash)
        flagRepresented<Model>(iInfo, jInfo, iNode, jNode, flag);
      else
        break;
    }
  }
}

template<typename Node>
GFL_GLOBAL
void copyByInfoIdxKernel(gfl::ArrayView<Node> const * const dst,
                         gfl::ArrayView<Node> const * const src,
                         gfl::ArrayView<NodeInfo> const * const nodesInfo) {
  using namespace gfl;

  assert(nodesInfo->size() <= src->size());
  assert(nodesInfo->size() <= dst->size());

  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    NodeInfo & info = nodesInfo->at(i);
    Node const & sNode = src->at(info.idx);
    Node & dNode = dst->at(i);
    dNode = sNode;
    info.idx = i;
  }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcOutLabelsKernel(Model const * const model,
                         gfl::ArrayView<Node> * const nodes,
                         gfl::i32 * branchingFactor,
                         gfl::f64 const primal,
                         gfl::f64 const dual,
                         DDContext const ddCtx) {
  using namespace gfl;

  __shared__ i32 branchingFactor_s;

  if (threadIdx.x == 0) {
    branchingFactor_s = 0;
  }
  __syncthreads();

  i32 branchingFactor_r = 0;
  auto [begin, end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodes->size());
  for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x) {
    Node & node = nodes->at(i);
    node.outLabels(model->lgf(node.state(), primal, dual, ddCtx));
    auto [minl, maxl, nlabels] = node.outLabels().summary();
    branchingFactor_r = max<i32>(branchingFactor_r, nlabels);
  }
  atomicMax_block(&branchingFactor_s, branchingFactor_r);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicMax(branchingFactor, branchingFactor_s);
  }
}

template<typename Model, typename Node>
GFL_GLOBAL
void copyBestTargetNodeKernel(Model const * const model, ExpansionData<Node> * const data) {
  using namespace gfl;

  assert(gridDim.x == 1);
  assert(blockDim.x == 1);

  auto & bestTargetNode = data->bestTargetNode;
  auto & children = data->children;

  bestTargetNode.reset();
  if (not children.empty() and model->isTarget(children.front().state())) {
    for (auto i  = 0; i < children.size(); i += 1) {
      Node const & node = children[i];
      if (not bestTargetNode.has_value() or isBetter<Model>(node.g(), bestTargetNode.value().g())) {
        bestTargetNode = node;
      }
    }
  }
}