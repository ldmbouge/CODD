#pragma once

#include "ExpansionData.hpp"
#include "ExpansionKernels.cuh"

template<typename Model, typename Node>
void expandParents(Model const * model,
                  gfl::f64 const primal,
                  ExpansionData<Node> * data,
                  gfl::i32 * flagged,
                  gfl::ArrayView<gfl::u8> const * const cubTmpBuffer) {
  using namespace gfl;
  auto & children = data->children;
  auto & childrenInfo = data->childrenInfo;
  auto & tmpInfo = data->tmpInfo;

  constexpr u8 IsNotChildren = 1;
  constexpr u8 IsChildren = 0;

  i32 const maxParents = data->parents.capacity();
  i32 const maxChildren = children.capacity();
  i32 const blockSize = 256;
  i32 const gridSizeParents = maxParents;
  i32 const gridSizeChildren = ceilDiv<i32>(maxChildren, blockSize);

  resizeToKernel<<<1, 1>>>(&children, maxChildren);
  CHECK_LAST_CUDA_ERROR();
  resizeToKernel<<<1, 1>>>(&childrenInfo, maxChildren);
  CHECK_LAST_CUDA_ERROR();
  resizeToKernel<<<1, 1>>>(&tmpInfo, maxChildren);
  CHECK_LAST_CUDA_ERROR();
  setFlagKernel<<<gridSizeChildren, blockSize>>>(IsNotChildren, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  expandParentsKernel<<<gridSizeParents, blockSize>>>(model, data, primal, IsChildren);
  CHECK_LAST_CUDA_ERROR();
  setValueKernel<<<1, 1>>>(flagged, 0);
  CHECK_LAST_CUDA_ERROR();
  countFlaggedKernel<<<gridSizeChildren, blockSize>>>(IsChildren, flagged, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  sortKernel<NodeInfo::FlagDecomposer><<<1, 1>>>(&childrenInfo, &tmpInfo, cubTmpBuffer);
  CHECK_LAST_CUDA_ERROR();
  swapKernel<<<1, 1>>>(&childrenInfo, &tmpInfo);
  CHECK_LAST_CUDA_ERROR();
  resizeToKernel<<<1, 1>>>(&childrenInfo, flagged);
  CHECK_LAST_CUDA_ERROR();
}

template<typename Model, typename Node>
void filterRepresentedChildren(ExpansionData<Node> * data,
                               gfl::i32 * flagged,
                               gfl::ArrayView<gfl::u8> * const cubTmpBuffer) {
  using namespace gfl;
  auto & children = data->children;
  auto & childrenInfo = data->childrenInfo;
  auto & tmpInfo = data->tmpInfo;

  constexpr u8 IsRepresented = 1;
  constexpr u8 IsNotRepresented = 0;

  i32 const maxChildren = children.capacity();
  i32 const blockSize = 256;
  i32 const gridSize = ceilDiv<i32>(maxChildren, blockSize);

  resizeToKernel<<<1, 1>>>(&tmpInfo, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  calcHashKernel<Model><<<gridSize, blockSize>>>(&children, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  sortKernel<NodeInfo::HashDecomposer><<<1, 1>>>(&childrenInfo, &tmpInfo, cubTmpBuffer);
  CHECK_LAST_CUDA_ERROR();
  swapKernel<<<1, 1>>>(&childrenInfo, &tmpInfo);
  CHECK_LAST_CUDA_ERROR();
  setFlagKernel<<<gridSize, blockSize>>>(IsNotRepresented, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  flagRepresentedKernel<Model><<<gridSize, blockSize>>>(IsRepresented, &children, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  setValueKernel<<<1, 1>>>(flagged, 0);
  CHECK_LAST_CUDA_ERROR();
  countFlaggedKernel<<<gridSize, blockSize>>>(IsNotRepresented, flagged, &childrenInfo);
  CHECK_LAST_CUDA_ERROR();
  sortKernel<NodeInfo::FlagDecomposer><<<1, 1>>>(&childrenInfo, &tmpInfo, cubTmpBuffer);
  CHECK_LAST_CUDA_ERROR();
  swapKernel<<<1, 1>>>(&childrenInfo, &tmpInfo);
  CHECK_LAST_CUDA_ERROR();
  resizeToKernel<<<1, 1>>>(&childrenInfo, flagged);
  CHECK_LAST_CUDA_ERROR();
}