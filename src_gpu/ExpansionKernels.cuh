#pragma once

#include <GFL.hpp>

#include "Contexts.hpp"
#include "ExpansionData.hpp"

template<typename T>
GFL_GLOBAL
void resizeToKernel( gfl::VectorView<T> * const v, gfl::i64 const count)
{
    v->resizeTo(count);
}

template<typename T>
GFL_GLOBAL
void resizeToKernel(gfl::VectorView<T> * const v, gfl::i64 const * const countPtr)
{
    v->resizeTo(*countPtr);
}

template<typename T>
GFL_GLOBAL
void shrinkToKernel( gfl::VectorView<T> * const v, gfl::i64 const count)
{
    using namespace gfl;
    v->resizeTo(min<i64>(count, v->size()));
}

GFL_GLOBAL
void resetInfoIdxKernel(gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        info.idx = scast<i32>(i);
    }
}

template<typename Node>
GFL_GLOBAL
void swapParentsAndChildrenKernel(ExpansionData<Node> * const expData)
{

    expData->swapParentsAndChildren();
}

template<typename T>
GFL_GLOBAL
void setValueKernel( T * const t, T const v)
{
    *t = v;
}

template<typename T>
GFL_GLOBAL
void copyValueKernel( T * const t, T const * const v)
{
    *t = *v;
}

template<typename Model, typename Node>
GFL_GLOBAL
void expandParentsKernel(
    Model const * const model,
    ExpansionData<Node> * const expData,
    gfl::f64 const primal,
    gfl::f64 const flag,
    gfl::i32 const branchFactor)
{
    using namespace gfl;

    auto const & parents    = expData->parents;
    auto const & parentsInfo = expData->parentInfo;
    auto & children         = expData->children;
    auto & childrenInfo     = expData->childrenInfo;

    i32 const pIdx = blockIdx.x;
    if (pIdx < parentsInfo.size())
    {
        Node const & pNode = parents[pIdx];
        auto const & pOutLabels = pNode.outLabels();
        auto [minl, maxl, nlabels] = pOutLabels.summary();
        assert(nlabels <= branchFactor);

        for(i32 label = minl + threadIdx.x; label <= maxl; label += blockDim.x)
        {
            if (pOutLabels.contains(label))
            {
                auto const cState = model->stf(pNode.state(), label);
                if (cState.has_value())
                {
                    f64 const tCost = model->scf(pNode.state(), label);
                    f64 const cG    = pNode.g() + tCost;
                    f64 cH          = pNode.f() - cG;
                    if constexpr (Model::has_heur)
                    {
                        f64 const h = model->h(cState.value(), BBCtx);
                        cH = worse<Model>(cH, h);
                    }
                    if (isBetter<Model>(cG + cH, primal))
                    {
                        i64 const offset = pIdx * branchFactor + (label - minl);
                        children[offset] = Node(cState.value(), cG, cH, label, pNode);
                        childrenInfo[offset] = NodeInfo(offset, pIdx,flag);
                    }
                    else
                    {
                        //printf("%.2f beat by primal %.2f\n",cG + cH, primal);
                    }

                }
            }
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcHashKernel(
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodes->size() >= nodesInfo->size());
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
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
void sortKernel(
    Buffer * const inBuffer,
    Buffer * const outBuffer,
    gfl::ArrayView<gfl::u8> const * const cubAuxMem,
    bool reverse = false)
{
    using namespace gfl;

    assert(cubAuxMem != nullptr);
    assert(inBuffer != nullptr);
    assert(outBuffer != nullptr);
    assert(inBuffer->size() <= outBuffer->size());

    // Debug
    auto auxDataMemSize = scast<size_t>(cubAuxMem->dataMemSize());

    if (reverse)
        cub::DeviceRadixSort::SortKeysDescending(
            cubAuxMem->data(), auxDataMemSize,
            inBuffer->data(), outBuffer->data(),
            inBuffer->size(),
            KeyDecomposer{});
    else
        cub::DeviceRadixSort::SortKeys(
            cubAuxMem->data(), auxDataMemSize,
            inBuffer->data(), outBuffer->data(),
            inBuffer->size(),
            KeyDecomposer{});
}

template<typename T>
GFL_GLOBAL
void swapKernel(T * const a, T * const b) { T::swap(*a,*b); }





GFL_GLOBAL
void setFlagKernel(
    gfl::i8 const flag,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        info.flag = flag;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void flagRepresentedKernel(
    gfl::i64 const flag,
    gfl::ArrayView<Node>  const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
    )
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & iInfo = nodesInfo->at(i);
        Node const & iNode = nodes->at(iInfo.idx);
        for (i64 j = i + 1; j < nodesInfo->size(); j += 1)
        {
            NodeInfo & jInfo = nodesInfo->at(j);
            Node const & jNode = nodes->at(jInfo.idx);
            if (iInfo.hash == jInfo.hash)
                flagRepresented<Model>(iInfo,jInfo, iNode, jNode, flag);
            else
                break;
        }
    }
}

GFL_GLOBAL
void countFlaggedKernel(
    gfl::u8 const flag,
    gfl::i64 * const count,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
    )
{
    using namespace gfl;
    __shared__ i32 count_s;

    if (threadIdx.x == 0)
    {
        count_s = 0;
    }
    __syncthreads();

    i32 count_r = 0;
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo const & info = nodesInfo->at(i);
        count_r += info.flag == flag;
    }
    if (count_r > 0)
    {
        atomicAdd_block(&count_s, count_r);
    }
    __syncthreads();

    if (threadIdx.x == 0 and count_s > 0)
    {
        atomicAdd(rcast<llu*>(count), scast<llu>(count_s));
    }
}

template<typename Node>
GFL_GLOBAL
void copyByInfoIdxKernel(
    gfl::ArrayView<Node> const * const dst,
    gfl::ArrayView<Node> const * const  src,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodesInfo->size() <= src->size());
    assert(nodesInfo->size() <= dst->size());

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node & sNode = src->at(info.idx);
        Node & dNode = dst->at(i);
        dNode = sNode;
        info.idx = i;
    }
}

template<typename Node>
GFL_GLOBAL
void saveByCutsetMarkKernel(
    CutsetData<Node> * const cutset,
    gfl::ArrayView<Node> const * const parents,
    gfl::ArrayView<NodeInfo> const * const parentsInfo)
{
    using namespace gfl;

    ArrayView<Node> mark = cutset->mark();



    if (not cutset->saved() and cutset->toSave())
    {
        assert(parentsInfo->size() <= mark.size());

        if(blockIdx.x == 0  and threadIdx.x == 0)
        {
            cutset->saved(true);
        }

        auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, parentsInfo->size());
        for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
        {
            NodeInfo & info = parentsInfo->at(i);
            Node & sNode    = parents->at(info.idx);
            Node & dNode    = mark.at(i);
            dNode  = sNode;
        }
    }
}
template<typename Node>
GFL_GLOBAL
void copyByCutsetMarkKernel(
    CutsetData<Node> * const cutset,
    gfl::ArrayView<Node> const * const src,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    ArrayView<Node> mark = cutset->mark();

    assert(nodesInfo->size() <= src->size());
    assert(nodesInfo->size() <= mark.size());

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node & sNode    = src->at(info.idx);
        Node & dNode    = mark.at(i);
        dNode  = sNode;
    }
}

template<typename T>
GFL_GLOBAL
void copyAll(
    gfl::ArrayView<T> const * const dst,
    gfl::ArrayView<T> const * const src)
{
    using namespace gfl;

    assert(dst->size() >= src->size());

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, src->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        dst->at(i)= src->at(i);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void copyBestTargetsKernel(
    gfl::optional<Node> * const bestTarget,
    gfl::optional<Node> * const bestExactTarget,
    gfl::ArrayView<Node> const * const targets,
    gfl::ArrayView<NodeInfo> const * const targetsInfo)
{
    using namespace gfl;

    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    // Has been set by termination detection
    bestTarget->reset();

    for (i64 i = 0; i < targetsInfo->size(); ++i)
    {
        NodeInfo const & info = targetsInfo->at(i);
        Node const & node = targets->at(info.idx);

        // Best overall (any node)
        f64 const g = node.g();
        if (not bestTarget->has_value() or
             isBetter<Model>(g, bestTarget->value().g()))
        {
            *bestTarget = node;
            //printf("[DBG] Best found with value %.2f\n", node.g());
        }

        // Best exact (non-approximated)
        if (not node.approximated())
        {
            if (not bestExactTarget->has_value() or
                isBetter<Model>(node.g(), bestExactTarget->value().g()))
            {
                //printf("[DBG] Exact found with value %.2f\n", node.f());
                *bestExactTarget = node;
            }
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setScoreAsGKernel(
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo,
    gfl::f64 const lambda)
{
    using namespace gfl;

    assert(nodes->size() >= nodesInfo->size());
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        info.score = score<Model>(node.g());
        info.score =
            node.approximated() ? info.score
                                : boostScore<Model>(info.score, lambda);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setScoreAsFKernel(
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo,
    gfl::f64 const lambda)
{
    using namespace gfl;

    assert(nodes->size() >= nodesInfo->size());
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        info.score = score<Model>(node.f());
        info.score =
            node.approximated() ? info.score
                                : boostScore<Model>(info.score, lambda);
    }
}

template<typename Node>
GFL_GLOBAL
void setApproximatedFlagKernel(
    gfl::i64 const flag,
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodes->size() == nodesInfo->size());
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        if (node.approximated())
        {
            info.flag = flag;
        }
    }
}

template<typename Node>
GFL_GLOBAL
void flagToSaveKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const parentInfo,
    gfl::i64 const width,
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<NodeInfo> const * const childrenInfo
)
{
    using namespace gfl;

    assert(children->size() >= childrenInfo->size());

    // Merge a prefix down to one, so that the total number of nodes is width
    i64 const prefixSize = childrenInfo->size() - (width - 1);
    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, prefixSize);
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        i64 const j = (width - 1) + i;
        NodeInfo const & info = childrenInfo->at(j);
        Node const & node = children->at(info.idx);
        if (not node.ancestorInCutset() and node.depth() > 1)
        {
            parentInfo->at(info.pIdx).flag = flag;
        }
    }
}

template<typename Node>
GFL_GLOBAL
void flagChildrenToSaveKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const suffixChildrenInfo,
    gfl::ArrayView<Node> const * const children
)
{
    using namespace gfl;

    assert(children->size() >= suffixChildrenInfo->size());

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, suffixChildrenInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = suffixChildrenInfo->at(i);
        Node const & node = children->at(info.idx);
        if (not node.approximated())
        {
            info.flag = flag;
        }
    }
}


template< typename Node>
GFL_GLOBAL
void updateAncInCutKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const parentInfo,
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<NodeInfo> const * const childrenInfo)
{
    using namespace gfl;

    assert(children->size() >= childrenInfo->size());

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, childrenInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo const & info = childrenInfo->at(i);
        Node  & node = children->at(info.idx);
        if (parentInfo->at(info.pIdx).flag == flag and node.depth() > 1)
        {
            node.ancestorInCutset(true);
        }
    }
}


template<typename Node>
GFL_GLOBAL
void assertCopyPrecondKernel(
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<NodeInfo> const * const tmpInfo,
    CutsetData<Node> const * const cutset)
{
    printf("children.size()=%d tmpInfo.size()=%d cutset.lastSegment.size()=%d\n",
       children->size(), tmpInfo->size(), cutset->lastSegmentPtr()->size());
    assert(children->size() >= tmpInfo->size());
    assert(cutset->lastSegmentPtr()->size() == tmpInfo->size());
}

template<typename Node>
GFL_GLOBAL
void markAndResizeByKernel( CutsetData<Node> * const cutset, gfl::i64 const * const count)
{
    cutset->markAndResizeBy(*count);
}

template<typename Node>
GFL_GLOBAL
void markAndResizeByKernel(
    CutsetData<Node> * const cutset,
    gfl::i64 const width,
    gfl::ArrayView<NodeInfo> const * const parentsInfo,
    gfl::ArrayView<NodeInfo> const * const childrenInfo)
{
    if (childrenInfo->size() > width and not cutset->saved())
    {
        cutset->toSave(true);
        cutset->markAndResizeBy(parentsInfo->size());
    }
}

template<typename Node>
GFL_GLOBAL
void initSuffix(
    gfl::i64 const width,
    gfl::ArrayView<NodeInfo> const * const info,
    gfl::ArrayView<Node> * const suffix)
{
    using namespace gfl;


    i64 const prefixSize = min<i64>(width - 1, info->size());
    *suffix = info->slice(prefixSize, info->size());
}

GFL_GLOBAL
void ceilKernel(
    gfl::i64 * const val,
    gfl::i64 const divider)
{
    *val = gfl::ceil<gfl::i64>(*val, divider);
}

template<typename Model, typename Node>
GFL_HOST_DEVICE
void mergeNodeWith(Node & main, Node const & toMerge)
{
    main.state(Model::smf(main.state(), toMerge.state()));
    main.g(better<Model>(main.g(), toMerge.g()));
    main.h(better<Model>(main.h(), toMerge.h()));
}

template<typename Model, typename Node>
GFL_GLOBAL
void reduceByInfoKernel(
    gfl::ArrayView<Node> * const nodes,
    gfl::ArrayView<NodeInfo> const * const inInfo,
    gfl::ArrayView<NodeInfo> * const outInfo,
    gfl::i64 const * const count)
{
    using namespace gfl;

    i32 constexpr reductionFactor = 32 * 32;
    assert(blockDim.x == 32);
    assert(gridDim.x * reductionFactor >= *count);

    __shared__ i32 tmpIdx[32];   // 128 bytes only

    i32 const begin = blockIdx.x * reductionFactor;
    i32 const end = min<i32>(begin + reductionFactor, *count);
    i32 const nodesOfBlock = end - begin;
    i32 const tIdx = threadIdx.x;

    if (tIdx < nodesOfBlock)
    {
        NodeInfo const & fInfo = inInfo->at(begin + tIdx);
        Node & fNode_r = nodes->at(fInfo.idx);      // ← reference to global memory
        fNode_r.approximated(true);
        for (i64 i = begin + tIdx + blockDim.x; i < end; i += blockDim.x)
        {
            NodeInfo const & iInfo = inInfo->at(i);
            Node const & iNode = nodes->at(iInfo.idx);
            mergeNodeWith<Model>(fNode_r, iNode);   // ← writes directly to global
        }
        tmpIdx[tIdx] = fInfo.idx;                   // ← share only the index
    }
    __syncwarp();

    if (threadIdx.x == 0 and nodesOfBlock > 0)
    {
        Node & fNode_r = nodes->at(tmpIdx[0]);      // ← reference to global memory
        for (i64 i = 1; i < min<i64>(nodesOfBlock, 32); ++i)
        {
            Node const & iNode = nodes->at(tmpIdx[i]);
            mergeNodeWith<Model>(fNode_r, iNode);
        }
        NodeInfo const & rInfo = inInfo->at(begin);
        outInfo->at(blockIdx.x) = rInfo;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void reduceByInfoKernelOld(
    gfl::ArrayView<Node> * const nodes,
    gfl::ArrayView<NodeInfo> const * const inInfo,
    gfl::ArrayView<NodeInfo> * const outInfo,
    gfl::i64 const * const count)
{
    using namespace gfl;

    i32 constexpr reductionFactor = 32 * 32; // Each thread merges 32 elements
    assert(blockDim.x == 32);
    assert(gridDim.x * reductionFactor >= *count);

    __shared__ Node tmpNodes[32];

    i32 const begin = blockIdx.x * reductionFactor;
    i32 const end = min<i32>(begin + reductionFactor, *count);
    i32 const nodesOfBlock = end - begin;
    i32 const tIdx = threadIdx.x;
    if (tIdx < nodesOfBlock)
    {
        NodeInfo const & fInfo = inInfo->at(begin + threadIdx.x);
        Node fNode_r = nodes->at(fInfo.idx);
        fNode_r.approximated(true);
        for (i64 i = begin + threadIdx.x + blockDim.x; i < end; i += blockDim.x)
        {
            NodeInfo const & iInfo = inInfo->at(i);
            Node const & iNode = nodes->at(iInfo.idx);
            mergeNodeWith<Model>(fNode_r, iNode);
        }
        tmpNodes[threadIdx.x] = fNode_r;
    }
    __syncwarp();

    if (threadIdx.x == 0 and nodesOfBlock > 0)
    {

        Node fNode_r = tmpNodes[0];
        for (i64 i = 1; i < min<i64>(nodesOfBlock,32); ++i)
        {
            Node const iNode = tmpNodes[i];
            //printf("Reading node with %.2f at slot %d\n", iNode.g(),i);
            mergeNodeWith<Model>(fNode_r, iNode);
        }
        NodeInfo const & rInfo = inInfo->at(begin);
        nodes->at(rInfo.idx) = fNode_r;
        outInfo->at(blockIdx.x) = rInfo;
        //printf("Writing node with %.2f at slot %d (idx %ld)\n", fNode_r.g(), blockIdx.x, rInfo.idx);
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void reduceByInfoSeqKernel(
    gfl::ArrayView<Node> * const children,
    gfl::ArrayView<NodeInfo> const * const inInfo,
    gfl::i64 const * const count)
{
    using namespace gfl;

    if (*count > 0)
    {
        NodeInfo const & rInfo = inInfo->at(0);
        Node & fNode_r = children->at(rInfo.idx);
        fNode_r.approximated(true);
        for (i64 i = 1; i < *count; ++i)
        {
            NodeInfo const & iInfo = inInfo->at(i);
            Node const & iNode = children->at(iInfo.idx);
            mergeNodeWith<Model>(fNode_r, iNode);
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcOutLabelsKernel(
        Model const * const model,
        gfl::ArrayView<Node> const * const nodes,
        gfl::ArrayView<NodeInfo> const * const nodesInfo,
        gfl::f64 const primal,
        gfl::f64 const dual,
        DDContext const ddCtx)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo const & info = nodesInfo->at(i);
        Node & node = nodes->at(info.idx);
        node.outLabels(model->lgf(node.state(), primal, dual, ddCtx));
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcOutLabelsKernel(
        Model const * const model,
        gfl::ArrayView<Node> const * const nodes,
        gfl::f64 const primal,
        gfl::f64 const dual,
        DDContext const ddCtx)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, nodes->size());
    for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        Node & node = nodes->at(i);
        node.outLabels(model->lgf(node.state(), primal, dual, ddCtx));
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setHKernel(
    gfl::optional<Node> const * bestTarget,
    gfl::ArrayView<Node> const * cutset)
{
    using namespace gfl;

    __shared__ f64 f;

    if (bestTarget->has_value())
    {
        if (threadIdx.x == 0)
        {
            f = bestTarget->value().g();
        }
        __syncthreads();

        auto [begin,end] = calcSlice<i64>(blockIdx.x, gridDim.x, cutset->size());
        for (i64 i = begin + threadIdx.x; i < end; i += blockDim.x)
        {

            Node & node = cutset->at(i);
            f64 const h = f - node.g();
            node.h(worse<Model>(h, node.h()));
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void checkForTargetOldKernel(
    bool * const found,
    Model const * const model,
    gfl::VectorView<Node> const * nodes)
{
    assert(nodes != nullptr);
    assert(not nodes->empty());
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    *found = nodes->at(0).isTarget(model);
}

template<typename Model, typename Node>
GFL_GLOBAL
void checkForTargetKernel(
    Model const * const model,
    gfl::optional<Node> * const target,
    gfl::ArrayView<Node> const * nodes,
    gfl::ArrayView<NodeInfo> const * nodesInfo)
{
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    if (not nodesInfo->empty())
    {
        NodeInfo const & info = nodesInfo->at(0);
        Node const & node = nodes->at(info.idx);
        if (node.isTarget(model)) *target = node;
    }
}

template<typename T>
GFL_GLOBAL
void printKernel(gfl::i64 const i, gfl::ArrayView<T> const * array)
{
    assert(array != nullptr);
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    printf("STEP %d\n",i);
    for (auto const & a : *array) {a.print(); printf("\n");}
}

template<typename T>
GFL_GLOBAL
void checkKernel(gfl::i64 const i,
    gfl::ArrayView<NodeInfo> const * nodesInfo,
    gfl::ArrayView<T> const * nodes)
{
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    bool found = nodesInfo->empty();
    for (auto const & info : *nodesInfo)
    {
        auto const & n = nodes->at(info.idx);
        if (i) printf("Node %.2f = %.2f + %.2f\n", n.f(), n.g(), n.h());
        if (n.g() <= 106) found = true;
    }
    printf("---\n");
    assert(found or i);
}


template<typename Node>
GFL_GLOBAL
void printKernel(gfl::i64 const i, gfl::ArrayView<NodeInfo> const * infos, gfl::ArrayView<Node> const * nodes)
{
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    printf("STEP %ld\n",i);
    for (auto const & j : *infos)
    {
        j.print(); printf(" | ");
        nodes->at(j.idx).print(); printf("\n");
    }
}

template<typename T>
GFL_GLOBAL
void assertNotEmptyKernel(gfl::ArrayView<T> const * array)
{
    assert(array != nullptr);
    assert(not array->empty());

}