#pragma once

#include <GFL.hpp>

#include "Contexts.hpp"
#include "ExpansionData.hpp"

template<typename Model, typename Node>
GFL_GLOBAL
void expandParentsKernel(
    Model const * const model,
    ExpansionData<Node> * const expData,
    gfl::f64 const primal,
    gfl::i32 const branchFactor
    )
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto & children = expData->children;
    auto & childrenInfo = expData->childrenInfo;

    assert(blockDim.x == 32);

    __shared__ i32 nChildren_s;
    __shared__ Node children_s[32];
    __shared__ NodeInfo childrenInfo_s[32];
    __shared__ i32 nodeOffset_g;
    __shared__ i32 infoOffset_g;

    i32 const tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    i32 const pIdx = tIdx / branchFactor;
    i32 const label = tIdx % branchFactor;

    if (threadIdx.x == 0)
    {
        nChildren_s = 0;
        nodeOffset_g = -1;
        infoOffset_g = -1;
    }
    __syncthreads();


    if (pIdx < parents.size())
    {
        Node const & pNode = parents[pIdx];
        auto const & pLabels = pNode.labels();

        if (pLabels.contains(label))
        {
            // Transition
            auto const cState = model->stf(pNode.state(), label);
            if (cState.has_value())
            {
                f64 const tCost = model->scf(pNode.state(), label);
                f64 const cG = pNode.g() + tCost;
                f64 cH = pNode.f() - cG;
                if constexpr (Model::has_heur)
                {
                    f64 const h = model->h(cState.value(), BBCtx);
                    cH = tighter<Model>(cH,h);
                }
                // Conditions to keep the child
                if (isBetter<Model>(cG + cH,primal))
                {
                    i32 const cIdx_s = atomicAdd_block(&nChildren_s,1);
                    children_s[cIdx_s] = Node(cState.value(), cG, cH, label, pNode);
                    childrenInfo_s[cIdx_s] = NodeInfo(-1, pIdx);
                }
            }
        }
    }
    __syncthreads();

    if (nChildren_s > 0)
    {

        if (threadIdx.x == 0)
        {
            infoOffset_g = childrenInfo.resizeByAtomic(nChildren_s);
            nodeOffset_g = children.resizeByAtomic(nChildren_s);
        }
        __syncthreads();
        if (threadIdx.x < nChildren_s)
        {
            //printf("Adding child\n");
            i32 const nIdx_g = nodeOffset_g + threadIdx.x;
            i32 const iIdx_g = infoOffset_g + threadIdx.x;
            childrenInfo_s[threadIdx.x].idx = nIdx_g;
            childrenInfo[iIdx_g] = childrenInfo_s[threadIdx.x];
            children[nIdx_g] = children_s[threadIdx.x];
        }
    }

}


template<typename Model, typename Node>
GFL_GLOBAL
void expandParentsNewKernel(
    Model const * const model,
    ExpansionData<Node> * const expData,
    gfl::f64 const primal,
    gfl::i32 const branchFactor
    )
{
    using namespace gfl;

    auto const & parents = expData->parents;
    auto const & parentsInfo = expData->parentInfo;
    auto & children = expData->children;
    auto & childrenInfo = expData->childrenInfo;

    assert(blockDim.x == 32);

    __shared__ i32 nChildren_s;
    __shared__ Node children_s[32];
    __shared__ NodeInfo childrenInfo_s[32];
    __shared__ i32 nodeOffset_g;
    __shared__ i32 infoOffset_g;

    i32 const tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    i32 const pIdx = tIdx / branchFactor;
    i32 const lIdx = tIdx % branchFactor;

    if (threadIdx.x == 0)
    {
        nChildren_s = 0;
        nodeOffset_g = -1;
        infoOffset_g = -1;
    }
    __syncthreads();


    if (pIdx < parentsInfo.size())
    {
        NodeInfo const & pInfo = parentsInfo[pIdx];
        Node const & pNode = parents[pInfo.idx];
        auto const & pLabels = pNode.labels();
        auto const label = pLabels.smallest() + lIdx;
        if (pLabels.contains(label))
        {
            // Transition
            auto const cState = model->stf(pNode.state(), label);
            if (cState.has_value())
            {
                f64 const tCost = model->scf(pNode.state(), label);
                f64 const cG = pNode.g() + tCost;
                f64 cH = pNode.f() - cG;
                if constexpr (Model::has_heur)
                {
                    f64 const h = model->h(cState.value(), BBCtx);
                    cH = tighter<Model>(cH,h);
                }
                // Conditions to keep the child
                if (isBetter<Model>(cG + cH,primal))
                {
                    i32 const cIdx_s = atomicAdd_block(&nChildren_s,1);
                    children_s[cIdx_s] = Node(cState.value(), cG, cH, label, pNode);
                    childrenInfo_s[cIdx_s] = NodeInfo(-1, pIdx);
                }
            }
        }
    }
    __syncthreads();

    if (nChildren_s > 0)
    {

        if (threadIdx.x == 0)
        {
            infoOffset_g = childrenInfo.resizeByAtomic(nChildren_s);
            nodeOffset_g = children.resizeByAtomic(nChildren_s);
        }
        __syncthreads();
        if (threadIdx.x < nChildren_s)
        {
            //printf("Adding child\n");
            i32 const nIdx_g = nodeOffset_g + threadIdx.x;
            i32 const iIdx_g = infoOffset_g + threadIdx.x;
            childrenInfo_s[threadIdx.x].idx = nIdx_g;
            childrenInfo[iIdx_g] = childrenInfo_s[threadIdx.x];
            children[nIdx_g] = children_s[threadIdx.x];
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

    assert(nodes->size() == nodesInfo->size());
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
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
    assert(inBuffer->size() == outBuffer->size());

    size_t auxDataMemSize = scast<size_t>(cubAuxMem->dataMemSize());
    cudaStream_t gpuSortQueue;
    cudaStreamCreateWithFlags(&gpuSortQueue, cudaStreamNonBlocking);
    if (reverse)
        cub::DeviceRadixSort::SortKeysDescending(
            cubAuxMem->data(),
            auxDataMemSize,
            inBuffer->data(),
            outBuffer->data(),
            inBuffer->size(),
            KeyDecomposer{},
            gpuSortQueue);
    else
        cub::DeviceRadixSort::SortKeys(
            cubAuxMem->data(),
            auxDataMemSize,
            inBuffer->data(),
            outBuffer->data(),
            inBuffer->size(),
            KeyDecomposer{},
            gpuSortQueue);
    cudaStreamDestroy(gpuSortQueue);
}

template<typename T>
GFL_GLOBAL
void swapKernel(T * const a, T * const b) { T::swap(*a,*b); }

template<typename T>
GFL_GLOBAL
void resizeToKernel(
    gfl::VectorView<T> * const v,
    gfl::i32 const count)
{ v->resizeTo(count); }

template<typename T>
GFL_GLOBAL
void resizeToKernel(
    gfl::VectorView<T> * const v,
    gfl::i32 const * const count)
{ v->resizeTo(*count); }

template<typename T>
GFL_GLOBAL
void setValueKernel( T * const t, T const v) { *t = v; }

GFL_GLOBAL
void setFlagKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        info.flag = flag;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void flagRepresentedChildrenKernel(
    gfl::i64 const flag,
    gfl::ArrayView<Node>  const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
    )
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & iInfo = nodesInfo->at(i);
        Node const & iNode = nodes->at(iInfo.idx);
        for (i32 j = i + 1; j < nodesInfo->size(); j += 1)
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
    gfl::i64 const flag,
    gfl::i32 * const count,
    gfl::ArrayView<NodeInfo> const * const nodesInfo
    )
{
    using namespace gfl;
    __shared__ i32 count_s;

    if (threadIdx.x == 0) { count_s = 0;}
    __syncthreads();

    i32 count_r = 0;
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x,nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo const & info = nodesInfo->at(i);
        count_r += info.flag == flag;
    }
    if (count_r > 0) atomicAdd_block(&count_s, count_r);
    __syncthreads();

    if (threadIdx.x == 0 and count_s > 0) { atomicAdd(count,count_s); }
}

template<typename Node>
GFL_GLOBAL
void copyByInfoKernel(
    gfl::ArrayView<Node> const * const dst,
    gfl::ArrayView<Node> const * const  src,
    gfl::ArrayView<NodeInfo> const * const nodesInfo,
    bool const updateInfoIdx = true)
{
    using namespace gfl;

    assert(src->size() >= nodesInfo->size());
    assert(dst->size() == nodesInfo->size());

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node & sNode = src->at(info.idx);
        Node & dNode = dst->at(i);
        dNode = sNode;
        if (updateInfoIdx) info.idx = i;
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

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, src->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        dst->at(i)= src->at(i);
    }
}

template<typename Node>
GFL_GLOBAL
void copyBestTargetsKernel(
    gfl::optional<Node> * const bestTarget,
    gfl::optional<Node> * const bestExactTarget,
    gfl::ArrayView<Node> const  * const  targets,
    gfl::ArrayView<NodeInfo> const * const targetsInfo)
{
    using namespace gfl;

    NodeInfo const & bestInfo = targetsInfo->at(0);
    *bestTarget = targets->at(bestInfo.idx);

    for (i32 i = 0; i < targetsInfo->size(); ++i)
    {
        NodeInfo const & targetInfo = targetsInfo->at(0);
        Node const & target = targets->at(targetInfo.idx);
        if (not target.approximated())
        {
            *bestExactTarget = target;
            break;
        }
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setScoreGKernel(
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodes->size() >= nodesInfo->size());
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        info.score = score<Model>(node.g());;
    }
}
template<typename Model, typename Node>
GFL_GLOBAL
void setScoreMergeKernel(
    Model const * const model,
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    //assert(nodes->size() == nodesInfo->size());
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        info.score = score<Model>(node.g());
        //f32 const gScore = score<Model>(node.g());
        // f32 simScore = 0.0;
        // f64 bestGScore = score<Model>(worst<Model>());
        // for (auto const & n : *nodes)
        // {
        //     simScore += model->ssf(n.state(), node.state());
        //     bestGScore = min<f64>(bestGScore,score<Model>(n.g()));
        // }
        // assert(simScore >= 0.0);
        // assert(nodes->size() > 0);
        // f32 const simScoreNorm = simScore / nodes->size();
        // assert(simScoreNorm >= 0.0);
        // assert(simScoreNorm <= 1.0);
        // f32 const gScoreNorm = gScore / fabs(bestGScore);
        // f32 const score = gScoreNorm * (1.0 - 0.05*simScoreNorm);
        //printf("%.3f -> %.3f -> %.3f\n", gScore, gScoreNorm, score);
        // info.score = score;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setScoreFKernel(
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodes->size() >= nodesInfo->size());
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        info.score = score<Model>(node.f());
    }
}

template<typename Node>
GFL_GLOBAL
void setApproximatedFlagKernel(
    gfl::i32 const flag,
    gfl::ArrayView<Node> const * const nodes,
    gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    assert(nodes->size() == nodesInfo->size());
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        Node const & node = nodes->at(info.idx);
        if (node.approximated())
        {
            info.flag = flag;
        }
    }
}

GFL_GLOBAL
void resetInfoIdxKernel(gfl::ArrayView<NodeInfo> const * const nodesInfo)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo & info = nodesInfo->at(i);
        info.idx = i;
    }
}

template<typename Node>
GFL_GLOBAL
void flagParentsToSaveKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const parentInfo,
    gfl::i32 const width,
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<NodeInfo> const * const childrenInfo
)
{
    using namespace gfl;

    assert(children->size() >= childrenInfo->size());

    // Merge a suffix so that the total number of nodes is width
    i32 const nChildrenToMerge = childrenInfo->size() - (width - 1);
    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nChildrenToMerge);
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        i32 const j = (width - 1) + i;
        NodeInfo const & info = childrenInfo->at(j);
        Node const & node = children->at(info.idx);
        if (not node.ancestorInCutset())
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

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, suffixChildrenInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
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
void updateAncestorKernel(
    gfl::i64 const flag,
    gfl::ArrayView<NodeInfo> const * const parentInfo,
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<NodeInfo> const * const childrenInfo)
{
    using namespace gfl;

    assert(children->size() >= childrenInfo->size());

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, childrenInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        auto const & info = childrenInfo->at(i);
        auto & node = children->at(info.idx);
        if (parentInfo->at(info.pIdx).flag == flag)
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
void resizeByKernel(
    CutsetData<Node> * const cutset,
    gfl::i32 const * const count)
{ cutset->addSegment(*count);}

template<typename Node>
GFL_HOST_DEVICE
void initChildrenPrefix(
    gfl::i32 const width,
    gfl::ArrayView<Node> const * const children,
    gfl::ArrayView<Node> * const childrenPrefix)
{
    using namespace gfl;

    i32 const prefixSize = width - 1;
    *childrenPrefix = children->slice(prefixSize, children->size());
}


GFL_GLOBAL
void initRootInfoKernel(
    gfl::VectorView<NodeInfo> * const nodesInfo)
{
    assert(nodesInfo != nullptr);
    assert(blockDim.x == 1);
    assert(gridDim.x == 1);

    using namespace gfl;
    nodesInfo->resizeTo(1);
    nodesInfo->at(0).idx = 0;
}

template<typename Model, typename Node>
GFL_HOST_DEVICE
void mergeNodeWith(Node & main, Node const & toMerge)
{
    main.state(Model::smf(main.state(), toMerge.state()));
    main.g(better<Model>(main.g(), toMerge.g()));
    main.h(looser<Model>(main.h(), toMerge.h()));
}

template<typename Model, typename Node>
GFL_GLOBAL
void reduceByInfoKernel(
    gfl::ArrayView<Node> * const children,
    gfl::ArrayView<NodeInfo> const * const inInfo,
    gfl::ArrayView<NodeInfo> * const outInfo,
    gfl::i32 const count)
{
    using namespace gfl;
    assert(blockDim.x == 32);
    __shared__ Node tmpNodes[32];

    auto [begin, end] = calcSlice<i32>(blockIdx.x, gridDim.x, count);
    i32 const nodesOfBlock = min<i32>(blockDim.x, end - begin);

    if (threadIdx.x < nodesOfBlock)
    {
        NodeInfo const & fInfo = inInfo->at(begin + threadIdx.x);
        Node fNode_r = children->at(fInfo.idx);
        fNode_r.approximated(true);
        for (i32 i = begin + threadIdx.x + blockDim.x; i < end; i += blockDim.x)
        {
            NodeInfo const & iInfo = inInfo->at(i);
            Node const & iNode = children->at(iInfo.idx);
            mergeNodeWith<Model>(fNode_r, iNode);
        }
        tmpNodes[threadIdx.x] = fNode_r;
    }
    __syncthreads();

    if (threadIdx.x == 0 and nodesOfBlock > 0)
    {
        Node fNode_r = tmpNodes[0];
        for (i32 i = 1; i < nodesOfBlock; ++i)
        {
            Node const & iNode = tmpNodes[i];
            mergeNodeWith<Model>(fNode_r, iNode);
        }
        NodeInfo const & rInfo = inInfo->at(begin);
        children->at(rInfo.idx) = fNode_r;
        outInfo->at(blockIdx.x) = rInfo;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void reduceByInfoSeqKernel(
    gfl::ArrayView<Node> * const children,
    gfl::ArrayView<NodeInfo> const * const inInfo,
    gfl::i32 const count)
{
    using namespace gfl;

    NodeInfo const & rInfo = inInfo->at(0);
    Node fNode_r = children->at(rInfo.idx);
    for (i32 i = 1; i < count; ++i)
    {
        NodeInfo const & iInfo = inInfo->at(i);
        Node const & iNode = children->at(iInfo.idx);
        mergeNodeWith<Model>(fNode_r, iNode);
    }
    children->at(rInfo.idx) = fNode_r;
}

template<typename Model, typename Node>
GFL_GLOBAL
__global__ void reductionKernel(
    gfl::ArrayView<Node> const * const inBuffer,
    gfl::ArrayView<Node> const * const outBuffer,
    gfl::i32 const count)
{
    using namespace gfl;

    assert(blockDim.x == 32);
    __shared__ Node tmpNodes[32];
    //assert(blockDim.x * sizeof(Node) <= getSharedMemSize());

    auto [begin, end] = calcSlice<int>(blockIdx.x, gridDim.x, count);
    i32 const nodesOfBlock = min<i32>(blockDim.x, end - begin);
    if (threadIdx.x < nodesOfBlock)
    {
        Node fNode_r = inBuffer->at(begin + threadIdx.x);
        fNode_r.approximated(true);
        for (i32 i = begin + threadIdx.x + blockDim.x; i < end; i += blockDim.x)
        {
            Node const & iNode = inBuffer->at(i);
            mergeNodeWith<Model>( fNode_r, iNode);
        }
        tmpNodes[threadIdx.x] = fNode_r;
    }
    __syncthreads();

    if (threadIdx.x == 0 and nodesOfBlock > 0)
    {
        Node fNode_r = tmpNodes[0];
        for (int i = 1; i < nodesOfBlock; ++i)
        {
            Node const & iNode = tmpNodes[i];
            mergeNodeWith<Model>( fNode_r, iNode);
        }
        outBuffer->at(blockIdx.x) = fNode_r;
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void reductionSeqKernel(
    gfl::ArrayView<Node> const * const inBuffer,
    gfl::i32 const count)
{
    using namespace gfl;

    assert(blockDim.x == 1);
    assert(gridDim.x == 1);
    assert(count > 0);

    Node result = inBuffer->at(0);
    result.approximated(true);
    for (i32 i = 1; i < count; ++i)
    {
        mergeNodeWith<Model>(result, inBuffer->at(i));
    }
    inBuffer->at(0) = result;
}

template<typename Model, typename Node>
GFL_GLOBAL
void calcOutLabelsNewKernel(
        Model const * const model,
        gfl::ArrayView<Node> const * const nodes,
        gfl::ArrayView<NodeInfo> const * const nodesInfo,
        gfl::f64 const primal,
        gfl::f64 const dual,
        DDContext const ddCtx)
{
    using namespace gfl;

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodesInfo->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        NodeInfo const & info = nodesInfo->at(i);
        Node & node = nodes->at(info.idx);
        node.labels(model->lgf(node.state(), primal, dual, ddCtx));
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

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, nodes->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        Node & node = nodes->at(i);
        node.labels(model->lgf(node.state(), primal, dual, ddCtx));
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void setHKernel(
    gfl::optional<Node> const * bestTarget,
    gfl::ArrayView<Node> const * cutset)
{
    using namespace gfl;

    assert(bestTarget->has_value());

    __shared__ f64 f;

    if (threadIdx.x == 0)
    {
        f = bestTarget->value().g();
    }
    __syncthreads();

    auto [begin,end] = calcSlice<i32>(blockIdx.x, gridDim.x, cutset->size());
    for (i32 i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        Node & node = cutset->at(i);
        f64 const h = f - node.g();
        node.h(tighter<Model>(node.h(), h));
    }
}

template<typename Model, typename Node>
GFL_GLOBAL
void checkForTargetKernel(
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
void checkForTargetNewKernel(
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
        Node & node = nodes->at(info.idx);
        if (node.isTarget(model)) *target = node;
    }
}



template<typename Model, typename Node>
GFL_GLOBAL
void checkForTargetKernel(
    Model const * const model,
    gfl::optional<Node> * const target,
    gfl::VectorView<Node> const * nodes)
{
    assert(nodes != nullptr);
    assert(not nodes->empty());
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    Node & n = nodes->at(0);
    if (n.isTarget(model)) *target = n;
}


template<typename T>
GFL_GLOBAL
void printKernel(gfl::i32 const i, gfl::ArrayView<T> const * array)
{
    assert(array != nullptr);
    assert(gridDim.x == 1);
    assert(blockDim.x == 1);

    using namespace gfl;

    printf("STEP %d\n",i);
    for (auto const & a : *array) {a.print(); printf("\n");}
}

GFL_GLOBAL
void assertLeqKernel(gfl::i32 const * i, gfl::i32 const * j)
{
    assert(*i <= *j);
}


GFL_GLOBAL
void assertLeqKernel(gfl::i32 const * i, gfl::i32 const  j)
{
    assert(*i <= j);
}
