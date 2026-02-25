#pragma once

#include "../src/Queue.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

template<typename Model, typename Node>
class QueueGpu : public Queue<Model,Node>
{
    std::vector<Node> cutsetNodesBuffer;
    std::vector<gfl::i32> cutsetOffsetsBuffer;

public:
    using Queue<Model,Node>::push;


    void pushFromGpu(gfl::tuple<gfl::ArrayView<Node>, gfl::ArrayView<gfl::i32>> const & cutset)
    {
        using namespace gfl;

        auto [nodesGpu, offsetsGpu] = cutset;
        auto & nodesCpu = cutsetNodesBuffer;
        auto & offsetsCpu = cutsetOffsetsBuffer;

        nodesCpu.resize(nodesGpu.size());
        CHECK_CUDA_ERROR(cudaMemcpy(
            nodesCpu.data(),
            nodesGpu.data(),
            nodesGpu.dataMemSize(),
            cudaMemcpyDeviceToHost));

        offsetsCpu.resize(offsetsGpu.size());
        CHECK_CUDA_ERROR(cudaMemcpy(
            offsetsCpu.data(),
            offsetsGpu.data(),
            offsetsGpu.dataMemSize(),
            cudaMemcpyDeviceToHost));

        for (i32 i = 0; i < offsetsCpu.size(); ++i)
        {
            i32 const begin = offsetsCpu.at(i);
            i32 const end   = (i + 1 < offsetsCpu.size()) ? offsetsCpu.at(i + 1) : nodesCpu.size();
            i32 const size = end - begin;
            ArrayView slice(size, nodesCpu.data() + begin);
            push(slice);
        }
    }
};
