#pragma once

#include "../src/Queue.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

template<typename Model, typename Node>
class QueueGpu : public Queue<Model,Node>
{
    std::vector<gfl::i32> offsetsCpu;
public:
    using Queue<Model,Node>::push;
    using Queue<Model,Node>::memPool_;


    void pushFromGpu(gfl::ArrayView<Node> const * cutsetGpu)
    {
        using namespace gfl;


        if (not cutsetGpu->empty())
        {
            ArrayView<Node> cutsetCpu(cutsetGpu->size(), new (&memPool_) Node[cutsetGpu->size()]);
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                cutsetCpu.data(),
                cutsetGpu->data(),
                cutsetGpu->dataMemSize(),
                cudaMemcpyDeviceToHost));
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            push(cutsetCpu);
        }
    }
};
