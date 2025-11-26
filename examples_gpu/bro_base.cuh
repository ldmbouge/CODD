#include "codd.hpp"
#include "LayerEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>
#include <span>

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory
inline
void printNodeInfo(std::vector<NodeInfo> const * const nodesInfo)
{
    for (auto const & ni : *nodesInfo)
    {
        printf("HASH = %lu, BOUND = %.1f, IDX = %ld, IS_REP = %d\n",
               ni.hash,
               ni.boundSrcToNode,
               ni.idx,
               ni.isRepresented);
    }
}

template<typename Node>
void printLabels(Node const & node)
{
    using namespace gfl;
    Array<u8>::print(node.labelsSrcToNode, node.labelsSrcToNode + node.nEdgesSrcToNode);
}

template<typename Model, typename Node>
int run_bro(int argc,char* argv[])
{
    using namespace gfl;
    using LayerHelperType = LayerHelper<Node>;
    using LayerInfoType   = LayerInfo<Node>;
    using LayerBufferType = std::vector<Node>;

    // Select GPU
    auto const gpuId = 0;
    cudaSetDevice(gpuId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, gpuId);

    // Parse arguments
    i64 width = -1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool gpu = false;
    bool sort = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for DIDP models");
    options.add_options("Available")
            ("w,width", "Beam width", cxxopts::value(width))
            ("g,gpu", "Use GPU acceleration", cxxopts::value(gpu))
            ("h,help", "Show this help message and exit")
            ("s,sort", "Sort nodes by cost", cxxopts::value(sort))
            ("i,instance", "Path to the instance file", cxxopts::value(instance))
            ("t,timeout", "Timeout in seconds", cxxopts::value(timeout));
    options.parse_positional({"instance"});
    options.custom_help("<OPTIONS>");
    options.positional_help("<INSTANCE>");
    auto const result = options.parse(argc, argv);

    // Option validation
    if (width == 0)
    {
        std::cerr << "Width must be bigger than 0" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (instance.empty())
    {
        std::cerr << "Missing instance file" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (result.count("help") > 0)
    {
        std::cout << options.help();
        exit(EXIT_SUCCESS);
    }

    auto * const readOnlyMem =
#ifdef __CUDACC__
        mallocManaged<void>(ReadOnlyMemSize);
#else
        mallocStd<void>(ReadOnlyMemSize);
#endif

    StackAllocator roAllocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new (roAllocator) Model();
    Model::parseFile(model, instance, roAllocator);

    std::cout << "Instance: " << instance << std::endl;
    if (not gpu)
    {
        std::cout << "GPU: False" << std::endl;
    }
    else
    {
        std::cout << "GPU: " <<  deviceProp.name << " (";
        printMemSize(deviceProp.totalGlobalMem);
        std::cout <<  " VRAM)" << std::endl;
    }
    std::cout << "Width: ";
    if (width <= 0)
        std::cout << "Auto" << std::endl;
    else
        std::cout << width << std::endl;

    // Search
    Node bestNode;
    f64 pBound = Model::worstValue();
    f64 dBound = Model::bestValue();

    // GPU
    LayerHelperType *        lh = gpu ? new LayerHelperType() : nullptr;
    LayerInfoType *   layerInfo = gpu ? mallocManaged<LayerInfoType>(sizeof(LayerInfoType)) : nullptr;
    StackAllocator * gAllocator = gpu ? new StackAllocator(mallocDevice(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Layers buffers
    std::vector<LayerBufferType> layers;
    std::vector<LabelsInfo>      labelsInfo;

    // Initialize first layer
    auto const rState = model->initial();
    auto const rLabels = model->lgf(rState, DDExact, pBound, dBound);
    layers.emplace_back().emplace_back(rState,rLabels);
    labelsInfo.emplace_back(rLabels.slc());

    // Let's goo!
    auto const start = RuntimeMonitor::cputime();
    i64 expandedNodes = 0;
    bool interrupted = false;
    bool newSolution = false;
    auto const isLayerEmpty = [](LayerBufferType const &l) { return l.empty(); };
    auto const isQueueEmpty = [&layers,&isLayerEmpty]{return std::all_of(layers.begin(), layers.end(), isLayerEmpty); };
    auto const lastNotEmpty = [&layers] {
        for (i32 i = layers.size() - 1; i >= 0; i -= 1)
        {
           if (not layers[i].empty())
           {
              return i;
           }
        }
        return -1;
    };
    while (not isQueueEmpty())
    {
        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }
        newSolution = false;

        // Grow number of layers on demand
        i32 const lIdx = lastNotEmpty();
        assert(lIdx >= 0);
        if (lIdx == layers.size() - 1)
        {
            layers.emplace_back();
            labelsInfo.emplace_back();
        }

        // Fragment
        auto & currentLayer = layers[lIdx];
        i64 const batchSize = gfl::min<i64>(currentLayer.size(), lh->getMaxParents(labelsInfo[lIdx].nLabels));
        i64 const fragmentSize = gfl::min<i64>(currentLayer.size(), width < 0 ? batchSize : width);
        auto const fragment = std::span(currentLayer.end() - fragmentSize, fragmentSize);
        expandedNodes += fragmentSize;

        // Batching
        i32 const nBatches = roundUpDivPosInt<i32>(fragmentSize, batchSize);
        assert(nBatches * batchSize >= fragmentSize);
        for (i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
        {
            i64 bBegin, bEnd;
            getBeginEnd(bBegin, bEnd, bIdx, nBatches, fragmentSize);
            i64 const currentBatchSize = bEnd - bBegin; // No + 1!
            auto const currentBatch = std::span(fragment.data() + bBegin, currentBatchSize);
            printf("[%7.2fs] Layer = %4d | Visited = %10ld | Nodes = %10ld -> %10ld",
                   RuntimeMonitor::elapsedSeconds(start),
                   lIdx,
                   expandedNodes,
                   currentLayer.size(),
                   currentLayer.size()-currentBatchSize);
            printf( " | MemSize = ");
            printMemSize(sizeof(Node) * currentLayer.size());
            printf( " | Cost = ");
            if (pBound != Model::worstValue())
            {
                printf("%7.2f", pBound);
            }
            else
            {
                printf("?");
            }
            printf(" | Batch %3d/%3d | BatchSize = %10ld\n", bIdx+1, nBatches, currentBatchSize);
            fflush(stdout);

            // Init
            LayerHelperType::clear(layerInfo, gAllocator);
            layerInfo->labelsInfo = labelsInfo[lIdx];
            LayerHelperType::initParents(currentBatchSize, layerInfo, gAllocator);
            cudaMemcpyAsync(
                    layerInfo->parents,
                    currentBatch.data(),
                    sizeof(Node) * currentBatchSize,
                    cudaMemcpyHostToDevice,
                    lh->gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();
            LayerHelperType::initChildren(layerInfo, gAllocator);
            LayerHelperType::initAux(layerInfo, gAllocator);

            // Children
            i32 blockSize = 32;
            dim3 gridSize = layerInfo->nParents;
            calcChildrenKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                    model,
                    layerInfo,
                    pBound,
                    DDCtx);
            CHECK_LAST_CUDA_ERROR();
            cudaStreamSynchronize(lh->gpuMainQueue);
            CHECK_LAST_CUDA_ERROR();

            i64 const nChildren = layerInfo->nChildren;
            if (nChildren > 0)
            {
               //  Representatives
                cub::DeviceRadixSort::SortKeys(
                        layerInfo->cubTmpMem,
                        layerInfo->cubTmpMemSize,
                        layerInfo->childrenInfo,
                        layerInfo->tmpChildrenInfo,
                        nChildren,
                        HashDecomposer{},
                        lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();

                blockSize = 128;
                gridSize = roundUpDivPosInt<i32>(nChildren, blockSize);
                calcRepKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                        nChildren,
                        layerInfo->children,
                        layerInfo->tmpChildrenInfo);
                CHECK_LAST_CUDA_ERROR();

                blockSize = 32;
                gridSize = roundUpDivPosInt<i32>(nChildren, blockSize);
                countRepKernel<Node><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                        layerInfo,
                        layerInfo->tmpChildrenInfo);
                CHECK_LAST_CUDA_ERROR();

                if (sort)
                    cub::DeviceRadixSort::SortKeys(
                            layerInfo->cubTmpMem,
                            layerInfo->cubTmpMemSize,
                            layerInfo->tmpChildrenInfo,
                            layerInfo->childrenInfo,
                            nChildren,
                            RepCostDecomposer{},
                            lh->gpuMainQueue);
                else
                    cub::DeviceRadixSort::SortKeys(
                            layerInfo->cubTmpMem,
                            layerInfo->cubTmpMemSize,
                            layerInfo->tmpChildrenInfo,
                            layerInfo->childrenInfo,
                            nChildren,
                            RepDecomposer{},
                            lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();

                swapPtr<Node><<<1,1,0,lh->gpuMainQueue>>>(&layerInfo->children, &layerInfo->tmpChildren);
                CHECK_LAST_CUDA_ERROR();

                blockSize = 128;
                gridSize = roundUpDivPosInt<i32>(nChildren, blockSize);
                copyRepKernel<Node><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(layerInfo, sort);
                CHECK_LAST_CUDA_ERROR();

                // Labels
                resetLabelsInfo<Node><<<1,1,0,lh->gpuMainQueue>>>(layerInfo);
                blockSize = 128;
                gridSize = roundUpDivPosInt<i32>(nChildren, blockSize);
                calcLabelsKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                        model,
                        layerInfo,
                        DDExact,
                        pBound,
                        dBound);
                CHECK_LAST_CUDA_ERROR();

                cudaStreamSynchronize(lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();
            }

            if (layerInfo->nRepresentatives > 0)
            {
                auto & nextLayer = layers[lIdx+1];
                i64 const nextLayerOldSize = nextLayer.size();
                nextLayer.resize(nextLayerOldSize + layerInfo->nRepresentatives);
                cudaMemcpy(
                        nextLayer.data() + nextLayerOldSize,
                        layerInfo->children,
                        sizeof(Node) * layerInfo->nRepresentatives,
                        cudaMemcpyDeviceToHost);
                CHECK_LAST_CUDA_ERROR();

                labelsInfo[lIdx+1].update(layerInfo->labelsInfo);

                if(model->isTarget(nextLayer[nextLayerOldSize].state))
                {
                    //printf("Checking targets...\n");
                    for (i64 i = nextLayerOldSize; i < nextLayer.size(); i += 1)
                    {
                        Node const & n = nextLayer[i];
                        if (Model::better(n.boundSrcToNode, pBound))
                        {
                            bestNode = n;
                            pBound = bestNode.boundSrcToNode;
                            newSolution = true;
                        }
                    }
                    nextLayer.resize(nextLayerOldSize);
                }
                if (sort and nextLayerOldSize > 0)
                {
                    // Reverse because we work on the tail of the vector
                    auto cmpByCost = [](Node const & a, Node const & b){return not Model::betterEq(a.boundSrcToNode,b.boundSrcToNode);};
                    //assert(std::is_sorted(nextLayer.data() + nextLayerOldSize, nextLayer.data() + nextLayer.size(), cmpByCost));
                    std::inplace_merge(nextLayer.data(), nextLayer.data() + nextLayerOldSize, nextLayer.data() + nextLayer.size(), cmpByCost);
                    assert(std::is_sorted(nextLayer.begin(), nextLayer.end(), cmpByCost));
                }
//                    for(Node const & n : nextLayer)
//                    {
//                        printf("%7.2f ", n.boundSrcToNode);
//                    }
//                    printf("\n");
//                fflush(stdout);
            }
        }
        currentLayer.resize(currentLayer.size() - fragmentSize);

        if (newSolution)
        {
            printf("[%7.2fs] SOLUTION     | Visited = %10ld | Cost = %7.2f | Value = ", expandedNodes, RuntimeMonitor::elapsedSeconds(start), bestNode.boundSrcToNode);
            printLabels(bestNode);
            printf("\n");
            fflush(stdout);
        }
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (pBound != Model::worstValue())
        {
            printf("COMPLETED    | Visited = %10ld | Cost = %7.2f | Value = ", expandedNodes, bestNode.boundSrcToNode);
            printLabels(bestNode);
            printf("\n");
        }
        else
        {
            printf("INFEASIBLE   | Visited = %10ld\n", expandedNodes);
        }
    }
    else
    {
        printf("TIMEOUT      | Visited = %10ld\n", expandedNodes);
    }
    fflush(stdout);

    return EXIT_SUCCESS;
}