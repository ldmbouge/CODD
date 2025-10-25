#include "codd.hpp"
#include "LayerEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>

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
    using LayerInfoType = LayerInfo<Node>;
    using LayerBufferType =  std::vector<Node>;

    // Select GPU
    cudaSetDevice(0);

    // Parse arguments
    i64 width = -1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool gpu = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for DIDP models");
    options.add_options("Available")
            ("w,width", "Beam width", cxxopts::value(width))
            ("g,gpu", "Use GPU acceleration", cxxopts::value(gpu))
            ("h,help", "Show this help message and exit")
            ("i,instance", "Path to the instance file", cxxopts::value(instance))
            ("t,timeout", "Timeout in seconds", cxxopts::value(timeout));
    options.parse_positional({"instance"});
    options.custom_help("<OPTIONS>");
    options.positional_help("<INSTANCE>");
    auto const result = options.parse(argc, argv);

    // Option validation
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

    auto *const readOnlyMem =
#ifdef __CUDACC__
        mallocManaged<void>(ReadOnlyMemSize);
#else
        mallocStd<void>(ReadOnlyMemSize);
#endif

    StackAllocator roAllocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new(roAllocator) Model();
    Model::parseFile(model, instance, roAllocator);

    std::cout << "Instance: " << instance << std::endl;
    std::cout << "GPU: " << (gpu ? "True" : "False") << std::endl;
    std::cout << "Beam Width: ";
    if (width <= 0)
        std::cout << "Auto" << std::endl;
    else
        std::cout << width << std::endl;
    //std::cout << "Node: "; printMemSize(sizeof(Node)); std::cout << std::endl;

    // Search
    Node bestNode;
    f64 pBound = Model::worstValue();
    f64 dBound = Model::bestValue();

    // GPU
    LayerHelperType         *lh = gpu ? new LayerHelperType() : nullptr;
    LayerInfoType *   layerInfo = gpu ? mallocManaged<LayerInfoType>(sizeof(LayerInfoType)) : nullptr;
    StackAllocator * gAllocator = gpu ? new StackAllocator(mallocDevice(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Layers buffers
    std::vector<LayerBufferType> layers;
    std::vector<LabelsInfo> labelsInfo;
    std::vector<i64> maxNodesToExpand;

    // Initialize first layer
    Node root;
    root.state = model->initial();
    if constexpr (Model::has_simple_lgf)
    {
        root.labels = model->lgf(root.state, DDExact);
    }
    else
    {
        root.labels = model->lgf(root.state, DDExact, pBound, dBound);
    }
    root.boundSrcToNode = 0;
    root.nEdgesSrcToNode = 0;
    layers.emplace_back();
    layers.back().push_back(root);
    LabelsInfo rootLabelsInfo;
    rootLabelsInfo.update(root.labels.slc());
    labelsInfo.push_back(rootLabelsInfo);
    maxNodesToExpand.push_back(1);

    // Let's goo!
    auto start = RuntimeMonitor::cputime();
    bool interrupted = false;
    bool newSolution = false;
    auto isLayerEmpty = [](LayerBufferType const &l) { return l.empty(); };
    auto isQueueEmpty = [&]{return std::all_of(layers.begin(), layers.end(), isLayerEmpty); };
    auto lastNotEmpty = [&]{
        i32 lIdx = -1;
        for(i32 i = 0; i < layers.size(); i += 1)
        {
            printf(" I %3d | S %10lu\n", i, layers[i].size());
           if (not layers[i].empty())
           {
              lIdx = i;
           }
        }
        return lIdx;
    };
    while (not isQueueEmpty())
    {
        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }
        newSolution = false;

        // Input
        i32 const layerIdx = lastNotEmpty();
        if (layerIdx == layers.size()-1)
        {
            layers.emplace_back();
            labelsInfo.emplace_back();
            maxNodesToExpand.push_back(1); // Not 0!
        }
        auto & currentLayer = layers[layerIdx];
        auto & nextLayer = layers[layerIdx + 1];

        // Processing
        {
            //Batching
            maxNodesToExpand[layerIdx] = gfl::max<i64>(maxNodesToExpand[layerIdx], lh->getMaxParents(labelsInfo[layerIdx].nLabels));
            if (width > 0 )
                maxNodesToExpand[layerIdx] = gfl::min<i64>(maxNodesToExpand[layerIdx], width);
            i64 const nParents = gfl::min<i64>(currentLayer.size(), width <= 0 ? maxNodesToExpand[layerIdx] : width);
            i32 const nBatches = roundUpDivPosInt<i32>(nParents, maxNodesToExpand[layerIdx]);
            i64 expandedNodes = 0;
            for (i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
            {

                i64 pBegin, pEnd;
                getBeginEnd(pBegin, pEnd, bIdx, nBatches, nParents);
                i64 const currentBatchSize = pEnd - pBegin; // No + 1!
                expandedNodes += currentBatchSize;
                Node const * const currentBatch = currentLayer.data() + currentLayer.size() - expandedNodes;

                printf("[%7.2fs] Layer %3d | Nodes = %10ld  -> %10ld | MemSize = ",
                       RuntimeMonitor::elapsedSeconds(start),
                       layerIdx,
                       currentLayer.size(),
                       currentLayer.size()-currentBatchSize);
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
                printf(" | Batch %3d/%3d | Width = %d\n", bIdx+1, nBatches, currentBatchSize);

                fflush(stdout);

                // Offload computation
                {
                    // Clear
                    LayerHelperType::clear(layerInfo, gAllocator);
                    layerInfo->labelsInfo = labelsInfo[layerIdx];

                    // Parents
                    LayerHelperType::initParents(currentBatchSize, layerInfo, gAllocator);
                    cudaMemcpyAsync(
                            layerInfo->parents,
                            currentBatch + pBegin,
                            sizeof(Node) * currentBatchSize,
                            cudaMemcpyHostToDevice,
                            lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

                    LayerHelperType::initChildren(layerInfo, gAllocator);
                    LayerHelperType::initAux(layerInfo, gAllocator);

                    i32 blockSize = 32;
                    dim3 gridSize;
                    std::tie(gridSize.x, gridSize.y, gridSize.z) = calcGridSize(layerInfo->nParents);
                    calcChildrenKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                            model,
                            layerInfo,
                            layerInfo->childrenInfo,
                            pBound,
                            DDCtx);
                    CHECK_LAST_CUDA_ERROR();
                    cudaStreamSynchronize(lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

                    if (layerInfo->nChildren > 0)
                    {
                       //  Representatives
                        cub::DeviceRadixSort::SortKeys(
                                layerInfo->cubTmpMem,
                                layerInfo->cubTmpMemSize,
                                layerInfo->childrenInfo,
                                layerInfo->tmpChildrenInfo,
                                layerInfo->nChildren,
                                HashDecomposer{},
                                lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        blockSize = 128;
                        gridSize = roundUpDivPosInt<i32>(layerInfo->nChildren, blockSize);
                        calcReprKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
                        CHECK_LAST_CUDA_ERROR();

                        cub::DeviceRadixSort::SortKeys(
                                layerInfo->cubTmpMem,
                                layerInfo->cubTmpMemSize,
                                layerInfo->tmpChildrenInfo,
                                layerInfo->childrenInfo,
                                layerInfo->nChildren,
                                IdxDecomposer{},
                                lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        cub::DeviceSelect::FlaggedIf(
                                layerInfo->cubTmpMem,
                                layerInfo->cubTmpMemSize,
                                layerInfo->children,
                                layerInfo->childrenInfo,
                                &layerInfo->nChildren,
                                layerInfo->nChildren,
                                SelectNotRep{},
                                lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        cudaStreamSynchronize(lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        // Labels
                        layerInfo->labelsInfo.reset();
                        blockSize = 128;
                        gridSize = roundUpDivPosInt<i32>(layerInfo->nChildren, blockSize);
                        calcLabelsKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                                model,
                                layerInfo,
                                DDExact,
                                pBound,
                                dBound);
                        CHECK_LAST_CUDA_ERROR();
                    }
                }

                // Retrieve nodes
                cudaStreamSynchronize(lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();

                if (layerInfo->nChildren > 0)
                {
                    i64 const nextLayerOldSize = nextLayer.size();
                    nextLayer.resize(nextLayerOldSize + layerInfo->nChildren);
                    cudaMemcpyAsync(
                            nextLayer.data() + nextLayerOldSize,
                            layerInfo->children,
                            sizeof(Node) * layerInfo->nChildren,
                            cudaMemcpyDeviceToHost,
                            lh->gpuAuxQueue);
                    CHECK_LAST_CUDA_ERROR();

                    labelsInfo[layerIdx+1].update(layerInfo->labelsInfo);

                    if(model->isTarget(nextLayer.back().state))
                    {
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

                }
            }

            currentLayer.resize(currentLayer.size() - expandedNodes);

            if (newSolution)
            {
                printf("[%7.2fs] SOLUTION  | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), bestNode.boundSrcToNode);
                printLabels(bestNode);
                printf("\n");
                fflush(stdout);
            }
        }
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (bestNode.boundSrcToNode != Model::worstValue())
        {
            printf("COMPLETED | Cost = %7.2f | Value = ", bestNode.boundSrcToNode);
            printLabels(bestNode);
            printf("\n");
        }
        else
        {
            printf("INFEASIBLE\n");
        }
    }
    else
    {
        printf("TIMEOUT\n");
    }
    fflush(stdout);

    return EXIT_SUCCESS;
}