#include "tsptw_model_1.hpp"
#include "codd.hpp"
#include "pool.hpp"
#include "vec.hpp"
#include "LayerEngine.cuh"
#include <StackAllocator.hpp>
#include <queue>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory

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

template<typename State, typename Labels>
void printLabels(LightNode<State, Labels> const & node)
{
    using namespace gfl;
    Array<u8>::print(node.labelsSrcToNode, node.labelsSrcToNode + node.nEdgesSrcToNode);
}

int main(int argc,char* argv[])
{
    using namespace gfl;

    using Model = TSPTW1;
    using State = Model::State;
    using Labels = Model::Labels;
    using Node = LightNode<State, Labels>;

    using LayerHelperType = LayerHelper<State, Labels>;
    using LayerInfoType = LayerInfo<State, Labels>;
    using LayerBufferType =  std::vector<Node>;

    // Select GPU
    cudaSetDevice(1);

    // Parse arguments
    i64 width = -1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool gpu = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for the TSPTW");
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
    std::cout << "Cities: " << model->n << std::endl;
    std::cout << "GPU: " << (gpu ? "True" : "False") << std::endl;
    std::cout << "Beam Width: ";
    if (width <= 0)
        std::cout << "Auto" << std::endl;
    else
        std::cout << width << std::endl;
    //std::cout << "Node: "; printMemSize(sizeof(Node)); std::cout << std::endl;

    // Search
    f64 primalBound = Model::worstValue();
    std::vector<Node> tmpLayer;
    std::vector<NodeInfo> tmpInfo;

    // GPU
    LayerHelperType *lh = gpu ? new LayerHelperType() : nullptr;
    LayerInfoType *layerInfo = gpu ? mallocManaged<LayerInfoType>(sizeof(LayerInfoType)) : nullptr;
    StackAllocator *gAllocator = gpu ? new StackAllocator(mallocDevice(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Layers buffers
    std::vector<LayerBufferType> layers;

    // Initialize first layer
    Node root;
    root.state = model->initial();
    root.boundSrcToNode = 0;
    root.nEdgesSrcToNode = 0;
    layers.emplace_back();
    layers.back().push_back(root);

    // Let's goo!
    auto start = RuntimeMonitor::cputime();
    Node bestNode;
    bestNode.boundSrcToNode = Model::worstValue();
    bool interrupted = false;
    bool newSolution = false;
    auto isLayerEmpty = [](LayerBufferType const &l) { return l.empty(); };
    auto isQueueEmpty = [&]{return std::all_of(layers.begin(), layers.end(), isLayerEmpty); };
    auto lastNotEmpty = [&]{ return std::find_if_not(layers.rbegin(), layers.rend(), isLayerEmpty); };
    while (not isQueueEmpty())
    {
        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }
        newSolution = false;

        // Input
        i32 const nLayers = layers.size();
        i32 const layerIdx = std::distance(layers.begin(), --lastNotEmpty().base()); // It is correct, do not ask.
        if (layerIdx == layers.size()-1)
        {
            layers.emplace_back();
            layers.back().reserve(layers[layerIdx].size());
        }
        auto & currentLayer = layers[layerIdx];
        auto & nextLayer = layers[layerIdx + 1];

        // Processing
        {
            //Batching
            i32 const fanOut = model->lgf(currentLayer.front().state, DDExact).size(); // Big assumption
            i32 const maxNodeToExpand = lh->getMaxParents(currentLayer.size(), fanOut);
            i64 const nParents = width <= 0 ? maxNodeToExpand : gfl::min<i64>(currentLayer.size(), width);
            i32 const nBatches = roundUpDivPosInt<i32>(nParents, maxNodeToExpand);
            i64 expandedNodes = 0;
            for (i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
            {

                i32 pBegin, pEnd;
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
                if(bestNode.boundSrcToNode != Model::worstValue())
                {

                    printf("%7.2f", bestNode.boundSrcToNode);
                }
                else
                {
                    printf("?");
                }
                printf(" | Batch %3d/%3d\n", bIdx+1, nBatches);

                fflush(stdout);

                // Offload computation
                {
                    // Clear
                    LayerHelperType::clear(layerInfo, gAllocator);

                    // Parents
                    LayerHelperType::initParents(currentBatchSize, layerInfo, gAllocator);
                    cudaMemcpyAsync(
                            layerInfo->parents,
                            currentBatch + pBegin,
                            sizeof(Node) * currentBatchSize,
                            cudaMemcpyHostToDevice,
                            lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

                    // Labels
                    i32 blockSize = 128;
                    dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
                    calcLabelsKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(model, layerInfo, DDExact);
                    CHECK_LAST_CUDA_ERROR();
                    cudaStreamSynchronize(lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

                    if (layerInfo->labelsPerParents > 0)
                    {
                        LayerHelperType::initChildren(layerInfo, gAllocator);
                        LayerHelperType::initAux(layerInfo, gAllocator);

                        blockSize = 32;
                        std::tie(gridSize.x, gridSize.y, gridSize.z) = calcGridSize(layerInfo->nParents);
                        calcChildrenKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(
                                model,
                                layerInfo,
                                layerInfo->childrenInfo,
                                primalBound,
                                DDCtx);
                        CHECK_LAST_CUDA_ERROR();
                        cudaStreamSynchronize(lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        if (layerInfo->nChildren > 0)
                        {
                            // Representatives
                            sortKernel<NodeInfo, HashDecomposer><<<1, 1, 0, lh->gpuMainQueue>>>(
                                    layerInfo->cubTmpMem,
                                    layerInfo->cubTmpMemSize,
                                    layerInfo->childrenInfo,
                                    layerInfo->tmpChildrenInfo,
                                    &layerInfo->nChildren);
                            CHECK_LAST_CUDA_ERROR();

                            blockSize = 128;
                            gridSize = roundUpDivPosInt<i32>(layerInfo->nChildren, blockSize);
                            calcReprKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
                            CHECK_LAST_CUDA_ERROR();
                        }
                    }
                }

                // Retrieve nodes
                if (layerInfo->nChildren > 0)
                {
                    tmpLayer.resize(layerInfo->nChildren);
                    cudaMemcpyAsync(
                            tmpLayer.data(),
                            layerInfo->children,
                            sizeof(Node) * layerInfo->nChildren,
                            cudaMemcpyDeviceToHost,
                            lh->gpuAuxQueue);
                    CHECK_LAST_CUDA_ERROR();

                    tmpInfo.resize(layerInfo->nChildren);
                    cudaMemcpyAsync(
                            tmpInfo.data(),
                            layerInfo->tmpChildrenInfo,
                            sizeof(NodeInfo) * layerInfo->nChildren,
                            cudaMemcpyDeviceToHost,
                            lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

                    cudaDeviceSynchronize();
                    CHECK_LAST_CUDA_ERROR();

                    i64 const tmpLayerSize = tmpLayer.size();
                    for (i64 i = 0; i < tmpLayerSize; i += 1)
                    {
                        if (not tmpInfo[i].isRepresented)
                        {
                            i32 const nIdx = tmpInfo[i].idx;
                            Node const &n = tmpLayer[nIdx];

                            if (not model->isTarget(n.state))
                            {
                                nextLayer.push_back(n);
                            }
                                else if (Model::better(n.boundSrcToNode, bestNode.boundSrcToNode))
                            {
                                bestNode = n;
                                primalBound = bestNode.boundSrcToNode;
                                newSolution = true;
                            }
                        }
                    }
                }
            }

            currentLayer.resize(currentLayer.size() - expandedNodes);
//            auto cmpByBnd = [](Node const & n1, Node const & n2) {return n1.boundSrcToNode > n2.boundSrcToNode;};
//            std::sort(currentLayer.begin(), currentLayer.end(), cmpByBnd);

            if (newSolution)
            {
                printf("[%7.2fs] SOLUTION  | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), bestNode.boundSrcToNode);
                printLabels(bestNode);
                printf("\n");
                fflush(stdout);
            };

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