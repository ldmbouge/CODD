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

    //auto cmpByCost = [](Node const & n1, Node const & n2) { return n1.cost < n2.cost; };
    //using LayerBufferType = std::priority_queue<Node,std::vector<Node>,decltype(cmpByCost)>;
    using LayerBufferType = std::vector<Node>;

    // Select GPU
    cudaSetDevice(0);

    // Parse arguments
    i64 initWidth = 1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool gpu = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for the TSPTW");
    options.add_options("Available")
            ("w,width", "Beam width", cxxopts::value(initWidth))
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
    auto *const model = new(roAllocator) Model();
    Model::parseFile(model, instance, roAllocator);

    std::cout << "Instance: " << instance << std::endl;
    std::cout << "Cities: " << model->n << std::endl;
    //std::cout << "Node: "; printMemSize(sizeof(Node)); std::cout << std::endl;
    std::cout << "GPU: " << (gpu ? "True" : "False") << std::endl;

    // Search
    f64 primalBound = Model::worstValue();

    // GPU
    LayerHelperType *lh = gpu ? new LayerHelperType() : nullptr;
    LayerInfoType *layerInfo = gpu ? mallocManaged<LayerInfoType>(sizeof(LayerInfoType)) : nullptr;
    StackAllocator *gAllocator = gpu ? new StackAllocator(mallocDevice(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Data structures
    std::vector<Node> * currentLayer = new std::vector<Node>();
    std::vector<Node> * nextLayer = new std::vector<Node>();
    std::vector<Node> tmpBuffer;
    std::vector<NodeInfo> nodesInfo;

    // Let's goo!
    i64 width = initWidth;
    bool isExact = false;
    Node bestNode;
    bestNode.boundSrcToNode = Model::worstValue();
    bool interrupted = false;
    auto start = RuntimeMonitor::cputime();
    while ((not isExact) and (not interrupted))
    {
        isExact = true;

        // Initialize first layer
        Node root;
        root.state = model->initial();
        root.boundSrcToNode = 0;
        root.nEdgesSrcToNode = 0;
        currentLayer->push_back(root);

        i32 layerIdx = 0;
        while (not currentLayer->empty())
        {

            printf("[%7.2fs] Layer %3d | Nodes %10ld | MemSize ", RuntimeMonitor::elapsedSeconds(start), layerIdx, currentLayer->size());
            printMemSize(sizeof(Node) * currentLayer->size());
            printf("\n");
            fflush(stdout);

            // Batching
            i64 const nParents = currentLayer->size();
            i32 const fanOut = model->lgf(currentLayer->front().state, DDExact).size(); // Big assumption
            i32 const maxBatchSize = lh->getMaxParents(nParents, fanOut);
            i32 const nBatches = roundUpDivPosInt<i32>(nParents, maxBatchSize);
            nextLayer->clear();
            for (i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
            {
                if (RuntimeMonitor::elapsedSeconds(start) > timeout)
                {
                    interrupted = true;
                    break;
                }

                // Offload computation
                {
                    i32 pBegin, pEnd;
                    getBeginEnd(pBegin, pEnd, bIdx, nBatches, nParents);
                    i32 const batchSize = pEnd - pBegin; // No + 1!

                    // Clear
                    LayerHelperType::clear(layerInfo, gAllocator);

                    // Parents
                    LayerHelperType::initParents(batchSize, layerInfo, gAllocator);
                    cudaMemcpyAsync(
                            layerInfo->parents,
                            currentLayer->data() + pBegin,
                            sizeof(Node) * batchSize,
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
                        //std::tie(gridSize.x, gridSize.y, gridSize.z) = calcGridSize(layerInfo->nParents);
                        gridSize = layerInfo->nParents;
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
                {
                    if (layerInfo->nChildren > 0)
                    {
                        tmpBuffer.resize(layerInfo->nChildren);
                        cudaMemcpyAsync(
                                tmpBuffer.data(),
                                layerInfo->children,
                                sizeof(Node) * layerInfo->nChildren,
                                cudaMemcpyDeviceToHost,
                                lh->gpuAuxQueue);
                        CHECK_LAST_CUDA_ERROR();

                        nodesInfo.resize(layerInfo->nChildren);
                        cudaMemcpyAsync(
                                nodesInfo.data(),
                                layerInfo->tmpChildrenInfo,
                                sizeof(NodeInfo) * layerInfo->nChildren,
                                cudaMemcpyDeviceToHost,
                                lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        cudaDeviceSynchronize();
                        CHECK_LAST_CUDA_ERROR();

                        i64 const tmpBufferSize = tmpBuffer.size();
                        for (i64 i = 0; i < tmpBufferSize; i += 1)
                        {
                            if (not nodesInfo[i].isRepresented)
                            {
                                i32 const nIdx = nodesInfo[i].idx;
                                nextLayer->push_back(tmpBuffer[nIdx]);
                            }
                        }
                    }
                }
            }

            //auto cmpByCost = [](Node const & n1,Node const & n2){return n1.cost< n2.cost;};
            //std::sort(nextLayer->begin(),nextLayer->end(), cmpByCost);
            i64 nextLayerSize = gfl::min<i64>(width, nextLayer->size());
            isExact = isExact and nextLayer->size() <= nextLayerSize;
            nextLayer->resize(nextLayerSize);

            Node const & bestNodeInLayer = nextLayer->front();
            if (model->isTarget(bestNodeInLayer.state) and Model::better(bestNodeInLayer.boundSrcToNode, bestNode.boundSrcToNode))
            {
                bestNode = bestNodeInLayer;
                printf("[%7.2fs] BETTER SOLUTION Cost = %7.2f | Value = ",  RuntimeMonitor::elapsedSeconds(start), bestNode.boundSrcToNode);
                printLabels(bestNode);
                printf("\n");
                primalBound = bestNode.boundSrcToNode;
            }

            std::swap(currentLayer, nextLayer);
            layerIdx += 1;
        }

        width *= 2;
        printf("[%7.2fs] INCREASING WIDTH TO %d\n",  RuntimeMonitor::elapsedSeconds(start), width);
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (bestNode.boundSrcToNode != Model::worstValue())
        {
            printf("COMPLETED Cost = %7.2f | Value = ", bestNode.boundSrcToNode);
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