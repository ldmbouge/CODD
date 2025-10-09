#include "tsptw_model_1.hpp"
#include "codd.hpp"
#include "LayerEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>

constexpr auto static ReadOnlyMemSize{48 * 1024}; // Cached in shared memory

void printNodeInfo(std::vector<NodeInfo> const * const nodesInfo)
{
    for (auto const & ni : *nodesInfo)
    {
        printf("HASH = %lu, BOUND = %.1f, COST = %.1f, ID = %ld, IDX = %d, IS_REP = %d\n",
               ni.hash,
               ni.boundSrcToNode,
               ni.cost,
               ni.id,
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

    // Select GPU
    cudaSetDevice(0);

    // Parse arguments
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool gpu = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for the TSPTW");
    options.add_options("Available")
            ("g,gpu",      "Use GPU acceleration", cxxopts::value(gpu))
            ("h,help"    , "Show this help message and exit")
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
    std::cout << "GPU: " << (gpu ? "True" : "False") << std::endl;

    // Search
    f64 primalBound = numeric_limits<f64>::max();
    std::vector<Node> * currentLayer = new std::vector<Node>();
    std::vector<Node> * nextLayer = new std::vector<Node>();
    std::vector<Node> tmpLayer;
    std::vector<NodeInfo> nextInfo;

    // GPU
    LayerHelperType * lh = gpu ? new LayerHelperType() : nullptr;
    LayerInfoType * layerInfo = gpu ? mallocManaged<LayerInfoType>(sizeof(LayerInfoType)) : nullptr;
    StackAllocator * gAllocator = gpu ? new StackAllocator(mallocDevice(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Initialize first layer
    Node root;
    root.state = model->initial();
    root.boundSrcToNode = 0;
    root.nEdgesSrcToNode = 0;
    currentLayer->push_back(root);

    // Top-down construction
    i32 layerIdx = 0;
    auto start = RuntimeMonitor::cputime();
    bool interrupted = false;
    Node bestNode;
    bestNode.boundSrcToNode = Model::worstValue();
    while (not currentLayer->empty())
    {
        auto elapsed = RuntimeMonitor::elapsedSeconds(start);
        if (elapsed > timeout)
        {
            interrupted = true;
            break;
        }

        // Init next layer
        i32 const nodesFanOut = model->lgf(currentLayer->front().state, DDExact).size(); // Big assumption
        i32 const maxNextLayerSize = currentLayer->size() * nodesFanOut;
        tmpLayer.clear();
        nextInfo.clear();
        nextLayer->clear();
        tmpLayer.reserve(maxNextLayerSize);
        nextInfo.reserve(maxNextLayerSize);
        nextLayer->reserve(maxNextLayerSize);
        Node bestNodeInlayer;
        bestNodeInlayer.boundSrcToNode = Model::worstValue();

        if (gpu)
        {
            i32 const nParents = currentLayer->size();
            i32 const maxParentsPerBatch = lh->getMaxParents(nParents, nodesFanOut) ;
            i32 const nBatches = roundUpDivPosInt<i32>(nParents,maxParentsPerBatch);
            for(i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
            {
                i32 pBegin, pEnd;
                getBeginEnd(pBegin, pEnd, bIdx, nBatches, nParents);
                i32 const currentBatchSize = pEnd - pBegin; // No + 1!

                // Offload computation
                {
                    // Clear
                    LayerHelperType::clear(layerInfo, gAllocator);

                    // Parents
                    LayerHelperType::initParents(currentBatchSize, layerInfo, gAllocator);
                    cudaMemcpyAsync(
                            layerInfo->parents,
                            currentLayer->data() + pBegin,
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
                        blockSize = roundUpToMultiple<i32>(layerInfo->labelsPerParents, 32);
                        gridSize = layerInfo->nParents;
                        i32 shrMemSize = sizeof(Node) * layerInfo->labelsPerParents + StackAllocator::DefaultAlign +
                                         sizeof(NodeInfo) * layerInfo->labelsPerParents;
                        calcChildrenKernel<Model><<<gridSize, blockSize, shrMemSize, lh->gpuMainQueue>>>(
                                model,
                                layerInfo,
                                layerInfo->childrenInfo,
                                primalBound,
                                DDCtx);
                        CHECK_LAST_CUDA_ERROR();
                        cudaEventRecord(lh->childrenOk, lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        // Representatives
                        sortKernel<NodeInfo, HashDecomposer><<<1, 1, 0, lh->gpuMainQueue>>>(
                                layerInfo->cubTmpMem,
                                layerInfo->cubTmpMemSize,
                                layerInfo->childrenInfo,
                                layerInfo->tmpChildrenInfo,
                                &layerInfo->nChildren);
                        CHECK_LAST_CUDA_ERROR();

                        blockSize = 128;
                        gridSize = roundUpDivPosInt<i32>(layerInfo->nParents * layerInfo->labelsPerParents, blockSize);
                        calcReprKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(layerInfo, layerInfo->tmpChildrenInfo);
                        CHECK_LAST_CUDA_ERROR();
                    }
                }

                // Retrieve nodes
                {
                    cudaEventSynchronize(lh->childrenOk);
                    CHECK_LAST_CUDA_ERROR();
                    if (layerInfo->nChildren > 0)
                    {
                        i32 const oldSize = tmpLayer.size();

                        tmpLayer.resize(oldSize+layerInfo->nChildren);
                        cudaMemcpyAsync(
                                tmpLayer.data() + oldSize,
                                layerInfo->children,
                                sizeof(Node) * layerInfo->nChildren,
                                cudaMemcpyDeviceToHost,
                                lh->gpuAuxQueue);
                        CHECK_LAST_CUDA_ERROR();

                        nextInfo.resize(oldSize+layerInfo->nChildren);
                        cudaMemcpyAsync(
                                nextInfo.data() + oldSize,
                                layerInfo->tmpChildrenInfo,
                                sizeof(NodeInfo) * layerInfo->nChildren,
                                cudaMemcpyDeviceToHost,
                                lh->gpuMainQueue);
                        CHECK_LAST_CUDA_ERROR();

                        cudaDeviceSynchronize();
                        CHECK_LAST_CUDA_ERROR();

                        if(nBatches > 1)
                        {
                            auto const firstNode = tmpLayer[nextInfo[oldSize].idx];
                            printf("           Batch = %3d | Nodes = %9d -> %9d | Cost = %7.2f\n", bIdx, currentBatchSize, layerInfo->nChildren, firstNode.boundSrcToNode);
                            fflush(stdout);
                        }
                    }
                }
            }
        }
        else
        {
            // Labels
            i32 const currentLayerSize = currentLayer->size();
            i32 maxLabel = 0;
            i32 labelsPerParent = 0;
            for (i32 i = 0; i < currentLayerSize; i += 1)
            {
                Node &n = currentLayer->at(i);
                n.labels = model->lgf(n.state, DDExact);
                auto [smallest, largest, count] = n.labels.slc();
                maxLabel = gfl::max<i32>(maxLabel, largest);
                labelsPerParent = gfl::max<i32>(labelsPerParent, count);
            }

            // Calculate next layer
            for (i32 pIdx = 0; pIdx < currentLayerSize; pIdx += 1)
            {
                Node & pNode = currentLayer->at(pIdx);
                for (auto const label: pNode.labels)
                {
                    auto cState = model->stf(pNode.state, label);
                    if (cState.has_value())
                    {
                        f64 cBoundSrcToNode = pNode.boundSrcToNode + model->scf(pNode.state, label);
                        f64 cHeuristicNodeToSink = Model::has_local ? model->local(cState.value(), DDCtx) : 0;
                        f64 cCost = cBoundSrcToNode + cHeuristicNodeToSink;
                        if (Model::better(cCost, primalBound))
                        {
                            // Node
                            tmpLayer.resize(tmpLayer.size() + 1);
                            auto & cNode = tmpLayer.back();
                            cNode.state = cState.value();
                            cNode.boundSrcToNode = cBoundSrcToNode;
                            memcpy(cNode.labelsSrcToNode, pNode.labelsSrcToNode, sizeof(cNode.labelsSrcToNode));
                            cNode.labelsSrcToNode[pNode.nEdgesSrcToNode] = label;
                            cNode.nEdgesSrcToNode = pNode.nEdgesSrcToNode + 1;

                            // NodeInfo
                            nextInfo.resize(nextInfo.size() + 1);
                            auto & cInfo = nextInfo.back();
                            cInfo.hash = Model::has_dom ? Model::domHash(cNode.state) : State::hash(cNode.state);
                            cInfo.boundSrcToNode = cBoundSrcToNode;
                            cInfo.cost = cCost;
                            cInfo.id = pIdx * (maxLabel + 1) + label;
                            cInfo.idx = nextInfo.size() - 1;
                            cInfo.isRepresented = false;
                        }
                    }
                }
            }

            // Sort by hash
            auto cmpByHash = [](NodeInfo const &ni1, NodeInfo const &ni2)
            { return ni1.hash < ni2.hash; };
            std::sort(nextInfo.begin(), nextInfo.end(), cmpByHash);

            // Find representatives
            i32 const tmpLayerSize = tmpLayer.size();
            for (i32 i = 0; i < tmpLayerSize; i += 1)
            {
                auto & iInfo = nextInfo[i];
                auto const iState = tmpLayer[iInfo.idx].state;
                assert(iInfo.idx >= 0);
                assert(iInfo.idx < tmpLayerSize);
                for (i32 j = i + 1; j < tmpLayerSize; j += 1)
                {
                    auto &jInfo = nextInfo[j];
                    auto const jState = tmpLayer[jInfo.idx].state;
                    assert(jInfo.idx >= 0);
                    assert(jInfo.idx < tmpLayerSize);
                    if (iInfo.hash == jInfo.hash)
                    {
                        checkStatePair<Model>(iInfo, iState, jInfo, jState);
                    } else
                    {
                        break;
                    }
                }
            }
        }

        i32 const tmpLayerSize = tmpLayer.size();
        if (tmpLayerSize > 0)
        {
            for (i32 niIdx = 0; niIdx < tmpLayerSize; niIdx += 1)
            {
                if (not nextInfo[niIdx].isRepresented)
                {
                    i32 const nIdx = nextInfo[niIdx].idx;
                    nextLayer->resize(nextLayer->size() + 1);
                    Node const &n = tmpLayer[nIdx];
                    nextLayer->back() = n;
                    if (Model::better(n.boundSrcToNode, bestNodeInlayer.boundSrcToNode))
                    {
                        bestNodeInlayer = n;
                    }
                }
            }
        }
        if (model->isTarget(bestNodeInlayer.state) and Model::better(bestNodeInlayer.boundSrcToNode, bestNode.boundSrcToNode))
        {
            bestNode = bestNodeInlayer;
        }

        if (not nextLayer->empty())
        {
            printf("[%7.2fs] Layer = %3d | Nodes = %9lu -> %9lu | Cost = %7.2f\n", elapsed, layerIdx, currentLayer->size(), nextLayer->size(), bestNodeInlayer.boundSrcToNode);
            fflush(stdout);
        }

        std::swap(currentLayer,nextLayer);
        layerIdx += 1;
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (bestNode.boundSrcToNode != Model::worstValue())
        {
            printf("COMPLETED\n");
            printf("Cost = %7.2f | Value = ", bestNode.boundSrcToNode);
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