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
    for(auto const & ni : *nodesInfo)
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
    LayerInfoType * layerInfo = nullptr;
    StackAllocator * gAllocator = gpu ? new StackAllocator(mallocManaged(lh->gpuMemSize), lh->gpuMemSize) : nullptr;

    // Initialize first layer
    Node root;
    root.state = model->initial();
    root.boundSrcToNode = 0;
    root.nEdgesSrcToNode = 0;
    currentLayer->push_back(root);

    // Top-down construction
    i32 layerIdx = 0;
    auto start = RuntimeMonitor::cputime();
    while (not currentLayer->empty())
    {
        auto elapsed = RuntimeMonitor::elapsedSince(start);
        if (elapsed / 1000 > timeout) break;

        if (gpu)
        {
            // Offload computation
            {
                // Clear
                LayerHelperType::clear(&layerInfo, gAllocator);

                // Parents
                LayerHelperType::initParents(*currentLayer, layerInfo, gAllocator);
                cudaMemPrefetchAsync(
                        gAllocator->getMem(),
                        gAllocator->calcUsedMemSize(),
                        lh->memLocGpu,
                        0,
                        lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();

                // Labels
                i32 blockSize = 128;
                dim3 gridSize = roundUpDivPosInt<i32>(layerInfo->nParents, blockSize);
                calcLabelsKernel<Model><<<gridSize, blockSize, 0, lh->gpuMainQueue>>>(model, layerInfo, DDExact);
                CHECK_LAST_CUDA_ERROR();
                cudaMemPrefetchAsync(
                        layerInfo,
                        sizeof(LayerInfoType),
                        lh->memLocCpu,
                        0,
                        lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();
                cudaStreamSynchronize(lh->gpuMainQueue);
                CHECK_LAST_CUDA_ERROR();

                if (layerInfo->labelsPerParents > 0)
                {
                    LayerHelperType::initChildren(layerInfo, gAllocator);
                    LayerHelperType::initAux(layerInfo, gAllocator);
                    cudaMemPrefetchAsync(
                            layerInfo,
                            sizeof(LayerInfoType),
                            lh->memLocGpu,
                            0,
                            lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();

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

                    sortKernel<NodeInfo, RepBoundDecomposer><<<1, 1, 0, lh->gpuMainQueue>>>(
                            layerInfo->cubTmpMem,
                            layerInfo->cubTmpMemSize,
                            layerInfo->tmpChildrenInfo,
                            layerInfo->childrenInfo,
                            &layerInfo->nChildren);
                    CHECK_LAST_CUDA_ERROR();
                }
            }

            // Retrieve nodes
            {
                cudaEventSynchronize(lh->childrenOk);
                CHECK_LAST_CUDA_ERROR();
                cudaMemPrefetchAsync(
                        layerInfo,
                        sizeof(LayerInfoType),
                        lh->memLocCpu,
                        0,
                        lh->gpuAuxQueue);
                CHECK_LAST_CUDA_ERROR();
                cudaStreamSynchronize(lh->gpuAuxQueue);
                if (layerInfo->nChildren > 0)
                {
                    CHECK_LAST_CUDA_ERROR();
                    cudaMemPrefetchAsync(
                            layerInfo->children,
                            sizeof(Node) * layerInfo->nChildren,
                            lh->memLocCpu,
                            0,
                            lh->gpuAuxQueue);
                    CHECK_LAST_CUDA_ERROR();
                    cudaMemPrefetchAsync(
                            layerInfo->childrenInfo,
                            sizeof(NodeInfo) * layerInfo->nChildren,
                            lh->memLocCpu,
                            0,
                            lh->gpuMainQueue);
                    CHECK_LAST_CUDA_ERROR();
                    cudaDeviceSynchronize();
                    CHECK_LAST_CUDA_ERROR();
                }

            }

            nextLayer->clear();
            nextLayer->reserve(layerInfo->nChildren);
            for (i32 niIdx = 0; niIdx < layerInfo->nChildren; niIdx += 1)
            {
                if (not layerInfo->childrenInfo[niIdx].isRepresented)
                {
                    i32 const nIdx = layerInfo->childrenInfo[niIdx].idx;
                    nextLayer->push_back(layerInfo->children[nIdx]);
                }
                else
                {
                    break;
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

            // Initialize aux
            i32 const nextLayerMaxSize = currentLayer->size() * labelsPerParent;
            tmpLayer.clear();
            nextInfo.clear();
            tmpLayer.reserve(nextLayerMaxSize);
            nextInfo.reserve(nextLayerMaxSize);

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

            // Sort by hash
            auto cmpByRepBound = [](NodeInfo const &ni1, NodeInfo const &ni2)
            {
                tuple<i32, f64> key1 = {ni1.isRepresented, ni1.boundSrcToNode};
                tuple<i32, f64> key2 = {ni2.isRepresented, ni2.boundSrcToNode};
                return key1 < key2;
            };
            std::sort(nextInfo.begin(), nextInfo.end(), cmpByRepBound);

            nextLayer->clear();
            nextLayer->reserve(tmpLayerSize);
            for (i32 niIdx = 0; niIdx < tmpLayerSize; niIdx += 1)
            {
                if (not nextInfo[niIdx].isRepresented)
                {
                    i32 const nIdx = nextInfo[niIdx].idx;
                    nextLayer->resize(nextLayer->size()+1);
                    nextLayer->back() = tmpLayer[nIdx];
                }
                else
                {
                    break;
                }
            }
        }

        if (not nextLayer->empty())
        {
            printf("[%7.2fs] Layer = %3d | Nodes = %9lu -> %9lu ", elapsed / 1000, layerIdx, currentLayer->size(), nextLayer->size());
            Node const & best = nextLayer->front();
            printf(" | Cost = %7.2f | Value = ", best.boundSrcToNode);
            gfl::Array<u8>::print(best.labelsSrcToNode, best.labelsSrcToNode + best.nEdgesSrcToNode);
            printf("\n");
            fflush(stdout);
        }

        std::swap(currentLayer,nextLayer);
        layerIdx += 1;
    }

    return EXIT_SUCCESS;
}