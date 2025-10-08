#include "tsptw_model_1.hpp"
#include "codd.hpp"
#include "common.h"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>

constexpr auto static ReadOnlyMemSize{48 * 1024}; // Cached in shared memory

template<typename Model>
void checkStatePair(NodeInfo & iInfo, typename Model::State const & iState, NodeInfo & jInfo, typename Model::State const & jState)
{
    using namespace gfl;

    if (Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode))
    {
        if (Model::State::equal(iState, jState))
        {
            jInfo.isRepresented = static_cast<i32>(true);
        }
    }
    if (Model::has_dom and Model::betterEq(jInfo.boundSrcToNode, iInfo.boundSrcToNode))
    {
        if (Model::dom(jState, iState))
        {
            iInfo.isRepresented = static_cast<i32>(true);
        }
    }
    if (Model::has_dom and Model::betterEq(iInfo.boundSrcToNode, jInfo.boundSrcToNode))
    {
        if (Model::dom(iState, jState))
        {
            jInfo.isRepresented = static_cast<i32>(true);
        }
    }
}


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
    
    using TSPTWModel = TSPTW1;
    using State = TSPTWModel::State;
    using Labels = TSPTWModel::Labels;
    using LNode = LightNode<State, Labels>;

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

    StackAllocator allocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new (allocator) TSPTWModel();
    TSPTWModel::parseFile(model, instance, allocator);

    std::cout << "Instance: " << instance << std::endl;
    std::cout << "GPU: " << (gpu ? "True" : "False") << std::endl;

    // Search
    f64 primalBound = numeric_limits<f64>::max();
    std::list<std::vector<LNode>> layers;

    // Initialize first layer
    LNode root;
    root.state = model->initial();
    root.boundSrcToNode = 0;
    root.nEdgesSrcToNode = 0;
    std::vector<LNode> firstLayer = {root};

    // Top-down construction
    std::vector<LNode> tmpLayer;
    std::vector<NodeInfo> nodeInfoNext;
    std::vector<LNode> * currentLayer = &firstLayer;
    i32 layerIdx = 0;
    auto start = RuntimeMonitor::cputime();
    while (not currentLayer->empty())
    {
        auto elapsed = RuntimeMonitor::elapsedSince(start);
        if (elapsed / 1000 > timeout)
        {
            break;
        }

        // Labels
        i32 const currentLayerSize = currentLayer->size();
        i32 maxLabel = 0;
        i32 labelsPerParent = 0;
        for (i32 i = 0; i < currentLayerSize; i += 1)
        {
            LNode & n = currentLayer->at(i);
            n.labels = model->lgf(n.state, DDExact);
            auto [smallest,largest,count] = n.labels.slc();
            maxLabel = gfl::max<i32>(maxLabel, largest);
            labelsPerParent = gfl::max<i32>(labelsPerParent, count);
        }

        // Initialize next layer
        i32 const nextLayerMaxSize = currentLayer->size() * labelsPerParent;
        tmpLayer.clear();
        tmpLayer.reserve(nextLayerMaxSize);
        nodeInfoNext.clear();
        nodeInfoNext.reserve(nextLayerMaxSize);

        // Calculate next layer
        for (i32 pIdx = 0; pIdx < currentLayerSize; pIdx += 1)
        {
            LNode & pNode = currentLayer->at(pIdx);
            for (auto const label : pNode.labels)
            {
                auto cState = model->stf(pNode.state, label);
                if (cState.has_value())
                {
                    f64 cBoundSrcToNode = pNode.boundSrcToNode + model->scf(pNode.state, label);
                    f64 cHeuristicNodeToSink = TSPTWModel::has_local ? model->local(cState.value(), DDCtx) : 0;
                    f64 cCost = cBoundSrcToNode + cHeuristicNodeToSink;
                    if (TSPTWModel::better(cCost, primalBound))
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
                        nodeInfoNext.resize(nodeInfoNext.size() + 1);
                        auto & cInfo = nodeInfoNext.back();
                        cInfo.hash = TSPTWModel::has_dom ? TSPTWModel::domHash(cNode.state) : State::hash(cNode.state);
                        cInfo.boundSrcToNode = cBoundSrcToNode;
                        cInfo.cost = cCost;
                        cInfo.id = pIdx * (maxLabel + 1) + label;
                        cInfo.idx = nodeInfoNext.size() - 1;
                        cInfo.isRepresented = false;
                    }
                }
            }
        }

//        printf("After Child Gen\n");
//        printNodeInfo(&nodeInfoNext);

        // Sort by hash
        auto cmpByHash = [](NodeInfo const & ni1, NodeInfo const & ni2) {return ni1.hash < ni2.hash;};
        std::sort(nodeInfoNext.begin(), nodeInfoNext.end(), cmpByHash);

//        printf("After Sort By Hash\n");
//        printNodeInfo(&nodeInfoNext);

        // Find representatives
        i32 const tmpLayerSize = tmpLayer.size();
        for (i32 i = 0; i < tmpLayerSize; i += 1)
        {
            auto & iInfo = nodeInfoNext[i];
            auto const iState = tmpLayer[iInfo.idx].state;
            assert(iInfo.idx >= 0);
            assert(iInfo.idx < tmpLayerSize);
            for (i32 j = i + 1; j < tmpLayerSize; j += 1)
            {
                auto & jInfo = nodeInfoNext[j];
                auto const jState = tmpLayer[jInfo.idx].state;
                assert(jInfo.idx >= 0);
                assert(jInfo.idx < tmpLayerSize);
                if (iInfo.hash == jInfo.hash)
                {
                    checkStatePair<TSPTWModel>(iInfo,iState,jInfo,jState);
                }
                else
                {
                    break;
                }
            }
        }

//        printf("After Find Rep\n");
//        printNodeInfo(&nodeInfoNext);

        // Sort by hash
        auto cmpByRepCost = [](NodeInfo const & ni1, NodeInfo const & ni2) {
            tuple<i32,f64> key1 = {ni1.isRepresented, ni1.cost};
            tuple<i32,f64> key2 = {ni2.isRepresented, ni2.cost};
            return key1 < key2;
        };
        std::sort(nodeInfoNext.begin(), nodeInfoNext.end(), cmpByRepCost);

//        printf("After Sort By Rep\n");
//        printNodeInfo(&nodeInfoNext);

        layers.emplace_back();
        std::vector<LNode> * nextLayer = & layers.back();
        nextLayer->reserve(tmpLayerSize);
        for(i32 niIdx = 0; niIdx < tmpLayerSize; niIdx += 1)
        {
            if (not nodeInfoNext[niIdx].isRepresented)
            {
                i32 const nIdx = nodeInfoNext[niIdx].idx;
                nextLayer->push_back(tmpLayer[nIdx]);
            }
            else
            {
                break;
            }
        }

        printf("[%7.2f] Layer = %2d | Nodes = %9lu -> %9lu ", elapsed / 1000, layerIdx, currentLayer->size(), nextLayer->size());
        if (not nextLayer->empty())
        {
            LNode const & best = nextLayer->front();
            printf(" | Cost = %7.2f | Value = ", best.boundSrcToNode);
            gfl::Array<u8>::print(best.labelsSrcToNode, best.labelsSrcToNode + best.nEdgesSrcToNode);
            printf("\n");
        }
        fflush(stdout);

        currentLayer = nextLayer;
        layerIdx += 1;
    }

    return EXIT_SUCCESS;
}