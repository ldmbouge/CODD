#include "codd.hpp"
#include "LayerEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <Array.hpp>
#include <span>

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory
constexpr auto static CpuMemSize{8ll * 1024ll * 1024ll * 1024ll}; // Same size GPU memory: 48 - 4 for runtime!)

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

template <typename Model, typename LayerInfoType>
void filterChildren(LayerInfoType * layerInfo, bool sort)
{
    using namespace gfl;

    i64 const & nChildren = layerInfo->nChildren;
    auto cmpByHash = [](NodeInfo const & a, NodeInfo const & b){return a.hash < b.hash;};
    std::sort(
        layerInfo->childrenInfo,
        layerInfo->childrenInfo + nChildren,
        cmpByHash);

    calcRep<Model>(
                nChildren,
                layerInfo->children,
                layerInfo->childrenInfo);

    countRep<Node>(
                layerInfo,
                layerInfo->childrenInfo);

    if (sort) {
        auto cmpByRepCost = [](NodeInfo const &a, NodeInfo const &b)
        {
            std::pair<u32, f64> const aa = {a.isRepresented, a.boundSrcToNode};
            std::pair<u32, f64> const bb = {b.isRepresented, b.boundSrcToNode};
            return aa < bb;
        };
        std::sort(
            layerInfo->childrenI using namespace gfl;

    i64 const & nChildren = layerInfo->nChildren;
    auto cmpByHash = [](NodeInfo const & a, NodeInfo const & b){return a.hash < b.hash;};
    std::sort(
        layerInfo->childrenInfo,
        layerInfo->childrenInfo + nChildren,
        cmpByHash);

    calcRep<Model>(
                nChildren,
                layerInfo->children,
                layerInfo->childrenInfo);

    countRep<Node>(
                layerInfo,
                layerInfo->childrenInfo);

    if (sort) {
        auto cmpByRepCost = [](NodeInfo const &a, NodeInfo const &b)
        {
            std::pair<u32, f64> constnfo,
            layerInfo->childrenInfo + nChildren,
            cmpByRepCost);
    }
    else {
        auto cmpByRep = [](NodeInfo const &a, NodeInfo const &b)
        {
            return a.isRepresented < b.isRepresented;
        };
        std::sort(
            layerInfo->childrenInfo,
            layerInfo->childrenInfo + nChildren,
            cmpByRep);
    }

    swapPtr(&layerInfo->children, &layerInfo->tmpChildren);
}

template <typename State>
float getStateSim(State const & a, State const & b)
{
    using namespace gfl;
    return 0.0;
}

template <typename State>
float getScore(State const & sBase, State const & sToEval)
{
    using namespace gfl;
    f64 const cBase = sBase.boundSrcToNode;
    f64 const cDiff = sToEval.boundSrcToNode - cBase;  // Cost difference
    f64 const sSim = getStateSim(sBase,sToEval);   // State similarity between 0.0 and 1.0, 0.0 <-> identical
    f64 const alpha = 1.0;                             // Scaling factor, default at 1.0

    //TODO Check branchless version

    if (cDiff < cBase * 0.1)           // Very similar cost, ignore the cost component
        return cBase * (alpha + sSim); // The more different is the state the better

    if (sSim < 0.1)                     // Very similar states, ignore state difference
        return cBase + (alpha * cDiff); // The smaller the cost difference the better

    return cBase + (alpha * cDiff * sSim); // The more the state is different, the more we "discount" the cost difference
}

template <typename Model, typename LayerInfoType>
void mergeChildren(LayerInfoType * layerInfo, gfl::i32 const width)
{
    // 1) Set $score(s) = cost(s)$ for each state $s$.
    // 2) Sort the states by increasing score.
    // 3) Divide the array of sorted states in chunks.
    // 4) For each chunk $C = [s_1,s_2,...,s_k]$:
    // 4.1) $s_1$ is the best state in the chunk by cost.
    // 4.2) Set $score(s_i) to promote (within the chunk) states with low cost that are different from $s_1$.
    // 5) Sort the states by increasing score. // This should promote (globally) heterogeneous states with low cost.
    // 6) Keep the best $w-1$ states, and merge all the others in 1 state.
    using namespace gfl;

    for(i64 i = 0; i < layerInfo->nChildren; i += 1)
    {
        i64 const cIdx = layerInfo->childrenInfo[i].idx;
        layerInfo->childrenInfo[i].score = layerInfo->children[cIdx].boundSrcToNode;
    }

    auto cmpByScore = [](NodeInfo const &a, NodeInfo const &b) { return a.score < b.score; };
    std::sort(layerInfo->childrenInfo, layerInfo->childrenInfo + layerInfo->nChildren, cmpByScore);

    i64 const nChunks = width;
    for (i32 chkIdx = 0; chkIdx < nChunks; chkIdx += 1)
    {
        i64 chkBegin, chkEnd;
        getBeginEnd(chkBegin,chkEnd,chkIdx, nChunks,layerInfo->nParents);
        i64 const baseIdx = layerInfo.childrenInfo[chkBegin].idx;
        auto const & sBase = layerInfo->children[baseIdx].state;
        for (i64 cIdx = chkBegin + 1; cIdx < chkEnd; cIdx += 1)
        {
            i64 const toEvalIdx = layerInfo.childrenInfo[cIdx].idx;
            auto const & sToEval = layerInfo->children[toEvalIdx];
            layerInfo->childrenInfo[cIdx].score = getScore(sBase,sToEval);
        }
    }

    std::sort(layerInfo->childrenInfo, layerInfo->childrenInfo + layerInfo->nChildren, cmpByScore);

    i64 const baseIdx = layerInfo->childrenInfo[width-1].idx;
    auto & nBase = layerInfo->children[baseIdx];
    nBase.isNotExact = 1;
    for (i64 cIdx = width; cIdx < layerInfo->nChildren; cIdx += 1)
    {
        i64 const toMergeIdx = layerInfo->childrenInfo[cIdx].idx;
        auto const & nToMerge = layerInfo->children[toMergeIdx];
        nBase.state = Model::smf(nBase.state, nToMerge.state);
        nBase.boundSrcToNode = Model::better(nBase.boundSrcToNode, nToMerge.boundSrcToNode) ?
                                    nToMerge.boundSrcToNode :
                                    nBase.boundSrcToNode;
    }
}

template<typename Node>
void printLabels(Node const & node)
{
    using namespace gfl;
    Array<u8>::print(node.labelsSrcToNode, node.labelsSrcToNode + node.nEdgesSrcToNode);
}

template<typename Model, typename Node>
int run_cadds_relaxed_seq(int argc,char* argv[])
{
    using namespace gfl;
    using LayerHelperType = LayerHelper<Node>;
    using LayerInfoType   = LayerInfo<Node>;
    using LayerBufferType = std::vector<Node>;

    // Parse arguments
    i64 width = -1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    bool sort = false;
    std::string instance;
    cxxopts::Options options("", "A C++ solver for DIDP models");
    options.add_options("Available")
            ("w,width", "Beam width", cxxopts::value(width))
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

    auto * const readOnlyMem = mallocStd<void>(ReadOnlyMemSize);
    StackAllocator roAllocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new (roAllocator) Model();
    Model::parseFile(model, instance, roAllocator);

    std::cout << "Instance: " << instance << std::endl;
    std::cout << "GPU: False" << std::endl;
    std::cout << "Width: ";
    if (width <= 0)
        std::cout << "Auto" << std::endl;
    else
        std::cout << width << std::endl;

    // Search
    Node bestNode;
    f64 pBound = Model::worstValue();
    f64 dBound = Model::bestValue();

    LayerInfoType *   layerInfo = mallocStd<LayerInfoType>(sizeof(LayerInfoType));
    StackAllocator * gAllocator = new StackAllocator(mallocStd(CpuMemSize), CpuMemSize);

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
        i64 const batchSize = gfl::min<i64>(currentLayer.size(), LayerHelperType::getMaxParents(labelsInfo[lIdx].nLabels, CpuMemSize, false));
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
                   currentLayer.size()-fragmentSize);
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
            i64 qSize = 0;
            for (auto const & l : layers)
            {
                qSize += l.size();
            }
            printf(" | Batch %3d/%3d | BatchSize = %10ld | Q = %10ld\n", bIdx+1, nBatches, currentBatchSize, qSize);
            fflush(stdout);

            // Init
            LayerHelperType::clear(layerInfo, gAllocator);
            layerInfo->labelsInfo = labelsInfo[lIdx];
            LayerHelperType::initParents(currentBatchSize, layerInfo, gAllocator);
            memcpy(layerInfo->parents,currentBatch.data(),sizeof(Node) * currentBatchSize);
            LayerHelperType::initChildren(layerInfo, gAllocator);

            // ### Algorithm
            // // 1) Given $n$ roots, construct $d$, the relaxed MDD.
            // 1.1) Backup the roots (see 3)
            // 1.2) Calculate children
            // 1.3) If no children, set the worst possible dual bound and go to 2)
            // 1.4) If solution layer, save the best dual bound and go to 2)
            // 1.5) Filter children by eq/dom
            // 1.6) Merge children until $nChildren <= width$
            // 1.7) Parents <- Children and go to 1.2)

            // 2) If $dualBound(d) > primalBound$ return no children, otherwise go to 3).

            // 3) Calculate the children of the $n$ roots and return them.
            // 3.1) Restore the roots
            // 3.2) Calculate children
            // 3.3) Filter children by eq/dom
            // 3.4) Set dual bound of each child.
            // 3.5) Return the children.
            // ###

            // 1.1) Backup the roots (see 3)
            memcpy(layerInfo->tmpParents,layerInfo->parents,sizeof(Node) * layerInfo->nParents);

            while (true)
            {
                // 1.2) Calculate children
                calcChildren(model,layerInfo,pBound,DDCtx);

                // 1.3) If no children, set the worst possible dual bound and go to 2)
                if (layerInfo->nChildren == 0)
                {
                    layerInfo->dBound = model->worstValue();
                    break;
                }

                // 1.4) If solution layer, save the best dual bound and go to 2)
                if (model->isTarget(layerInfo->children[0]))
                {
                    for (i64 cIdx = 0; cIdx < layerInfo->nChildren; cIdx += 1)
                    {
                        i64 const cBound = layerInfo->children[cIdx].boundSrcToNode;
                        layerInfo->dBound = model->isBetter(cBound, layerInfo->dBound) ?
                                            cBound:
                                            layerInfo->dBound;
                    }
                    break;
                }
                // 1.5) Filter children by eq/dom
                filterChildren(layerInfo,sort);

                // 1.6) Merge children
                layerInfo->nChildren = layerInfo->nRepresentatives;
                if (layerInfo->nChildren > width)
                {
                    mergeChildren(layerInfo, width);
                    layerInfo->nChildren = width;
                }

                // 1.7) Parents <- Children and go to 1.2)
                copyNodes(&layerInfo->nChildren, layerInfo->parents, layerInfo->children, layerInfo->childrenInfo);
                layerInfo->nParents = layerInfo->nChildren;
                layerInfo->nChildren = 0;
                layerInfo->nRepresentatives = 0;
            }

            // 2) If $dualBound(d) > primalBound$ return no children, otherwise go to 3).
            if (layerInfo->dBound > pBound)
            {




                // Labels
                layerInfo->labelsInfo.reset();
                calcLabels(
                        model,
                        layerInfo,
                        DDExact,
                        pBound,
                        dBound);
            }

            if (layerInfo->nRepresentatives > 0)
            {
                auto & nextLayer = layers[lIdx+1];
                i64 const nextLayerOldSize = nextLayer.size();
                nextLayer.resize(nextLayerOldSize + layerInfo->nRepresentatives);
                memcpy(nextLayer.data() + nextLayerOldSize,
                       layerInfo->children,
                       sizeof(Node) * layerInfo->nRepresentatives);

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
            printf("[%7.2fs] SOLUTION     | Visited = %10ld | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), expandedNodes, bestNode.boundSrcToNode);
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