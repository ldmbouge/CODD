#pragma once

#include "codd.hpp"
#include "BatchEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <span>

#include "BatchInfo.cuh"
#include "LayersHelper.cuh"

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory
constexpr auto static CpuMemSize{8ll * 1024ll * 1024ll * 1024ll}; // Same size GPU memory: 48 - 4 for runtime!)

template<typename Model, typename Node>
void printLog(double elapsed,
              gfl::i32 lIdx,
              gfl::i64 lSize,
              gfl::i64 fSize,
              gfl::f64 pBound,
              gfl::i32 bIdx,
              gfl::i64 bSize,
              gfl::i32 nBatches,
              gfl::i64 nExpanded,
              gfl::i64 qSize
        )
{
    using namespace gfl;
    printf("[%7.2fs] ", elapsed);
    printf("Layer = %4d (%10ld -> %10ld) | ",lIdx, lSize, lSize - fSize);

    printf( "Cost = ");
    if (pBound != Model::worstValue())
    {
        printf("%7.2f", pBound);
    }
    else
    {
        printf("?");
    }
    printf( " | ");

    printf("Batch %3d/%3d of Size %10ld ", bIdx+1, nBatches, bSize);
    printf( "(");
    printMemSize(sizeof(Node) * bSize);
    printf( ") | ");

    printf("Expanded = %10ld | Queue = %10ld\n", nExpanded, qSize);
    fflush(stdout);
}

template<typename Model, typename Node>
int run_cadds_relaxed_seq(int argc,char* argv[])
{
    using namespace gfl;
    using BatchInfoType   = BatchInfo<Node>;

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

    BatchInfoType * batchInfo = mallocStd<BatchInfoType>(sizeof(BatchInfoType));
    StackAllocator * gAllocator = new StackAllocator(mallocStd(CpuMemSize), CpuMemSize);

    // Layers buffers
    LayersHelper<Node> exactLayers;
    LayersHelper<Node> auxLayer;

    // Initialize first layer
    auto const rState = model->initial();
    auto const rLabels = model->lgf(rState, DDExact, pBound, dBound);
    exactLayers.getLayer(0).emplace_back(rState,rLabels);
    exactLayers.getLabelsInfo(0).emplace_back(rLabels.slc());

    // Let's goo!
    auto const start = RuntimeMonitor::cputime();
    i64 expandedNodes = 0;
    bool interrupted = false;
    bool newSolution = false;
    while (not exactLayers.allLayersEmpty())
    {
        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }
        newSolution = false;

        // Grow number of layers on demand
        i32 const currentExactLayerIdx = exactLayers.calcDeepestNotEmpty();

        // Fragment
        auto & currentExactLayer = exactLayers.getLayer(currentExactLayerIdx);
        auto & currentExactLabelsInfo = exactLayers.getLabelsInfo(currentExactLayerIdx);
        i64 const batchSize = gfl::min<i64>(currentExactLayer.size(), BatchInfoType::calcMaxParents(currentExactLabelsInfo.nLabels, CpuMemSize, false));
        i64 const fragmentSize = gfl::min<i64>(currentExactLayer.size(), width < 0 ? batchSize : width);
        auto const fragment = std::span(currentExactLayer.end() - fragmentSize, fragmentSize);

        // Batching
        i32 const nBatches = roundUpDivPosInt<i32>(fragmentSize, batchSize);
        assert(nBatches * batchSize >= fragmentSize);
        for (i32 bIdx = 0; bIdx < nBatches; bIdx += 1)
        {
            i64 bBegin, bEnd;
            getBeginEnd(bBegin, bEnd, bIdx, nBatches, fragmentSize);
            i64 const currentBatchSize = bEnd - bBegin; // No + 1!
            auto const currentBatch = std::span(fragment.data() + bBegin, currentBatchSize);
            expandedNodes += currentBatchSize;

            printLog<Model,Node>(
                    RuntimeMonitor::elapsedSeconds(start),
                    currentExactLayerIdx,
                    currentExactLayer.size(),
                    fragmentSize,
                    pBound,
                    bIdx,currentBatchSize,nBatches,
                    expandedNodes,
                    exactLayers.countAllNodes());

            auto & tmpLayer = auxLayer.getLayer(0);
            tmpLayer.clear();
            tmpLayer.reserve(currentBatchSize);
            memcpy(tmpLayer.data(),currentBatch.data() ,sizeof(Node) * currentBatchSize);
            BatchEngine<Node>::initBatch(batchInfo,gAllocator,tmpLayer,exactLayers.getLabelsInfo(currentExactLayer));

            while (true)
            {
                BatchEngine<Node>::processBatchRelaxed(model,pBound,dBound,batchInfo);
                if (batchInfo->nChildren > 0)
                {
                    tmpLayer.clear();
                    tmpLayer.reserve(batchInfo->nChildren);
                    memcpy(tmpLayer.data(),batchInfo->children,sizeof(Node) * batchInfo->nChildren);
                    if (not model->isTarget(tmpLayer[0].state))
                    {
                        assert(batchInfo->labelsInfo.nLabels <= exactLayers.getLabelsInfo(currentExactLayer).nLabels);
                        BatchEngine<Node>::initParentsWithChildren(batchInfo);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    break;
                }
            }

            // I know that the only child I have is the best
            if (batchInfo->nChildren > 0)
            {
                // Update bounds and solution
                Node const & tmpNode = batchInfo->children[0];
                auto const & tmpBound = tmpNode.boundSrcToNode;
                dBound = Model::calcBetter(tmpBound,dBound);
                if ((not tmpNode.isNotExact) and Model::isBetter(tmpBound,pBound))
                {
                    pBound = tmpBound;
                    bestNode = tmpNode;
                    newSolution = true;
                }

                // If necessary, enqueue children for further expansion
                if (tmpBound < pBound)
                {
                    tmpLayer.clear();
                    tmpLayer.reserve(currentBatchSize);
                    memcpy(tmpLayer.data(),currentBatch.data() ,sizeof(Node) * currentBatchSize);
                    BatchEngine<Node>::initBatch(batchInfo,gAllocator,tmpLayer,exactLayers.getLabelsInfo(currentExactLayer));
                    BatchEngine<Node>::processBatchExact(model,pBound,dBound,batchInfo);

                    if (batchInfo->nChildren > 0)
                    {
                        auto & nextExactLayer = exactLayers.getLayer(currentExactLayerIdx+1);
                        i64 const nextExactLayerOldSize = nextExactLayer.size();
                        nextExactLayer.resize(nextExactLayerOldSize + batchInfo->nChildren);
                        memcpy(nextExactLayer.data() + nextExactLayerOldSize,
                               batchInfo->children,
                               sizeof(Node) * batchInfo->nChildren);

                        exactLayers.getLabelsInfo(currentExactLayerIdx+1).update(batchInfo->labelsInfo);
                        Node const & tmpNodeExact = nextExactLayer[nextExactLayerOldSize];
                        if (model->isTarget(tmpNodeExact.state))
                        {
                            if (Model::better(tmpNodeExact.boundSrcToNode, pBound))
                            {
                                bestNode = tmpNodeExact;
                                pBound = bestNode.boundSrcToNode;
                                newSolution = true;
                            }
                            nextExactLayer.resize(nextExactLayerOldSize);
                        }
                        if (sort and nextExactLayerOldSize > 0)
                        {
                            // Reverse because we work on the tail of the vector
                            auto cmpByBound = [](Node const & a, Node const & b){return not Model::betterEq(a.boundSrcToNode,b.boundSrcToNode);};
                            //assert(std::is_sorted(nextLayer.data() + nextLayerOldSize, nextLayer.data() + nextLayer.size(), cmpByCost));
                            std::inplace_merge(nextExactLayer.data(), nextExactLayer.data() + nextExactLayerOldSize, nextExactLayer.data() + nextExactLayer.size(), cmpByBound);
                            assert(std::is_sorted(nextLayer.begin(), nextLayer.end(), cmpByCost));
                        }
                    }
                }
            }
        }
        currentExactLayer.resize(currentExactLayer.size() - fragmentSize);

        if (newSolution)
        {
            printf("[%7.2fs] SOLUTION     | Visited = %10ld | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), expandedNodes, bestNode.boundSrcToNode);
            Node::printLabels(bestNode);
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
            Node::printLabels(bestNode);
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