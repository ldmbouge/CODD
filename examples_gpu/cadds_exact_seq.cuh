#pragma once

#include "codd.hpp"
#include "BatchFunctions.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <span>

#include "BatchInfo.cuh"

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory
constexpr auto static CpuMemSize{8ll * 1024ll * 1024ll * 1024ll}; // Same size GPU memory: 48 - 4 for runtime!)

template<typename Model, typename Node>
int run_cadds_exact_seq(int argc,char* argv[])
{
    using namespace gfl;
    using BatchInfoHelperType = BatchInfoHelper<Node>;
    using BatchInfoType   = BatchInfo<Node>;
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

    BatchInfoType * batchInfo = mallocStd<BatchInfoType>(sizeof(BatchInfoType));
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

            BatchEngine<Node>::initBatch(batchInfo,gAllocator,currentBatch,labelsInfo[lIdx]);
            BatchEngine<Node>::processBatchExact(batchInfo,pBound,dBound,currentBatch);

            if (batchInfo->nChildren > 0)
            {
                auto & nextLayer = layers[lIdx+1];
                i64 const nextLayerOldSize = nextLayer.size();
                nextLayer.resize(nextLayerOldSize + batchInfo->nChildren);
                memcpy(nextLayer.data() + nextLayerOldSize,
                       batchInfo->children,
                       sizeof(Node) * batchInfo->nChildren);

                labelsInfo[lIdx+1].update(batchInfo->labelsInfo);

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