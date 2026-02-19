#pragma once

#include "codd.hpp"
#include "LayerEngine.cuh"
#include <ArenaAllocator.hpp>
#include <cxxopts.hpp>
#include <Memory.hpp>
#include <Types.hpp>

#include "ExpansionInfo.cuh"
#include "LayersHelper.cuh"
#include "BoundsHelpers.cuh"

template<typename Model, typename Node>
void printLog(double elapsed,
              gfl::i32 lIdx,
              gfl::i64 lSize,
              gfl::i64 fSize,
              gfl::f64 pBound,
              gfl::f64 dBound,
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

    printf( "P/D = ");
    if (isValid<Model>(pBound))
    {
        printf("%7.2f", pBound);
    }
    else
    {
        printf("?");
    }
    printf( "/");
    if (isValid<Model>(dBound))
    {
        printf("%7.2f", dBound);
    }
    else
    {
        printf("?");
    }
    printf( " | ");

    printf("Batch %3d/%3d of Size %10ld ", bIdx+1, nBatches, bSize);
    // printf( "(");
    // printMemSize(sizeof(Node) * bSize);
    // printf( ") | ");

    printf("Expanded = %10ld | Queue = %10ld\n", nExpanded, qSize);
    fflush(stdout);
}


template<typename Model>
void printProgress(double elapsed,
              gfl::f64 pBound,
              gfl::f64 dBound,
              gfl::i64 nExpanded,
              gfl::i64 qSize
        )
{
    using namespace gfl;
    printf("[%7.2fs] ", elapsed);
    printf( "P/D = ");
    if (isValid<Model>(pBound))
    {
        printf("%7.2f", pBound);
    }
    else
    {
        printf("      ?");
    }
    printf( "/");
    if (isValid<Model>(dBound))
    {
        printf("%7.2f", dBound);
    }
    else
    {
        printf("      ?");
    }
    printf( " | ");

    printf("Expanded = %10ld | Queue = %10ld\n", nExpanded, qSize);
    fflush(stdout);
}

template<typename Model, typename Node>
int run_cadds_relaxed_seq(int argc, char* argv[])
{
    using namespace xuda;
    using BatchInfoType = ExpansionInfo<Node>;

    // Parse arguments
    i64 width = -1;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    std::string instance;
    cxxopts::Options options("", "A C++ solver for DIDP models");
    options.add_options("Available")
            ("w,width", "DD width", cxxopts::value(width))
            ("h,help", "Show this help message and exit")
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

    constexpr auto static ReadOnlyMemSize{256 * 1024}; // Aim to be cached in shared memory
    auto * const readOnlyMem = mallocStd<void>(ReadOnlyMemSize);
    StackAllocator roAllocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new (roAllocator) Model();
    Model::parseFile(model, instance, roAllocator);

    std::cout << "Instance: " << instance << std::endl;
    std::cout << "GPU: False" << std::endl;
    std::cout << "Width: " << width << std::endl;

    // Solutions and bounds
    Node bestNode;
    f64 pBound = worstValue<Model>();
    f64 dBound = bestValue<Model>();

    constexpr auto static DeviceMemSize{8ll * 1024ll * 1024ll * 1024ll}; // Same size GPU memory: 48 - 4 for runtime!)
    BatchInfoType * const batchInfo = mallocStd<BatchInfoType>(sizeof(BatchInfoType));
    StackAllocator * const gAllocator = new StackAllocator(mallocStd(DeviceMemSize), DeviceMemSize);

    // Layers buffers
    LayersHelper<Node,Model> layers;

    // Initialize first layer with root
    auto const rState =  model->initial();
    Node const root(
       ,rState,
        model->lgf(rState, DDExact, pBound, dBound),
        Model::has_local ? model->local(rState, DDCtx) : bestValue<Model>());
    layers.addToLayer(0,root);

    // Search info
    i64 expandedNodes = 0;
    bool interrupted = false;
    bool newSolution = false;
    i64 iteration = 0;

    constexpr f64 printProgressInterval = 5;
    auto const start = RuntimeMonitor::cputime();
    auto lastPrintProgress = RuntimeMonitor::cputime();
    while (not layers.allLayersEmpty() and isWorst<Model>(pBound,dBound))
    {
        iteration += 1;

        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }

        // Pull node
        i32 const currentLayerIdx = layers.calcDeepestMostPromising();
        auto const currentNode = layers.getLayer(currentLayerIdx).back();
        layers.removeSuffixFromLayer(currentLayerIdx, 1);
        expandedNodes += 1;

        if (RuntimeMonitor::elapsedSeconds(lastPrintProgress) > printProgressInterval)
        {
            printProgress<Model>(
                RuntimeMonitor::elapsedSeconds(start),
                pBound,
                dBound,
                expandedNodes,
                layers.countAllNodes());
            lastPrintProgress = RuntimeMonitor::cputime();
        }
        LayerEngine<Node>::initBatchSwappable(
            batchInfo,
            gAllocator,
            layers.getLabelsInfo(currentLayerIdx),
            width,
            Model::max_depth);
        LayerEngine<Node>::compileRelaxed(
            model,
            batchInfo,
            pBound,
            dBound,
            width);

        // If there is a terminal node, it is the best one
        assert(batchInfo->nChildren >= 0);
        assert(batchInfo->nChildren <= 1);
        if (batchInfo->nChildren > 0)
        {
            // Update bound and solution
            Node const & tNode = batchInfo->children[0];
            assert(tmpNode.gValue == tmpNode.fValue);
            if (isBetter<Model>(tNode.fValue,pBound))
            {
                if (not tNode.isApproximated)  // Better solution found
                {
                    pBound = tNode.fValue;
                    bestNode = tNode;
                    newSolution = true;
                }
                else
                {
                    for (i64 i = 0; i < batchInfo->cutsetSize; )
                    {
                        // Find nodes in the same layer
                        i64 const lIdx = batchInfo->cutset[i].nEdgesSrcToNode-1;
                        i64 j = i + 1;
                        for ( ; j < batchInfo->cutsetSize; j += 1)
                        {
                            if (batchInfo->cutset[j].nEdgesSrcToNode-1 != lIdx)
                            {
                                break;
                            }
                        }

                        // Collect them in a (partial) cutset
                        std::span<Node> const cutset(batchInfo->cutset + i, j - i);
                        assert(std::all_of(cutset.begin(), cutset.end(), [](auto const & n) { return n.nEdgesSrcToNode-1 == cutsetLayerIdx;}));

                        // Add cutset to the layer
                        layers.addToLayer(lIdx, cutset);

                        // Next set of nodes
                        i = j;
                    }
                }
            }
        }

        if (newSolution)
        {
            printf("[%7.2fs] SOLUTION              | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), expandedNodes, bestNode.fValue);
            Node::printLabels(bestNode);
            printf("\n");
            fflush(stdout);
            Node::printLabels(bestNode);
            printf("\n");
            fflush(stdout);
        }
        auto const tmpBound = layers.calcBestBound();
        if (isWorst<Model>(tmpBound,dBound) and isValid<Model>(tmpBound))
        {
            printf("[%7.2fs] TIGHTENING            | Dual = %7.2f -> %7.2f\n",
                RuntimeMonitor::elapsedSeconds(start),
                dBound,
                tmpBound,
                expandedNodes,
                layers.countAllNodes());
            dBound = tmpBound;
        }
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (pBound != worstValue<Model>())
        {
            printf("COMPLETED    | Expanded = %10ld | Queue = %10ld | Cost = %7.2f | Value = ", expandedNodes,  layers.countAllNodes(), bestNode.gValue);
            Node::printLabels(bestNode);
            printf("\n");
        }
        else
        {
            printf("INFEASIBLE   | Expanded = %10ld\n", expandedNodes);
        }
    }
    else
    {
        printf("TIMEOUT      | Expanded = %10ld\n", expandedNodes);
    }
    fflush(stdout);

    return EXIT_SUCCESS;
}