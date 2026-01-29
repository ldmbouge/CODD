#pragma once

#include "codd.hpp"
#include "BatchEngine.cuh"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>
#include <span>

#include "BatchInfo.cuh"
#include "LayersHelper.cuh"
#include "BoundsHelpers.cuh"

constexpr auto static ReadOnlyMemSize{256 * 1024}; // Cached in shared memory
constexpr auto static CpuMemSize{8ll * 1024ll * 1024ll * 1024ll}; // Same size GPU memory: 48 - 4 for runtime!)

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
    f64 pBound = worstValue<Model>();
    f64 dBound = bestValue<Model>();

    BatchInfoType * batchInfo = mallocStd<BatchInfoType>(sizeof(BatchInfoType));
    StackAllocator * gAllocator = new StackAllocator(mallocStd(CpuMemSize), CpuMemSize);

    // Layers buffers l<?
    LayersHelper<Node,Model> exactLayers;
    LayersHelper<Node,Model> auxLayer;

    // Initialize first layer
    auto const rState = model->initial();
    auto const rLabels = model->lgf(rState, DDExact, pBound, dBound);
    exactLayers.getLayer(0).emplace_back(rState,rLabels,bestValue<Model>());
    exactLayers.getLabelsInfo(0).update(rLabels.slc());
    exactLayers.updateBound(0);

    // Let's goo!
    auto const start = RuntimeMonitor::cputime();
    i64 expandedNodes = 0;
    bool interrupted = false;
    bool newSolution = false;
    i64 iteration = 0;
    f64 const printProgressInterval = 5;
    auto lastPrintProgress = RuntimeMonitor::cputime();
    while (not exactLayers.allLayersEmpty() and isWorst<Model>(pBound,dBound))
    {
        iteration += 1;

        if (RuntimeMonitor::elapsedSeconds(start) > timeout)
        {
            interrupted = true;
            break;
        }
        newSolution = false;

        bool dive = not (isValid<Model>(pBound)); // and iteration % 10 < 1;
        i32 const currentExactLayerIdx = dive ?
            exactLayers.calcDeepestNotEmpty() :
            exactLayers.calcDeepestMostPromising();

        // Fragment
        auto & currentExactLayer = exactLayers.getLayer(currentExactLayerIdx);
        auto & currentExactLabelsInfo = exactLayers.getLabelsInfo(currentExactLayerIdx);
        i64 const batchSize = gfl::min<i64>(currentExactLayer.size(), BatchInfoType::calcMaxParents(currentExactLabelsInfo.nLabels, CpuMemSize, false));
        i32 suffixLength = 1;
        while(suffixLength < currentExactLayer.size() and currentExactLayer.at(currentExactLayer.size() - 1 - suffixLength).heuristicBound == currentExactLayer.back().heuristicBound and suffixLength < width)
        {
            suffixLength += 1;
        }
        //printf("Suffix length: %i\n", suffixLength);
        i64 const fragmentSize = gfl::min<i64>(currentExactLayer.size(), width);; //gfl::min<i64>(currentExactLayer.size(), width < 0 ? batchSize : width);
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

            // printLog<Model,Node>(
            //         RuntimeMonitor::elapsedSeconds(start),
            //         currentExactLayerIdx,
            //         currentExactLayer.size(),
            //         fragmentSize,
            //         pBound,
            //         dBound,
            //         bIdx,currentBatchSize,nBatches,
            //         expandedNodes,
            //         exactLayers.countAllNodes());

            if (RuntimeMonitor::elapsedSeconds(lastPrintProgress) > printProgressInterval)
            {
                printProgress<Model>(
                    RuntimeMonitor::elapsedSeconds(start),
                    pBound,
                    dBound,
                    expandedNodes,
                    exactLayers.countAllNodes());
                lastPrintProgress = RuntimeMonitor::cputime();
            }

            auto & tmpLayer = auxLayer.getLayer(0);
            tmpLayer.clear();
            tmpLayer.resize(currentBatchSize);
            memcpy(tmpLayer.data(),currentBatch.data() ,sizeof(Node) * currentBatchSize);
            BatchEngine<Node>::initBatch(batchInfo,gAllocator,tmpLayer,exactLayers.getLabelsInfo(currentExactLayerIdx));

            i64 relaxedLayerIdx = currentExactLayerIdx;
            //printf("           Processing relaxation...");
            while (true)
            {
                relaxedLayerIdx += 1;
                BatchEngine<Node>::processBatchRelaxed(model,pBound,dBound,width,batchInfo);
                if (batchInfo->nChildren > 0)
                {
                    tmpLayer.clear();
                    tmpLayer.resize(batchInfo->nChildren);
                    memcpy(tmpLayer.data(),batchInfo->children,sizeof(Node) * batchInfo->nChildren);
                    if (not model->isTarget(tmpLayer[0].state))
                    {
                        LabelsInfo const li = batchInfo->labelsInfo;
                        assert(batchInfo->nChildren <= BatchInfoType::calcMaxParents(li.nLabels, CpuMemSize, false));
                        BatchEngine<Node>::initBatch(batchInfo,gAllocator,tmpLayer,li);
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
            // printf(" (%ld layers)\n", relaxedLayerIdx - currentExactLayerIdx);

            // I know that the only child I have is the best
            if (batchInfo->nChildren > 0)
            {
                // Update bound and solution
                Node const & tmpNode = batchInfo->children[0];
                if (isBetter<Model>(tmpNode.heuristicBound,pBound))
                {
                    if (not tmpNode.isNotExact)
                    {
                        pBound = tmpNode.sumEdgesSrcToNode;
                        bestNode = tmpNode;
                        newSolution = true;
                    }
                    else
                    {
                        tmpLayer.clear();
                        tmpLayer.resize(currentBatchSize);
                        // if(currentBatch.size() != 1)
                        // {
                        //     printf("[ERROR] Batch size should be 1, it is %d\n", currentBatch.size());
                        //     exit(1);
                        // }
                        checkNode(currentBatch[0]);
                        currentBatch[0].heuristicBound = tmpNode.heuristicBound;
                        checkNode(currentBatch[0]);
                        memcpy(tmpLayer.data(),currentBatch.data() ,sizeof(Node) * currentBatchSize);
                        BatchEngine<Node>::initBatch(batchInfo,gAllocator,tmpLayer,exactLayers.getLabelsInfo(currentExactLayerIdx));
                        BatchEngine<Node>::processBatchExact(model,pBound,dBound,batchInfo,sort);

                        if (batchInfo->nChildren > 0)
                        {
                            auto const nextExactLayerIdx = currentExactLayerIdx+1;
                            auto & nextExactLayer = exactLayers.getLayer(nextExactLayerIdx);
                            auto & nextExactLabelsInfo =  exactLayers.getLabelsInfo(nextExactLayerIdx);
                            i64 const nextExactLayerOldSize = nextExactLayer.size();
                            nextExactLayer.resize(nextExactLayerOldSize + batchInfo->nChildren);
                            memcpy(nextExactLayer.data() + nextExactLayerOldSize,
                                   batchInfo->children,
                                   sizeof(Node) * batchInfo->nChildren);

                            nextExactLabelsInfo.update(batchInfo->labelsInfo);
                            Node const & tmpNodeExact = nextExactLayer[nextExactLayerOldSize];
                            if (model->isTarget(tmpNodeExact.state))
                            {
                                if (isBetter<Model>(tmpNodeExact.sumEdgesSrcToNode, pBound))
                                {
                                    bestNode = tmpNodeExact;
                                    pBound = bestNode.sumEdgesSrcToNode;
                                    newSolution = true;
                                }
                                nextExactLayer.resize(nextExactLayerOldSize);
                            }
                            else if (sort)
                            {
                                // Reverse because we work on the tail of the vector
                                auto cmpByBound = [](Node const & a, Node const & b){return isWorst<Model>(a.heuristicBound,b.heuristicBound);};
                                assert(std::is_sorted(nextExactLayer.data() + nextExactLayerOldSize, nextExactLayer.data() + nextExactLayer.size(), cmpByBound));
                                std::inplace_merge(nextExactLayer.data(), nextExactLayer.data() + nextExactLayerOldSize, nextExactLayer.data() + nextExactLayer.size(), cmpByBound);
                                assert (std::is_sorted(nextExactLayer.begin(), nextExactLayer.end(), cmpByBound));
                            }
                        }
                    }
                }
                else
                {
                    //printf("           Pruned %ld nodes\n", currentBatchSize);
                    //printNodes(batchInfo->nChildren, batchInfo->children);
                    int nBPrefix = 0;
                    for (int i = 0; i < batchInfo->nChildren; i += 1)
                    {
                        auto & const tmptmp = batchInfo->children[i];
                        gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                        bool isPrefix = true;
                        for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                        {
                            isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                        }
                        if (isPrefix)
                        {
                            nBPrefix+= 1;
                        }
                    }
                    if (nBPrefix > 0)
                    {
                        printf("OPT ANCESTOR FILTERING!!!!\n");
                        fflush(stdout);
                    }
                }
            }
            else
            {
                   //printf("           Discarded %ld nodes\n", currentBatchSize);
                int nBPrefix = 0;
                for (int i = 0; i < batchInfo->nChildren; i += 1)
                {
                    auto & const tmptmp = batchInfo->children[i];
                    gfl::u8 opt[] = {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0};
                    bool isPrefix = true;
                    for (gfl::i32 i = 0; i < tmptmp.nEdgesSrcToNode; i += 1)
                    {
                        isPrefix = isPrefix and tmptmp.labelsSrcToNode[i] == opt[i];
                    }
                    if (isPrefix)
                    {
                        nBPrefix+= 1;
                    }
                }
                if (nBPrefix > 0)
                {
                    printf("OPT ANCESTOR FILTERING!!!!\n");
                    fflush(stdout);
                }
            }
        }
        exactLayers.getLayer(currentExactLayerIdx).resize(exactLayers.getLayer(currentExactLayerIdx).size() - fragmentSize);
        if (newSolution)
        {
            printf("[%7.2fs] SOLUTION     | Cost = %7.2f | Value = ", RuntimeMonitor::elapsedSeconds(start), expandedNodes, bestNode.sumEdgesSrcToNode);
            Node::printLabels(bestNode);
            printf("\n");
            fflush(stdout);
            Node::printLabels(bestNode);
            printf("\n");
            fflush(stdout);
        }

        exactLayers.updateBound(currentExactLayerIdx);
        exactLayers.updateBound(currentExactLayerIdx+1);
        auto const tmpBound = exactLayers.calcBestBound();
        if (isWorst<Model>(tmpBound,dBound))
        {
            printf("[%7.2fs] TIGHTENING   | Dual = %7.2f -> %7.2f | Expanded = %10ld | Queue = %10ld\n",
                RuntimeMonitor::elapsedSeconds(start),
                dBound,
                tmpBound,
                expandedNodes,
                exactLayers.countAllNodes());
            dBound = tmpBound;
        }
    }

    printf("[%7.2fs] ", RuntimeMonitor::elapsedSeconds(start));
    if (not interrupted)
    {
        if (isValid<Model>(pBound))
        {
            printf("COMPLETED    | Expanded = %10ld | Cost = %7.2f | Value = ", expandedNodes, bestNode.sumEdgesSrcToNode);
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