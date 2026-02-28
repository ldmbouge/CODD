#pragma once

#include <GFL.hpp>

#include "CliManager.hpp"
#include "QueueGpu.cuh"
#include "BnBManager.hpp"
#include "LogManager.hpp"
#include "StatsManager.hpp"
#include "ExpansionEngineSeq.hpp"
#include "ExpansionEngineGpu.cuh"

template<typename Model, typename Node, int BranchFactor, int Depth>
int runHybrid(int argc, char* argv[])
{
    using namespace gfl;
    using RelEngGpu = ExpansionEngineGpu<Model,Node>;
    using ResEngCpu = ExpansionEngineSeq<Model,Node>;

    CliManager cli("CODD", "A C++ solver for DIDP models.");
    cli.parse(argc, argv);
    std::cout << "Instance: " << cli.instance() << std::endl;
    std::cout << "Width: " << cli.width() << std::endl;
    std::cout << "Engine: GPU (Relaxed) + CPU (Restricted)"  << std::endl;

    // Model (+ Instance) creation
    constexpr i32 modelMemSize = 256 * 1024; // Small, it MUST fit in shared memory
    assert(modelMemSize > sizeof(Model)); // At least, instance data not included
    ArenaAllocator modelAlloc(modelMemSize, cudaReserveManaged(modelMemSize));
    Model * const model = new (modelAlloc) Model();
    model->init(cli.instance(), modelAlloc);

    // Expansion engines
    constexpr i32 engMemSize = 2048; // Small, it MUST be on managed memory
    assert(engMemSize > sizeof(RelEngGpu));
    ArenaAllocator relEngAlloc(engMemSize, cudaReserveManaged(engMemSize));
    ArenaAllocator relBuffAlloc(cli.memSize(), cudaReserveDevice(cli.memSize()));
    RelEngGpu * const relEng = new (relEngAlloc) RelEngGpu();
    relEng->initRelaxedExpansion(cli.width(), BranchFactor, Depth, relBuffAlloc);
    printf("Relaxed working memory: ");
    printMemSize(relBuffAlloc.usedSize());
    printf("\n");

    assert(engMemSize > sizeof(ResEngCpu));
    ArenaAllocator resEngAlloc(engMemSize, heapReserve(engMemSize));
    ArenaAllocator resBuffAlloc(cli.memSize(), heapReserve(cli.memSize()));
    ResEngCpu * const resEng = new (resEngAlloc) ResEngCpu();
    i32 const cpuWidth = 4096; //cli.width();
    resEng->initRestrictedExpansion(cpuWidth, BranchFactor, resBuffAlloc);
    printf("Restricted Working memory: ");
    printMemSize(resBuffAlloc.usedSize());
    printf("\n");

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats(cli.timeout());

    // Layers
    QueueGpu<Model,Node> queue;
    queue.push(Node::makeRoot(model));

    // Log manager
    LogManager<Model,Node> log(bnb,queue,stats);

    std::vector<Node> parentsBuffer;
    parentsBuffer.reserve(cli.pop());

    std::vector<Node> testBuffer;

    // BnB search
    log.header();
    stats.start();

    while (not queue.empty() and stats.elapsed<sec>() <= cli.timeout() and not bnb.solved())
    {
        f64 const lambda = isValid<Model>(bnb.primal()) ? 1.5 : 2.5;
        parentsBuffer.clear();
        while (not queue.empty() and parentsBuffer.size() < cli.pop())
        {
            if (parentsBuffer.empty())
            {
                Node const * node = queue.pullBest();
                parentsBuffer.push_back(*node);
            }
            else
            {
                Node const * node = queue.peekBest();

                // //Restricted
                //  if (isValid<Model>(bnb.primal()))
                // {
                //     testBuffer.clear();
                //     testBuffer.push_back(*node);
                //     resEng->expandRestricted(model, testBuffer, bnb.primal(), bnb.dual());
                //     auto [resBestTarget, _] = resEng->getTargets();
                //     if (resBestTarget.has_value())
                //     {
                //         bnb.primal(resBestTarget.value());
                //     }
                //     if (resEng->exact)
                //     {
                //         queue.pullBest();
                //         log.progress();
                //         //pruned++;
                //         continue;
                //     }
                // }

                if (parentsBuffer.back().depth() == node->depth())
                {
                    node = queue.pullBest();
                    parentsBuffer.push_back(*node);
                }
                else
                {
                    break;
                }
            }
        }
        fflush(stdout);
        if (not parentsBuffer.empty())
        {

            printf("\rExpanding depth=%d f=%.2f qsize=%lld\033[K",
              parentsBuffer.front().depth(),
              parentsBuffer.front().f(),
              queue.size());
            fflush(stdout);

            bnb.dual(parentsBuffer.front().f());

            // Relaxed
            relEng->expandRelaxed(model, parentsBuffer, bnb.primal(), bnb.dual(), lambda);
            auto [relBestTarget, relBestExactTarget] = relEng->getTargets();
            if (relBestExactTarget.has_value())
            {
                bnb.primal(relBestExactTarget.value());
            }
            if (relBestTarget.has_value())
            {
                if (not bnb.pruneAncestor(relBestTarget.value()))
                {
                    bnb.dual(relBestTarget.value());
                    auto const & cutset = relEng->cutData.nodes();
                    queue.pushFromGpu(cutset);
                }
            }
            log.progress();
        }
    }
    stats.end();
    log.summary();

    return EXIT_SUCCESS;
}
