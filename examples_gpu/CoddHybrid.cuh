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

    if (cli.width()  < BranchFactor)
    {
        printf("WARNING: Width too small, increased to %d\n.", BranchFactor);
        cli.width(BranchFactor);
    }


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
    i32 const cpuWidth = 1024; //cli.width();
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
    i32 adjToPop = cli.pop();

    i32 prundedByDual = 0;
    i32 prundedByRes = 0;

    while (not queue.empty() and stats.elapsed<sec>() <= cli.timeout() and not bnb.solved())
    {
        parentsBuffer.clear();
        prundedByDual = 0;
        prundedByRes = 0;

        while (not queue.empty() and parentsBuffer.size() < adjToPop)
        {
            {
                Node const * node = queue.peekBest();
                if (isBetterEq<Model>(node->f(), bnb.primal()) or not bnb.hasPrimal())
                {

                    //Restricted
                      // testBuffer.clear();
                      // testBuffer.push_back(*node);
                      // {
                      //     TIMED_SCOPE_N("ResDD");
                      //     resEng->expandRestricted(model, testBuffer, bnb.primal(), bnb.dual());
                      // }
                      // auto [resBestTarget, _] = resEng->getTargets();
                      // if (resBestTarget.has_value())
                      // {
                      //     printf("Find exact from Res with value: %.3f\n", resBestTarget.value().g());
                      //     bnb.primal(resBestTarget.value());
                      // }
                      // if (resEng->exact)
                      // {
                      //     queue.pullBest();
                      //     prundedByRes += 1;
                      //     log.progress();
                      //     continue;
                      // }

                    if (parentsBuffer.empty() or
                       (parentsBuffer.size() < adjToPop and parentsBuffer.back().f() == node->f() and parentsBuffer.back().depth() == node->depth()))
                    {
                        node = queue.pullBest();
                        parentsBuffer.push_back(*node);
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    prundedByDual += 1;
                    queue.pullBest();
                    log.progress();
                }
            }
            log.progress();
        }

        printf("Final nodes %d (D = %d RS = %d  T = %d) nodes with value %.3f at depth %d\n",
               (int) parentsBuffer.size(),
               prundedByDual,
               prundedByRes,
               prundedByRes + prundedByDual + (int)parentsBuffer.size(),
               parentsBuffer.front().f(),
               parentsBuffer.front().depth());

        if (not parentsBuffer.empty())
        {

            //fflush(stdout);

            bnb.dual(parentsBuffer.front().f());

            printf("[DBG] Offloading %ld nodes with f %.2f (Remaining %ld)\n", parentsBuffer.size(), parentsBuffer.back().f(), queue.size());

            f64 const fDepth = scast<f32>(parentsBuffer.front().depth()) / scast<f32>(Depth) ; //1.5;//isValid<Model>(bnb.primal()) ? 1.5 : 3.0;
            f64 const lambda = 1.0;// - fDepth; //bnb.hasPrimal() ? 1.5 : 5.0;


            //printf("Going on GPU with %d nodes\n", parentsBuffer.size());
            {
                //TIMED_SCOPE_N("RelDD");
                relEng->expandRelaxed(model, parentsBuffer, bnb.primal(), bnb.dual(), lambda);
            }

            // Avoid loops
            if (parentsBuffer.size() * 1.1 >= relEng->cutData.nodes()->size() and
                parentsBuffer.size() * 0.9 <= relEng->cutData.nodes()->size())
            {
                adjToPop = ceil<i32>(adjToPop,2);
            }
            else
            {
                adjToPop = cli.pop();
            }

            auto [relBestTarget, relBestExactTarget] = relEng->getTargets();
            if (relBestExactTarget.has_value())
            {
                Node const & bestExt = relBestExactTarget.value();
                printf("[DBG] REL exact value: %.3f\n", bestExt.g());
                bnb.primal(bestExt);
            }
            if (relBestTarget.has_value())
            {
                Node const & bestOverall = relBestTarget.value();
                printf("[DBG] REL best value: %.3f (Exact %d)\n", bestOverall.g(), not bestOverall.approximated());; // G is correct!
                if (isBetter<Model>(bestOverall.g(), bnb.primal()))
                {
                    if (bestOverall.approximated())
                    {
                        auto const & cutset = relEng->cutData.nodes();
                        printf("[DBG] Pushing %lld nodes to the queue\n", cutset->size());
                        queue.pushFromGpu(cutset, bnb.primal());
                    }
                }
            }
            log.progress();
        }
    }
    stats.end();
    log.summary();
    //Timer::summary();

    return EXIT_SUCCESS;
}
