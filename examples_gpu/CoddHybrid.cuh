#pragma once

#include <GFL.hpp>

#include "CliManager.hpp"
#include "BlockingQueue.cuh"
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

    // if (cli.width()  < BranchFactor)
    // {
    //     printf("WARNING: Width too small, increased to %d\n.", BranchFactor);
    //     cli.width(BranchFactor);
    // }


    std::cout << "Instance: " << cli.instance() << std::endl;
    std::cout << "Width: " << cli.width() << std::endl;
    std::cout << "Lambda: " << cli.lambda() << std::endl;
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
    // Restricted engines - one per validator thread
    constexpr i32 nValidators = 16;
    i32 const cpuWidth = 4096 ; //cli.width();
    std::array<ResEngCpu*, nValidators> resEngs;
    for (i32 i = 0; i < nValidators; ++i)
    {
        ArenaAllocator resEngAlloc(engMemSize, heapReserve(engMemSize));
        ArenaAllocator resBuffAlloc(cli.memSize(), heapReserve(cli.memSize()));
        resEngs[i] = new (resEngAlloc) ResEngCpu();
        resEngs[i]->initRestrictedExpansion(cpuWidth, BranchFactor, resBuffAlloc);
    }

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats(cli.timeout());

    // Layers
    std::atomic<i32> inFlight{0};
    std::mutex              sharedMutex;
    std::condition_variable sharedCv;

    BlockingQueue<Queue<Model,Node>,    Model> readyQueue(inFlight, sharedMutex, sharedCv);
    BlockingQueue<QueueGpu<Model,Node>, Model> stashQueue(inFlight, sharedMutex, sharedCv);
    readyQueue.push(Node::makeRoot(model));

    // Log manager
    LogManager<Model,Node,decltype(readyQueue)> log(bnb,readyQueue,stats);

    // ── Validator threads ────────────────────────────────────────────

    auto validatorFn = [&](ResEngCpu * resEng)
    {
        std::vector<Node> localBuffer;
        while (auto node = stashQueue.pop())   // blocking
        {
            localBuffer.clear();
            localBuffer.push_back(**node);
            resEng->expandRestricted(model, localBuffer, bnb.primal(), bnb.dual());
            auto [best, _] = resEng->getTargets();
            if (best) bnb.primal(best.value());
            if (not resEng->exact)
                readyQueue.pushNoCount(*node);      // moving between queues, no inFlight change
            else
                stashQueue.done();
        }
    };
    std::array<std::thread, nValidators> validators;
    for (i32 i = 0; i < nValidators; ++i)
        validators[i] = std::thread(validatorFn, resEngs[i]);


    std::vector<Node> parentsBuffer;
    parentsBuffer.reserve(cli.pop());

    // BnB search
    log.header();
    stats.start();

    i32 adjToPop = cli.pop();
    while (
        stats.elapsed<sec>() <= cli.timeout()
        and not bnb.solved()
        )
    {
        // Collect batch from readyQueue
        parentsBuffer.clear();
        auto first = readyQueue.popOrDone(); // returns nullopt if empty AND inFlight==0
        if (not first) break;
        if (first)
            parentsBuffer.push_back(**first);
        else
            break;
        while (parentsBuffer.size() < adjToPop)
        {
            auto node = readyQueue.popIf(parentsBuffer.back().f(), parentsBuffer.back().depth());  // blocks until non-empty, checks f()
            if (node)
                parentsBuffer.push_back(**node);
            else
                break;
        }

        printf("[DBG] Offloading %d nodes with f = %.2f\n",
               (int) parentsBuffer.size(),
               parentsBuffer.front().f());
        fflush(stdout);

        if (not parentsBuffer.empty())
        {
            bnb.dual(parentsBuffer.front().f());

            //printf("[DBG] Offloading %ld nodes with f %.2f (Remaining %ld)\n", parentsBuffer.size(), parentsBuffer.back().f(), queue.size());

            // f64 const fDepth = scast<f32>(parentsBuffer.front().depth()) / scast<f32>(Depth) ; //1.5;//isValid<Model>(bnb.primal()) ? 1.5 : 3.0;
            // f64 const lambda = 1.0 + 2 * fDepth; //bnb.hasPrimal() ? 1.5 : 5.0;


            //printf("Going on GPU with %d nodes\n", parentsBuffer.size());
            {
                //TIMED_SCOPE_N("RelDD");
                relEng->expandRelaxed(model, parentsBuffer, bnb.primal(), bnb.dual(), cli.lambda());
            }

            // Avoid loops
            if (parentsBuffer.size() * 1.1 >= relEng->cutData.nodes()->size() and
                parentsBuffer.size() * 0.9 <= relEng->cutData.nodes()->size())
            {
                adjToPop = ceil<i32>(adjToPop,10);
            }
            else
            {
                adjToPop = cli.pop();
            }

            auto [relBestTarget, relBestExactTarget] = relEng->getTargets();
            if (relBestExactTarget.has_value())
            {
                Node const & bestExt = relBestExactTarget.value();
                //printf("[DBG] REL exact value: %.3f\n", bestExt.g());
                bnb.primal(bestExt);
            }
            if (relBestTarget.has_value())
            {
                Node const & bestOverall = relBestTarget.value();
                //printf("[DBG] REL best value: %.3f (Exact %d)\n", bestOverall.g(), not bestOverall.approximated());; // G is correct!
                if (isBetter<Model>(bestOverall.g(), bnb.primal()))
                {
                    if (bestOverall.approximated())
                    {
                        auto const & cutset = relEng->cutData.nodes();
                        stashQueue.pushFromGpu(cutset, bnb.primal());
                    }
                }
            }
            else
            {
                //printf("[DBG] REL no node survived \n");
            }
            log.progress();
        }
    }

    stashQueue.stop();
    for (auto & t : validators) t.join();
    readyQueue.stop();

    stats.end();
    log.summary();
    //Timer::summary();

    return EXIT_SUCCESS;
}
