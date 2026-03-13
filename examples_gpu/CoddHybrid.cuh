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
    using EngGpu = ExpansionEngineGpu<Model,Node>;
    using EngCpu = ExpansionEngineSeq<Model,Node>;

    CliManager cli("CODD", "A C++ solver for DIDP models.");
    cli.parse(argc, argv);
    std::cout << "Instance: " << cli.instance() << std::endl;
    std::cout << "Engine: GPU (Relaxed) + CPU (Restricted)"  << std::endl;
    std::cout << "Relaxed Width: " << cli.gpuWidth() << std::endl;
    std::cout << "Relaxed Buffer: " << cli.pop() << std::endl;
    std::cout << "Restricted Width: " << cli.cpuWidth() << std::endl;
    std::cout << "Validators: " << cli.validators() << std::endl;
    std::cout << "Lambda: " << cli.lambda() << std::endl;

    // Model (+ Instance) creation
    constexpr i32 modelMemSize = 512 * 1024 * 1024;
    assert(modelMemSize > sizeof(Model)); // At least, instance data not included
    ArenaAllocator modelAlloc(modelMemSize, cudaReserveManaged(modelMemSize));
    Model * const model = new (modelAlloc) Model();
    model->init(cli.instance(), modelAlloc);

    // Expansion engines
    i32 engMemSize = sizeof(EngGpu) + DefaultAlign;
    ArenaAllocator engGpuAlloc(engMemSize * 2, cudaReserveManaged(engMemSize * 2));
    ArenaAllocator buffGpuAlloc(cli.memSize(), cudaReserveDevice(cli.memSize()));
    EngGpu * const resEngGpu = new (engGpuAlloc) EngGpu();
    resEngGpu->initRestrictedExpansion(cli.gpuWidth(), BranchFactor, buffGpuAlloc);
    buffGpuAlloc.clear();
    EngGpu * const relEngGpu = new (engGpuAlloc) EngGpu();
    relEngGpu->initRelaxedExpansion(cli.gpuWidth(), BranchFactor, Depth, buffGpuAlloc);
    printf("GPU Relaxed/Restricted working memory: ");
    printMemSize(buffGpuAlloc.usedSize());
    printf("\n");

    // Relaxed + Restricted engines - per validator thread
    engMemSize = sizeof(EngCpu) + DefaultAlign;
    std::vector<EngCpu*> resEngsCpu;
    std::vector<EngCpu*> relEngsCpu;
    ArenaAllocator engCpuAlloc(engMemSize * 2  * cli.validators() , heapReserve(engMemSize * 2 * cli.validators()));
    ArenaAllocator buffCpuAlloc(cli.memSize(), heapReserve(cli.memSize()));
    std::vector<f64> inFlightBestF;
    for (i32 i = 0; i < cli.validators(); ++i)
    {
        resEngsCpu.push_back(new (engCpuAlloc) EngCpu());
        relEngsCpu.push_back(new (engCpuAlloc) EngCpu());
        inFlightBestF.push_back(best<Model>());
        resEngsCpu[i]->initRestrictedExpansion(cli.cpuWidth(), BranchFactor, buffCpuAlloc);
        relEngsCpu[i]->initRelaxedExpansion(32, BranchFactor, Depth, buffCpuAlloc);
    }
    printf("CPU Relaxed + Restricted working memory: ");
    printMemSize(buffCpuAlloc.usedSize());
    printf("\n");

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats(cli.timeout());

    // Queues
    std::mutex              queueMutex;
    std::condition_variable queueCV;
    std::atomic<i32> inFlight = 0;
    BlockingQueue<Queue<Model,Node>, Model> readyQueue(queueMutex, queueCV, inFlight);
    BlockingQueue<Queue<Model,Node>, Model> pendingQueue(queueMutex, queueCV, inFlight);
    bnb.onPrimal([&]{readyQueue.drainInto(pendingQueue);});


    // Log manager
    LogManager<Model, Node, decltype(readyQueue), decltype(pendingQueue)> log(bnb, readyQueue, pendingQueue, stats);
    // ── Validator threads ────────────────────────────────────────────

    auto validatorFn = [&](f64 & bestF, EngCpu * relEngCpu, EngCpu * resEngCpu)
    {
        constexpr i32 nodesToPull = 25;
        std::vector<Node const *> bufferIn;
        std::vector<Node const *> bufferOut;
        bufferIn.reserve(nodesToPull);
        bufferOut.reserve(nodesToPull);

        while (true)
        {
            bufferIn.clear();
            bufferOut.clear();

            if (not pendingQueue.pullUpTo(nodesToPull, bufferIn, bestF, bnb.primal())) break;
            for (auto & node : bufferIn)
            {
                std::vector<Node> nodes;
                nodes.push_back(*node);

                // Pruning by not connection to the sink
                relEngCpu->expandRelaxed(model, nodes, bnb.primal(), bnb.dual(), 1.0, false);
                auto [relBestTarget, relBestExactTarget] = relEngCpu->getTargets();
                if (not relBestTarget.has_value()) { continue; }

                // Pruning by exactness or primal
                resEngCpu->expandRestricted(model, nodes, bnb.primal(), bnb.dual());
                auto [resBestTarget, _] = resEngCpu->getTargets();
                if (resBestTarget.has_value())
                {
                    assert(not resBestTarget.value().approximated());
                    bnb.primal(resBestTarget.value());
                }
                if (not resEngCpu->exact or not resEngCpu->completed)
                {
                    bufferOut.push_back(node);
                }
            }
            readyQueue.push(bufferOut, bestF);   // inFlight += pushed
            pendingQueue.done(bufferIn.size());
        }
    };

    std::vector<Node> parentsBuffer;
    parentsBuffer.reserve(cli.pop());
    Pool nodesPoll;
    Node const * root = Node::makeRoot(model);
    pendingQueue.push(root);

    std::vector<std::thread> validators;
    for (i32 i = 0; i < cli.validators(); ++i)
    {
        validators.push_back(std::thread(
            validatorFn,
            std::ref(inFlightBestF[i]),
            relEngsCpu[i],
            resEngsCpu[i])
        );
    }

    std::atomic<bool> searchDone{false};
    std::thread logThread([&]()
    {
        while (not searchDone.load())
        {
            log.progress();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
    });

    auto const getDual = [&]
    {
        std::unique_lock lock(queueMutex);
        f64 dual = worst<Model>();
        if (not readyQueue.emptyUnlocked())   dual = better<Model>(dual, readyQueue.peekBestUnlocked()->f());
        if (not pendingQueue.emptyUnlocked()) dual = better<Model>(dual, pendingQueue.peekBestUnlocked()->f());
        for (auto const & f : inFlightBestF)
            dual = better<Model>(dual, f);
        return dual;
    };

    // BnB search
    i32 patientPullAdj = 5;
    bool firstRestricted = false;
    log.header();
    stats.start();
    i32 pullGpu = cli.pop();
    while (stats.elapsed<sec>() <= cli.timeout() and not bnb.solved())
    {
        if (not firstRestricted)
        {
            parentsBuffer.push_back(*readyQueue.peekBestBlocking(inFlight));
            resEngGpu->expandRestricted(model, parentsBuffer, bnb.primal(), bnb.dual());
            auto [resBestTarget, resBestExactTarget] = resEngGpu->getTargets();
            if (resBestTarget.has_value())
            {
                assert(not resBestTarget.value().approximated());
                bnb.primal(resBestTarget.value());
            }
            parentsBuffer.clear();
            firstRestricted = true;
        }

        // Collect batch from readyQueue
        parentsBuffer.clear();
        if (not readyQueue.pullUpTo(pullGpu, parentsBuffer, bnb.primal())) break;
        if (not parentsBuffer.empty())
        {
            //printf("[DBG] Offloading %ld nodes with f %.2f (Remaining %ld)\n", parentsBuffer.size(), parentsBuffer.back().f(), queue.size());
            relEngGpu->expandRelaxed(model, parentsBuffer, bnb.primal(), bnb.dual(), cli.lambda());
            // Avoid loops
            if (parentsBuffer.size() * 1.1 >= relEngGpu->cutData.nodes()->size() and
                parentsBuffer.size() * 0.9 <= relEngGpu->cutData.nodes()->size())
            {
                pullGpu = ceil<i32>(pullGpu,10);
                printf("[INFO] Loop detected, reducing pull size to %d\n", pullGpu);
                fflush(stdout);
            }
            else
            {
                pullGpu = cli.pop();
            }

            auto [relBestTarget, relBestExactTarget] = relEngGpu->getTargets();
            if (relBestExactTarget.has_value())
            {
                Node const & bestExt = relBestExactTarget.value();
                //printf("[DBG] REL exact value: %.3f\n", bestExt.g());
                bnb.primal(bestExt);
            }
            if (relBestTarget.has_value())
            {
                Node const & bestOverall = relBestTarget.value();
               // printf("[DBG] REL best value: %.3f (Exact %d)\n", bestOverall.g(), not bestOverall.approximated());; // G is correct!
                if (isBetter<Model>(bestOverall.g(), bnb.primal()))
                {
                    if (bestOverall.approximated())
                    {
                        pendingQueue.push(relEngGpu->cutData.fragments(), bestOverall.g(),  bnb.primal());
                    }
                }
            }
            readyQueue.done(parentsBuffer.size());
        }
        bnb.dual(getDual());
    }

    searchDone.store(true);
    logThread.join();

    pendingQueue.stop();
    for (auto & t : validators) t.join();
    readyQueue.stop();

    stats.end();
    log.summary();
    //Timer::summary();

    return EXIT_SUCCESS;
}
