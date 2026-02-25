#pragma once

#include <GFL.hpp>

#include "CliManager.hpp"
#include "ExpansionEngineSeq.hpp"
#include "Queue.hpp"
#include "BnBManager.hpp"
#include "LogManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node, int BranchFactor, int Depth>
int runRelaxedSeq(int argc, char* argv[])
{
    using namespace gfl;
    using ExpansionEngine = ExpansionEngineSeq<Model,Node>;

    CliManager cli("CODD", "A C++ solver for DIDP models.");
    cli.parse(argc, argv);
    std::cout << "Instance: " << cli.instance() << std::endl;
    std::cout << "Width: " << cli.width() << std::endl;
    std::cout << "Engine: Sequential"  << std::endl;
    // Allocators
    constexpr i32 modelMemSize = 256 * 1024; // Small, it MUST fit in shared memory
    assert(modelMemSize > sizeof(Model)); // At least, instance data not included
    ArenaAllocator modelAlloc(modelMemSize, heapReserve(modelMemSize));
    constexpr i32 engMemSize = 1024; // Small, it MUST be on managed memory
    assert(engMemSize > sizeof(ExpansionEngine));
    ArenaAllocator engAlloc(engMemSize, heapReserve(engMemSize));
    ArenaAllocator buffAlloc(cli.memSize(), heapReserve(cli.memSize()));

    // Model (+ Instance) creation
    Model * const model = new (modelAlloc) Model();
    model->init(cli.instance(), modelAlloc);

    // Expansion engine
    ExpansionEngine * const eng = new (engAlloc) ExpansionEngine();
    eng->initRelaxedExpansion(cli.width(), BranchFactor, Depth, buffAlloc);

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats(cli.timeout());

    // Layers
    Queue<Model,Node> queue;
    queue.push(Node::makeRoot(model));

    // Log manager
    LogManager<Model,Node> log(bnb,queue,stats);

    // BnB search
    log.header();
    stats.start();
    int cnt = 0;
    while (not queue.empty() and stats.elapsed<sec>() <= cli.timeout() and not bnb.solved())
    {
       //std::cout << "ITER CNT:" << cnt++ << " ----------------------------------------\n";
        Node const node = queue.pullBest();
        //std::cout << "NODE:";Node::print(node);std::cout << "\n";
        bnb.dual(queue.bestDual());
        assert(bnb.consistent());

        eng->fullyExpandRelaxed(model, node, bnb.primal(), bnb.dual());

        // Check relaxation and manage cutset
        auto [bestTarget, bestExactTarget] = eng->getTargets();
        if (bestExactTarget.has_value())
        {
            bnb.primal(bestExactTarget.value());
        }
        if (bestTarget.has_value())
        {
            if (not bnb.pruneAncestor(bestTarget.value()))
            {
                //std::cout << "who is a bad boy?"; Node::print(trg); std::cout << "\n";
                bnb.dual(bestTarget.value());
                assert(bnb.consistent());
                auto const & cutset = eng->cutset();
                queue.push(cutset);
            }
        }
        log.progress();
    }
    stats.end();
    log.summary();

    return EXIT_SUCCESS;
}
