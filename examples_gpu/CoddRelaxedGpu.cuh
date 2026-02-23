#pragma once

#include <GFL.hpp>

#include "CliManager.hpp"
#include "Queue.hpp"
#include "BnBManager.hpp"
#include "LogManager.hpp"
#include "StatsManager.hpp"
#include "ExpansionEngineGpu.cuh"

template<typename Model, typename Node, int BranchFactor, int Depth>
int runRelaxedGpu(int argc, char* argv[])
{
    using namespace gfl;
    using ExpansionEngine = ExpansionEngineGpu<Model,Node>;

    CliManager cli("CODD", "A C++ solver for DIDP models.");
    cli.parse(argc, argv);
    std::cout << "Instance: " << cli.instance() << std::endl;
    std::cout << "Width: " << cli.width() << std::endl;
    std::cout << "Engine: GPU"  << std::endl;
    // Allocators
    constexpr i32 modelMemSize = 256 * 1024; // Small, it MUST fit in shared memory
    assert(modelMemSize > sizeof(Model)); // At least, instance data not included
    ArenaAllocator modelAlloc(modelMemSize, cudaReserveManaged(modelMemSize));
    constexpr i32 engMemSize = 1024; // Small, it MUST be on managed memory
    assert(engMemSize > sizeof(ExpansionEngine));
    ArenaAllocator engAlloc(engMemSize, cudaReserveManaged(engMemSize));
    ArenaAllocator buffAlloc(cli.memSize(), cudaReserveDevice(cli.memSize()));

    // Model (+ Instance) creation
    Model * const model = new (modelAlloc) Model();
    model->init(cli.instance(), modelAlloc);

    // Expansion engine
    ExpansionEngine * const eng = new (engAlloc) ExpansionEngine();
    eng->initRelaxedExpansion(cli.width(), BranchFactor, Depth, buffAlloc);

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats;

    // Log manager
    LogManager<Model,Node> log;
    bnb.onPrimal([&log,&stats,&bnb]{log.primal(stats,bnb);});
    bnb.onDual([&log,&stats,&bnb]{log.dual(stats,bnb);});

    // Layers
    Queue<Model,Node> queue;
    queue.onPush([&](i32 const n){stats.inserted(n);});
    queue.onPull([&](i32 const n){stats.extracted(n);});
    queue.push(Node::makeRoot(model));

    // BnB search
    log.header();
    stats.start();
    while (stats.elapsed<sec>() <= cli.timeout() and not bnb.solved())
    {
        Node const node = queue.pullBest();
        bnb.dual(queue.bestDual());
        assert(bnb.consistent());

        eng->expandRelaxed(model, node, bnb.primal(), bnb.dual());

        // Check relaxation and manage cutset
        if (eng->hasTarget())
        {
            Node const trg = eng->getTargetFromGpu();
            if (not bnb.pruneAncestor(trg))
            {
                bnb.primal(trg);
                assert(bnb.consistent());
                auto const & cutset = eng->cutset();
                queue.pushFromGpu(cutset);
            }
        }
        log.progress(stats,bnb);
    }
    stats.end();
    log.summary(stats,bnb,cli.timeout());

    return EXIT_SUCCESS;
}
