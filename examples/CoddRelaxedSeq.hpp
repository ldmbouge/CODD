#pragma once

#include <GFL.hpp>

#include "CliManager.hpp"
#include "ExpansionEngine.hpp"
#include "Queue.hpp"
#include "BnBManager.hpp"
#include "LogManager.hpp"
#include "StatsManager.hpp"

template<typename Model, typename Node, int BranchFactor, int Depth>
int runRelaxedSeq(int argc, char* argv[])
{
    using namespace gfl;
    using ExpansionEngine = ExpansionEngine<Model,Node>;

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
    eng->initFullRelaxedExpansion(cli.width(), BranchFactor, Depth, buffAlloc);

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats;

    // Log manager
    LogManager<Model,Node> log;

    // Layers
    Queue<Model,Node> queue;
    queue.onPush([&](i32 const n){stats.inserted(n);});
    queue.onPull([&](i32 const n){stats.extracted(n);});
    auto const state = model->initial();
    auto const outLabels = model->lgf(state, worst<Model>(), best<Model>(), DDExact);
    f64 const h =  Model::has_heur ? model->h(state, DDInit) : best<Model>();
    Node const root(state, outLabels, h);
    queue.push(root);

    // BnB search
    log.header();
    stats.start();
    while (
        not queue.empty() and
        stats.elapsed<sec>() <= cli.timeout() and
        not bnb.gapClosed())
    {
        // Expand
        Node const & node = queue.pullBest();
        eng->fullyExpandRelaxed(model, node, bnb.primal(), bnb.dual());

        // Check relaxation and manage cutset
        if (eng->hasTarget())
        {
            Node const & trg_node = eng->getTarget();
            if (not bnb.pruneAncestor(model, trg_node))
            {
                bnb.updatePrimal(model, trg_node);
                auto const & cutset = eng->cutset();
                queue.push(cutset);
                bnb.updateDual(queue.bestDual());
            }
        }
        log.log(stats,bnb);
    }
    stats.end();
    log.summary(stats,bnb,cli.timeout());

    return EXIT_SUCCESS;
}