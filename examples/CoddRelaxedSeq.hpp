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
    std::cout << "Instance: " << cli.instance_path << std::endl;
    std::cout << "Width: " << cli.width << std::endl;
    std::cout << "Engine: Sequential";

    // Allocators
    constexpr i32 model_mem_size = 256 * 1024; // Small, it MUST fit in shared memory
    assert(model_mem_size > sizeof(Model)); // At least, instance data not included
    ArenaAllocator model_alloc(model_mem_size, heapReserve(model_mem_size));
    constexpr i32 engine_mem_size = 1024; // Small, it MUST be on managed memory
    assert(engine_mem_size > sizeof(ExpansionEngine));
    ArenaAllocator engine_alloc(engine_mem_size, heapReserve(engine_mem_size));
    ArenaAllocator buffers_alloc(cli.mem_size, heapReserve(cli.mem_size));

    // Model (+ Instance) creation
    Model * const model = new (model_alloc) Model();
    model->init(cli.instance_path, model_alloc);

    // Expansion engine
    ExpansionEngine * const eng = new (engine_alloc) ExpansionEngine();
    eng->initFullRelaxedExpansion(cli.width, BranchFactor, Depth, buffers_alloc);

    // Bounds and solutions manager
    BnBManager<Model,Node> bnb;

    // Search statistics
    StatsManager stats;

    // Log manager
    LogManager<Model,Node> log;

    // Layers
    Queue<Model,Node> queue;
    auto const state = model->initial();
    auto const outLabels = model->lgf(state, worst<Model>(), best<Model>(), DDExact);
    f64 const f =  Model::has_heur ? model->h(state, DDInit) : best<Model>();
    Node const root(state, outLabels, f);
    queue.push(root);

    // BnB search
    log.header();
    stats.start();
    while (stats.elapsed<sec>() <= cli.timeout and bnb.gapClosed())
    {
        // Expand
        assert(not queue.empty());
        Node const & node = queue.pullBest();
        stats.extracted(1);
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
                stats.inserted(cutset.size());
            }
        }
        log.log(stats,bnb);
    }
    log.summary(stats,bnb, cli.timeout);

    return EXIT_SUCCESS;
}