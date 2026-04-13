#pragma once

#include <chrono>
#include <thread>
#include <GFL.hpp>
#include <cxxopts.hpp>

#include "CliManager.hpp"
#include "ExpansionEngineSeq.hpp"
#include "LayeredQueue.hpp"
#include "LogManager.hpp"
#include "SolutionManager.hpp"
#include "StatsManager.hpp"

template<typename Model>
int runCaddsSeq(gfl::Ptr<Model> model, CliManager const & cli) {
  using namespace gfl;
  using Node = LNode<typename Model::State, typename Model::OutLabels, Model::Depth>;
  using Engine = ExpansionEngineSeq<Model,Node>;
  static_assert(std::is_trivially_copyable_v<Node>);

  // Print information
  std::cout << "Search: CADD" << std::endl;
  std::cout << "Engine: Sequential" << std::endl;
  cli.print(std::cout);

  // Expansion engine
  PoolAllocator<Heap> engAlloc;
  Engine * const eng = new (engAlloc) Engine();
  ArenaAllocator buffAlloc(Heap::reserve(cli.memSize()), cli.memSize());

  // Queue
  LayeredQueue<Node> queue;
  Node const * root = Node::makeRoot(model.get());
  queue.push(0, root->outLabels().size(), root);

  // Misc
  StatsManager stats(cli.timeout());
  SolutionsManager<Model, Node> sols(model);
  LogManager<Model, Node> log(sols, queue, stats);

  // CADDs
  log.run();
  stats.startTime();
  while (not queue.empty() and (stats.elapsed() <= cli.timeout())) {
    i32 const lIdx = queue.deepestNotEmpty();
    i32 const branchingFactor = queue.getBranchingFactor(lIdx);
    i32 const maxBatchSize = Engine::getBatchSize(buffAlloc.totalSize(), branchingFactor);
    i32 toPull = cli.isFragmentSizeAuto() ? maxBatchSize : cli.fragmentSize();

    // Batching
    while (toPull > 0) {
      ArrayView<Node> const batch = queue.pullUpTo(lIdx, min<i32>(toPull,maxBatchSize));
      toPull -= batch.size() == maxBatchSize ? maxBatchSize : toPull;
      buffAlloc.clear();
      eng->allocData(batch.size(), branchingFactor, buffAlloc);
      eng->expand(model, batch, sols.solutionCost());
      auto const expansion = eng->getExpansion();
      if (not expansion.empty()) {
        Node const * const bestTargetNode = eng->getBestTargetNode();
        if (bestTargetNode) {
          sols.solution(*bestTargetNode);
        } else {
          queue.push(lIdx + 1, eng->getBranchingFactor(), expansion);
        }
      }
    }
  }
  stats.endTime();
  log.stop();

  return EXIT_SUCCESS;
}