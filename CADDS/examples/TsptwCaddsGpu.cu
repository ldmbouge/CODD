#include <CliManager.hpp>
#include <CaddsGpu.cuh>
#include "Tsptw.hpp"

int main(int argc, char * argv[]) {
  using namespace gfl;
  CliManager cli;
  cli.parse(argc, argv);
  PoolAllocator<Managed> alloc;
  auto const model = alloc.makePtr<Tsptw>(cli.instancePath(), alloc);
  return runCaddsGpu(model, cli);
}