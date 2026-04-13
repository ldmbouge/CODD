#include <CaddsSeq.hpp>
#include <CliManager.hpp>
#include "Tsptw.hpp"

int main(int argc, char * argv[]) {
  using namespace gfl;
  CliManager cli;
  cli.parse(argc, argv);
  PoolAllocator<Heap> alloc;
  auto const model =  alloc.makePtr<Tsptw>(cli.instancePath(), alloc);
  return runCaddsSeq(model, cli);
}