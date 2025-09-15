#include "searchRestrictedOnlyNoQ.hpp"
#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

void BAndBRestrictedOnlyNoQ::search(Bounds& bnds)
{
   // Setup

   auto bbPool = _theDD->makeNDAllocator();
   using namespace std;
   std::streamsize ss = cout.precision();
   auto start = RuntimeMonitor::cputime();
   [[maybe_unused]] auto last = start;
   cout << "B&B searching..." << endl;
   bnds.attach(_theDD);
   double optTime = 0.0;
   bnds.onSolution([ss,start,&optTime](const auto& lbls) {
      optTime = RuntimeMonitor::elapsedSince(start);
      //std::cout << "TIME:" << setprecision(ss) << optTime << "\n";
   });
   AbstractDD::Ptr restricted = _theDD->duplicate();
   WidthBounded* ddr;
   restricted->setStrategy(ddr = new Restricted(_mxw)); // forget the discard tracking

   ANode::Ptr rootNode = bbPool->cloneNode(restricted->init());

   if (restricted->hasLocal()) {
      auto dualRootValue = restricted->local(rootNode,LocalContext::BBCtx);
      cout << "dual@root:" << std::fixed << dualRootValue << "\n";
      rootNode->setBackwardBound(dualRootValue);
   }

   unsigned nIter = 0;
   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   
   bool exact = false;
   long ttl = 0;
   while(!exact) {
      std::cout << "trying width=" << ddr->getWidth() << "...\n";
      restricted->apply(rootNode,bnds);
      exact = restricted->isExact();
      if (!exact) ddr->setWidth(ddr->getWidth() << 1);
      nIter++;
      ttl += restricted->nbNodes();
      std::cout << "Expanded:" << restricted->nbNodes() << "\n";
   }
   
   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "TOTAL # nodes:" << ttl << "\n";
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #iterations:" <<  nIter
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\tWidth: " << _mxw << "/" << ddr->getWidth()
        //<< "\nSol: " << bnds
        << "\n";
}
