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
   unsigned int nbSeen = 0;
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
   restricted->setStrategy(ddr = new Restricted(_mxw));

   auto hOrder = [restricted](const QNode& a,const QNode& b) {
      return restricted->isBetter(a.bound,b.bound);
   };
   ANode::Ptr rootNode = bbPool->cloneNode(restricted->init());

   if (restricted->hasLocal()) {
      auto primalRootValue = restricted->local(rootNode,LocalContext::BBCtx);
      cout << "primal@root:" << primalRootValue << "\n";
      rootNode->setBackwardBound(primalRootValue);
   }

   unsigned nIter = 0;
   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   
   bool exact = false;
   while(!exact) {
      std::cout << "trying width=" << ddr->getWidth() << "...\n";
      restricted->apply(rootNode,bnds);
      exact = restricted->isExact();
      ddr->setWidth(ddr->getWidth() << 1);
      nIter++;
   }
   
   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #iterations:" <<  nIter
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\tWidth: " << _mxw << "/" << ddr->getWidth()
        //<< "\nSol: " << bnds
        << "\n";
}