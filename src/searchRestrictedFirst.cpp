#include "searchRestrictedFirst.hpp"
#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

void BAndBRestrictedFirst::search(Bounds& bnds)
{
   // Setup

   auto bbPool = _theDD->makeNDAllocator();
   using namespace std;
   unsigned int nbSeen = 0;
   std::streamsize ss = cout.precision();
   auto start = RuntimeMonitor::cputime();
   auto last = start;
   cout << "B&B searching..." << endl;
   bnds.attach(_theDD);
   double optTime = 0.0;
   bnds.onSolution([ss,start,&optTime](const auto& lbls) {
      optTime = RuntimeMonitor::elapsedSince(start);
      std::cout << "TIME:" << setprecision(ss) << optTime << "\n";
   });
   AbstractDD::Ptr relaxed = _theDD->duplicate();
   AbstractDD::Ptr restricted = _theDD->duplicate();

   relaxed->setStrategy(new Relaxed(_mxw));
   restricted->setStrategy(new Restricted(_mxw));

   auto hOrder = [restricted](const QNode& a,const QNode& b) {
      return restricted->isBetter(a.bound,b.bound);
   };
   Heap<QNode,decltype(hOrder)> pq(bbPool->get(),64000,hOrder);
   ANode::Ptr rootNode = bbPool->cloneNode(restricted->init());

   if (restricted->hasLocal()) {
      auto primalRootValue = restricted->local(rootNode,LocalContext::BBCtx);
      cout << "primal@root:" << primalRootValue << "\n";
      rootNode->setBackwardBound(primalRootValue);
      pq.insertHeap(QNode { rootNode, primalRootValue } );   
   } else {
      pq.insertHeap(QNode { rootNode, restricted->initialWorst() } );   
   }


   // Main Loop
   bool dualBetter = false;

   while(!pq.empty()) {
      auto bbn = pq.extractMax();

      // std::cout << "dequeued: ";
      // restricted->printNode(std::cout, bbn.node);
      // std::cout << std::endl;

      auto curPrimal = bbn.bound;
      //bnds.setPrimalG(bbn.node->getBound(),curPrimal); 
      //TODO: ask if makes sense to update global here

      auto compPrimal = bbn.node->getBound() + restricted->local(bbn.node, LocalContext::DDInit);
      if (!restricted->isBetterEQ(compPrimal,curPrimal)) {
         //cout<< " primal comp improve!\n";
         compPrimal = compPrimal;
      }
      dualBetter = false;

      // cout << "CURPRIMAL:" << curPrimal << "\t DUAL:" << bnds.getDual()
      //      << " isBetter:" << restricted->isBetter(curPrimal,bnds.getDual()) << "\n";
      // if (!restricted->isBetter(curPrimal,bnds.getDual())) {
      //    bbPool->release(bbn.node);
      //    continue;
      // }

      bool primalBetter = restricted->apply(bbn.node,bnds,true);
      //cout << "primalBetter? " << primalBetter << endl;
      if(primalBetter) {
         dualBetter = relaxed->apply(bbn.node, bnds);
         if (!restricted->isExact() && !relaxed->isExact()) {
            
            auto discardSet = restricted->theDiscardedSet();
            //cout << "discarded set: " << discardSet << endl;
            for(auto n : discardSet) {
               
               // std::cout << "discarded: ";
               // restricted->printNode(std::cout, n);
               // std::cout << std::endl;

               // if (!restricted->isBetter(n->getBound() + n->getBackwardBound(),bnds.getDual())) {
               //    continue; // the loop over the discard set! Not the main loop
               // }

               auto nd = bbPool->cloneNode(n);

               if(nd) {
                  assert(nd->getBound() == n->getBound());
                  double bwd;

                  // if (restricted->hasLocal()) {
                  //    auto newBnd = restricted->local(nd,LocalContext::BBCtx);
                  //    if (!restricted->isBetter(newBnd,n->getBackwardBound()))
                  //       bwd = newBnd;
                  //    else bwd = n->getBackwardBound();
                  // } else bwd = n->getBackwardBound();
                  bwd = n->getBackwardBound();
                  
                  const auto insKey = n->getBound() + bwd;
                  const auto improve = true; //restricted->isBetter(insKey,bnds.getDual());
                  if(improve) {
                     //cout << "insert: " << nd << endl;
                     pq.insertHeap(QNode {nd, insKey });
                  }
               }
            }

         }
      }
      bbPool->release(bbn.node);
   }

   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << endl
        << "Sol: " << bnds << endl
        << "\t LIM?:" << (pq.size() > 0)
        << "\n";
}