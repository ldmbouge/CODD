#include "searchRestrictedOnly.hpp"
#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

void BAndBRestrictedOnly::search(Bounds& bnds)
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
   AbstractDD::Ptr restricted = _theDD->duplicate();

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

      // cout << "CURPRIMAL:" << curPrimal << "\t DUAL:" << bnds.getDual()
      //      << " isBetter:" << restricted->isBetter(curPrimal,bnds.getDual()) << "\n";
      // if (!restricted->isBetter(curPrimal,bnds.getDual())) {
      //    bbPool->release(bbn.node);
      //    continue;
      // }

      restricted->apply(bbn.node,bnds,true);
      //cout << "primalBetter? " << primalBetter << endl;
      //if (!restricted->isExact()) {
            
         auto discardSet = restricted->theDiscardedSet();
         cout << "discarded set: " << discardSet << endl;
         for(auto n : discardSet) {
               
            std::cout << "discarded: ";
            restricted->printNode(std::cout, n);
            std::cout << std::endl;

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
                  cout << "insert: " << endl << "\t";
                  restricted->printNode(std::cout, nd);
                  cout << endl;
               }
            } else {
               cout << "clone failed: " << endl << "\t";
               restricted->printNode(std::cout, n);
               cout << endl;
               //exit(-42);
            }
         }

      //}
      bbPool->release(bbn.node);
   }

   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << endl
        << "Sol: " << bnds << endl
        << "\t LIM?:" << (pq.size() > 0)
        << "\n";
}

// breaksRF 4     -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 1 1], 1323
// longest prefix -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 

//discarded: 24,T,<19,439>,B=819,BB=0,LBLS:[0:1 1:1 2:1 3:1 4:0 5:1 6:1 7:1 8:0 9:0 10:1 11:1 12:1 13:1 14:1 15:1 16:1 17:1 18:0 ]
//discarded: 10,T,<19,439>,B=684,BB=0,LBLS:[0:1 1:1 2:0 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 11:1 12:1 13:0 14:0 15:0 16:0 17:1 18:0 ]
