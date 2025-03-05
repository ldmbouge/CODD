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
      //std::cout << "TIME:" << setprecision(ss) << optTime << "\n";
   });
   WidthBounded* ddr[2];

   AbstractDD::Ptr restricted = _theDD->duplicate();
   restricted->setStrategy(ddr[0] = new Restricted(_mxw));

   AbstractDD::Ptr relaxed = _theDD->duplicate();
   relaxed->setStrategy(ddr[1] = new Relaxed(_mxw));


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

   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   while(!pq.empty()) {
      auto bbn = pq.extractMax();

      // std::cout << "dequeued: ";
      // restricted->printNode(std::cout, bbn.node);
      // std::cout << std::endl;

      
      auto curDual = bbn.bound;
      bnds.setDual(bbn.node->getBound(),curDual);

      // auto compDual = bbn.node->getBound() + restricted->local(bbn.node, LocalContext::DDInit);
      // //cout << "DUAL KEY:" << curDual << " dualCOMP:" << compDual << "\n";
      // if (!restricted->isBetterEQ(compDual,curDual)) {
      //    //cout<< " dual comp improve!\n";
      //    curDual = compDual;
      // }
      ttlNode++;
      // cout << "CURDUAL:" << curDual << "\t PRIMAL:" << bnds.getPrimal()
      //        << " isBetter:" << restricted->isBetter(curDual,bnds.getPrimal()) << "\n";
      // if (!restricted->isBetter(curDual,bnds.getPrimal())) {
      //    bbPool->release(bbn.node);
      //    continue;
      // }
      nNode++;
      restricted->apply(bbn.node,bnds);
            
      auto discardSet = restricted->theDiscardedSet();
      //cout << "discarded set: " << discardSet << endl;
      for(auto n : discardSet) {
            
         std::cout << "discarded: ";
         restricted->printNode(std::cout, n);
         std::cout << std::endl;

         if(relaxed->hasLocal()) {
            //std::cout << "has local" << std::endl;
            double localDual = relaxed->local(n, LocalContext::BBCtx);
            std::cout << "isBetter(primal=" << bnds.getPrimal() << ", bnd=" << n->getBound() << "+local=" << localDual << "="<< n->getBound()+localDual <<") = " << relaxed->isBetterEQ(bnds.getPrimal(), n->getBound() + localDual) << std::endl;
            if(relaxed->isBetterEQ(bnds.getPrimal(), n->getBound() + localDual)) {
               continue;
               std::cout << "local skip" << std::endl;
            }
         }

         bool dualBetter = relaxed->apply(n, bnds);
         if(dualBetter) {
            // if (n == relaxed->getRoot()) { // the cutset is the root. Only way out: increase width.
            //    auto w = ddr[1]->getWidth() + 1;
            //    ddr[1]->setWidth(w);
            //    std::cout << "\t-->widening... " << w << "\n";
            // }
            auto nd = bbPool->cloneNode(n);
            std::cout << "cloned and got: " << nd << std::endl;
            if (nd) {
               assert(nd->getBound() == n->getBound());
               pq.insertHeap(QNode {nd, nd->getBound()+nd->getBackwardBound() });
            }
         } else {
            std::cout << "relaxed skip" << std::endl;
         }
         bbPool->release(bbn.node);
      }
   }

   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
        << "\t P/D:" << pruned << "/" << insDom
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\t LIM?:" << (pq.size() > 0)
        << "\t Seen:" << nbSeen
        << "\nSol: " << bnds
        << "\n";
}

// breaksRF 4     -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 1 1], 1323
// longest prefix -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 

//discarded: 24,T,<19,439>,B=819,BB=0,LBLS:[0:1 1:1 2:1 3:1 4:0 5:1 6:1 7:1 8:0 9:0 10:1 11:1 12:1 13:1 14:1 15:1 16:1 17:1 18:0 ]
//discarded: 10,T,<19,439>,B=684,BB=0,LBLS:[0:1 1:1 2:0 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 11:1 12:1 13:0 14:0 15:0 16:0 17:1 18:0 ]
