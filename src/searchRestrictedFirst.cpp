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

template<typename Heap>
std::tuple<int,bool,std::vector<ANode::Ptr>> filterDom(Bounds& bnds,
                                                       AbstractDD::Ptr dd,
                                                       const std::vector<ANode::Ptr>& nodes,
                                                       Heap* pq)
{
   bool newGuyDominated = false;
   std::vector<ANode::Ptr> survived;
   if (!dd->hasDominance()) {
      survived = nodes;
      return {0,false,survived};
   }
   auto end = nodes.rend();
   auto begin = nodes.rbegin();
   for (auto i = begin; i != end; i++) {
      auto n = *i;
      [[maybe_unused]] auto sz = nodes.size();
      for (auto j = begin; j != end; j++) {
         auto other = *j;
         if(i == j) continue;
         bool isObjDom = dd->isBetterEQ(other->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominatesEq(other,n);
         if (newGuyDominated) {
            break;          
         }
      }
      if(!newGuyDominated) {
         survived.push_back(n);
      }
   }

   int pruned = 0;
   for(auto n: nodes) {
      unsigned d = 0;
      auto pqSz = pq->size();
      auto allLocs = new Heap::LocType*[pqSz];
      for(unsigned k = 0;k < pqSz;k++) {
         auto other = (*pq)[k];
         bool isObjDom   = dd->isBetterEQ(other->value().node->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominatesEq(other->value().node,n);
         if (newGuyDominated) {
            goto prune;             
         }        
         bool objDom   = dd->isBetterEQ(n->getBound(),other->value().node->getBound());
         bool qnDominated = objDom && dd->dominatesEq(n,other->value().node);
         if (qnDominated)
            allLocs[d++] = other;
      }

      prune:
      if (d) {
         for(auto i =0u; i < d;i++) 
            pq->remove(allLocs[i]);
         pruned += d;
      }
      delete[]allLocs;
   }
   return {pruned,newGuyDominated,survived};
}

void BAndBRestrictedFirst::search(Bounds& bnds)
{
   // Setup
   [[maybe_unused]] static int nbRELAX = 0;
   auto bbPool = _theDD->makeNDAllocator();
   using namespace std;
   unsigned int nbSeen = 0;
   std::streamsize ss = cout.precision();
   auto start = RuntimeMonitor::cputime();
   [[maybe_unused]] auto last = start;
   cout << "B&B(RF) searching..." << endl;
   bnds.attach(_theDD);
   double optTime = 0.0;
   bnds.onSolution([start,&optTime](const auto& lbls) {
      optTime = RuntimeMonitor::elapsedSince(start);
      //std::cout << "TIME:" << setprecision(ss) << optTime << "\n";
   });
   WidthBounded* ddr[2];

   AbstractDD::Ptr restricted = _theDD->duplicate();
   restricted->setStrategy(ddr[0] = new Restricted(_mxw));

   AbstractDD::Ptr relaxed = _theDD->duplicate();
   relaxed->setStrategy(ddr[1] = new Relaxed(_mxw));// _mxw));


   auto hOrder = [restricted](const QNode& a,const QNode& b) {
      return restricted->isBetter(a.bound,b.bound);
   };
   Heap<QNode,decltype(hOrder)> pq(bbPool->get(),64000,hOrder);
   ANode::Ptr rootNode = bbPool->cloneNode(restricted->init());

   if (restricted->hasLocal()) {
      auto dualRootValue = restricted->local(rootNode,LocalContext::BBCtx);
      cout << "dual@root:" << dualRootValue << "\n";
      rootNode->setBackwardBound(dualRootValue);
      pq.insertHeap(QNode { rootNode, dualRootValue } );   
   } else {
      pq.insertHeap(QNode { rootNode, restricted->initialWorst() } );
   }
   const bool hasLocal = relaxed->hasLocal();
   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   bool primalBetter = false;
   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   while(!pq.empty()) {
      auto bbn = pq.extractMax();

      //std::cout << "dequeued: " << bbn.node << " ";
      //restricted->printNode(std::cout, bbn.node);
      //std::cout << std::endl;
     
      auto curDual = bbn.bound;
      bnds.setDual(bbn.node->getBound(),curDual);
      auto now = RuntimeMonitor::cputime();
      auto fs = RuntimeMonitor::elapsedMilliseconds(start,now);
      auto fl = RuntimeMonitor::elapsedMilliseconds(last,now);
      if (_timeLimit && _timeLimit(fs))         
         break;      
      if (primalBetter || fl > 5000) {
         double gap = 100 * std::abs(bnds.getPrimal() - curDual) / bnds.getPrimal();      
         cout << std::fixed << "B&B(" << setw(5) << nNode << ")\t " << setprecision(6);
         if (curDual == relaxed->initialWorst())
            cout << setw(7) << "-"  << "\t " << setw(7) << bnds.getPrimal() << "\t ";
         else
            cout << setw(7) << curDual << "\t " << setw(7) << bnds.getPrimal() << "\t ";
         if (gap > 100)
            cout << setw(6) << "-";
         else cout << setw(6) << setprecision(4) << gap << "%";
         cout << "\t time:" << setw(6) << setprecision(4) <<  fs / 1000.0 << "s";
         cout << "\n";
         last = RuntimeMonitor::cputime();
      }
      ttlNode++;
      nNode++;
      primalBetter = restricted->apply(bbn.node,bnds);
      using namespace RuntimeMonitor;
      //std::cout << "BACK restricted:"  << elapsedMilliseconds(start,cputime())/1000.0 << " " << bbn.node << " \n";
      
      auto discardSet = restricted->theDiscardedSet();
      std::vector<ANode::Ptr> survivedLocal = hasLocal ? filterLocal(bnds, relaxed, discardSet) : discardSet;
      auto [tmpPruned,newGuyDominated,survivedDom] = filterDom(bnds, relaxed, survivedLocal, &pq);
      insDom += survivedLocal.size() - survivedDom.size();
      pruned += tmpPruned;
      
      //std::cout << "discardSet   :" << discardSet.size() << "\n";
      //std::cout << "survivedDom  :" << survivedDom.size() << "\n";
      //std::cout << "discardSet FILTER:"  << elapsedMilliseconds(start,cputime())/1000.0 << " " << bbn.node << " \n";

      //std::cout << "survivedLocal:" << survivedLocal.size() << "\n";      
      [[maybe_unused]] int nbRELAX = 0;
      int nbINS = 0;
      for(auto n: survivedDom) {
         nbRELAX++;
         bool dualBetter = relaxed->apply(n, bnds);
         // std::cout << "Survivor:" << n->getBound() << " BWD:" << n->getBackwardBound() << " TTL:" << n->getTotalBound()
         //           << " dualBetter:" << dualBetter
         //           << " newGuyDominated:" << newGuyDominated
         //           << "\n";
         //double localDual = relaxed->local(n, LocalContext::BBCtx);
         //std::cout << "LOCAL(S):" << localDual << " SUM:" << n->getBound() + localDual << "\n"; 
         // std::cout << "RELAX:" << n->getBound() << " | " << n->getTotalBound()
         //           << " DOM?:" << (newGuyDominated ? "T" : "F")
         //           << " Got: " << dualBetter << " B@SINK:" << relaxed->currentOpt() << "\n";         
         //std::cout << "reaching relaxed DD. Got: " << dualBetter <<"\n";         
         if(dualBetter) {
            if(!newGuyDominated) {
               auto nd = bbPool->cloneNode(n);
               //std::cout << "cloned and got: " << nd << std::endl;
               if (nd) {
                  assert(nd->getBound() == n->getBound());
                  pq.insertHeap(QNode {nd, nd->getBound()+nd->getBackwardBound() });
                  nbINS++;
               }
            } else insDom++;
         }
      }
      //std::cout << "ADDQUEUE: " << nbINS << " out of " << nbRELAX << "\n";
      bbPool->release(bbn.node);
      //std::cout << "nbRELAX:" << nbRELAX << "\t PQ = " << pq.size() << "\n";
   }


   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << std::setprecision (std::numeric_limits<double>::digits10 + 1)
        << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
        << "\t P/D:" << pruned << "/" << insDom
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\t LIM?:" << (pq.size() > 0)
        << "\t Seen:" << nbSeen
        << "\n";

   //cout << "NBRELAX:" << nbRELAX << "\n";
        //<< "\nSol: " << bnds
        //<< "\n";
   // cout << ddr[0]->getWidth() << " " << nNode << " " << spent/1000 << " " << bnds.getPrimal() << endl;
}

// breaksRF 4     -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 1 0 1 1 1 1 1], 1323
// longest prefix -> [1 1 0 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 

//discarded: 24,T,<19,439>,B=819,BB=0,LBLS:[0:1 1:1 2:1 3:1 4:0 5:1 6:1 7:1 8:0 9:0 10:1 11:1 12:1 13:1 14:1 15:1 16:1 17:1 18:0 ]
//discarded: 10,T,<19,439>,B=684,BB=0,LBLS:[0:1 1:1 2:0 3:1 4:1 5:1 6:1 7:1 8:1 9:1 10:1 11:1 12:1 13:0 14:0 15:0 16:0 17:1 18:0 ]
