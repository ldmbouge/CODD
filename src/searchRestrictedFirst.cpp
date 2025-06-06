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

std::vector<ANode::Ptr> filterLocal(Bounds& bnds, AbstractDD::Ptr dd,const std::vector<ANode::Ptr>& nodes)
{
   std::vector<ANode::Ptr> survived;
   for(const auto& n: nodes) {
      double localDual = dd->local(n, LocalContext::BBCtx);
      if(!dd->isBetterEQ(bnds.getPrimal(), n->getBound() + localDual)) 
         survived.push_back(n);      
   }
   return survived;
}

template<typename Ord>
std::tuple<int,bool,std::vector<ANode::Ptr>>
filterDom(Bounds& bnds, AbstractDD::Ptr dd,const std::vector<ANode::Ptr>& nodes, Heap<QNode,Ord>* pq)
{
   std::vector<ANode::Ptr> survived;
   bool newGuyDominated = false;
   auto end = nodes.rend();
   auto begin = nodes.rbegin();
   for (auto i = begin; i != end; i++) { // loops backward on nodes
      auto n = *i;
      [[maybe_unused]] auto sz = nodes.size();
      //for (auto j = i+1; j != end; j++) {
      for (auto j = begin; j != end; j++) {
         auto other = *j;
         if(i == j) continue;
         bool isObjDom = dd->isBetterEQ(other->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominates(other,n);
         if (newGuyDominated) {
            break;          
         }
         // bool objDom   = dd->isBetterEQ(n->getBound(),(*j)->getBound());
         // bool qnDominated = objDom && dd->dominates(n,*j);
         // if (!qnDominated)
         //    survived->push_back(n);
      }
      if(!newGuyDominated) {
         survived.push_back(n);
      }
   }

   int pruned = 0;
   for(const auto& n: nodes) {
      unsigned d = 0;
      auto pqSz = pq->size();
      auto allLocs = new Heap<QNode,Ord>::LocType*[pqSz];
      for(unsigned k = 0;k < pqSz;k++) {
         auto other = (*pq)[k];
         bool isObjDom   = dd->isBetterEQ(other->value().node->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominates(other->value().node,n);
         if (newGuyDominated) {
            goto prune;  // n is dominated by something in the queue. kill what n dominates now            
         }        
         bool objDom   = dd->isBetterEQ(n->getBound(),other->value().node->getBound());
         bool qnDominated = objDom && dd->dominates(n,other->value().node); // n dominates a node in the QUEUE. Kill queue node
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
   return {pruned,newGuyDominated,std::move(survived)};
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
   bnds.onSolution([ss,start,&optTime](const auto& lbls) {
      optTime = RuntimeMonitor::elapsedSince(start);
      //std::cout << "TIME:" << setprecision(ss) << optTime << "\n";
   });
   WidthBounded* ddr[2];

   AbstractDD::Ptr restricted = _theDD->duplicate();
   restricted->setStrategy(ddr[0] = new Restricted(_mxw));  //->killDominance(); // doesn't help either

   AbstractDD::Ptr relaxed = _theDD->duplicate();   
   relaxed->setStrategy(ddr[1] = new Relaxed(_mxw)); //->killDominance(); // doesn't help

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

   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   while(!pq.empty()) {
      auto bbn = pq.extractMax();

      auto curDual = bbn.bound;
      bnds.setDual(bbn.node->getBound(),curDual);

      ttlNode++;
      nNode++;
      restricted->apply(bbn.node,bnds);

      auto discardSet = restricted->theDiscardedSet();

      std::vector<ANode::Ptr> survivedLocal = relaxed->hasLocal() ? filterLocal(bnds, relaxed, discardSet) : discardSet;
      
      std::vector<ANode::Ptr> survivedDom;
      bool newGuyDominated = false;
      if (relaxed->hasDominance()) {
         auto [tmpPruned,newGuyDominated,survivedDom] = filterDom<decltype(hOrder)>(bnds, relaxed, survivedLocal, &pq);
         insDom += survivedLocal.size() - survivedDom.size();
         pruned += tmpPruned;
      } else {
         survivedDom = survivedLocal;
      }
      //std::cout << "discardSet   :" << discardSet.size() << "\n";
      //std::cout << "survivedLocal:" << survivedLocal.size() << "\n";      
      //std::cout << "survivedDom  :" << survivedDom.size() << "\n";
      [[maybe_unused]] int nbRELAX = 0;
      for(const auto& n: survivedDom) {
         nbRELAX++;
         bool dualBetter = relaxed->apply(n, bnds);
         // std::cout << "Survivor:" << n->getBound() << " BWD:" << n->getBackwardBound() << " TTL:" << n->getTotalBound()
         //           << " dualBetter:" << dualBetter
         //           << " newGuyDominated:" << newGuyDominated
         //           << "\n";
         //double localDual = relaxed->local(n, LocalContext::BBCtx);
         //std::cout << "LOCAL(S):" << localDual << " SUM:" << n->getBound() + localDual << "\n"; 
         //std::cout << "reaching relaxed DD. Got: " << dualBetter << " B@SINK:" << relaxed->currentOpt() << "\n";         
         //std::cout << "reaching relaxed DD. Got: " << dualBetter <<"\n";         
         if(dualBetter) {
            if(!newGuyDominated) {
               auto nd = bbPool->cloneNode(n);
               //std::cout << "cloned and got: " << nd << std::endl;
               if (nd) {
                  assert(nd->getBound() == n->getBound());
                  pq.insertHeap(QNode {nd, nd->getBound()+nd->getBackwardBound() });
               }
            } else insDom++;
         }
         bbPool->release(bbn.node);
      }
      //std::cout << "nbRELAX:" << nbRELAX << "\t PQ = " << pq.size() << "\n";
   }


   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << std::setprecision (std::numeric_limits<double>::digits10 + 1) << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
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
