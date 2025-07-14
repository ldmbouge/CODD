#include "searchRestrictedFirstThreaded.hpp"
#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include <thread>
#include <queue>
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

struct TQNode {
   ANode::Ptr node;
   double    bound;
   friend std::ostream& operator<<(std::ostream& os,const TQNode& q) {
      return os << "TQNode[(" << q.node->getId() << ','
                << q.node->getBound() << ','
                << q.node->getBackwardBound() << ")," << q.bound << "]";
   }
};

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
         newGuyDominated = isObjDom && dd->dominates(other,n);
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
         newGuyDominated = isObjDom && dd->dominates(other->value().node,n);
         if (newGuyDominated) {
            goto prune;             
         }        
         bool objDom   = dd->isBetterEQ(n->getBound(),other->value().node->getBound());
         bool qnDominated = objDom && dd->dominates(n,other->value().node);
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

struct MyNode {
   int value;
   bool valid;
   friend std::ostream& operator<<(std::ostream& os,const MyNode& q) {
      return os << "Node[" << q.value << "(" << q.valid << ")]";
   }
};
void BAndBRestrictedFirstThreaded::search(Bounds& bnds)
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
   restricted->setStrategy(ddr[0] = new Restricted(_mxw));
   //restricted->killDominance(); // doesn't help either

   AbstractDD::Ptr relaxed = _theDD->duplicate();
   //relaxed->killDominance();
   relaxed->setStrategy(ddr[1] = new Relaxed(_mxw)); //512= >100s 1024=10s 2048=2s


   auto hOrder = [restricted](const TQNode& a,const TQNode& b) {
      return restricted->isBetter(a.bound,b.bound);
   };
   ThreadSafeHeap<TQNode,decltype(hOrder)> pq(bbPool->get(),64000,hOrder); // the main B&B queue
   ThreadSafeHeap<TQNode,decltype(hOrder)> unvetted(bbPool->get(),64000,hOrder); // the queue where new nodes are sent to be vetted or culled before added to the main queue

   ANode::Ptr rootNode = bbPool->cloneNode(restricted->init());

   if (restricted->hasLocal()) {
      auto dualRootValue = restricted->local(rootNode,LocalContext::BBCtx);
      cout << "dual@root:" << dualRootValue << "\n";
      rootNode->setBackwardBound(dualRootValue);
      pq.insertHeap(TQNode { rootNode, dualRootValue } );   
   } else {
      pq.insertHeap(TQNode { rootNode, restricted->initialWorst() } );
   }
   const bool hasLocal = relaxed->hasLocal();
   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   bool primalBetter = false;

   int nDualCulled = 0;
   std::thread dualCuller([&unvetted, &pq, &relaxed, &bnds, &nDualCulled]() {
      unvetted.vetHeap([&relaxed, &bnds, &nDualCulled](TQNode& candidate){ 
         bool dualBetter = relaxed->apply(candidate.node,bnds); 
         if(dualBetter) nDualCulled++;
         return dualBetter;
      }, pq);
   });

   // Main Loop
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   const auto& pred = [&](){ 
      if(pq.empty() && unvetted.empty()) { 
         pq.setDone();
         unvetted.setDone();
         return false;
      } 
      return pq.empty();
   };
   pq.onArrival(pred, [&](){
      //std::cout << "waiting room: " << unvetted.size() << "   vetted: " << pq.size() << "\n";
      auto bbnOpt = pq.extractMax();
      TQNode bbn;
      if(bbnOpt.has_value()) {
         bbn = bbnOpt.value();
      } else if(pq.empty() && unvetted.empty()) { 
         pq.setDone();
         unvetted.setDone();
         return;
      } else {
         return; // if there is no valid node, but there are stolen nodes, we have to wait for stolen nodes to be returned
      }

      auto curDual = bbn.bound;
      bnds.setDual(bbn.node->getBound(),curDual);
      auto now = RuntimeMonitor::cputime();
      auto fs = RuntimeMonitor::elapsedMilliseconds(start,now);
      auto fl = RuntimeMonitor::elapsedMilliseconds(last,now);
      if (_timeLimit && _timeLimit(fs)) {        
         pq.setDone();
         unvetted.setDone();
         return;      
      }
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

      auto discardSet = restricted->theDiscardedSet();
      //std::cout << "comp discard\n";
      // std::vector<ANode::Ptr> survivedLocal = hasLocal ? filterLocal(bnds, relaxed, discardSet) : discardSet;
      // auto [tmpPruned,newGuyDominated,survivedDom] = filterDom(bnds, relaxed, survivedLocal, &pq);
      // insDom += survivedLocal.size() - survivedDom.size();
      // pruned += tmpPruned;
      //std::cout << "inserting: " << discardSet << "\n";
      int i = 32;
      unvetted.insertAllIf(discardSet, [&bbPool, &bbn, &pq, &i](ANode::Ptr n){
         // std::cout << "try insert\n";
         auto nd = bbPool->cloneNode(n);
         std::optional<TQNode> ret = std::nullopt;
         if (nd) {
            assert(nd->getBound() == n->getBound());
            ret = TQNode {nd, nd->getBound()+nd->getBackwardBound() };
            // std::cout << "insert\n";
            if((i--) > 0) {
               pq.insertHeap(ret.value());
               ret = std::nullopt;
            }
         }
         bbPool->release(bbn.node);
         return ret;
      });
      // std::cout << "done inserting\n";
   });

   dualCuller.join();

   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << std::setprecision (std::numeric_limits<double>::digits10 + 1)
        << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
        << "\t P/D:" << pruned << "/" << insDom
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\t LIM?:" << (pq.size() > 0)
        << "\t Seen:" << nbSeen
        << "\t Dual Culled:" << nDualCulled
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
