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

void filterLocal(Bounds& bnds, AbstractDD::Ptr dd, std::vector<ANode::Ptr> nodes, std::vector<ANode::Ptr>* survived)
{
   for(auto n: nodes) {
      double localDual = dd->local(n, LocalContext::BBCtx);
      if(!dd->isBetterEQ(bnds.getPrimal(), n->getBound() + localDual)) {
         survived->push_back(n);
      }
   }
}

template<typename Ord>
int filterDom(Bounds& bnds, AbstractDD::Ptr dd, std::vector<ANode::Ptr> nodes, Heap<QNode,Ord>* pq, std::vector<ANode::Ptr>* survived)
{
   bool newGuyDominated   = false;
   auto end = nodes.rend();
   auto begin = nodes.rbegin();
   for (auto i = begin; i != end; i++) {
      auto n = *i;
      auto sz = nodes.size();

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
         survived->push_back(n);
      }
   }

   int pruned = 0;
   for(auto n: nodes) {
      unsigned d = 0;
      auto pqSz = pq->size();
      auto allLocs = new Heap<QNode,Ord>::LocType*[pqSz];
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
   return pruned;
}

void BAndBRestrictedFirst::search(Bounds& bnds)
{
   // Setup
   static int nbRELAX = 0;
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

      //std::cout << "dequeued: ";
      // restricted->printNode(std::cout, bbn.node);
      //std::cout << std::endl;

      
      auto curDual = bbn.bound;
      bnds.setDual(bbn.node->getBound(),curDual);

      // if(relaxed->hasLocal()){
      //    auto compDual = bbn.node->getBound() + relaxed->local(bbn.node,LocalContext::DDInit);
      //    //cout << "DUAL KEY:" << curDual << " dualCOMP:" << compDual << "\n";
      //    if (!relaxed->isBetterEQ(compDual,curDual)) {
      //       //cout<< " dual comp improve!\n";
      //       curDual = compDual;
      //    }
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


      /*      cout << "discarded set: " << discardSet.size() << endl;
      struct {
         bool operator()(ANode::Ptr a,ANode::Ptr b) const {
            return a->getBound() < b->getBound();
         }
      } custom;
      std::sort(discardSet.begin(),discardSet.end(),custom);
      const int last = discardSet.size()-1;
      cout << "FIRST:" << discardSet[0]->getTotalBound()   << "\n";
      cout << "LAST :" << discardSet[last]->getTotalBound() << "\n";
      
      std::vector<ANode::Ptr> survivedLocal;
      std::vector<ANode::Ptr> survivedDom;
      
      bool newGuyDominated = false;
      if (relaxed->hasDominance()) {
         int tmpPruned = filterDom<decltype(hOrder)>(bnds, relaxed, discardSet, &pq, &survivedDom);
         insDom += discardSet.size() - survivedDom.size() - tmpPruned;
         pruned += tmpPruned;
      }
      */
      std::vector<ANode::Ptr> survivedLocal;
      if(relaxed->hasLocal()) {
         filterLocal(bnds, relaxed, discardSet, &survivedLocal);
      } else {
         survivedLocal = discardSet;
      }
      /*
      std::cout << "discardSet   :" << discardSet.size() << "\n";
      std::cout << "survivedDom  :" << survivedDom.size() << "\n";
      std::cout << "survivedLocal:" << survivedLocal.size() << "\n";
      */
      int nbRELAX = 0;
      for(auto n: survivedLocal) {
         nbRELAX++;
         //bool dualBetter = relaxed->apply(n, bnds);
         // std::cout << "reaching relaxed DD. Got: " << dualBetter << " B@SINK:" << relaxed->currentOpt() << "\n";         
         //std::cout << "reaching relaxed DD. Got: \n";         
         if(true) { // || dualBetter) {
            //if(!newGuyDominated) {
            auto nd = bbPool->cloneNode(n);
            //std::cout << "cloned and got: " << nd << std::endl;
            if (nd) {
               assert(nd->getBound() == n->getBound());
               pq.insertHeap(QNode {nd, nd->getBound()+nd->getBackwardBound() });
            }
            //} else insDom++;
         }
         bbPool->release(bbn.node);
      }
      //std::cout << "nbRELAX:" << nbRELAX << "\t PQ = " << pq.size() << "\n";
   }


   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
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
