#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include "RuntimeMonitor.hpp"

struct QNode {
   ANode::Ptr node;
   double    bound;
   friend std::ostream& operator<<(std::ostream& os,const QNode& q) {
      return os << "QNode[(" << q.node->getId() << ','
                << q.node->getBound() << ','
                << q.node->getBackwardBound() << ")," << q.bound << "]";
   }
};

void BAndB::search(Bounds& bnds)
{
   Pool mem;
   using namespace std;
   std::streamsize ss = cout.precision();
   auto start = RuntimeMonitor::cputime();
   auto last = start;
   cout << "B&B searching..." << endl;
   bnds.attach(_theDD);
   double optTime = 0.0;
   bnds.onSolution([start,&optTime](const auto& lbls) {
      optTime = RuntimeMonitor::elapsedSince(start);
      std::cout << "TIME:" << optTime << "\n";
   });
   AbstractDD::Ptr relaxed = _theDD->duplicate();
   AbstractDD::Ptr restricted = _theDD->duplicate();
   WidthBounded* ddr[2];
   _theDD->setStrategy(new Exact);
   relaxed->setStrategy(ddr[0] = new Relaxed(_mxw));
   restricted->setStrategy(ddr[1] = new Restricted(_mxw));
   auto hOrder = [this](const QNode& a,const QNode& b) {
      return _theDD->isBetter(a.bound,b.bound);
   };
   Heap<QNode,decltype(hOrder)> pq(&mem,64000,hOrder);   
   pq.insertHeap(QNode { _theDD->init(), _theDD->initialWorst() } );
   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   bool primalBetter = false;
   cout << "B&B Nodes          " << setw(6) << "LB\t " << setw(6) << "UB\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   while(!pq.empty()) {
      auto bbn = pq.extractMax();
      auto now = RuntimeMonitor::cputime();
      auto fs = RuntimeMonitor::elapsedMilliseconds(start,now);
      auto fl = RuntimeMonitor::elapsedMilliseconds(last,now);
      if (_timeLimit && _timeLimit(fs))         
         break;
      if (primalBetter || fl > 5000) {
         double gap = 100 * (bnds.getPrimal() - bbn.bound) / bnds.getPrimal();      
         cout << "B&B(" << setw(5) << nNode << ")\t ";
         if (bbn.bound == _theDD->initialWorst())
            cout << setw(6) << "-"  << "\t " << setw(6) << bnds.getPrimal() << "\t ";
         else
            cout << setw(6) << bbn.bound << "\t " << setw(6) << bnds.getPrimal() << "\t ";
         if (gap > 100)
            cout << setw(6) << "-";
         else cout << setw(6) << setprecision(4) << gap << "%";
         cout << "\t time:" << setw(6) << setprecision(4) <<  fs / 1000.0 << "s";
         cout << "\n";
         last = RuntimeMonitor::cputime();
      }
      primalBetter = false;
#ifndef _NDEBUG     
      cout << "BOUNDS NOW: " << bnds << endl;
      cout << "EXTRACTED:  " << bbn.node->getId() << " ::: ";
      _theDD->printNode(cout,bbn.node);
      cout << "\t(" << bbn.bound << ")" << " SZ:" << pq.size() << endl;
#endif
      ttlNode++;
      if (!_theDD->isBetter(bbn.bound,bnds.getPrimal()))
         continue;
      nNode++;
      bool dualBetter = relaxed->apply(bbn.node,bnds);
      if (dualBetter) {
         primalBetter = restricted->apply(bbn.node,bnds);
         
         if (!restricted->isExact() && !relaxed->isExact()) {
            for(auto n : relaxed->computeCutSet()) {
               if (n == relaxed->getRoot()) { // the cutset is the root. Only way out: increase width.
                  for(auto i=0u;i < sizeof(ddr)/sizeof(WidthBounded*);i++)
                     ddr[i]->setWidth(ddr[i]->getWidth() + 1);
               }
               // use the bound in n (the ones in nd are _reset_ when duplicate occurs????)
               bool newGuyDominated = false;
               if (!_theDD->isBetter(n->getBound() + n->getBackwardBound(),bnds.getPrimal()))  {
                  //std::cout << "in Cutset. Not good enough: " << n->getBound() + n->getBackwardBound() << "\n";
                  continue;
               }
               if (_theDD->hasDominance()) {
                  unsigned d = 0;
                  auto pqSz = pq.size();
                  auto allLocs = new Heap<QNode,decltype(hOrder)>::LocType*[pqSz];
                  for(unsigned k = 0;k < pqSz;k++) {
                     auto bbn = pq[k];
                     bool isObjDom   = _theDD->isBetterEQ(bbn->value().node->getBound(),n->getBound());
                     newGuyDominated = isObjDom && _theDD->dominates(bbn->value().node,n);
                     if (newGuyDominated)
                        break;                     
                     bool objDom   = _theDD->isBetterEQ(n->getBound(),bbn->value().node->getBound());
                     bool qnDominated = objDom && _theDD->dominates(n,bbn->value().node);
                     if (qnDominated)
                        allLocs[d++] = bbn;
                  }
                  if (d) {
                     //std::cout << "new BBNode Dominated " << d << " BB nodes" << std::endl;
                     for(auto i =0u; i < d;i++) 
                        pq.remove(allLocs[i]);
                     pruned += d;
                  }
                  delete[]allLocs;
               }
               if (!newGuyDominated) {
                  auto nd = _theDD->duplicate(n); // we got a duplicate of the node.
                  assert(nd->getBound() == n->getBound());
                  pq.insertHeap(QNode {nd, n->getBound()+n->getBackwardBound()});
               }
               else insDom++;
            }
         }
      }
   }
   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
        << "\t P/D:" << pruned << "/" << insDom
        << "\t Time:" << optTime << "/" << spent << "ms"
        << "\t LIM?:" << (pq.size() > 0)
        << "\n";
}


/*
void BAndB::search()
{
   Pool mem;
   using namespace std;
   cout << "B&B searching..." << endl;
   AbstractDD::Ptr relaxed = _theDD->duplicate();
   AbstractDD::Ptr restricted = _theDD->duplicate();
   _theDD->setStrategy(new Exact);
   relaxed->setStrategy(new Relaxed(_mxw));
   restricted->setStrategy(new Restricted(_mxw));
   Heap<QNode,QNode> pq(&mem,32);
   pq.insertHeap(QNode { _theDD->init(), _theDD->initialWorst() } );
   Bounds bnds(_theDD);
   while(!pq.empty()) {
      auto bbn = pq.extractMax();
#ifndef _NDEBUG      
      cout << "BOUNDS NOW: " << bnds << endl;
      cout << "EXTRACTED:  " << *bbn.node << "\t(" << bbn.bound << ")" << endl;
#endif      
      restricted->reset();
      restricted->makeInitFrom(bbn.node);
      restricted->compute();
      bool primalBetter = _theDD->isBetter(restricted->currentOpt(),bnds.getPrimal());
      if (primalBetter)
         restricted->update(bnds);
      if (!restricted->isExact()) {
         relaxed->reset();
         relaxed->makeInitFrom(bbn.node);
         relaxed->compute();
         bool improving = _theDD->isBetter(relaxed->currentOpt(),bnds.getPrimal());
         if (improving) {
            relaxed->update(bnds);
            if (!relaxed->isExact())
               for(auto n : relaxed->computeCutSet()) {
                  auto nd = _theDD->duplicate(n); // we got a duplicate of the node.
                  pq.insertHeap(QNode {nd, relaxed->currentOpt()});
               }
         }         
      }
   }
   cout << "Done: " << bnds << "\n";
}
*/  
