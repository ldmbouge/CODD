#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>

struct QNode {
   ANode::Ptr node;
   double    bound;
   bool operator()(const QNode& a,const QNode& b) {
      return a.bound < b.bound;
   }
};

void BAndB::search()
{
   Pool mem;
   using namespace std;
   cout << "B&B searching..." << endl;
   AbstractDD::Ptr relaxed = _theDD->duplicate();
   AbstractDD::Ptr restricted = _theDD->duplicate();
   _theDD->setStrategy(new Exact);
   _theDD->compute();
   //_theDD->display("Exact (test)");
   cout << "EXACT INCUMBENT:" << _theDD->incumbent() << "\n";
   relaxed->setStrategy(new Relaxed(_mxw));
   restricted->setStrategy(new Restricted(_mxw));
   Heap<QNode,QNode> pq(&mem,32);
   pq.insertHeap(QNode { _theDD->init(), _theDD->initialWorst() } );
   Bounds bnds(_theDD);
   while(!pq.empty()) {
      auto bbn = pq.extractMax();
      cout << "BOUNDS NOW: " << bnds << endl;
      cout << "EXTRACTED:  " << *bbn.node << "\t(" << bbn.bound << ")" << endl;
      restricted->reset();
      restricted->makeInitFrom(bbn.node);
      restricted->compute();
      bool primalBetter = _theDD->isBetter(restricted->currentOpt(),bnds.getPrimal());
      if (primalBetter)
         restricted->update(bnds);
      //cout << "BNDs AFTER restricted:" << bnds << endl;
      //cout << "Restricted:" << (restricted->isExact() ? "EXACT" : "INEXACT") << endl;
      if (!restricted->isExact()) {
         relaxed->reset();
         relaxed->makeInitFrom(bbn.node);
         relaxed->compute();
         //relaxed->display("Relaxed");
         //cout << "before improve test: P=" << bnds.getPrimal() << " CR=" << relaxed->currentOpt() << endl;
         bool improving = _theDD->isBetter(relaxed->currentOpt(),bnds.getPrimal());
         //cout << "AFTER Relax:" << (relaxed->isExact() ? "EXACT" : "INEXACT") << endl;
         if (improving) {
            //cout << "relax Improving? " << (improving ? "YES" : "NO") << endl;
            relaxed->update(bnds);
            //cout << "AFTER Relax:" << bnds << endl;
            if (!relaxed->isExact())
               for(auto n : relaxed->computeCutSet()) {
                  auto nd = _theDD->duplicate(n); // we got a duplicate of the node.
                  //pq.insertHeap(QNode {nd, nd->getBound()});
                  pq.insertHeap(QNode {nd, relaxed->currentOpt()});
               }
         }         
      }
   }
   cout << "Done: " << bnds << "\n";
}
