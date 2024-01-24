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
   _theDD->setStrategy(new Exact);
   relaxed->setStrategy(new Relaxed(_mxw));
   Heap<QNode,QNode> pq(&mem,32);
   pq.insertHeap(QNode { _theDD->init(), _theDD->initialBest() } );
   Bounds bnds(_theDD);
   while(!pq.empty()) {
      auto bbn = pq.extractMax();     
      AbstractDD::Ptr restricted = _theDD->duplicate();
      restricted->setStrategy(new Restricted(_mxw));
      restricted->makeInitFrom(bbn.node);
      restricted->compute();
      restricted->update(bnds);
      cout << "AFTER restricted:" << bnds << endl;
      cout << (restricted->isExact() ? "EXACT" : "INEXACT") << endl;
      if (!restricted->isExact()) {
         relaxed->reset();
         relaxed->makeInitFrom(bbn.node);
         relaxed->compute();
         bool improving = _theDD->better(relaxed->currentOpt(),bnds.getPrimal());
         cout << "Improving? " << (improving ? "YES" : "NO") << endl;
         relaxed->update(bnds);
         cout << "AFTER Relax:" << bnds << endl;
         cout << "Dual " << (relaxed->isExact() ? "EXACT" : "INEXACT") << endl;
         if (improving) {
            auto cs = relaxed->computeCutSet();
            cout << "cutSet:" << endl;
            for(auto n : cs) {
               cout << "\tCSN:" << *n << endl;
               auto nd = _theDD->duplicate(n); // we got a duplicate of the node.
               pq.insertHeap(QNode {nd, nd->getBound()});
            }
         }         
      }
   }
   
}
