#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>

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
   cout << "B&B searching..." << endl;
   bnds.attach(_theDD);
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
   unsigned nNode = 0;
   while(!pq.empty()) {
      auto bbn = pq.extractMax();
#ifndef _NDEBUG     
      cout << "BOUNDS NOW: " << bnds << endl;
      cout << "EXTRACTED:  " << bbn.node->getId() << " ::: ";
      _theDD->printNode(cout,bbn.node);
      cout << "\t(" << bbn.bound << ")" << " SZ:" << pq.size() << endl;
#endif
      if (!_theDD->isBetter(bbn.bound,bnds.getPrimal()))
         continue;
      nNode++;

      // relaxed->makeInitFrom(bbn.node);
      // relaxed->compute();
      // bool dualBetter = _theDD->isBetter(relaxed->currentOpt(),bnds.getPrimal());
      // if (dualBetter) {
      //    relaxed->update(bnds);
         
      //    restricted->makeInitFrom(bbn.node);
      //    restricted->compute();
      //    bool primalBetter = _theDD->isBetter(restricted->currentOpt(),bnds.getPrimal());
      //    if (primalBetter)
      //       restricted->update(bnds);

      bool dualBetter = relaxed->apply(bbn.node,bnds);
      if (dualBetter) {         
         bool primalBetter = restricted->apply(bbn.node,bnds);
         
         if (!restricted->isExact() && !relaxed->isExact()) {
            for(auto n : relaxed->computeCutSet()) {
               auto nd = _theDD->duplicate(n); // we got a duplicate of the node.
               if (n == relaxed->getRoot()) { // the cutset is the root. Only way out: increase width.
                  for(auto i=0u;i < sizeof(ddr)/sizeof(WidthBounded*);i++)
                     ddr[i]->setWidth(ddr[i]->getWidth() + 1);
               }
               // use the bound in n (the ones in nd are _reset_ when duplicate occurs????)
               assert(nd->getBound() == n->getBound());
               bool newGuyDominated = false;
               if (_theDD->hasDominance()) {
                  unsigned d = 0;
                  auto pqSz = pq.size();
                  auto allLocs = new Heap<QNode,decltype(hOrder)>::LocType*[pqSz];
                  for(unsigned k = 0;k < pqSz;k++) {
                     auto bbn = pq[k];
                     bool isObjDom   = _theDD->isBetterEQ(bbn->value().node->getBound(),nd->getBound());
                     bool isStateDom = _theDD->dominates(bbn->value().node,nd);
                     newGuyDominated = isObjDom && isStateDom;
                     if (newGuyDominated)
                        break;                     
                     bool objDom   = _theDD->isBetterEQ(nd->getBound(),bbn->value().node->getBound());
                     bool stateDom = _theDD->dominates(nd,bbn->value().node);
                     bool qnDominated = objDom && stateDom;
                     if (qnDominated)
                        allLocs[d++] = bbn;
                  }
                  if (d) {
                     //std::cout << "new BBNode Dominated " << d << " BB nodes" << std::endl;
                     for(auto i =0u; i < d;i++) 
                        pq.remove(allLocs[i]);                    
                  }
                  delete[]allLocs;
               }
               if (!newGuyDominated)
                  pq.insertHeap(QNode {nd, n->getBound()+n->getBackwardBound()});
            }
         }
      }
   }
   cout << "Done: " << bnds << "\t #nodes:" <<  nNode << "\n";
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
