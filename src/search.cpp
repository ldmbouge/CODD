#include "search.hpp"
#include "node.hpp"
#include "heap.hpp"
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <stdlib.h>
#include "RuntimeMonitor.hpp"
#include "pool.hpp"

void BAndB::search(Bounds& bnds)
{
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
   AbstractDD::Ptr relaxed = _theDD->duplicate();
   AbstractDD::Ptr restricted = _theDD->duplicate();
   WidthBounded* ddr[2];
   //_theDD->setStrategy(new Exact);
   relaxed->setStrategy(ddr[0] = new Relaxed(_mxw));
   restricted->setStrategy(ddr[1] = new Restricted(_mxw));
   auto hOrder = [relaxed](const QNode& a,const QNode& b) {
      return relaxed->isBetter(a.bound,b.bound);
   };
   Heap<QNode,decltype(hOrder)> pq(bbPool->get(),64000,hOrder);
   ANode::Ptr rootNode = bbPool->cloneNode(relaxed->init());
   if (relaxed->hasLocal()) {
      auto dualRootValue = relaxed->local(rootNode,LocalContext::BBCtx);
      cout << "dual@root:" << dualRootValue << "\n";
      rootNode->setBackwardBound(dualRootValue);
      pq.insertHeap(QNode { rootNode, dualRootValue } );   
   } else {
      pq.insertHeap(QNode { rootNode, relaxed->initialWorst() } );   
   }
   unsigned nNode = 0,ttlNode = 0,insDom=0,pruned=0;
   bool primalBetter = false;
   cout << "B&B Nodes          " << setw(6) << "Dual\t " << setw(6) << "Primal\t Gap(%)\n";
   cout << "----------------------------------------------\n";
   while(!pq.empty()) {

       // cout << "----------------------------------------------------------------------" << "\n";
       // cout << "B&B HEAP\n";
       // pq.printHeap(cout,[relaxed](std::ostream& os,const QNode& n) -> std::ostream& {
       //    relaxed->printNode(os,n.node);
       //    os << " KEY:" << n.bound;
       //    return os;
       // }) << "\n";
       // cout << "----------------------------------------------------------------------" << "\n";
      
      auto bbn = pq.extractMax();
      //std::cout << "bbn node dequeued: " << bbn << std::endl;
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
      auto compDual = bbn.node->getBound() + relaxed->local(bbn.node,LocalContext::DDInit);
      //cout << "DUAL KEY:" << curDual << " dualCOMP:" << compDual << "\n";
      if (!relaxed->isBetterEQ(compDual,curDual)) {
         //cout<< " dual comp improve!\n";
         curDual = compDual;
      }
      primalBetter = false;
#ifndef _NDEBUG
      cout << "BOUNDS NOW: " << bnds << endl; 
      cout << "EXTRACTED:  " << bbn.node->getId() << " ::: ";
      relaxed->printNode(cout,bbn.node);
      cout << "\t(" << curDual << ")" << " SZ:" << pq.size() << endl;
#endif
      ttlNode++;
      //   cout << "CURDUAL:" << curDual << "\t PRIMAL:" << bnds.getPrimal()
      //        << " isBetter:" << relaxed->isBetter(curDual,bnds.getPrimal()) << "\n";
      if (!relaxed->isBetter(curDual,bnds.getPrimal())) {
         bbPool->release(bbn.node);
         continue;
      }
      nNode++;
      //cout << "relaxed->apply: " << bbn.node->getBound() << "\n";
      bool dualBetter = relaxed->apply(bbn.node,bnds);
#ifndef _NDEBUG
      cout << "relaxed ran..." << "\n";
      relaxed->printNode(cout,bbn.node);      
#endif
      //cout << "dualBetter? " << dualBetter << "\n";
      if (dualBetter) { //if global opt is better than current primal
         primalBetter = restricted->apply(bbn.node,bnds);
         
         if (!restricted->isExact() && !relaxed->isExact()) {
            auto cutSet = relaxed->computeCutSet();
            //int k = 0;
            for(auto n : cutSet) {
               //std::cout << "CUTSET(" << k++ << ") ";
               //std::cout << "CUTSET ";               
               //relaxed->printNode(std::cout,n);
               
               if (n == relaxed->getRoot()) { // the cutset is the root. Only way out: increase width.
                  auto w = ddr[0]->getWidth() + 1;
                  ddr[0]->setWidth(w);
                  std::cout << "\t-->widening... " << w << " CUTSET SIZE:" << cutSet.size() <<  "\n";
               }
               // use the bound in n (the ones in nd are _reset_ when duplicate occurs????)
               bool newGuyDominated = false;
               if (!relaxed->isBetter(n->getBound() + n->getBackwardBound(),bnds.getPrimal()))  {
                  // std::cout << "\tDISCARD PRIMAL=" << bnds.getPrimal();
                  // relaxed->printNode(std::cout,n);
                  // std::cout << "\n";
                  continue; // the loop over the cutset! Not the main loop
               }
               if (relaxed->hasDominance()) {
                  unsigned d = 0;
                  auto pqSz = pq.size();
                  auto allLocs = new Heap<QNode,decltype(hOrder)>::LocType*[pqSz];
                  for(unsigned k = 0;k < pqSz;k++) {
                     auto bbn = pq[k];
                     bool isObjDom   = relaxed->isBetterEQ(bbn->value().node->getBound(),n->getBound());
                     newGuyDominated = isObjDom && relaxed->dominates(bbn->value().node,n);
                     if (newGuyDominated)
                        break;                     
                     bool objDom   = relaxed->isBetterEQ(n->getBound(),bbn->value().node->getBound());
                     bool qnDominated = objDom && relaxed->dominates(n,bbn->value().node);
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
               assert(n->isExact());
               if (!newGuyDominated) {
                  auto nd = bbPool->cloneNode(n);
                  
                  //std::cout << "CLONED and got:"  << nd << "\n";
                  
                  if (nd) { // the node creation could return *NOTHING* if it was already created
                     assert(nd->getBound() == n->getBound());
                     double bwd;
                     if (relaxed->hasLocal()) {
                        auto newBnd = relaxed->local(nd,LocalContext::BBCtx);
                        // if (newBnd > n->getBackwardBound())                           
                        //std::cout<<"New:"<< newBnd << " backward:" << n->getBackwardBound() << "\n";
                        //if (newBnd <= n->getBackwardBound())
                        if (!relaxed->isBetter(newBnd,n->getBackwardBound()))
                           bwd = newBnd;
                        else bwd = n->getBackwardBound();
                        //bwd = n->getBackwardBound();
                     } else bwd = n->getBackwardBound();
                     
                     const auto insKey = n->getBound() + bwd;
                     const auto improve = relaxed->isBetter(insKey,bnds.getPrimal());

                     // std::cout<< "CLONE VALUE:" << insKey << " bwd:" << bwd << " PRIMAL:" << bnds.getPrimal()
                     //          << " IMPROVED:" << (improve ? "T" : "F") << "\n";

                     if (improve) 
                        pq.insertHeap(QNode {nd, insKey }); //std::min(insKey,curDual)});
                  } else nbSeen++;
               }
               else insDom++;
            }
         }
      } //else 
      //std::cout << "DB:F " <<  "Primal:" << bnds.getPrimal() << " Dual:" << relaxed->currentOpt()  << "\n";      
      bbPool->release(bbn.node);
   }
   cout << setprecision(ss);
   auto spent = RuntimeMonitor::elapsedSince(start);
   cout << "Done(" << _mxw << "):" << bnds.getPrimal() << "\t #nodes:" <<  nNode << "/" << ttlNode
        << "\t P/D:" << pruned << "/" << insDom
        << "\t Time:" << optTime/1000 << "/" << spent/1000 << "s"
        << "\t LIM?:" << (pq.size() > 0)
        << "\t Seen:" << nbSeen
        << "\n";
        //<< "sol: " << bnds << endl;

   //cout << ddr[0]->getWidth() << " " << nNode << "/" << ttlNode << " " << spent/1000 << " " << bnds.getPrimal() << endl;
}
