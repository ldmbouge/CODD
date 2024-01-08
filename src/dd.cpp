#include "dd.hpp"
#include "util.h"
#include <iostream>
#include "heap.hpp"

void AbstractDD::addArc(Edge::Ptr e)
{
   e->_from->addArc(e);
   e->_to->addArc(e);
}


void AbstractDD::doIt()
{
   auto root = init();
   auto sink = target();
   auto cur = root;
   _an.push_back(root);
   _an.push_back(sink);
   int l = 9;
   std::cout << "DOIT Loop\n"; 
   while (neq(cur,sink)) {
      auto nxt = transition(cur,l--);
      _an.push_back(nxt);
      Edge::Ptr e = new (_mem) Edge(cur,nxt);
      addArc(e);
      nxt->print(std::cout);std::cout << std::endl;
      cur = nxt;
   }
   _root = root;
   _trg  = sink;
   print(std::cout);
}

struct DNode {
   ANode::Ptr node;
   unsigned   degree;
   bool operator()(const DNode& a,const DNode& b) {
      return a.degree < b.degree;
   }
   DNode& operator--(int) {
      assert(degree > 0);
      degree--;
      return *this;
   }
   bool operator==(const DNode& n) const noexcept { return node ==n.node;}
   friend std::ostream& operator<<(std::ostream& os,const DNode& n) {
      os << "<K=" << n.degree << ",N=";
      n.node->print(os);
      return os << ">";      
   }  
};


void AbstractDD::print(std::ostream& os)
{
   std::cout << "PRINTING: --------------------------------------------------\n";
   Heap<DNode,DNode> h(_mem,1000);
   for(ANode::Ptr n : _an) {
      h.insert({n,n->nbParents()});
      std::cout << *n << " #PAR:" << n->nbParents() << "\n";
   }
   h.buildHeap();
   while (h.size() > 0) {
      auto cur = h.extractMax();
      for(auto ki = cur.node->beginKids(); ki != cur.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         os << "EDGE:" << *k << std::endl;
         auto at = h.find({k->_to,0});
         assert(at!=-1);
         h.decrease(at);
      }
   }
}

