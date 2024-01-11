#include "dd.hpp"
#include "util.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <unistd.h>
#include "heap.hpp"
#include "queue.hpp"

AbstractDD::~AbstractDD()
{
   std::cout << "AbstractDD::~AbstractDD(" << this << ")\n";
   for(ANode::Ptr p : _an)
      p->~ANode();
   _an.clear();
   _root = _trg = nullptr;
   _mem->clear();
}

void AbstractDD::addArc(Edge::Ptr e)
{
   e->_from->addArc(e);
   e->_to->addArc(e);
}

void printMe(ANode* n)
{
   n->print(std::cout);
   std::cout << std::endl;
}

void AbstractDD::compute()
{
   assert(_strat);
   _strat->compute();
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
   display();
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


void AbstractDD::saveGraph(std::ostream& os)
{
   Heap<DNode,DNode> h(_mem,1000);
   for(ANode::Ptr n : _an) 
      h.insert({n,n->nbParents()});   
   h.buildHeap();
   std::string colors[2] = {"green","red"};
   os << "digraph MDD {" << std::endl;
   os << " node [style=filled gradientangle=270];\n"; 
   while (h.size() > 0) {
      auto cur = h.extractMax();
      os << "\"" << *cur.node << "\" [color=\"" << colors[0] << "\"];\n";      
      for(auto ki = cur.node->beginKids(); ki != cur.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = h.find({k->_to,0});
         h.decrease(at);
         ANode::Ptr from = k->_from;
         ANode::Ptr to   = k->_to;
         os << "\"" << *from << "\" ->" << "\"" << *to << "\"";
         os << " [ label=\"" << k->_obj << "(" << k->_lbl << ")" << "\" ];\n";
      }
   }
   os << "}" << std::endl;
}

void AbstractDD::display()
{
   char buf[256] = "/tmp/dotfile-XXXXXXX";
   mkstemp(buf);
   std::ofstream out(buf);
   std::string base = buf;
   base = base + ".pdf";
   saveGraph(out);
   out.close();
   pid_t child = fork();
   if (child == 0) {
      int st = execlp("dot","dot","-Tpdf","-O",buf,0);
      printf("execl status: %d\n",st);
   }
   int st = 0;
   waitpid(child,&st,0);
   child = fork();
   if (child == 0) {
      execlp("open","open",base.c_str(),0);
      printf("execl2 status: %d\n",st);
   }
   unlink(buf);
}

void AbstractDD::computeBest()
{
   Heap<DNode,DNode> h(_mem,1000);
   for(ANode::Ptr n : _an) 
      h.insert({n,n->nbParents()});
   h.buildHeap();
   while (h.size() > 0) {
      auto n = h.extractMax();
      double cur = (n.node->nbParents() == 0) ? 0 : initialBest();
      for(auto pi = n.node->beginPar();pi != n.node->endPar();pi++) {
         Edge::Ptr e = *pi;
         std::cout << "EDGE:" << *e << std::endl;
         auto ep = e->_from->_bound + e->_obj;
         cur = better(cur,ep);
      }
      std::cout << "\tCOMPUTED:" << cur << " for ";
      n.node->print(std::cout);
      std::cout << "\n";
      n.node->setBound(cur);
      for(auto ki = n.node->beginKids(); ki != n.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = h.find({k->_to,0});
         assert(at!=-1);
         h.decrease(at);
      }
   }
   std::cout << "B@SINK:" << _trg->getBound() << std::endl;
}

// ----------------------------------------------------------------------
// Exact DD Strategy

void Exact::compute()
{
   auto root = _dd->init();
   auto sink = _dd->target();
   CQueue<ANode::Ptr> qn(32);
   qn.enQueue(root);
   while (!qn.empty()) {
      auto p = qn.deQueue();
      std::set<int> al {};
      for(auto k = p->beginKids(); k != p->endKids();k++) 
         al.insert((*k)->_lbl);
      for(auto l : _dd->_labels) {
         if (al.contains(l)) continue;
         auto child = _dd->transition(p,l); // we get back a new node, or an already existing one.
         if (child) {
            const bool newNode = child->nbParents()==0; // is this a newly created node?
            auto theCost = _dd->cost(p,l);
            Edge::Ptr e = new (_dd->_mem) Edge(p,child,l);
            e->_obj = theCost;
            _dd->addArc(e); // connect to new node
            if (_dd->neq(child,sink)) {
               if (newNode)
                  qn.enQueue(child);
            }
         }
      }
   }
   _dd->computeBest();
   _dd->print(std::cout);
}

void Restricted::truncate(std::vector<ANode::Ptr>& layer)
{
   auto from = layer.begin();
   std::advance(from,_mxw);
   for(auto start=from;start != layer.end();start++) {
      (*start)->disconnect();
      auto at = std::find(_dd->_an.begin(),_dd->_an.end(),*start);
      _dd->_an.erase(at);
   }
   layer.erase(from,layer.end());
}

std::vector<ANode::Ptr> Restricted::pullLayer(CQueue<ANode::Ptr>& qn)
{
   ANode::Ptr n = qn.deQueue();
   std::vector<ANode::Ptr> lk {};
   const unsigned cL = n->getLayer();
   lk.push_back(n);
   while(!qn.empty()) {
      if (qn.peek()->getLayer() != cL)
         break;
      n = qn.deQueue();
      lk.push_back(n);         
   }
   return lk;
}

void Restricted::compute()
{
   auto root = _dd->init();
   auto sink = _dd->target();
   CQueue<ANode::Ptr> qn(32);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) {
      std::vector<ANode::Ptr> lk = pullLayer(qn); // We have in lk the queue content for layer cL
      if (lk.size() > _mxw) 
         truncate(lk);
      for(auto p : lk) { // loop over layer lk. p is a "parent" node.
         std::set<int> remLabels = _dd->_labels; // deep copy
         for(auto k = p->beginKids(); k != p->endKids();k++) 
            remLabels.erase((*k)->_lbl);
         for(auto l : remLabels) {
            auto child = _dd->transition(p,l); // we get back a new node, or an already existing one.
            if (child) {
               const bool newNode = child->nbParents()==0; // is this a newly created node?
               auto theCost = _dd->cost(p,l);
               Edge::Ptr e = new (_dd->_mem) Edge(p,child,l);
               e->_obj = theCost;
               _dd->addArc(e); // connect to new node
               if (_dd->neq(child,sink)) {
                  if (newNode)
                     qn.enQueue(child);
               }
               child->setLayer(std::max(child->getLayer(),p->getLayer()+1));
            }            
         }
      }
   }
   _dd->computeBest();
   _dd->print(std::cout);   
}
