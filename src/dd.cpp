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


Bounds::Bounds(std::shared_ptr<AbstractDD> dd)
{
   _primal = dd->initialBest();
   _dual   = dd->initialWorst();      
}

AbstractDD::AbstractDD(const std::set<int>& labels)
   : _mem(new Pool),_labels(labels),_exact(true)
{}

AbstractDD::~AbstractDD()
{
   std::cout << "AbstractDD::~AbstractDD(" << this << ")\n";
   delete _mem;
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

void display(AbstractDD* dd)
{
   dd->display("Debug");
}

void AbstractDD::setStrategy(Strategy* s)
{
   _strat = s;
   _strat->_dd = this;
}

void AbstractDD::compute()
{
   assert(_strat);
   _strat->compute();
}

std::vector<int> AbstractDD::incumbent()
{
   std::vector<int> inc {};
   for(auto it= _trg->beginOptLabels();it != _trg->endOptLabels();it++)
      inc.push_back(*it);
   return inc;
}

std::vector<ANode::Ptr> AbstractDD::computeCutSet()
{
   return _strat->computeCutSet();
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


void AbstractDD::print(std::ostream& os,std::string gLabel)
{
   display(gLabel);
   std::cout << "PRINTING: --------------------------------------------------\n";
   Heap<DNode,DNode> h(_mem,1000);
   for(ANode::Ptr n : _an) {
      h.insert({n,n->nbParents()});
      //std::cout << *n << " #PAR:" << n->nbParents() << "\n";
   }
   h.buildHeap();
   while (h.size() > 0) {
      auto cur = h.extractMax();
      for(auto ki = cur.node->beginKids(); ki != cur.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         os << "EDGE:" << *k << std::endl;
         auto at = h.find({k->_to,0});
         assert(at!= nullptr);
         h.decrease(at);
      }
   }
}


void AbstractDD::saveGraph(std::ostream& os,std::string gLabel)
{
   Heap<DNode,DNode> h(_mem,1000);
   for(ANode::Ptr n : _an) 
      h.insert({n,n->nbParents()});   
   h.buildHeap();
   std::string colors[2] = {"red","green"};
   os << "digraph MDD {" << std::endl;
   os << "label=\"" << gLabel << "\";\n"; 
   os << " node [style=filled gradientangle=270];\n"; 
   while (h.size() > 0) {
      auto cur = h.extractMax();
      os << "\"" << *cur.node << "\" [fillcolor=\"" << colors[cur.node->isExact()] << "\"";
      if (eq(cur.node,_trg))
         os << ",shape=box,color=black";
      os << "];\n";      
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

void AbstractDD::display(std::string gLabel)
{
   char buf[256] = "/tmp/dotfile-XXXXXXX";
   mkstemp(buf);
   std::ofstream out(buf);
   std::string base = buf;
   base = base + ".pdf";
   saveGraph(out,gLabel);
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
      //std::cout << "\tCOMPUTE START: " << *n.node << "\n";
      double cur = (n.node->nbParents() == 0) ? n.node->getBound() : initialBest();
      for(auto pi = n.node->beginPar();pi != n.node->endPar();pi++) {
         Edge::Ptr e = *pi;
         //std::cout << "\tEDGE:" << *e << std::endl;
         auto ep = e->_from->_bound + e->_obj;
         cur = better(cur,ep);
         if (cur==ep) {
            n.node->_optLabels = e->_from->_optLabels;
            n.node->_optLabels.push_back(e->_lbl);
         }
      }
      //std::cout << "\tCOMPUTED:" << cur << " for " << *n.node << "\n";
      n.node->setBound(cur);
      for(auto ki = n.node->beginKids(); ki != n.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = h.find({k->_to,0});
         assert(at!=nullptr);
         h.decrease(at);
      }
   }
   std::cout << "B@SINK:" << _trg->getBound() << "\tLBL:" << _trg->_optLabels << std::endl;
}

// ----------------------------------------------------------------------
// Exact DD Strategy

std::set<int> Strategy::remainingLabels(ANode::Ptr p)
{
   std::set<int> remLabels = _dd->_labels; // deep copy
   for(auto k = p->beginKids(); k != p->endKids();k++) 
      remLabels.erase((*k)->_lbl);
   return remLabels;
}


void Exact::compute()
{
   auto root = _dd->init();
   auto sink = _dd->target();
   _dd->_exact = true;
   CQueue<ANode::Ptr> qn(32);
   qn.enQueue(root);
   while (!qn.empty()) {
      auto p = qn.deQueue();
      std::set<int> remLabels = remainingLabels(p);
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
         }
      }
   }
   _dd->computeBest();
   //_dd->print(std::cout,"Exact");
}

std::list<ANode::Ptr> WidthBounded::pullLayer(CQueue<ANode::Ptr>& qn)
{
   ANode::Ptr n = qn.deQueue();
   std::list<ANode::Ptr> lk {};
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

// ----------------------------------------------------------------------
// Restricted DD Strategy

void Restricted::truncate(std::list<ANode::Ptr>& layer)
{
   _dd->_exact = false;
   auto from = layer.begin();
   std::advance(from,_mxw);
   for(auto start=from;start != layer.end();start++) {
      (*start)->disconnect();
      auto at = std::find(_dd->_an.begin(),_dd->_an.end(),*start);
      _dd->_an.erase(at);
   }
   layer.erase(from,layer.end());
}

void Restricted::compute()
{
   _dd->_exact = true;
   auto root = _dd->init();
   auto sink = _dd->target();
   CQueue<ANode::Ptr> qn(32);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) {
      std::list<ANode::Ptr> lk = pullLayer(qn); // We have in lk the queue content for layer cL
      if (lk.size() > _mxw) 
         truncate(lk);
      for(auto p : lk) { // loop over layer lk. p is a "parent" node.
         std::set<int> remLabels = remainingLabels(p);
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
   //_dd->print(std::cout,"Restricted");   
}

// ----------------------------------------------------------------------
// Relaxed DD Strategy

void Relaxed::transferArcs(ANode::Ptr donor,ANode::Ptr receiver)
{
   for(auto ei = donor->beginPar(); ei != donor->endPar();ei++) {
      auto ep = *ei;
      Edge::Ptr ne = new (_dd->_mem) Edge(ep->_from,receiver,ep->_lbl);
      ne->_obj = ep->_obj;
      _dd->addArc(ne);      
   }
   for(auto ei = donor->beginKids(); ei != donor->endKids();ei++) {
      auto ep = *ei;
      Edge::Ptr ne = new (_dd->_mem) Edge(receiver,ep->_to,ep->_lbl);
      ne->_obj = ep->_obj;
      _dd->addArc(ne);      
   }
   donor->disconnect();
}

std::list<ANode::Ptr> Relaxed::mergeLayer(std::list<ANode::Ptr>& layer)
{
   const auto mergesNeeded = layer.size() - _mxw;
   auto mergesDone = 0u;
   std::list<ANode::Ptr> delayed;
   std::list<ANode::Ptr> final;
   while (mergesDone < mergesNeeded && layer.size() > 0) {
      ANode::Ptr n1 = layer.front();
      layer.pop_front();
      std::list<ANode::Ptr> toMerge = {n1};
      ANode::Ptr mNode = nullptr;
      bool newNode = true;
      for(auto i = layer.begin();i != layer.end();i++) {
         auto n2 = *i;
         assert(n1->getLayer() == n2->getLayer());
         mNode = _dd->merge(n1,n2);
         if (mNode) {
            toMerge.push_back(n2);
            layer.erase(i);
            break;
         }
      }
      if (toMerge.size() == 2) {
         mergesDone++;
         newNode = mNode->nbParents()==0; // is this a newly created node?
         const bool sameLayer = mNode->getLayer() == n1->getLayer();
         mNode->setLayer(std::max(mNode->getLayer(),n1->getLayer()));
         mNode->setExact(false);
         _dd->_exact = false;
         for(auto d : toMerge) {
            if (d == mNode) continue; // skip in case the node itself is the merged one
            transferArcs(d,mNode);
            auto at = std::find(_dd->_an.begin(),_dd->_an.end(),d);
            _dd->_an.erase(at);
         }
         if (sameLayer) //   newNode && 
            layer.push_back(mNode);
         else delayed.push_back(mNode);
      } else final.push_back(n1);
   }
   layer.insert(layer.end(),final.begin(),final.end());
   return delayed;
}

void Relaxed::compute()
{
   _dd->_exact = true;
   auto root = _dd->init();
   auto sink = _dd->target();
   CQueue<ANode::Ptr> qn(32);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) {
      std::list<ANode::Ptr> lk = pullLayer(qn); // We have in lk the queue content for layer cL
      std::list<ANode::Ptr> delay;
      if (lk.size() > _mxw) 
         delay = mergeLayer(lk);
      for(auto p : delay)
         qn.enQueue(p);
      for(auto p : lk) { // loop over layer lk. p is a "parent" node.
         std::set<int> remLabels = remainingLabels(p);
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
   //_dd->print(std::cout,"Relaxed");
}

std::vector<ANode::Ptr> Relaxed::computeCutSet()
{
   std::vector<ANode::Ptr> cs = {};
   auto mark = _dd->_mem->mark();
   CQueue<ANode::Ptr> qn(32);
   qn.enQueue(_dd->_root);
   while (qn.size() > 0) {
      auto cur = qn.deQueue();
      if (cur->isExact()) {
         bool akExact = true;
         for(auto ki = cur->beginKids(); akExact && ki != cur->endKids();ki++) 
            akExact = (*ki)->_to->isExact();
         if (akExact) {
            for(auto ki = cur->beginKids(); akExact && ki != cur->endKids();ki++) {
               qn.enQueue((*ki)->_to);
            }
         } else cs.push_back(cur);
      }
   }
   _dd->_mem->clear(mark);
   return cs;
}
