#include "dd.hpp"
#include "node.hpp"
#include "util.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <limits>
#include <cstdio>
#include <unistd.h>
#include "heap.hpp"
#include "queue.hpp"
#include <sys/types.h>
#include <sys/wait.h>
#include <stdlib.h>


Bounds::Bounds(std::shared_ptr<AbstractDD> dd)
{
   _primal = dd->initialBest();
}

void Bounds::attach(std::shared_ptr<AbstractDD> dd)
{
   _primal = dd->initialBest();
}

AbstractDD::AbstractDD(const GNSet& labels)
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

void display(AbstractDD* dd)
{
   dd->display();
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
   computeBest(_strat->getName());
   computeBestBackward(_strat->getName());
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
   display();
   std::cout << "PRINTING: --------------------------------------------------\n";
   Heap<DNode,DNode> h(_mem,1000);
   for(auto n : _an) {
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
   for(auto n : _an) 
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

void AbstractDD::display()
{
   const std::string gLabel = _strat->getName();
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

void AbstractDD::computeBest(const std::string m)
{
   //std::cout << "ANSZ:" << _an.size() << "\n";
   Heap<DNode,DNode> h(_mem,1000);
   unsigned mxId = 0;
   for(auto n : _an) 
      mxId = std::max(n->getId(),mxId);
   auto nl = new (_mem) Heap<DNode,DNode>::Location*[mxId+1];
   memset(nl,0,sizeof(Heap<DNode,DNode>::Location*)*(mxId+1));
   for(auto n : _an) {
      nl[n->getId()] = h.insert({n,n->nbParents()});
      if (n != _root)
         n->setBound(initialBest());
   }
   h.buildHeap();
   while (h.size() > 0) {
      auto n = h.extractMax();
      //std::cout << "\tCOMPUTE START: " << *n.node << "\n";
      double cur = (n.node->nbParents() == 0) ? n.node->getBound() : initialBest();
      for(auto pi = n.node->beginPar();pi != n.node->endPar();pi++) {
         Edge::Ptr e = *pi;
         auto ep = e->_from->_bound + e->_obj;
         //std::cout << "\tEDGE:" << *e << " EP=" << ep << std::endl;
         if (isBetter(ep,cur)) {
            cur = ep;
            n.node->_optLabels = e->_from->_optLabels;
            n.node->_optLabels.push_back(e->_lbl);
         }
      }
      //std::cout << "\tCOMPUTED:" << cur << " for " << *n.node << "\n";
      n.node->setBound(cur);
      for(auto ki = n.node->beginKids(); ki != n.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = nl[k->_to->getId()];
         assert(at!=nullptr);
         h.decrease(at);
      }
   }
#ifndef _NDEBUG     
   std::cout << '\t' << m << " B@SINK:" << _trg->getBound() << "\tLBL:" << _trg->_optLabels << std::endl;
#endif   
}

void AbstractDD::computeBestBackward(const std::string m)
{
   Heap<DNode,DNode> h(_mem,1000);
   unsigned mxId = 0;
   for(auto n : _an) 
      mxId = std::max(n->getId(),mxId);
   auto nl = new (_mem) Heap<DNode,DNode>::Location*[mxId+1];
   memset(nl,0,sizeof(Heap<DNode,DNode>::Location*)*(mxId+1));
   for(auto n : _an) {
      nl[n->getId()] = h.insert({n,n->nbChildren()});
      if (n != _trg)
         n->setBackwardBound(initialBest());
   }
   h.buildHeap();
   while (h.size() > 0) {
      auto n = h.extractMax();
      //std::cout << "\tCOMPUTE START: " << *n.node << "\n";
      double cur = (n.node->nbChildren() == 0) ? n.node->getBackwardBound() : initialBest();
      for(auto ci = n.node->beginKids();ci != n.node->endKids();ci++) {
         Edge::Ptr e = *ci;
         auto ep = e->_to->_bbound + e->_obj;
         //std::cout << "\tEDGE:" << *e << " EP=" << ep << std::endl;
         if (isBetter(ep,cur)) {
            cur = ep;
         }
      }
      //std::cout << "\tCOMPUTED:" << cur << " for " << *n.node << "\n";
      n.node->setBackwardBound(cur);
      for(auto pi = n.node->beginPar(); pi != n.node->endPar();pi++) {
         Edge::Ptr k = *pi;
         auto at = nl[k->_from->getId()];
         assert(at!=nullptr);
         h.decrease(at);
      }
   }
#ifndef _NDEBUG     
   std::cout << '\t' << m << " B@ROOT:" << _trg->getBackwardBound() << std::endl;
#endif   
}


// ----------------------------------------------------------------------
// Exact DD Strategy

GNSet Strategy::remainingLabels(ANode::Ptr p)
{
   GNSet remLabels = _dd->_labels; // deep copy
   for(auto k = p->beginKids(); k != p->endKids();k++) 
      remLabels.remove((*k)->_lbl);
   return remLabels;
}


void Exact::compute()
{
   auto root = _dd->init();
   _dd->target();
   _dd->_exact = true;
   CQueue<ANode::Ptr> qn(32);
   qn.enQueue(root);
   while (!qn.empty()) {
      auto p = qn.deQueue();
      auto remLabels = remainingLabels(p);
      for(auto l : remLabels) {     
         auto child = _dd->transition(p,l); // we get back a new node, or an already existing one.
         if (child) {
            const bool newNode = child->nbParents()==0; // is this a newly created node?
            auto theCost = _dd->cost(p,l);
            Edge::Ptr e = new (_dd->_mem) Edge(p,child,l);
            e->_obj = theCost;
            _dd->addArc(e); // connect to new node
            if (!_dd->eqSink(child)) {
               if (newNode)
                  qn.enQueue(child);
            }
         }
      }
   }
}

NDArray& WidthBounded::pullLayer(CQueue<ANode::Ptr>& qn)
{
   ANode::Ptr n = qn.deQueue();
   _nda.clear();
   const unsigned cL = n->getLayer();
   _nda.push_back(n);
   while(!qn.empty()) {
      if (qn.peek()->getLayer() != cL)
         break;
      n = qn.deQueue();
      _nda.push_back(n);         
   }
   return _nda;
}

// ----------------------------------------------------------------------
// Restricted DD Strategy

void Restricted::truncate(NDArray& layer)
{
   /*   layer.sort([](const ANode::Ptr& a,const ANode::Ptr& b) {
       return a->getBound() >= b->getBound();
   });
   */
   _dd->_exact = false;
   auto from = layer.at(_mxw);
   for(auto start=from;start != layer.end();start++) {
      (*start)->disconnect();
      _dd->_an.remove(*start);
   }
   layer.eraseSuffix(from);
}

void Restricted::compute()
{
   _dd->_exact = true;
   auto root = _dd->init();
   _dd->target();
   CQueue<ANode::Ptr> qn(32);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) { 
      auto& lk = pullLayer(qn); // We have in lk the queue content for layer cL
      if (lk.size() > _mxw) 
         truncate(lk);
      for(auto p : lk) { // loop over layer lk. p is a "parent" node.
         auto remLabels = remainingLabels(p);
         for(auto l : remLabels) {
            auto child = _dd->transition(p,l); // we get back a new node, or an already existing one.
            if (child) {
               const bool newNode = child->nbParents()==0; // is this a newly created node?
               auto theCost = _dd->cost(p,l);
               Edge::Ptr e = new (_dd->_mem) Edge(p,child,l);
               e->_obj = theCost;
               _dd->addArc(e); // connect to new node
               if (!_dd->eqSink(child)) {
                  if (newNode)
                     qn.enQueue(child);
               }
               child->setLayer(std::max(child->getLayer(),p->getLayer()+1));
            }            
         }
      }
   }
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

std::list<ANode::Ptr> Relaxed::mergeLayer(NDArray& layer)
{
   layer.sort([](const ANode::Ptr& a,const ANode::Ptr& b) {
      return a->getBound() >= b->getBound();
      //return a->getId() >= b->getId();      
   });
   
   const auto mergesNeeded = layer.size() - _mxw;
   auto mergesDone = 0u;
   std::list<ANode::Ptr> delayed;
   NDArray final;

   while (mergesDone < mergesNeeded && layer.size() > 0) {      
      ANode::Ptr n1 = layer.front();
      layer.pop_front();
      ANode::Ptr toMerge[2] = {n1,nullptr};
      ANode::Ptr mNode = nullptr;
      for(auto i = layer.begin();i != layer.end();i++) {
         auto n2 = *i;
         assert(n1->getLayer() == n2->getLayer());
         mNode = _dd->merge(n1,n2);
         if (mNode) {
            toMerge[1] = n2;
            layer.erase(i);
            break;
         }
      }
      if (toMerge[1]) {
         mergesDone++;
         const bool newNode = mNode->nbParents()==0; // is this a newly created node? 
         const bool sameLayer = mNode->getLayer() == n1->getLayer();
         mNode->setLayer(std::max(mNode->getLayer(),n1->getLayer()));
         mNode->setExact(false);
         _dd->_exact = false;
         for(auto i = 0; i < 2;i++) {
            auto d = toMerge[i];            
            if (d != mNode) { // skip in case the node itself is the merged one
               transferArcs(d,mNode);
               _dd->_an.remove(d);
            }
         }
         if (sameLayer) {
            if (newNode)
               layer.push_back(mNode);
         } else delayed.push_back(mNode);
      } else final.push_back(n1);
   }
   for(auto n : final) layer.push_back(n);
   return delayed;
}

void Relaxed::compute()
{
   _dd->_exact = true;
   auto root = _dd->init();
   _dd->target();
   CQueue<ANode::Ptr> qn(32);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) {
      auto& lk = pullLayer(qn); // We have in lk the queue content for layer cL

      for(auto nd : lk) {
         double cur = (nd->nbParents() == 0) ? nd->getBound() : _dd->initialBest();
         for(auto pi = nd->beginPar();pi != nd->endPar();pi++) {
            Edge::Ptr e = *pi;
            auto ep = e->_from->getBound() + e->_obj;
            if (_dd->isBetter(ep,cur)) 
               cur = ep;            
         }        
         nd->setBound(cur);
      }
      
      std::list<ANode::Ptr> delay; // nodes whose layer was "increase". Need to go back in queue
      if (lk.size() > _mxw) 
         delay = mergeLayer(lk);
      for(auto p : delay)
         qn.enQueue(p);
      for(auto p : lk) { // loop over layer lk. p is a "parent" node.
         auto remLabels = remainingLabels(p);
         for(auto l : remLabels) {
            auto child = _dd->transition(p,l); // we get back a new node, or an already existing one.
            if (child) {
               const bool newNode = child->nbParents()==0; // is this a newly created node?
               auto theCost = _dd->cost(p,l);
               Edge::Ptr e = new (_dd->_mem) Edge(p,child,l);
               e->_obj = theCost;
               _dd->addArc(e); // connect to new node
               if (!_dd->eqSink(child)) {
                  if (newNode)
                     qn.enQueue(child);
               }
               child->setLayer(std::max(child->getLayer(),p->getLayer()+1));
            }            
         }
      }
   }
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
