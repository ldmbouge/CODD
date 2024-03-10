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
   Heap<DNode> h(_mem,1000,[](const DNode& a,const DNode& b) { return a.degree < b.degree;});
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
   Heap<DNode> h(_mem,1000,[](const DNode& a,const DNode& b) { return a.degree < b.degree;});
   for(auto n : _an) 
      h.insert({n,n->nbParents()});   
   h.buildHeap();
   std::string colors[2] = {"red","green"};
   os << "digraph MDD {" << std::endl;
   os << "label=\"" << gLabel << "\";\n"; 
   os << " node [style=filled gradientangle=270];\n"; 
   while (h.size() > 0) {
      auto cur = h.extractMax();
      os << "\"";
      printNode(os,cur.node);
      os << "\" [fillcolor=\"" << colors[cur.node->isExact()] << "\"";
      if (eq(cur.node,_trg))
         os << ",shape=box,color=black";
      os << "];\n";      
      for(auto ki = cur.node->beginKids(); ki != cur.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = h.find({k->_to,0});
         h.decrease(at);
         ANode::Ptr from = k->_from;
         ANode::Ptr to   = k->_to;
         os << "\"";
         printNode(os,from);
         os << "\" ->" << "\"";
         printNode(os,to);
         os << "\"";
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
      int st = execlp("dot","dot","-Tpdf","-O",buf,NULL);
      printf("execl status: %d\n",st);
   }
   int st = 0;
   waitpid(child,&st,0);
   child = fork();
   if (child == 0) {
      execlp("open","open",base.c_str(),NULL);
      printf("execl2 status: %d\n",st);
   }
   //unlink(buf);
}

typedef Heap<DNode> DegHeap;

void AbstractDD::computeBest(const std::string m)
{
   //std::cout << "ANSZ:" << _an.size() << "\n";
   DegHeap h(_mem,1000,[](const DNode& a,const DNode& b) { return a.degree < b.degree;});
   unsigned mxId = 0;
   for(auto n : _an) 
      mxId = std::max(n->getId(),mxId);
   auto nl = new (_mem) DegHeap::Location*[mxId+1];
   memset(nl,0,sizeof(DegHeap::Location*)*(mxId+1));
   for(auto n : _an) {
      nl[n->getId()] = h.insert({n,n->nbParents()});
      if (n != _root)
         n->setBound(initialBest());
   }
   h.buildHeap();
   while (h.size() > 0) {
      auto n = h.extractMax();
      double cur = (n.node->nbParents() == 0) ? n.node->getBound() : initialBest();
      //std::cout << "\tCOMPUTE BEST: " << n.node->getId() << " \033[31;1m" << n.node->getLayer() << "\033[0m #P:" << n.node->nbParents() << "\n";
      for(auto pi = n.node->beginPar();pi != n.node->endPar();pi++) {
         Edge::Ptr e = *pi;
         auto ep = e->_from->_bound + e->_obj;
         //std::cout << "\t   EDGE:(" << *e << ") SP=" << e->_from->_bound << " EP=" << ep << std::endl;
         if (isBetter(ep,cur)) {
            cur = ep;
            n.node->_optLabels = e->_from->_optLabels;
            n.node->_optLabels.push_back(e->_lbl);
         }
      }
      //std::cout << "\tCOMPUTED:" << cur << " for " << n.node->getId() << "\tHELD:" << n.node->getBound() << "\n";
      n.node->setBound(cur);
      for(auto ki = n.node->beginKids(); ki != n.node->endKids();ki++) {
         Edge::Ptr k = *ki;
         auto at = nl[k->_to->getId()];
         assert(at!=nullptr);
         h.decrease(at);
      }
   }
#ifndef _NDEBUG     
   std::cout << '\t' << m << " B@SINK:" << _trg->getBound() << "\tLBL:[";
   for(auto l : _trg->_optLabels)
      std::cout << l;
   std::cout << "]" << std::endl;
#endif   
}

void AbstractDD::computeBestBackward(const std::string m)
{
   DegHeap h(_mem,1000,[](const DNode& a,const DNode& b) { return a.degree < b.degree;});
   unsigned mxId = 0;
   for(auto n : _an) 
      mxId = std::max(n->getId(),mxId);
   auto nl = new (_mem) DegHeap::Location*[mxId+1];
   memset(nl,0,sizeof(DegHeap::Location*)*(mxId+1));
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

auto Strategy::remainingLabels(ANode::Ptr p)
{
   return _dd->getLabels(p);
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
   _dd->computeBest(getName());
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

std::size_t WidthBounded::estimate(CQueue<ANode::Ptr>& qn)
{
   ANode::Ptr n = qn.peek();
   auto layer = n->getLayer();
   std::size_t nb = 0;
   qn.doOnAll([layer,&nb](auto aNodeLoc) {
      // std::cout << "aNodeLoc->value()->getLayer() = " << aNodeLoc->value()->getLayer();
      // std::cout << " TRG: " << layer << "\n";
      nb += aNodeLoc->value()->getLayer()==layer;
   });
   return nb;
}

// ----------------------------------------------------------------------
// Restricted DD Strategy

void Restricted::truncate(NDArray& layer)
{
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
               child->setLayer(std::max(child->getLayer(),p->getLayer()+1));
               if (!_dd->eqSink(child)) {
                  if (newNode) {
                     qn.enQueue(child);
                     auto nbNode = estimate(qn);
                     //std::cout << "#NODES: " << nbNode << "\n";
                     if (nbNode > _mxw) {
                        //std::cout << "JUMP..." << nbNode << '/' << _mxw << "\n";
                        goto next;
                     }
                  }
               }
            }            
         }
      }         
   next:;
   }
   _dd->computeBest(getName());
}

// ----------------------------------------------------------------------
// Relaxed DD Strategy

void Relaxed::transferArcs(ANode::Ptr donor,ANode::Ptr receiver)
{
   for(auto ei = donor->beginPar(); ei != donor->endPar();ei++) {
      auto ep = *ei;
      ep->_to = receiver;
      _dd->addArc(ep);
   }
   assert(donor->nbChildren()==0);
   for(auto ei = donor->beginKids(); ei != donor->endKids();ei++) {
      auto ep = *ei;
      ep->_from = receiver;
      _dd->addArc(ep);
   }
   donor->disconnect();
}

NDAction Relaxed::mergePair(ANode::Ptr mNode,ANode::Ptr toMerge[2])
{
   assert(toMerge[0] != nullptr && toMerge[1] != nullptr);
   // Neither toMerge[0], nor toMerge[1] are in the layer
   // mNode could be
   // 1. A brand new node NOT in the layer
   // 2. A node already in the layer but not in toMerge
   // 3. A node already in the layer that is one of toMerge[i] 
   const bool newNode = mNode->nbParents()==0; // is this a brand new node
   if (newNode) mNode->setLayer(toMerge[0]->getLayer()); // give it a layer
   const bool sameLayer = mNode->getLayer() == toMerge[0]->getLayer(); // layer not changing?
   mNode->setLayer(std::max(mNode->getLayer(),toMerge[0]->getLayer()));// possibly set layer
   mNode->setExact(false); // surely inexact now 
   _dd->_exact = false;    // DD inexact as well
   bool addIt = false;
   for(auto i = 0; i < 2;i++) {
      auto d = toMerge[i];            
      if (d != mNode) { // skip in case the node itself is the merged one
         transferArcs(d,mNode);
         _dd->_an.remove(d);
      } else addIt = true; // d == mNode -> so we do need to add it (toMerge were removed)
   }
   return NDAction { mNode, sameLayer ? ((newNode || addIt) ?
                                         NDAction::InFront : NDAction::Noop)
                     : NDAction::Delay };
}

ANode::Ptr Relaxed::mergeOne(auto& layer,auto& skip)
{   
   auto i = layer.begin();
   auto n1 = *i;   
   ANode::Ptr toMerge[2] = {n1,nullptr};
   ANode::Ptr mNode = nullptr;
   auto j = i;
   for(++j;j != layer.end();++j) {
      auto n2 = *j;
      assert(n1->getLayer() == n2->getLayer());
      assert(n1->nbChildren()==0);
      assert(n2->nbChildren()==0);         
      mNode = _dd->merge(n1,n2);
      if (mNode) {
         assert(mNode->nbChildren()==0);         
         toMerge[1] = n2;
         break;
      }
   }
   if (toMerge[1]) {
      assert(mNode != nullptr);
      i = layer.erase(i);
      j = layer.erase(j);
      NDAction act = mergePair(mNode,toMerge);
      switch(act.act) {
         case NDAction::Delay:
            return act.node;
         case NDAction::InFront:
            layer.push_front(act.node);
            return nullptr;
         case NDAction::Noop:
            return nullptr;
      }
   } else {
      assert(mNode == nullptr);
      i = layer.erase(i);
      skip.push_back(n1);
   }
   return nullptr;
}


template <typename Fun> void Relaxed::mergeLayer(auto& layer,Fun f)
{
   std::list<ANode::Ptr> skip;
   while (skip.size() + layer.size() > _mxw && layer.size() > 0) {
      auto dn = mergeOne(layer,skip); // skipped nodes are not willing to  merge with anything.
      if (dn) f(dn); // delayed node saw an increase in layer. Back in the overall queue via f
   }
   layer.splice(layer.begin(),std::move(skip)); // put the skipped guys back in
}

void Relaxed::tighten(ANode::Ptr nd) noexcept
{
   double cur  = (nd->nbParents() == 0 && nd != _dd->_trg) ? nd->getBound() : _dd->initialBest();
   for(auto pi = nd->beginPar();pi != nd->endPar();pi++) {
      Edge::Ptr e = *pi;
      auto ep = e->_from->getBound() + e->_obj;
      if (_dd->isBetter(ep,cur)) {
         cur = ep;
         nd->_optLabels = e->_from->_optLabels;
         nd->_optLabels.push_back(e->_lbl);
      }
   }  
   nd->setBound(cur);
}

class LQueue {
   std::list<ANode::Ptr> _next;
   std::list<ANode::Ptr> _rest;
   Relaxed&              _dd;
   void insertInNext(ANode::Ptr n) {
      _next.push_front(n);
   }
public:
   LQueue(Relaxed& dd) : _next(),_rest(),_dd(dd) {}
   void enQueue(ANode::Ptr n) noexcept {
      if (_next.size() == 0)
         _next.push_back(n);
      else {
         if (_next.front()->getLayer() == n->getLayer()) {
            insertInNext(n);
            //eager currently disabled. It seems to cause an issue with knapsack (loose optimality)
            //and it's not -yet- clear why.            
            if (0 && _next.size() > 2 *  _dd.getWidth()) {
               _next.sort([dd = _dd.theDD()](const ANode::Ptr& a,const ANode::Ptr& b) {
                  return !dd->isBetter(a->getBound(),b->getBound());
               });
               _dd.mergeLayer(_next,[this](ANode::Ptr dn)  {
                  _rest.push_back(dn);
               });
            }
         } else 
            _rest.push_back(n);         
      }
   }
   bool empty() const noexcept {
      return size() == 0;
   }
   std::size_t size() const noexcept {
      return _next.size() + _rest.size();
   }
   std::list<ANode::Ptr> pullLayer() noexcept {
      std::list<ANode::Ptr> retVal = std::move(_next);

      //sort and collapse the layer via merging when the layer is pulled.
      retVal.sort([dd = _dd.theDD()](const ANode::Ptr& a,const ANode::Ptr& b) {
         return !dd->isBetter(a->getBound(),b->getBound());
      });
      _dd.mergeLayer(retVal,[this](ANode::Ptr dn)  {
         _rest.push_back(dn);
      });

      auto layer = (_rest.size() > 0) ? _rest.front()->getLayer() : -1;
      for(auto i = _rest.begin(); i != _rest.end();) {
         if ((*i)->getLayer() != layer)            
            break;
         std::cout << "adding: " << (*i)->getId() << "\n" << std::flush;
         _next.push_back(*i);
         i = _rest.erase(i);
      }
      return retVal;
   }   
};

void Relaxed::compute()
{
   _dd->_exact = true;
   auto root = _dd->init();
   _dd->target();
   LQueue qn(*this);
   root->setLayer(0);
   qn.enQueue(root);
   while (!qn.empty()) {
      auto lk = qn.pullLayer();

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
               auto ep = p->getBound() + e->_obj;
               if (_dd->isBetter(ep,child->getBound())) {
                  child->setBound(ep);
                  child->_optLabels = p->_optLabels;
                  child->_optLabels.push_back(e->_lbl);
               }
               child->setLayer(std::max(child->getLayer(),p->getLayer()+1));               
               if (!_dd->eqSink(child)) {
                  if (newNode)
                     qn.enQueue(child);
               }
            }            
         }
      }
   }
   _dd->computeBest(getName());
   //tighten(_dd->_trg);
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
