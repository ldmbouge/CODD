#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"
#include "heap.hpp"

struct QNode {
   ANode::Ptr node;
   double    bound;
   friend std::ostream& operator<<(std::ostream& os,const QNode& q) {
      return os << "QNode[(" << q.node->getId() << ','
                << q.node->getBound() << ','
                << q.node->getBackwardBound() << ")," << q.bound << "]";
   }
};

std::vector<ANode::Ptr> filterLocal(Bounds& bnds, AbstractDD::Ptr dd, std::vector<ANode::Ptr> nodes);
template<typename Ord>
int filterDom(bool &newGuyDominated, Bounds& bnds, AbstractDD::Ptr dd, std::vector<ANode::Ptr> nodes, Heap<QNode,Ord>* pq, std::vector<ANode::Ptr>* survived)
{
   newGuyDominated = false;
   auto end = nodes.rend();
   auto begin = nodes.rbegin();
   for (auto i = begin; i != end; i++) {
      auto n = *i;
      [[maybe_unused]] auto sz = nodes.size();
      for (auto j = begin; j != end; j++) {
         auto other = *j;
         if(i == j) continue;
         bool isObjDom = dd->isBetterEQ(other->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominates(other,n);
         if (newGuyDominated) {
            break;          
         }
      }
      if(!newGuyDominated) {
         survived->push_back(n);
      }
   }

   int pruned = 0;
   for(auto n: nodes) {
      unsigned d = 0;
      auto pqSz = pq->size();
      auto allLocs = new Heap<QNode,Ord>::LocType*[pqSz];
      for(unsigned k = 0;k < pqSz;k++) {
         auto other = (*pq)[k];
         bool isObjDom   = dd->isBetterEQ(other->value().node->getBound(),n->getBound());
         newGuyDominated = isObjDom && dd->dominates(other->value().node,n);
         if (newGuyDominated) {
            goto prune;             
         }        
         bool objDom   = dd->isBetterEQ(n->getBound(),other->value().node->getBound());
         bool qnDominated = objDom && dd->dominates(n,other->value().node);
         if (qnDominated)
            allLocs[d++] = other;
      }

      prune:
      if (d) {
         for(auto i =0u; i < d;i++) 
            pq->remove(allLocs[i]);
         pruned += d;
      }
      delete[]allLocs;
   }
   return pruned;
}

class BAndB {
protected:
   AbstractDD::Ptr _theDD;
   const unsigned    _mxw;
   std::function<bool(double)> _timeLimit;
public:
   BAndB(AbstractDD::Ptr dd,const unsigned width)
      : _theDD(dd),_mxw(width),_timeLimit(nullptr) {}
   ~BAndB() {}
   void search(Bounds& bnds);
   void setTimeLimit(std::function<bool(double)> lim) { _timeLimit = lim;}
};  

#endif
