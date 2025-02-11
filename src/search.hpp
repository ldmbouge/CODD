#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"

struct QNode {
   ANode::Ptr node;
   double    bound;
   friend std::ostream& operator<<(std::ostream& os,const QNode& q) {
      return os << "QNode[(" << q.node->getId() << ','
                << q.node->getBound() << ','
                << q.node->getBackwardBound() << ")," << q.bound << "]";
   }
};

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
