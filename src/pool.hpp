#ifndef __POOL_H
#define __POOL_H

#include "node.hpp"

class LPool {
   Pool::Ptr _mem;
   unsigned  _id;
   std::stack<ANode::Ptr> _free;
public:
   typedef LPool* Ptr; // space saving measure
   LPool(Pool::Ptr mem) : _mem(mem),_id(0) {}
   unsigned grabId() noexcept     { return _id++;}
   Pool::Ptr get() const noexcept {  return _mem;}
   ANode::Ptr claimNode() {
      if (_free.empty())
         return nullptr;
      else {
         ANode::Ptr nd = _free.top();
         _free.pop();
         return nd;
      }
   }
   void release(ANode::Ptr n);
};

#endif
