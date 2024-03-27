#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include <functional>
#include "dd.hpp"
#include "store.hpp"

class BAndB {
   LPool::Ptr _mem;
   AbstractDD::Ptr _theDD;
   const unsigned  _mxw;
   std::function<bool(double)> _timeLimit;
public:
   BAndB(AbstractDD::Ptr dd,const unsigned width)
      : _mem(new LPool(new Pool)),_theDD(dd),_mxw(width),_timeLimit(nullptr) {}
   ~BAndB() {
      delete _mem;
   }
   void search(Bounds& bnds);
   void setTimeLimit(std::function<bool(double)> lim) { _timeLimit = lim;}
};  

#endif
