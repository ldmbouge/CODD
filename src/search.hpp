#ifndef __SEARCH_HPP__
#define __SEARCH_HPP__

#include "dd.hpp"

class BAndB {
   AbstractDD::Ptr _theDD;
   const unsigned  _mxw;
public:
   BAndB(AbstractDD::Ptr dd,const unsigned width) : _theDD(dd),_mxw(width) {}
   void search();
};  

#endif
