/*
 * ddOpt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * ddOpt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2023. by Laurent Michel.
 */


#ifndef __MINICPP_HEAP_H
#define __MINICPP_HEAP_H

#include <algorithm>
#include <iostream>
#include <functional>
#include "store.hpp"

template <class T,class Ord = std::greater<T>> class Heap {
   Pool::Ptr _pool;
   Ord _ord;
   T* _tab;
   int _mxs;
   int  _at;
   void refloat(int p) noexcept {
      while(p>0) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_tab[l],_tab[p]))
            best = l;
         else best = p;
         if (r < _at && _ord(_tab[r],_tab[best]))
            best = r;
         if (best != p) 
            std::swap(_tab[p],_tab[best]);
         p = p / 2;
      }
   }
   void heapify(int p) noexcept {
      while(true) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_tab[l],_tab[p]))
            best = l;
         else best = p;
         if (r < _at && _ord(_tab[r],_tab[best]))
            best = r;
         if (best != p) {
            std::swap(_tab[p],_tab[best]);
            p = best;
         } else break;
      }
   }
public:
   Heap(Pool::Ptr p,int sz) : _pool(p) {
      _mxs = sz;
      _at =  1;
      _tab = new (_pool) T[_mxs];
      for(int i=0u;i<_mxs;++i) _tab[i] = T();
   }
   void clear() noexcept { _at = 1;}
   unsigned size() const noexcept { return (unsigned)(_at - 1);}
   bool empty() const noexcept { return _at == 1;}
   const T& operator[](int i) const noexcept { return _tab[i+1];}
   void insert(const T& v) noexcept {
      if (_at >= _mxs) {
         T* nt = new (_pool) T[_mxs << 1];
         for(int i=0;i<_mxs;++i) nt[i] = _tab[i];
         _tab = nt;
         _mxs <<= 1;
      }
      _tab[_at++] = v;
   }
   int find(const T& v) {
      for(int i=1;i<_mxs;i++)
         if (_tab[i] == v)
            return i;
      return -1;
   }
   void decrease(int at) {
      _tab[at]--;
      refloat(at);
   }
   void insertHeap(const T& v) noexcept {
      insert(v);
      heapify(_at-1);
   }
   void buildHeap() noexcept {
      for(int i = _at / 2;i > 0;--i)
         heapify(i);
   }
   T extractMax() noexcept {
      T rv = _tab[1];
      _tab[1] = _tab[--_at];
      heapify(1);
      return rv;
   }
   friend std::ostream& operator<<(std::ostream& os,const Heap<T,Ord>& h) {
      for(int i=1;i < h._at;i++) 
         os << i << ":" << h._tab[i] << "\n";      
      return os;
   }
};

#endif
