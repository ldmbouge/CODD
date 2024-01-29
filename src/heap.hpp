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
   class Location {
      T _val;
      int _p;
      friend class Heap<T,Ord>;
   public:
      Location() {}
      Location(const T& val,int p) : _val(val),_p(p) {}
      T value() const noexcept { return _val;}
      T operator*() const noexcept { return _val;}      
   };
   Pool::Ptr     _pool; // where memory comes from
   Ord            _ord; // ordering object
   Location**    _data; // location pointers
   Location**    _lseg; // list of location segments
   // current size of segment and insertion point
   int            _mxs; // maximal size
   int             _at; // actual size (where next element goes)
   // segments for location allocation
   int             _ms; // maximum number of segments 
   int            _nbs; // number of segments
   void resize() noexcept {
      const int newSize = _mxs << 1; // double
      Location* newSeg = new (_pool) Location[_mxs];
      Location** nd    = new (_pool) Location*[newSize];
      for(int i=0u;i < _at;i++) {
         nd[i] = _data[i];
      }
      for(int i=_mxs; i < newSize;i++) {
         nd[i] = newSeg + i - _mxs;
         nd[i]->_p = i;
      }
      if (_nbs >= _ms) { // make sure we have room to track segment
         Location** nls = new (_pool) Location*[_ms << 1];
         for(auto i=0;i < _ms;i++)
            nls[i] = _lseg[i];
         _lseg = nls;
         _ms <<= 1;
      }
      _lseg[_nbs++] = newSeg;
      _mxs = newSize;
      _data = nd;
   }
   void refloat(int p) noexcept {
      while(p>0) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_data[l]->_val,_data[p]->_val))
            best = l;
         else best = p;
         if (r < _at && _ord(_data[r]->_val,_data[best]->_val))
            best = r;
         if (best != p)  {
            std::swap(_data[p],_data[best]);
            _data[p]->_p = p;
            _data[best]->_p = best;
         }
         p = p / 2;
      }
   }
   void heapify(int p) noexcept {
      while(true) {
         int l = p * 2,r = p * 2 + 1;
         int best;
         if (l < _at && _ord(_data[l]->_val,_data[p]->_val))
            best = l;
         else best = p;
         if (r < _at && _ord(_data[r]->_val,_data[best]->_val))
            best = r;
         if (best != p) {
            std::swap(_data[p],_data[best]);
            _data[p]->_p = p;
            _data[best]->_p = best;
            p = best;
         } else break;
      }
   }
public:
   Heap(Pool::Ptr p,int sz)
      : _pool(p),_mxs(sz),_at(1),_ms(32),_nbs(0)
   {
      _lseg = new (_pool) Location*[_ms];
      auto cs = _lseg[_nbs++] = new (_pool) Location[_mxs];
      _data = new (_pool) Location*[_mxs];
      for(int i=0u;i<_mxs;++i) {         
         _data[i]  = cs + i;
         *_data[i] = Location { T(), i };
      }
   }
   void clear() noexcept { _at = 1;}
   unsigned size() const noexcept { return (unsigned)(_at - 1);}
   bool empty() const noexcept { return _at == 1;}
   Location* operator[](int i) const noexcept {
      return _data[i+1];
   }
   Location* insert(const T& v) noexcept {
      if (_at >= _mxs) resize();
      _data[_at]->_val = v;
      _data[_at]->_p = _at;
      ++_at;
      return _data[_at-1];
   }
   Location* find(const T& v) {
      for(int i=1;i<_mxs;i++)
         if (_data[i]->_val == v)
            return _data[i];
      return nullptr;
   }
   void decrease(Location* at) {
      at->_val--;
      refloat(at->_p);
   }
   const Location* insertHeap(const T& v) noexcept {
      auto loc = insert(v);
      heapify(_at-1);
      return loc;
   }
   void buildHeap() noexcept {
      for(int i = _at / 2;i > 0;--i)
         heapify(i);
   }
   T extractMax() noexcept {
      Location* rv = _data[1];
      _data[1] = _data[--_at];
      _data[1]->_p = 1;
      heapify(1);
      return rv->_val;
   }
   friend std::ostream& operator<<(std::ostream& os,const Heap<T,Ord>& h) {
      for(int i=1;i < h._at;i++) 
         os << i << ":" << h._data[i]->_val << "\n";      
      return os;
   }
};

#endif
