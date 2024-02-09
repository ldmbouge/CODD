/*
 * C++ddOpt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * C++ddOpt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2023. by Laurent Michel.
 */

#ifndef __QUEUE_H
#define __QUEUE_H

#include <iostream>
#include <utility>

template <typename T> class CQueue;

template <typename T> class Location {
   T   _val;
   int _pos;
   friend class CQueue<T>;
public:
   T value() const noexcept { return _val;}
   T operator*() const noexcept { return _val;}
};

template <typename T> class CQueue  {
   int _mxs;
   int _enter;
   int _exit;
   int _mask;
   int _mxSeg;
   int _nbs;
   int _cnt;
   Location<T>*  _locs;
   Location<T>** _vlocs;
   Location<T>** _data;
   void resize() {
      const int newSize = _mxs << 1;
      Location<T>* nl = new Location<T>[_mxs];
      Location<T>** nd = new Location<T>*[newSize];
      int at = 0,cur = _exit;
      do {
         nd[at] = _data[cur];
         nd[at]->_pos = at;
         ++at;
         cur = (cur + 1) & _mask;
      } while (cur != _enter);
      nd[at++] = _data[cur];
      while (at < newSize) {
         nd[at] = nl + at - _mxs;
         ++at;
      }
      delete[]_data;
      if (_nbs >= _mxSeg) {
         Location<T>** nvl = new Location<T>*[_mxSeg << 1];
         for(int i=0;i < _mxSeg;++i) nvl[i] = _vlocs[i];
         delete[]_vlocs;
         _vlocs = nvl;
         _mxSeg <<= 1;
      }
      _vlocs[_nbs++] = nl;
      _data = nd;
      _exit = 0;
      _enter = _mxs - 1;
      _mxs = newSize;
      _mask = _mxs - 1;
   }
public:
   CQueue(int sz = 32) : _mxSeg(8),_nbs(1) {
      _mxs = 1;
      while (_mxs <= sz) _mxs <<= 1; // mxs must be a power of 2.
      _locs = new Location<T>[_mxs];
      _vlocs = new Location<T>*[_mxSeg];
      _data = new Location<T>*[_mxs];
      _vlocs[0] = _locs;
      for(auto i=0;i < _mxs;++i)
         _data[i] = _locs+i;
      _mask = _mxs - 1;
      _enter = _exit = 0;
      _cnt = 0;
   }
   ~CQueue() {
      for(int i=0u;i < _nbs;++i)
         delete[]_vlocs[i];
      delete[]_vlocs;
      delete[]_data;
   }
   void clear() noexcept {
      _enter = _exit = 0;
      _cnt=0;
   }
   int size() const noexcept   { return _cnt;}
   bool empty() const noexcept { return _enter == _exit;}
   Location<T>* enQueue(const T& v) {
      if (_cnt == _mxs - 1) resize();
      Location<T>* at = _data[_enter];
      at->_val = v;
      at->_pos = _enter;
      _enter = (_enter + 1) & _mask;
      ++_cnt;
      return at;
   }
   T deQueue() {
      if (_enter != _exit) {
         T rv = _data[_exit]->_val;
         _exit = (_exit + 1) & _mask;
         --_cnt;
         assert(_cnt >=0);
         return rv;
      } else {
         assert(_cnt == 0);
         return T();
      }
   }
   T peek() {
      if (_enter != _exit) 
         return _data[_exit]->_val;
      else return T();
   }
   bool retract(const T& val) {
      assert(_cnt > 0);
      int cur = _exit;
      do {
         if (_data[cur]->_val == val) {
            retract(_data[cur]);
            return true;
         }
         cur = (cur + 1) & _mask;
      } while (cur != _enter);
      return false;
   }
   void retract(Location<T>* loc) {
      if (loc==nullptr) return;
      assert(_cnt > 0);
      int at = loc->_pos;
      if (at == _exit)
         _exit = (_exit + 1) & _mask;
      else if (at == _enter)
         _enter = (_enter > 0) ? _enter - 1 : _mask;
      else {
         std::swap(_data[at],_data[_exit]);
         _data[at]->_pos = at;
         _exit = (_exit + 1) & _mask;
      }
      loc->_val = T();
      --_cnt;
      assert(_cnt>=0);
   }
   template <class Fun>
   void doOnAll(Fun f) {
      int k = _exit;
      int n = _cnt;
      while(n) {
         f(_data[k]);
         k = (k + 1) & _mask;
         n--;
      }
   }
};

#endif

