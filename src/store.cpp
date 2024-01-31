/*
 * DDOpt is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License  v3
 * as published by the Free Software Foundation.
 *
 * mini-cp is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY.
 * See the GNU Lesser General Public License  for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with mini-cp. If not, see http://www.gnu.org/licenses/lgpl-3.0.en.html
 *
 * Copyright (c)  2023. by Laurent Michel
 */

#include "store.hpp"
#include <algorithm>
#include <assert.h>

// ========================================================================

Pool::Pool(std::size_t defSize)
   : _store(0),
     _segSize(defSize),
     _top(0),
     _seg(0),
     _nbSeg(1),
     _mxs(32)
{
   _store = new Segment*[_mxs];
   _store[0] = new Segment(_segSize);
}

Pool::~Pool()
{
   for(auto i = 0u;i < _nbSeg;++i)
      delete _store[i];
   delete[] _store;
}

void* Pool::allocate(std::size_t sz)
{
   if (sz & 0xF)  // unaligned on 8 bytes boundary
      sz = (sz | 0xF) + 1; // increase to align
   assert((sz & 0xF) == 0 && sz != 0);           // check alignment
   auto s = _store[_seg];
   while(_top + sz > s->_sz) {
      assert(_top != 0);
      ++_seg;
      if (_seg >= _nbSeg) {
         if (_nbSeg == _mxs) {
            Segment** tab = new Segment*[_mxs << 1];
            for(auto i = 0u;i < _mxs;++i) tab[i] = _store[i];
            delete []_store;
            _store = tab;
            _mxs <<= 1;
         }
         _store[_nbSeg++] = new Segment(std::max(_segSize,sz));
      }
      _top = 0;
      s = _store[_seg];
   }
   void* ptr = s->_base + _top;
   _top   = _top + sz;
   return ptr;
}
