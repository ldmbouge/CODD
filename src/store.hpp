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
 * Copyright (c)  2018. by Laurent Michel
 */

#ifndef __STORE_H
#define __STORE_H

#include <vector>
#include "handle.hpp"
#include "stlAllocAdapter.hpp"
/**
 * Default segment size for storage management (4 Megs).
 */
#define SEGSIZE (1 << 22)

class Pool;
/**
 * @brief Abstract object representing a mark. Nothing is user visible
 * @see Pool
 */
class PoolMark {
   friend class Pool;
   size_t   _top;
   unsigned _seg;
   PoolMark(size_t top,unsigned seg) : _top(top),_seg(seg) {}
public:
   PoolMark() : _top(0),_seg(0) {}
};

/**
 * @brief Memory pool for the solver.
 *
 * The class is a segmented memory pool.
 * Objects are allocated on the pool with a stack like policy (LIFO).
 * There is no provision to automatically deallocate or shrink the pool.
 * Pool shrinkage is completely under the control of the programmer who
 * must take a `PoolMark` to remember the size of the allocator at some point
 * in time and later **clear** the allocator to restore its state to a chosen `PoolMark`.
 */
class Pool {
   struct Segment {
      char*      _base;
      std::size_t  _sz;
      Segment(std::size_t tsz) { _base = new char[tsz];_sz = tsz;}
      ~Segment()               { delete[] _base;}
      typedef std::shared_ptr<Segment> Ptr;
   };
   Segment** _store;
   const std::size_t   _segSize;
   size_t                  _top;
   unsigned                _seg;
   unsigned              _nbSeg;
   unsigned                _mxs;
public:
   Pool(std::size_t defSize = SEGSIZE);
   ~Pool();
   //typedef std::shared_ptr<Pool> Ptr;
   typedef Pool* Ptr; // space saving measure
   void* allocate(std::size_t sz);
   void free(void* ptr) {}
   /**
    * @brief Restore the pool to its very initial state (fully empty).
    */
   void clear() { _top = 0;_seg = 0;}
   /**
    * @brief Restore the pool to a specified mark. One can only shrink the pool.
    * @param m the mark to restore to.
    */
   void clear(const PoolMark& m) { _top = m._top;_seg = m._seg;}
   /**
    * @brief Capture the current mark and return it.
    * @return a PoolMark instance referring to the size of the allocator now.
    */
   PoolMark mark() const noexcept { return {_top,_seg};}
   std::size_t capacity() const noexcept { return _segSize;}
   std::size_t usage() const noexcept { return (_nbSeg - 1) * _segSize  + _top;}
};

inline void* operator new(std::size_t sz,Pool::Ptr store)
{
   return store->allocate(sz);
}

inline void* operator new[](std::size_t sz,Pool::Ptr store)
{
   return store->allocate(sz);
}

#endif
