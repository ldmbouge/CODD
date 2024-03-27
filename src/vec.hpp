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
 * Copyright (c)  2023. by Laurent Michel
 */

#ifndef __VEC_H
#define __VEC_H

#include <memory>
#include <iterator>
#include "store.hpp"
#include <assert.h>
#include <iostream>

template <class T,typename SZT = std::size_t> class Vec {
   Pool::Ptr    _mem;
   SZT           _sz;
   SZT          _msz;
   T*          _data;
   void makeValid(SZT p) {
      SZT newSize = (_msz==0) ? 1 : _msz;
      while (p >= newSize)
         newSize = newSize << 1;
      if (newSize != _msz) {
         T* nd = new (_mem) T[newSize];
         for(SZT i=0;i< _msz;i++)
            nd[i] = _data[i];
         _data = nd;
         _msz = newSize;
      }
   }
public:
   typedef handle_ptr<Pool> Ptr;
   typedef SZT IdxType;
   /**
    * Class to lazily resize the vector if written to at offset i, so that the size becomes i+1
    * (and it resizes as needed)
    */
   class TVecSetter {
      Vec& _v;
      SZT _at;
   public:
      TVecSetter(Vec& v,SZT at) : _v(v),_at(at) {}
      TVecSetter& operator=(T&& v) {
         _v.makeValid(_at);
         _v.at(_at,std::move(v));
         _v._sz = _at + 1;
         return *this;
      }
      template <class U> TVecSetter& operator=(const U& nv) {
         _v.makeValid(_at);
         _v.at(_at,nv);
         _v._sz = _at + 1;
         return *this;
      }
      operator T() const noexcept   { return _v._data[_at];}
      T operator->() const noexcept { return _v._data[_at];}
   };

   /**
    * Copy contructor transfers the content of v2 to this new node (on the pool mem)
    * So it's useful to copy a vector over to another pool.
    */
   Vec(Pool::Ptr mem,const Vec& v2) : _mem(mem),_sz(v2._sz),_msz(v2._msz)  {
      _data = (_msz > 0) ? new (_mem) T[_msz] : nullptr;
      for(auto i=0u;i < _sz;i++)
         _data[i] = v2._data[i];      
   }
   Vec(Pool::Ptr mem,SZT s = 0u)
      : _mem(mem),_sz(0),_msz(s) {      
      _data = (_msz > 0) ? new (mem) T[_msz] : nullptr;
   }
   Vec(const Vec& v2) : _mem(v2._mem),_sz(v2._sz),_msz(v2._msz) {
      _data = (_msz > 0) ? new (_mem) T[_msz] : nullptr;
      for(auto i=0u;i < _sz;i++)
         _data[i] = v2._data[i];
   }
   Vec(Vec&& v)
      : _mem(v._mem),_sz(v._sz),_msz(v._msz),
        _data(std::move(v._data))
   {}
   ~Vec() {}
   Vec& operator=(const Vec& v) {
      if (_msz < v._msz) {
         _data = (v._msz > 0) ? new (_mem) T[v._msz] : nullptr;
         _msz =  v._msz;
      }
      for(auto i=0u;i < v._sz;i++)
         _data[i] = v._data[i];
      _sz = v._sz;
      return *this;
   }
   Vec& operator=(Vec&& s) {
      _mem = std::move(s._mem);
      _sz = s._sz;
      _msz = s._msz;
      _data = std::move(s._data);
      return *this;
   }
   Pool::Ptr getPool() { return _mem;}
   void clear() noexcept {
      _sz = 0;
   }
   SZT size() const { return _sz;}
   SZT push_back(const T& p) {
      makeValid(_sz);
      at(_sz,p);
      return _sz++;
   }
   T pop_back() {
      T rv = _data[_sz - 1];
      _sz -= 1;
      return rv;
   }
   template <class Relink>
   SZT remove(SZT i,Relink rl) {
      assert(_sz > 0);
      assert(i >= 0 && i < _sz);
      if (i < _sz - 1)
         at(i,_data[_sz - 1]);
      rl(_data[i]);
      _sz -= 1;
      return _sz;
   }
   const T get(SZT i) const noexcept           { return _data[i];}
   const T operator[](SZT i) const noexcept    { return _data[i];}
   auto operator[](SZT i) noexcept             { return TVecSetter(*this,i);}
   void at(SZT i,const T& nv)                  { _data[i] = nv;}
   class iterator { 
      T*    _data;
      long   _num;
      iterator(T* d,long num = 0) : _data(d),_num(num) {}
   public:
      using iterator_category = std::input_iterator_tag;
      using value_type = T;
      using difference_type = long;
      using pointer = T*;
      using reference = T&;
      iterator& operator++()   { _num = _num + 1; return *this;}
      iterator operator++(int) { iterator retval = *this; ++(*this); return retval;}
      iterator& operator--()   { _num = _num - 1; return *this;}
      iterator operator--(int) { iterator retval = *this; --(*this); return retval;}
      bool operator==(iterator other) const {return _num == other._num;}
      bool operator!=(iterator other) const {return !(*this == other);}
      T& operator*() const {return _data[_num];}
      friend class Vec;
   };
   class const_iterator { 
      T*    _data;
      long   _num;
      const_iterator(T* d,long num = 0) : _data(d),_num(num) {}
   public:
      using iterator_category = std::input_iterator_tag;
      using value_type = T;
      using difference_type = short;
      using pointer = T*;
      using reference = T&;
      const_iterator& operator++()   { _num = _num + 1; return *this;}
      const_iterator operator++(int) { const_iterator retval = *this; ++(*this); return retval;}
      const_iterator& operator--()   { _num = _num - 1; return *this;}
      const_iterator operator--(int) { const_iterator retval = *this; --(*this); return retval;}
      bool operator==(const_iterator other) const {return _num == other._num;}
      bool operator!=(const_iterator other) const {return !(*this == other);}
      const T operator*() const {return _data[_num];}
      friend class Vec;
   };
   using reverse_iterator = std::reverse_iterator<iterator>;
   iterator begin() const { return iterator(_data,0);}
   iterator end()   const { return iterator(_data,_sz);}
   reverse_iterator rbegin() const { return reverse_iterator(end());}
   reverse_iterator rend()   const { return reverse_iterator(begin());}
   const_iterator cbegin() const { return const_iterator(_data,0);}
   const_iterator cend()   const { return const_iterator(_data,_sz);}
   reverse_iterator crbegin() const { return reverse_iterator(cend());}
   reverse_iterator crend()   const { return reverse_iterator(cbegin());}
   friend std::ostream& operator<<(std::ostream& os,const Vec<T,SZT>& v) {
      os << "[";
      for(auto i = 0u;i < v._sz;i++) {
         os << v[i];
         if (i < v._sz-1)
            os << ',';
      }
      return os << "]";
   }
};



#endif
