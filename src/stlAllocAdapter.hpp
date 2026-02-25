/*
 * mini-cp is free software: you can redistribute it and/or modify
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
 * Copyright (c)  2018. by Laurent Michel, Pierre Schaus, Pascal Van Hentenryck
 */

#ifndef __STLALLOC_ADAPTER_H
#define __STLALLOC_ADAPTER_H

#include <functional>
#include <memory>
#include <iostream>
#include <assert.h>

namespace stl {
   template <std::size_t N> class CoreAlloc {
      char     _buf[N];
      char*       _end;
      char*        _sp;
   public:
      CoreAlloc() : _end(_buf + N),_sp(_buf) {}
      void* allocate(std::size_t n) {
         std::cout << "\t\tCoreAlloc["<< N << "] available: "
                   << std::distance(_sp,_end) << std::endl;
         if (n <= std::distance(_sp,_end)) {
            char* ptr = _sp;
            _sp += n;
            return ptr;
         }
         throw std::bad_alloc();
      }
      void free(void* ptr) {}
      void* base() { return _buf;}
      constexpr std::size_t capacity() const { return N;} // expressed in bytes
   };

   template <class T,typename CA> class StackAdapter {
      CA _ca;
   public:  
      typedef T value_type;
      typedef T*  pointer;
      typedef const T* const_pointer;
      typedef T&  reference;
      typedef const T& const_reference;
      typedef std::size_t size_type;
      typedef std::ptrdiff_t difference_type;      
   public:
      using propagate_on_container_copy_assignment = std::true_type;
      using propagate_on_container_move_assignment = std::true_type;
      using propagate_on_container_swap = std::true_type;
     //      StackAdapter() {}
      explicit StackAdapter(CA ca) : _ca(ca) {}  
      pointer allocate(size_type n, const void* hint = nullptr) {
         //std::cout << "allocate called  : " << n << " bytes:" << n * sizeof(T) << std::endl;
         assert(_ca);
         return (pointer)_ca->allocate(n * sizeof(T));
      } 
      void deallocate(value_type* p, size_type n) {
         //std::cout << "deallocate called: " << p << " sz = " << n << std::endl;
         return;
      } 
      size_type max_size() const noexcept {
         assert(_ca);
         return _ca->capacity() / sizeof(value_type);
      } 
      template <class U, class... Args> void construct(U* p, Args&&... args) {
         new (p) U(std::forward<Args>(args)...);
      }
      template <class U> void destroy(U* p) {
         p->U::~U();
      } 
      pointer address(reference x) const {
         return std::addressof(x);
      } 	
      const_pointer address(const_reference x) const {
         return std::addressof(x);
      } 
      template <class U>
      struct rebind { typedef StackAdapter<U,CA> other; };

      template <class T2>
      friend bool operator == (const StackAdapter<T,CA>& lhs,const StackAdapter<T2,CA>& rhs) noexcept {
         return lhs._ca == rhs._ca;
      }
      template <class T2>
      friend bool operator!=(const StackAdapter<T,CA>& lhs,const StackAdapter<T2,CA>& rhs) noexcept {
         return !(lhs._ca == rhs._ca);
      }
   };
}

#endif
