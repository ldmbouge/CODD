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
 * Copyright (c)  2023. by Laurent Michel
 */

#ifndef __HANDLE_H
#define __HANDLE_H

#include <memory>

/**
 * @brief Templated smart pointer for Constraint Programming.
 * This class is used for the vast majority of objects created on behalf of a CP
 * solver instance. It is necessary for several reasons.
 * - First, when creating an STL container (e.g., `vector<T>`) to hold, for instance,
 *   integer variables (`var<int>`) one cannot use `T=var<int>`  directly. Indeed
 *   the variable would be owned by the vector and destroyed when the vector went out
 *   of scope leading to easy errors. Instead, one should use `T=var<int>*`, namely, one
 *   should store pointers to the variables in the vector instead.
 * - Second, once pointers to variables appear in STL containers, the question becomes: *who
 *   is in charge of deallocation?* The natural C++ answer is to use a smart pointer
 *   like `std::shared_ptr<T>` to resolve this problem. Unfortunately, this carries some
 *   overhead as reference counters must increase/decrease all the time.
 * 
 * The `handle_ptr<T>` addresses this issue. It can be used like a `shared_ptr<T>`, yet there
 * is no need to increase/decrease as the solver owns the object (the variable) for the entire
 * lifetime of the solver. So a `handle_ptr<T>` behaves like a vanilla `T*`.
 *
 * The vast majority of user visible instances of classes with instances allocated on the
 * solver heap define a public type `Ptr` that happens to be a `handle_ptr` to their own type,
 * making it easy to use.
 * The type provide the same abstraction as the STL for compatibility's sake.
 */
template <typename T>
class handle_ptr {
   T* _ptr;
public:
   template <typename DT> friend class handle_ptr;
   typedef T element_type;
   typedef T* pointer;
   handle_ptr() noexcept : _ptr(nullptr) {}
   handle_ptr(T* ptr) noexcept : _ptr(ptr) {}
   handle_ptr(const handle_ptr<T>& ptr) noexcept : _ptr(ptr._ptr)  {}
   handle_ptr(std::nullptr_t ptr)  noexcept : _ptr(ptr) {}
   handle_ptr(handle_ptr<T>&& ptr) noexcept : _ptr(std::move(ptr._ptr)) {}
   template <typename DT> handle_ptr(DT* ptr) noexcept : _ptr(ptr) {}
   template <typename DT> handle_ptr(const handle_ptr<DT>& ptr) noexcept : _ptr(ptr._ptr) {}
   template <typename DT> handle_ptr(handle_ptr<DT>&& ptr) noexcept : _ptr(std::move(ptr._ptr)) {}
   handle_ptr& operator=(const handle_ptr<T>& ptr) { _ptr = ptr._ptr;return *this;}
   handle_ptr& operator=(handle_ptr<T>&& ptr)      { _ptr = std::move(ptr._ptr);return *this;}
   handle_ptr& operator=(T* ptr)                   { _ptr = ptr;return *this;}  
   const T* get() const  noexcept { return _ptr;}
   T* get() noexcept { return _ptr;}
   T* operator->() const noexcept { return _ptr;}
   T& operator*() const noexcept  { return *_ptr;}
   template<typename SET> SET& operator*() const noexcept {return *_ptr;}
   operator bool() const noexcept { return _ptr != nullptr;}
   //operator const T*() const noexcept { return _ptr;}
   void dealloc() { delete _ptr;_ptr = nullptr;}
   void free()    { delete _ptr;_ptr = nullptr;}
   /**
    * Equality operator necessary to compare handle to abstract and concrete (sub) types.
    * @param p1  a smart pointer to a `T` instance to compare
    * @param p2 a smart pointer to a `X` instance to compare.
    * Typically \f$X <: T\f$
    */
   template<class X> friend bool operator==(const handle_ptr<T>& p1,const handle_ptr<X>& p2)
   {
      return p1._ptr == p2.get();
   }
};

/**
 * @brief Similar to the STL `make_shared`. It builds a handle on a heap allocated object and forwards the
 *  arguments.
 */
template <class T,class... Args>
inline handle_ptr<T> make_handle(Args&&... formals)
{
   return handle_ptr<T>(new T(std::forward<Args>(formals)...));
}


template <typename T>
class strict_ptr {
   T* _ptr;
public:
   template <typename DT> friend class strict_ptr;
   typedef T element_type;
   typedef T* pointer;
   strict_ptr() noexcept : _ptr(nullptr) {}
   strict_ptr(T* ptr) noexcept : _ptr(ptr) {}
   strict_ptr(const strict_ptr<T>& ptr) noexcept : _ptr(ptr._ptr)  {}
   strict_ptr(std::nullptr_t ptr)  noexcept : _ptr(ptr) {}
   strict_ptr(strict_ptr<T>&& ptr) noexcept : _ptr(std::move(ptr._ptr)) {}
   strict_ptr& operator=(const strict_ptr<T>& ptr) { _ptr = ptr._ptr;return *this;}
   strict_ptr& operator=(strict_ptr<T>&& ptr)      { _ptr = std::move(ptr._ptr);return *this;}
   strict_ptr& operator=(T* ptr)                   { _ptr = ptr;return *this;}
   T* get() const noexcept { return _ptr;}
   T* operator->() const noexcept { return _ptr;}
   T& operator*() const noexcept  { return *_ptr;}
   operator bool() const noexcept { return _ptr != nullptr;}
   void dealloc() { delete _ptr;_ptr = nullptr;}
   void free()    { delete _ptr;_ptr = nullptr;}
   /**
    * Equality operator necessary to compare strict to abstract and concrete (sub) types.
    * @param p1  a smart pointer to a `T` instance to compare
    * @param p2 a smart pointer to a `X` instance to compare.
    * Typically \f$X <: T\f$
    */
   template<class X> friend bool operator==(const strict_ptr<T>& p1,const strict_ptr<X>& p2)
   {
      return p1._ptr == p2.get();
   }
};


#endif
