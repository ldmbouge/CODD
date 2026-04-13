#pragma once

namespace gfl {
template<typename T>
class Ptr {
  T * _ptr;

public:
  Ptr() noexcept : _ptr(nullptr) {}
  Ptr(T * p) noexcept : _ptr(p) {}

  template<typename U>
  Ptr(Ptr<U> p) noexcept : _ptr(p.get()) {}


  T * get() const noexcept { return _ptr; }
  T * operator->() const noexcept { return _ptr; }
  T & operator*() const noexcept { return *_ptr; }
  operator T const * () const noexcept { return _ptr; }

  template<typename U>
  friend bool operator==(Ptr<T> a, Ptr<U> b) noexcept {
    return a._ptr == b.get();
  }
};
}