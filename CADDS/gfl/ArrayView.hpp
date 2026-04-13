#pragma once

#include <cassert>
#include <cstdio>

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl {
template<typename T>
class ArrayView {
protected:
  T * _data;
  i64 _size;

public:
  ArrayView() noexcept = default;

  GFL_HOST_DEVICE
  ArrayView(T * const data, i64 const size) noexcept :
    _data(data),
    _size(size)
  {
    assert(data != nullptr);
  }

  template<typename Allocator>
  ArrayView(i64 const size, Allocator & alloc) noexcept :
    ArrayView(alloc.template allocate<T>(size), size) {}

  GFL_HOST_DEVICE
  ArrayView(T * begin, T * end) noexcept :
    _data(begin),
    _size(scast<i64>(end - begin))
  {
    assert(begin != nullptr);
    assert(end != nullptr);
    assert(begin <= end);
  }

  GFL_HOST_DEVICE
  T * data() const noexcept { return _data; }

  GFL_HOST_DEVICE
  i64 size() const noexcept { return _size; }

  GFL_HOST_DEVICE
  bool empty() const noexcept { return _size == 0; }

  GFL_HOST_DEVICE
  static
  i64 dataMemSize(i64 const size) noexcept { return sizeof(T) * size; }

  GFL_HOST_DEVICE
  i64 dataMemSize() const noexcept { return dataMemSize(_size); }

  GFL_HOST_DEVICE
  T & at(i64 const idx) const noexcept {
    assert(idx >= 0);
    assert(idx < _size);
    return _data[idx];
  }

  GFL_HOST_DEVICE
  T * begin() const noexcept { return _data; }

  GFL_HOST_DEVICE
  T * end() const noexcept { return _data + _size; }

  GFL_HOST_DEVICE
  T & front() const noexcept {
    assert(_size > 0);
    return _data[0];
  }

  GFL_HOST_DEVICE
  T & back() const noexcept {
    assert(_size > 0);
    return _data[_size - 1];
  }

  GFL_HOST_DEVICE
  T & operator[](i64 const idx) const noexcept { return at(idx); }

  GFL_HOST_DEVICE static
  void swap(ArrayView & a, ArrayView & b) noexcept {
    T * const tmpData = a._data;
    a._data = b._data;
    b._data = tmpData;

    auto tmpSize = a._size;
    a._size = b._size;
    b._size = tmpSize;
  }

  GFL_HOST_DEVICE
  void swap(ArrayView & other) noexcept { swap(*this, other); }

  GFL_HOST_DEVICE
  ArrayView slice(i64 const begin, i64 const end) const noexcept {
    assert(0 <= begin);
    assert(begin <= end);
    assert(end <= _size);
    return ArrayView(_data + begin, end - begin);
  }

  GFL_HOST_DEVICE
  ArrayView slice(i64 const count) const noexcept {
    assert(count != 0);
    return count >= 0 ? slice(0, count)              // Prefix
                      : slice(_size - count, _size); // Suffix
  }

  friend
  std::ostream & operator<<(std::ostream & os, const ArrayView & a) {
    bool comma = false;
    for (T const * it = a.begin(); it != a.end(); ++it) {
      os << (comma ? "," : "");
      os << *it;
      comma = true;
    }
    return os;
  }

  GFL_HOST_DEVICE static
  void print(T const * begin, T const * end, char const * fmt = "%d") noexcept {
    assert(begin != nullptr);
    assert(end != nullptr);
    assert(begin <= end);

    bool comma = false;
    for (T const * it = begin; it != end; ++it) {
      printf(comma ? "," : "");
      printf(fmt, *it);
      comma = true;
    }
  }

  GFL_HOST_DEVICE
  void print(char const * fmt = "%d") const noexcept { print(begin(), end(), fmt); }
};
}
