#pragma once

#include <atomic>
#include <cstring>

#include "ArrayView.hpp"

namespace gfl {
template<typename T>
class VectorView : public ArrayView<T> {
public:
  using ArrayView<T>::at;

protected:
  using ArrayView<T>::_size;
  using ArrayView<T>::_data;
  i64 _capacity;

public:
  VectorView() noexcept = default;

  GFL_HOST_DEVICE
  VectorView(T * data, i64 const capacity) noexcept : ArrayView<T>(scast<i64>(0), data), _capacity(capacity) {
    assert(data != nullptr);
    assert(capacity > 0);
  }

  template<typename Allocator>
  GFL_HOST_DEVICE
  VectorView(i64 const capacity, Allocator & alloc) noexcept
      : ArrayView<T>(alloc.template allocate<T>(capacity), scast<i32>(0)), _capacity(capacity) {
    assert(capacity > 0);
    assert(_data != nullptr);
  }

  GFL_HOST_DEVICE
  i64 capacity() const noexcept { return _capacity; }

  GFL_HOST_DEVICE
  bool empty() const noexcept { return _size == 0; }

  GFL_HOST_DEVICE
  void clear() noexcept { _size = 0; }

  GFL_HOST_DEVICE static
  void swap(VectorView & a, VectorView & b) noexcept {
    ArrayView<T>::swap(a, b);
    i64 const tmpCapacity = a._capacity;
    a._capacity = b._capacity;
    b._capacity = tmpCapacity;
  }

  GFL_HOST_DEVICE
  void swap(VectorView & other) noexcept { swap(*this, other); }

  GFL_HOST_DEVICE
  i64 resizeTo(i64 const size) noexcept {
    assert(size >= 0);
    assert(size <= _capacity);
    i64 const oldSize = _size;
    _size = size;
    return oldSize;
  }

  GFL_HOST_DEVICE
  i64 resizeBy(i64 const delta) noexcept {
    i64 const oldSize = _size;
    assert(oldSize + delta >= 0);
    assert(oldSize + delta <= _capacity);
    _size += delta;
    return oldSize;
  }

  GFL_HOST_DEVICE
  void pushBack(ArrayView<T> const & elements) noexcept {
    i64 const oldSize = resizeBy(elements.size());
    std::memcpy(&at(oldSize), elements.data(), elements.dataMemSize());
  }

  GFL_HOST_DEVICE
  void pushBack(T const * value) noexcept {
    i64 const oldSize = resizeBy(1);
    std::memcpy(&at(oldSize), value, sizeof(T));
  }

#ifdef __CUDACC__
  void pushBackToGpuAsync(ArrayView<T> const & elements) noexcept {
    i64 const oldSize = resizeBy(elements.size());
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&at(oldSize), elements.data(), elements.dataMemSize(), cudaMemcpyHostToDevice));
  }

  void pushBackFromGpu(ArrayView<T> const & elements) noexcept {
    i64 const oldSize = resizeBy(elements.size());
    CHECK_CUDA_ERROR(cudaMemcpy(&at(oldSize), elements.data(), elements.dataMemSize(), cudaMemcpyDeviceToHost));
  }
#endif

  GFL_HOST_DEVICE
  ArrayView<T> popBack(i64 const count) noexcept {
    i64 const oldSize = resizeBy(-count);
    ArrayView<T> items(_data + _size, _data + oldSize);
    return items;
  }

  GFL_HOST_DEVICE
  T & popBack() noexcept { return popBack(1).front(); }
};
} // namespace gfl