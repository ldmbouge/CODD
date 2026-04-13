#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "DebugUtils.hpp"
#include "Types.hpp"

namespace gfl {

struct Heap {
  template<typename T = u8>
  static T * reserve(i64 const count = 1) noexcept {
    checkOrAbort(count > 0, "Heap::reserve: count must be > 0");
    void * memory = std::malloc(sizeof(T) * count);
    checkOrAbort(memory != nullptr, "Heap::reserve: malloc failed");
    return scast<T *>(memory);
  }

  template<typename T>
  static T * realloc(T * const ptr, i64 const count) noexcept {
    checkOrAbort(count > 0, "Heap::realloc: count must be > 0");
    void * memory = std::realloc(ptr, sizeof(T) * count);
    checkOrAbort(memory != nullptr, "Heap::realloc: realloc failed");
    return scast<T *>(memory);
  }

  template<typename T>
  static void release(T * & memory) noexcept {
    checkOrAbort(memory != nullptr, "Heap::release: null pointer");
    std::free(memory);
    memory = nullptr;
  }
};

#ifdef __CUDACC__

struct Host {
  template<typename T = u8>
  static T * reserve(i64 const count = 1) noexcept {
    checkOrAbort(count > 0, "Host::reserve: count must be > 0");
    void * memory = nullptr;
    cudaError_t const status = cudaMallocHost(&memory, sizeof(T) * count);
    checkOrAbort(status == cudaSuccess and memory != nullptr, "Host::reserve: cudaMallocHost failed");
    return scast<T *>(memory);
  }

  template<typename T>
  static
  void release(T * & memory) noexcept {
    checkOrAbort(memory != nullptr, "Host::release: null pointer");
    cudaError_t const status = cudaFreeHost(memory);
    checkOrAbort(status == cudaSuccess, "Host::release: cudaFreeHost failed");
    memory = nullptr;
  }
};

struct Device {
  template<typename T = u8>
  static
  T * reserve(i64 const count = 1) noexcept {
    checkOrAbort(count > 0, "Device::reserve: count must be > 0");
    void * memory = nullptr;
    cudaError_t const status = cudaMalloc(&memory, sizeof(T) * count);
    checkOrAbort(status == cudaSuccess and memory != nullptr, "Device::reserve: cudaMalloc failed");
    return scast<T *>(memory);
  }

  template<typename T>
  static
  void release(T * & memory) noexcept {
    checkOrAbort(memory != nullptr, "Device::release: null pointer");
    cudaError_t const status = cudaFree(memory);
    checkOrAbort(status == cudaSuccess, "Device::release: cudaFree failed");
    memory = nullptr;
  }
};

struct Managed {
  template<typename T = u8>
  static
  T * reserve(i64 const count) noexcept {
    checkOrAbort(count > 0, "Managed::reserve: count must be > 0");
    void * memory = nullptr;
    cudaError_t const status = cudaMallocManaged(&memory, sizeof(T) * count);
    checkOrAbort(status == cudaSuccess and memory != nullptr, "Managed::reserve: cudaMallocManaged failed");
    return scast<T *>(memory);
  }

  template<typename T>
  static
  void release(T * & memory) noexcept {
    checkOrAbort(memory != nullptr, "Managed::release: null pointer");
    cudaError_t const status = cudaFree(memory);
    checkOrAbort(status == cudaSuccess, "Managed::release: cudaFree failed");
    memory = nullptr;
  }
};
#endif

template<typename B>
concept MemoryBackend = requires(i64 n, u8 * & p) {
  { B::template reserve<u8>(n) } noexcept -> std::same_as<u8*>;
  { B::template release<u8>(p) }     noexcept;
};

static_assert(MemoryBackend<Heap>);
#ifdef __CUDACC__
static_assert(MemoryBackend<Host>);
static_assert(MemoryBackend<Device>);
static_assert(MemoryBackend<Managed>);
#endif
}