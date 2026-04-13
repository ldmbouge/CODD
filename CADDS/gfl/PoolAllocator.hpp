#pragma once
#include <vector>
#include "Memory.hpp"
#include "Ptr.hpp"

namespace gfl {
  template<MemoryBackend B>
  class PoolAllocator {

    constexpr static i32 DefaultChunkSize = 4llu * 1024 * 1024; // 4MB

    std::vector<ArenaAllocator> _chunks;
    i32 const _chunkSize;

  public:
    explicit
    PoolAllocator(i64 const chunkSize = DefaultChunkSize) noexcept : _chunkSize(chunkSize) {}


    template<typename T>
    T * allocate(i64 const count = 1, i32 const align = alignof(T)) {
       if (_chunks.empty() or not _chunks.back().template canAllocate<T>(count, align)) {
          i32 const memSize = sizeof(T) * count + align;
          u8 * mem = B::reserve(memSize);
          _chunks.emplace_back(mem, memSize);
       }
       return _chunks.back().template allocate<T>(count,align);
    }

    i64 size() const noexcept {
      i64 total = 0;
      for (auto const & c : _chunks)
        total += c.usedSize();
      return total;
    }

    template<typename T, typename... Args>
    Ptr<T> makePtr(Args &&... args) {
      T * p = allocate<T>();
      new (p) T(std::forward<Args>(args)...);
      return Ptr<T>(p);
    }
  };
}

template<gfl::MemoryBackend B>
void * operator new(std::size_t const size, gfl::PoolAllocator<B> & alloc) {
  using namespace gfl;
  return alloc.template allocate<u8>(scast<i64>(size));
}

template<gfl::MemoryBackend B>
void * operator new[](std::size_t const size, gfl::PoolAllocator<B> & alloc) {
  using namespace gfl;
  return alloc.template allocate<u8>(scast<i64>(size));
}

template<gfl::MemoryBackend B>
void operator delete(void *, gfl::PoolAllocator<B> &) noexcept {}

template<gfl::MemoryBackend B>
void operator delete[](void *, gfl::PoolAllocator<B> &) noexcept {}
