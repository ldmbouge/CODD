#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "Align.hpp"
#include "DebugUtils.hpp"
#include "FunQual.hpp"
#include "MathUtils.hpp"
#include "Types.hpp"

namespace gfl {
class ArenaAllocator {
protected:
  uptr const _begin;
  uptr _current;
  uptr const _end;

public:
  GFL_HOST_DEVICE
  ArenaAllocator(u8 * memory, i64 const size) noexcept
      : _begin(rcast<uptr>(memory)), _current(_begin), _end(_begin + scast<uptr>(size)) {
    assert(memory != nullptr);
    assert(size > 0);
  }

  template<typename T>
  GFL_HOST_DEVICE
  bool canAllocate(i64 const count = 1, i32 const align = alignof(T)) const noexcept {
    uptr const memory = roundUp<uptr>(_current, align);
    uptr const size = sizeof(T) * scast<uptr>(count);
    uptr const newCurrent = memory + size;
    return newCurrent < _end;
  }

  template<typename T>
  GFL_HOST_DEVICE
  T * allocate(i64 const count = 1, i32 const align = alignof(T)) noexcept {
    uptr const memory = roundUp<uptr>(_current, align);
    uptr const size = sizeof(T) * scast<uptr>(count);
    uptr const newCurrent = memory + size;
    checkOrAbort(newCurrent < _end,  "ArenaAllocator: out of memory");
    _current = newCurrent;
    return rcast<T *>(memory);
  }

  GFL_HOST_DEVICE
  void clear() noexcept { _current = _begin; }

  GFL_HOST_DEVICE
  u8 * mem() const noexcept { return rcast<u8 *>(_begin); }

  GFL_HOST_DEVICE
  u8 * freeMem() const noexcept { return rcast<u8 *>(_current); }

  GFL_HOST_DEVICE
  i64 freeSize() const noexcept { return scast<i64>(_end - _current); }

  GFL_HOST_DEVICE
  i64 usedSize() const noexcept { return scast<i64>(_current - _begin); }

  GFL_HOST_DEVICE
  i64 totalSize() const noexcept { return scast<i64>(_end - _begin); }
};
}

inline
void * operator new(std::size_t const size, gfl::ArenaAllocator & allocator) {
  using namespace gfl;
  return allocator.allocate<u8>(scast<i64>(size), DefaultAlign);
}

inline
void * operator new[](std::size_t const size, gfl::ArenaAllocator & allocator) {
  using namespace gfl;
  return allocator.allocate<u8>(scast<i64>(size), DefaultAlign);
}

inline
void operator delete(void *, gfl::ArenaAllocator &) noexcept {}

inline
void operator delete[](void *, gfl::ArenaAllocator &) noexcept {}