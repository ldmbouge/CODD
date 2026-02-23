#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "Align.hpp"
#include "Types.hpp"
#include "FunQual.hpp"
#include "Backend.hpp"
#include "MathUtils.hpp"

namespace gfl
{
    class ArenaAllocator
    {
    protected:
        uptr const begin {0};
        uptr current {0};
        uptr const end {0};

    public:
        ArenaAllocator() = default;

        ArenaAllocator(ArenaAllocator const &) = delete;
        ArenaAllocator(ArenaAllocator &&) = delete;
        ArenaAllocator& operator=(ArenaAllocator const &) = delete;
        ArenaAllocator& operator=(ArenaAllocator &&) = delete;

        GFL_HOST_DEVICE
        ArenaAllocator(i64 const size, u8 * memory) noexcept :
            begin(rcast<uptr>(memory)),
            current(begin),
            end(begin + scast<uptr>(size))
        {
            assert(memory != nullptr);
            assert(size > 0);
            assert(begin <= numeric_limits<uptr>::max() - scast<uptr>(size)); // Overflow
        }

        template<typename T>
        GFL_HOST_DEVICE
        T * allocate(i64 const count, i32 const align) noexcept
        {
            uptr const size = sizeof(T) * scast<uptr>(count);
            uptr const memory = roundUp<uptr>(current, align);
            uptr const newCurrent = memory + size;
            if (newCurrent > end)
            {
                constexpr char const * errorMsg = "ArenaAllocator::allocate failed: out of memory\n";
#ifdef __CUDA_ARCH__
                printf("%s", errorMsg);
#else
                std::fprintf(stderr, "%s", errorMsg);
                std::fflush(stderr);
#endif
                abort();
            }
            current = newCurrent;
            return rcast<T*>(memory);
        }

        template<typename T>
        GFL_HOST_DEVICE
        T * allocate(i64 const count = 1) noexcept
        {
            i32 const align = max<i32>(alignof(T), DefaultAlign);
            return allocate<T>(count, align);
        }

        GFL_HOST_DEVICE
        void clear() noexcept { current = begin; }

        GFL_HOST_DEVICE u8 const * getMarker() const noexcept { return rcast<u8*>(current); }

        GFL_HOST_DEVICE
        void rewindTo(u8 const * const marker) noexcept
        {
            uptr const addr = rcast<uptr>(marker);
            assert(begin <= addr);
            assert(addr <= current);
            current = addr;
        }

        GFL_HOST_DEVICE u8* mem() const noexcept { return rcast<u8*>(begin); }
        GFL_HOST_DEVICE u8* freeMem() const noexcept { return rcast<u8*>(current); }
        GFL_HOST_DEVICE i64 freeSize() const noexcept { return scast<i64>(end - current); }
        GFL_HOST_DEVICE i64 usedSize() const noexcept { return scast<i64>(current - begin); }
        GFL_HOST_DEVICE i64 totalSize() const noexcept { return scast<i64>(end - begin); }
    };
}

// Placement new overloads for ArenaAllocator
inline
void * operator new(std::size_t const size, gfl::ArenaAllocator & allocator)
{
    using namespace gfl;
    return allocator.allocate<u8>(scast<i64>(size), DefaultAlign);
}

inline
void * operator new[](std::size_t const size, gfl::ArenaAllocator & allocator)
{
    using namespace gfl;
    return allocator.allocate<u8>(scast<i64>(size), DefaultAlign);
}

// Required matching deletes (no-op, arena owns the memory)
inline
void operator delete(void*, gfl::ArenaAllocator&) noexcept {}

inline
void operator delete[](void*, gfl::ArenaAllocator&) noexcept {}
