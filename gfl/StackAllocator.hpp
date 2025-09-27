#pragma once

#include <cassert>

#include "Types.hpp"

namespace gfl
{
    class StackAllocator
    {
    public:
        constexpr i32 static DefaultAlign{16}; // 128-bit aligned

    protected:
        std::uintptr_t const begin;
        std::uintptr_t current;
        std::uintptr_t const end;

    public:
        GFL_HOST_DEVICE inline
        StackAllocator(void * memory, i64 size) noexcept;
        template <typename T>
        GFL_HOST_DEVICE
        T* allocate(i64 size = sizeof(T), i32 align = alignof(T)) noexcept;
        template <typename T>
        GFL_HOST_DEVICE
        T* allocateArray(i64 size, i32 align = alignof(T)) noexcept;
        void clear() noexcept { current = begin; }
        void* getMem() const noexcept { return reinterpret_cast<void*>(begin); }
        void* getFreeMem() const noexcept { return reinterpret_cast<void*>(current); }
        i64 calcFreeMemSize() const noexcept { return end - current; }
        i64 calcUsedMemSize() const noexcept { return current - begin; }
        i64 calcTotalMemSize() const noexcept { return end - begin; }
    };

    GFL_HOST_DEVICE inline
    StackAllocator::StackAllocator(void * const memory, i64 const size) noexcept :
        begin(reinterpret_cast<uintptr_t>(memory)),
        current(begin),
        end(begin + size)
    {
        assert(size > 0);
        assert(begin < end);
        assert(memory != nullptr);
    }

    template <typename T>
    GFL_HOST_DEVICE
    T* StackAllocator::allocate(i64 const size, i32 const align) noexcept
    {
        assert(size % sizeof(T) == 0);
        assert(align >= alignof(T));
        auto memory = current;
        auto const offset = memory % align;
        memory += offset != 0 ? align - offset : 0;
        current = memory + size;
        assert(current <= end);
        return reinterpret_cast<T*>(memory);
    }

    template <typename T>
    GFL_HOST_DEVICE
    T * StackAllocator::allocateArray(i64 const size, i32 const align) noexcept
    {
        return allocate<T>(sizeof(T) * size, align);
    }
}

inline
void * operator new(std::size_t size, gfl::StackAllocator & allocator)
{
   return allocator.allocate<gfl::u8>(size);
}

inline
void * operator new[](std::size_t size, gfl::StackAllocator & allocator)
{
    return allocator.allocate<gfl::u8>(size);
}