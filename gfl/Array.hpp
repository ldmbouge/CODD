#pragma once

#include <cstdio>
#include <cassert>

#include "Common.hpp"
#include "StackAllocator.hpp"
#include "Types.hpp"
#include "Utils.hpp"

namespace gfl
{
    template<typename T>
    class Array
    {
        protected:
            T * data;
            gfl::i32 size;

        public:
            GFL_HOST_DEVICE Array(T * data, i32 size) noexcept;
            GFL_HOST_DEVICE Array(StackAllocator & allocator, i32 size) noexcept;
            GFL_HOST_DEVICE T*  getData() const noexcept {return data;}
            GFL_HOST_DEVICE i32 getSize() const noexcept {return size;}
            GFL_HOST_DEVICE T * at(i32 index) const noexcept;
            GFL_HOST_DEVICE T * begin() const noexcept {return at(0);}
            GFL_HOST_DEVICE T * end() const noexcept {return at(size - 1) + 1;}
            GFL_HOST_DEVICE T & operator()(i32 const index) const noexcept {return *at(index);}
            GFL_HOST_DEVICE static void print(T const * begin, T const * end) noexcept;
            GFL_HOST_DEVICE void print() const noexcept {print(begin(),end());}
            GFL_HOST_DEVICE static u32 calcDataMemSize(i32 const size) noexcept {return size * sizeof(T);}
            GFL_HOST_DEVICE u32 calcDataMemSize() const noexcept {return calcDataMemSize(size);}
            GFL_HOST_DEVICE Array<T> & operator=(Array<T> & other) = delete;
            GFL_HOST_DEVICE Array<T> & operator=(Array<T> const & other) = delete;
            GFL_HOST_DEVICE Array<T> & operator=(Array<T> && other) noexcept;
    };

    template <typename T>
    GFL_HOST_DEVICE
    Array<T>::Array(T * const data, i32 const size) noexcept :
        data(data),
        size(size)
    {
        assert(data != nullptr);
        assert(size > 0);
    }

    template <typename T>
    GFL_HOST_DEVICE
    Array<T>::Array(StackAllocator & allocator, i32 const size) noexcept :
        Array<T>(allocator.allocate<T>(calcDataMemSize(size)), size)
    {}

    template<typename T>
    GFL_HOST_DEVICE
    T * Array<T>::at(i32 index) const noexcept
    {
        assert(index >= 0);
        assert(index < size);
        assert(size > 0);
        return data + index;
    }

    template<typename T>
    void Array<T>::print(T const * begin, T const * end) noexcept
    {
        for(T const * it = begin; it != end; it += 1)
        {
            printf(it != begin ? "," : "");
            printVal<T>(*it);
        }
    }

    template<typename T>
    GFL_HOST_DEVICE
    Array<T> & Array<T>::operator=(Array<T> && other) noexcept
    {
        if (this != &other)
        {
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
}