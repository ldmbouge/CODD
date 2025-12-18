#pragma once

#include "Array.hpp"

namespace gfl
{
    template<typename T>
    class Vector
    {
        protected:
            T * data;
            i64 size;
            i64 capacity;

        public:
            GFL_HOST_DEVICE Vector(T * data, i64 capacity) noexcept;
            GFL_HOST_DEVICE Vector(StackAllocator & allocator, i64 capacity) noexcept;
            GFL_HOST_DEVICE T * getData() const noexcept {return data;}
            GFL_HOST_DEVICE i64 getSize() const noexcept {return size;}
            GFL_HOST_DEVICE i64 getCapacity() const noexcept {return capacity;}
            GFL_HOST_DEVICE T * at(i32 index) const noexcept;
            GFL_HOST_DEVICE T * begin() const noexcept {return at(0);};
            GFL_HOST_DEVICE T * end() const noexcept {return at(size - 1) + 1;}
            GFL_HOST_DEVICE T & operator[](i32 const index) const noexcept {return *at(index);}
            GFL_HOST_DEVICE void pushBack(T t) noexcept;
            template <typename... Args>
            GFL_HOST_DEVICE void emplaceBack(Args&&... args) noexcept;
            GFL_HOST_DEVICE void popBack() noexcept {size = size > 0 ? size - 1 : 0;}
            GFL_HOST_DEVICE void resize(i32 size) noexcept;
            GFL_HOST_DEVICE void clear() noexcept {resize(0);}
            GFL_HOST_DEVICE void print() const noexcept {Array<T>::print(begin(), end());}
            GFL_HOST_DEVICE static u32 calcDataMemSize(i32 capacity) noexcept {return capacity * sizeof(T);}
            GFL_HOST_DEVICE u32 calcDataMemSize() const noexcept {return calcDataMemSize(capacity);}
            GFL_HOST_DEVICE Vector<T> & operator=(Vector<T> & other) = delete;
            GFL_HOST_DEVICE Vector<T> & operator=(Vector<T> const & other) = delete;
            GFL_HOST_DEVICE Vector<T> & operator=(Vector<T> && other) noexcept;


    };

    template<typename T>
    GFL_HOST_DEVICE
    Vector<T>::Vector( T * const data, i32 const capacity) noexcept :
            data(data),
            size(0),
            capacity(capacity)
    {
        assert(data != nullptr);
        assert(capacity > 0);
    }

    template<typename T>
    GFL_HOST_DEVICE
    Vector<T>::Vector(StackAllocator & allocator, i32 const capacity) noexcept :
        Vector<T>(allocator.allocate<T>(calcDataMemSize(capacity)), capacity)
    {}

    template<typename T>
    GFL_HOST_DEVICE
    T * Vector<T>::at( i32 const index) const noexcept
    {
        assert(index >= 0);
        assert(index < size);
        assert(size > 0);
        return data + index;
    }

    template<typename T>
    GFL_HOST_DEVICE
    void Vector<T>::pushBack(T t) noexcept
    {
        assert(size < capacity);
        size += 1;
        *at(size - 1) = t;
    }

    template<typename T>
    template<typename... Args>
    GFL_HOST_DEVICE
    void Vector<T>::emplaceBack(Args&&... args) noexcept
    {
        assert(size < capacity);
        size += 1;
        new (at(size - 1)) T(forward<Args>(args)...);
    }

    template<typename T>
    GFL_HOST_DEVICE
    void Vector<T>::resize(i32 const size) noexcept
    {
        assert(size < capacity);
        assert(size >= 0);
        this->size = size;
    }

    template<typename T>
    GFL_HOST_DEVICE
    Vector<T> & Vector<T>::operator=(Vector<T> && other) noexcept
    {
        if (this != &other)
        {
            data = other.data;
            size = other.size;
            capacity = other.capacity;
            other.data = nullptr;
            other.size = 0;
            other.capacity = 0;
        }
        return *this;
    }
}
