#pragma once

#include "VectorView.hpp"
#include "Memory.hpp"
#include <cassert>

namespace gfl
{
    template<typename T>
    class Vector : public VectorView<T>
    {
        using VectorView<T>::size_;
        using VectorView<T>::data_;
        using VectorView<T>::capacity_;

        static constexpr i64 MaxBytes    = 64ll * 1024ll * 1024ll * 1024ll; // 64GB
        static constexpr i64 MaxElements = MaxBytes / scast<i64>(sizeof(T));
        static constexpr i64 GrowthFactor  = 4;
        static constexpr i64 DefaultSize = 4096;

        void reserve(i64 const capacity) noexcept
        {
            if (capacity > capacity_)
            {

                i64 const newCapacity = capacity * GrowthFactor;
                checkOrAbort(newCapacity <= MaxElements, "Vector exceeded MaxElements");
                vmCommit(data_, newCapacity);   // pointer stays the same, no copy
                capacity_ = newCapacity;
            }
        }

    public:
        Vector() = delete;

        Vector(Vector const&) = delete;
        Vector& operator=(Vector const&) = delete;

        Vector(Vector&&) = delete;
        Vector& operator=(Vector&&) = delete;

        explicit
        Vector(i64 const capacity) noexcept :
            VectorView<T>(capacity, vmReserve<T>(MaxElements))
        {
            vmCommit(data_, capacity);
        }

        ~Vector() noexcept
        {
            assert(data_ != nullptr);
            vmRelease(data_, MaxElements);
        }

        i64 resizeTo(i64 const size) noexcept
        {
            reserve(size);
            return VectorView<T>::resizeTo(size);
        }

        i64 resizeBy(i64 const delta) noexcept
        {
            reserve(size_ + delta);
            return VectorView<T>::resizeBy(delta);
        }

        void pushBack(ArrayView<T> const& elements) noexcept
        {
            reserve(size_ + elements.size());
            VectorView<T>::pushBack(elements);
        }

        void pushBack(T const & value) noexcept
        {
            reserve(size_ + 1);
            VectorView<T>::pushBack(&value);
        }
    };
}