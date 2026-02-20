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

        static constexpr i64 MaxBytes    = 8ll * 1024ll * 1024ll * 1024ll; // 8GB
        static constexpr i64 MaxElements = MaxBytes / scast<i64>(sizeof(T));
        static constexpr i32 GrowthFactor  = 4;
        static constexpr i32 DefaultSize = 1024;

        void reserve(i32 const capacity) noexcept
        {
            if (capacity > capacity_)
            {

                i32 const newCapacity = capacity * GrowthFactor;
                printf("CAPACITY = %d -> %d\n", capacity_, newCapacity);
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
        Vector(i32 const capacity) noexcept :
            VectorView<T>(capacity, vmReserve<T>(MaxElements))
        {
            vmCommit(data_, capacity);
        }

        ~Vector() noexcept
        {
            assert(data_ != nullptr);
            vmRelease(data_, MaxElements);
        }

        i32 resizeTo(i32 const size) noexcept
        {
            reserve(size);
            return VectorView<T>::resizeTo(size);
        }

        GFL_HOST_DEVICE
        i32 resizeBy(i32 const delta) noexcept
        {
            reserve(size_ + delta);
            return VectorView<T>::resizeBy(delta);
        }

        GFL_HOST_DEVICE
        void pushBack(ArrayView<T> const& elements) noexcept
        {
            reserve(size_ + elements.size());
            VectorView<T>::pushBack(elements);
        }

        GFL_HOST_DEVICE
        void pushBack(T const & value) noexcept
        {
            reserve(size_ + 1);
            VectorView<T>::pushBack(value);
        }
    };
}