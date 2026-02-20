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

        static constexpr i64 MaxElements   = 256ll * 1024ll * 1024ll; // 256M elements
        static constexpr i32 GrowthFactor  = 4;
        static constexpr i32 DefaultSize = 1024;

        void reserve(i32 const capacity) noexcept
        {
            if (capacity > capacity_)
            {
                i32 const newCapacity = capacity * GrowthFactor;
                data_ = heapRealloc(data_, newCapacity);
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
            VectorView<T>(capacity, heapReserve<T>(capacity))
        {}

        ~Vector() noexcept
        {
            assert(data_ != nullptr);
            heapRelease(data_);
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