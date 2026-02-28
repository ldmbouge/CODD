#pragma once

#include <cstring>
#include <atomic>

#include "ArrayView.hpp"

namespace gfl
{
    template<typename T>
    class VectorView : public ArrayView<T>
    {
    public:
        using ArrayView<T>::at;

    protected:
        using ArrayView<T>::size_;
        using ArrayView<T>::data_;

        i64 capacity_{0};

    public:
        VectorView() noexcept = default;

        VectorView(VectorView const &) noexcept = default;
        VectorView& operator=(VectorView const &) noexcept = default;

        VectorView(VectorView&&) = default;
        VectorView& operator=(VectorView&&) = default;

        GFL_HOST_DEVICE
        VectorView(i64 const capacity, T * data) noexcept :
            ArrayView<T>(scast<i64>(0),data),
            capacity_(capacity)
        {
            assert(data != nullptr);
            assert(capacity > 0);
        }

        template<typename Allocator>
        GFL_HOST_DEVICE
        VectorView(i64 const capacity, Allocator & alloc) noexcept :
            ArrayView<T>(scast<i64>(0), alloc.template allocate<T>(capacity)),  // Start with size=0
            capacity_(capacity)
        {
            assert(capacity > 0);
            assert(data_ != nullptr);
        }

        GFL_HOST_DEVICE
        i64 capacity() const noexcept { return capacity_; }

        GFL_HOST_DEVICE
        bool empty() const noexcept { return size_ == 0; }

        GFL_HOST_DEVICE
        void clear() noexcept { size_ = 0; }

        GFL_HOST_DEVICE static
        void swap(VectorView & a, VectorView & b) noexcept
        {
            // Swap ArrayView members (data_ and size_)
            ArrayView<T>::swap(a, b);

            // Swap capacity_
            i64 const tmpCapacity = a.capacity_;
            a.capacity_ = b.capacity_;
            b.capacity_ = tmpCapacity;
        }

        GFL_HOST_DEVICE
        void swap(VectorView & other) noexcept { swap(*this, other); }

        GFL_HOST_DEVICE
        i64 resizeTo(i64 const size) noexcept
        {
            assert(size >= 0);
            assert(size <= capacity_);
            i64 const oldSize = size_;
            size_ = size;
            return oldSize;
        }

        GFL_HOST_DEVICE
        i64 resizeBy(i64 const delta) noexcept
        {
            i64 const oldSize = size_;
            assert(oldSize + delta >= 0);
            assert(oldSize + delta <= capacity_);
            size_ += delta;
            return oldSize;
        }

        GFL_HOST_DEVICE
        i64 resizeByAtomic(i64 const delta) noexcept
        {
#ifdef __CUDA_ARCH__
            i64 const oldSize = atomicAdd(rcast<llu*>(&size_), scast<llu>(delta));
#else
            std::atomic_ref<i64> atomicSize(size_);
            i64 const oldSize = atomicSize.fetch_add(delta, std::memory_order_relaxed);
#endif
            assert(oldSize + delta >= 0);
            assert(oldSize + delta <= capacity_);
            return oldSize;
        }

        GFL_HOST_DEVICE
        void pushBack(ArrayView<T> const & elements) noexcept
        {
            i64 const oldSize = resizeBy(elements.size());
            std::memcpy(&at(oldSize), elements.data(), elements.dataMemSize());
        }

        void pushBack(std::vector<T> const & elements) noexcept
        {
            i64 const oldSize = resizeBy(elements.size());
            std::memcpy(&at(oldSize), elements.data(), sizeof(T) * elements.size());
        }

        GFL_HOST_DEVICE
        void pushBack(T const * value) noexcept
        {
            i64 const oldSize = resizeBy(1);
            std::memcpy(&at(oldSize), value, sizeof(T));
        }

#ifdef __CUDACC__
        void pushBackGpuAsync(T const * values, i64 const count = 1) noexcept
        {
            i64 const oldSize = resizeBy(count);
            cudaMemcpyAsync(&at(oldSize), values, sizeof(T) * count, cudaMemcpyHostToDevice);
        }
#endif

        GFL_HOST_DEVICE
        void pushBackAtomic(ArrayView<T> const & items) noexcept
        {
            i64 const oldSize = resizeByAtomic(items.size());
            std::memcpy(&at(oldSize), items.data(), items.dataMemSize());
        }

        GFL_HOST_DEVICE
        void pushBackAtomic(T const & value) noexcept {{pushBackAtomic(ArrayView<T>(&value, 1));}}

        GFL_HOST_DEVICE
        ArrayView<T> popBack(i64 const count) noexcept
        {
            i64 const oldSize = resizeBy(-count);
            ArrayView<T> items(data_ + size_, data_ + oldSize);
            return items;
        }

        GFL_HOST_DEVICE
        T & popBack() noexcept { return popBack(1).front(); }

        GFL_HOST_DEVICE
        ArrayView<T> popBackAtomic(i64 const count) noexcept
        {
            i64 const oldSize = resizeByAtomic(-count);
            ArrayView<T> items(data_ + size_, data_ + oldSize);
            return items;
        }

        GFL_HOST_DEVICE
        T & popBackAtomic() noexcept { return popBackAtomic(1).front(); }
    };
}
