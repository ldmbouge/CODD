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

        i32 capacity_{0};

    public:
        VectorView() noexcept = default;

        VectorView(VectorView const &) noexcept = default;
        VectorView& operator=(VectorView const &) noexcept = default;

        VectorView(VectorView&&) = default;
        VectorView& operator=(VectorView&&) = default;

        GFL_HOST_DEVICE
        VectorView(i32 const capacity, T * data) noexcept :
            ArrayView<T>(0,data),
            capacity_(capacity)
        {
            assert(data != nullptr);
            assert(capacity > 0);
        }

        template<typename Allocator>
        GFL_HOST_DEVICE
        VectorView(i32 const capacity, Allocator & alloc) noexcept :
            ArrayView<T>(0, alloc.template allocate<T>(capacity)),  // Start with size=0
            capacity_(capacity)
        {
            assert(capacity > 0);
            assert(data_ != nullptr);
        }

        GFL_HOST_DEVICE
        i32 capacity() const noexcept { return capacity_; }

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
            i32 const tmpCapacity = a.capacity_;
            a.capacity_ = b.capacity_;
            b.capacity_ = tmpCapacity;
        }

        GFL_HOST_DEVICE
        void swap(VectorView & other) noexcept { swap(*this, other); }

        GFL_HOST_DEVICE
        i32 resizeTo(i32 const size) noexcept
        {
            assert(size >= 0);
            assert(size <= capacity_);
            i32 const oldSize = size_;
            size_ = size;
            return oldSize;
        }

        GFL_HOST_DEVICE
        i32 resizeBy(i32 const delta) noexcept
        {
            i32 const oldSize = size_;
            assert(oldSize + delta >= 0);
            assert(oldSize + delta <= capacity_);
            size_ += delta;
            return oldSize;
        }

        GFL_HOST_DEVICE
        i32 resizeByAtomic(i32 const delta) noexcept
        {
#ifdef __CUDA_ARCH__
            i32 const oldSize = atomicAdd(&size_, delta);
#else
            std::atomic_ref<i32> atomicSize(size_);
            i32 const oldSize = atomicSize.fetch_add(delta, std::memory_order_relaxed);
#endif
            assert(oldSize + delta >= 0);
            assert(oldSize + delta <= capacity_);
            return oldSize;
        }

        GFL_HOST_DEVICE
        void pushBack(ArrayView<T> const & elements) noexcept
        {
            i32 const oldSize = resizeBy(elements.size());
            std::memcpy(&at(oldSize), elements.data(), elements.dataMemSize());
        }

        GFL_HOST_DEVICE
        void pushBack(T const & value) noexcept
        {
            i32 const oldSize = resizeBy(1);
            std::memcpy(&at(oldSize), &value, sizeof(T));
        }

#ifdef __CUDACC__
        void pushBackGpu(T const & value) noexcept
        {
            i32 const oldSize = resizeBy(1);
            cudaMemcpy(&at(oldSize), &value, sizeof(T), cudaMemcpyHostToDevice);
        }
#endif

        GFL_HOST_DEVICE
        void pushBackAtomic(ArrayView<T> const & items) noexcept
        {
            i32 const oldSize = resizeByAtomic(items.size());
            std::memcpy(&at(oldSize), items.data(), items.dataMemSize());
        }

        GFL_HOST_DEVICE
        void pushBackAtomic(T const & value) noexcept {{pushBackAtomic(ArrayView<T>(&value, 1));}}

        GFL_HOST_DEVICE
        ArrayView<T> popBack(i32 const count) noexcept
        {
            i32 const oldSize = resizeBy(-count);
            ArrayView<T> items(data_ + size_, data_ + oldSize);
            return items;
        }

        GFL_HOST_DEVICE
        T & popBack() noexcept { return popBack(1).front(); }

        GFL_HOST_DEVICE
        ArrayView<T> popBackAtomic(i32 const count) noexcept
        {
            i32 const oldSize = resizeByAtomic(-count);
            ArrayView<T> items(data_ + size_, data_ + oldSize);
            return items;
        }

        GFL_HOST_DEVICE
        T & popBackAtomic() noexcept { return popBackAtomic(1).front(); }
    };
}
