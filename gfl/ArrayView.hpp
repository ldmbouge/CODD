#pragma once

#include <cassert>
#include <cstdio>
#include <functional>
#include <type_traits>

#include "FunQual.hpp"
#include "Types.hpp"

namespace gfl
{
    template<typename T>
    class ArrayView
    {
    protected:
        T* data_{nullptr};
        alignas(8) i64 size_{0};  // Aligned for atomic operations in VectorView

    public:
        ArrayView() noexcept = default;
        ~ArrayView() noexcept = default;

        ArrayView(ArrayView const&) noexcept = default;
        ArrayView& operator=(ArrayView const&) noexcept = default;

        ArrayView(ArrayView&&) = default;
        ArrayView& operator=(ArrayView&&) = default;

        GFL_HOST_DEVICE
        ArrayView(i64 const size, T * const data) noexcept :
            data_(data),
            size_(size)
        {
            assert(data != nullptr);
            assert(size >= 0);  // Allow size = 0 for VectorBase (starts empty)
        }

        template<typename Allocator>
        GFL_HOST_DEVICE
        ArrayView(i64 const size, Allocator & alloc) noexcept :
            ArrayView(size, alloc.template allocate<T>(size)) {}

        GFL_HOST_DEVICE
        ArrayView(T * begin, T * end) noexcept :
            data_(begin),
            size_(scast<i64>(end - begin))
        {
            assert(begin != nullptr);
            assert(end != nullptr);
            assert(begin <= end);
        }

        GFL_HOST_DEVICE
        T * data() const noexcept { return data_; }

        GFL_HOST_DEVICE
        i64 size() const noexcept { return size_; }

        GFL_HOST_DEVICE
        bool empty() const noexcept { return size_ == 0; } // Usefull for vectorview

#ifdef __CUDACC__
        // Needed for CUDA to access size at running time vs launch time.
        GFL_HOST_DEVICE
        i64 const * sizePtr() const noexcept { return &size_; }
#endif

        GFL_HOST_DEVICE static
        i64 dataMemSize(i64 const size) noexcept { return sizeof(T) * size; }

        GFL_HOST_DEVICE
        i64 dataMemSize() const noexcept { return dataMemSize(size_); }

        GFL_HOST_DEVICE
        T & at(i64 const idx) const noexcept
        {
            assert(idx >= 0);
            assert(idx < size_);
            return data_[idx];
        }

        GFL_HOST_DEVICE
        T * begin() const noexcept { return data_; }

        GFL_HOST_DEVICE
        T * end() const noexcept { return data_ + size_; }

        GFL_HOST_DEVICE
        T & front() const noexcept
        {
            assert(size_ > 0);
            return data_[0];
        }

        GFL_HOST_DEVICE
        T & back() const noexcept
        {
            assert(size_ > 0);
            return data_[size_ - 1];
        }

        GFL_HOST_DEVICE
        T & operator[](i64 const idx) const noexcept
        { return at(idx); }

        GFL_HOST_DEVICE static
        void swap(ArrayView & a, ArrayView & b) noexcept
        {
            T * const tmpData = a.data_;
            a.data_ = b.data_;
            b.data_ = tmpData;

            i64 tmpSize = a.size_;
            a.size_ = b.size_;
            b.size_ = tmpSize;
        }

        GFL_HOST_DEVICE
        void swap(ArrayView & other) noexcept { swap(*this, other); }


        GFL_HOST_DEVICE
        ArrayView slice(i64 const begin, i64 const end) const noexcept
        {
            assert(0 <= begin);
            assert(begin <= end);
            assert(end   <= size_);
            return ArrayView<T>(end - begin, data_ + begin);
        }

        GFL_HOST_DEVICE
        ArrayView slice(i64 const count) const noexcept
        {
            assert(count != 0);
            if (count >= 0) return slice(0,count);
            else return slice(size_ + count, size_);
        }
       friend std::ostream& operator<<(std::ostream& os,const ArrayView& av) {
            os << "[";
            for(T const * it = av.begin(); it != av.end(); ++it)
               os << *it;
            return os << "]";
       }
       
        GFL_HOST_DEVICE static
        void print(T const * begin, T const * end, char const * fmt = "%d") noexcept
        {
            assert(begin != nullptr);
            assert(end != nullptr);
            assert(begin <= end);

            bool comma = false;
            for (T const * it = begin; it != end; ++it)
            {
                printf(comma ? "," : "");
                printf(fmt, *it);
                comma = true;
            }
        }

        GFL_HOST_DEVICE static
        void print(T const * begin, i64 count, char const* fmt = "%d") noexcept
        { print(begin, begin + count, fmt); }

        GFL_HOST_DEVICE
        void print(char const* fmt = "%d") const noexcept
        { print(begin(), end(), fmt); }

        template<typename Fn>
        GFL_HOST_DEVICE static
        void print(T const * begin, T const * end, Fn toString) noexcept
        {
            assert(begin != nullptr);
            assert(end != nullptr);
            assert(begin <= end);
            bool comma = false;
            for (T const * it = begin; it != end; ++it)
            {
                printf(comma ? "," : "");
                printf("%s", toString(*it));
                comma = true;
            }
        }

        template<typename Fn>
        GFL_HOST_DEVICE void print(Fn toString) const noexcept
        { print(begin(), end(), toString); }
    };
}
