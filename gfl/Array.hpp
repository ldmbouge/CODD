#pragma once

#include "ArrayView.hpp"
#include "Memory.hpp"

namespace gfl
{
    template<typename T>
    class Array : public ArrayView<T>
    {
        using ArrayView<T>::size_;
        using ArrayView<T>::data_;

    public:
        Array() = delete;

        Array(Array const&) = delete;
        Array& operator=(Array const&) = delete;

        Array(Array && other) = delete;
        Array& operator=(Array && other) = delete;

        explicit
        Array(i64 const size) noexcept :
            ArrayView<T>(size, vmReserve<T>(size))
        {
            vmCommit(data_, size);
        }

        ~Array() noexcept
        {
            assert(data_ != nullptr);
            vmRelease(data_, size_);
        }
    };
}