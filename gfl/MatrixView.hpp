#pragma once

#include "ArrayView.hpp"

namespace xuda
{
    template<typename T>
    class MatrixView : public ArrayView<T>
    {
    protected:
        i32 rows_;
        i32 cols_;

    public:
        XUDA_HOST_DEVICE MatrixView(T* data, i32 rows, i32 cols) noexcept;

        template<typename Allocator>
        XUDA_HOST_DEVICE MatrixView(Allocator& alloc, i32 rows, i32 cols) noexcept;

        XUDA_HOST_DEVICE i32 rows() const noexcept { return rows_; }
        XUDA_HOST_DEVICE i32 cols() const noexcept { return cols_; }

        XUDA_HOST_DEVICE static i32 dataMemSize(i32 rows, i32 cols) noexcept { return sizeof(T) * rows * cols; }

        XUDA_HOST_DEVICE T& at(i32 i, i32 j) const noexcept;
        XUDA_HOST_DEVICE T& operator()(i32 i, i32 j) const noexcept { return at(i, j); }

        XUDA_HOST_DEVICE ArrayView<T> row(i32 i) const noexcept;
        XUDA_HOST_DEVICE MatrixView<T> reshape(i32 newRows, i32 newCols) const noexcept;
    };

    template<typename T>
    XUDA_HOST_DEVICE
    MatrixView<T>::MatrixView(T* data, i32 rows, i32 cols) noexcept
        : ArrayView<T>(data, rows * cols), rows_(rows), cols_(cols)
    {
        assert(rows > 0);
        assert(cols > 0);
    }

    template<typename T>
    template<typename Allocator>
    XUDA_HOST_DEVICE
    MatrixView<T>::MatrixView(Allocator& alloc, i32 rows, i32 cols) noexcept
        : ArrayView<T>(alloc, rows * cols), rows_(rows), cols_(cols)
    {
        assert(rows > 0);
        assert(cols > 0);
    }

    template<typename T>
    XUDA_HOST_DEVICE
    T& MatrixView<T>::at(i32 i, i32 j) const noexcept
    {
        assert(i >= 0);
        assert(i < rows_);
        assert(j >= 0);
        assert(j < cols_);
        return ArrayView<T>::at(i * cols_ + j);  // Reuses ArrayView::at()
    }

    template<typename T>
    XUDA_HOST_DEVICE
    ArrayView<T> MatrixView<T>::row(i32 i) const noexcept
    {
        assert(i >= 0);
        assert(i < rows_);
        return ArrayView<T>(data_ + i * cols_, cols_);
    }

    template<typename T>
    XUDA_HOST_DEVICE
    MatrixView<T> MatrixView<T>::reshape(i32 newRows, i32 newCols) const noexcept
    {
        assert(newRows > 0);
        assert(newCols > 0);
        assert(newRows * newCols == size_);
        return MatrixView<T>(data_, newRows, newCols);
    }
}