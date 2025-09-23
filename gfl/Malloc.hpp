#pragma once

#include <cassert>
#include <cstdlib>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "Types.hpp"

namespace gfl
{
    template<typename T>
    T * mallocStd(i64 const size) noexcept
    {
        void * memory = std::malloc(size);
        assert(memory != nullptr);
        return static_cast<T *>(memory);
    }

    inline
    void freeStd(void * memory) noexcept
    {
        assert(memory != nullptr);
        free(memory);
        memory = nullptr;
    }

#ifdef __CUDACC__
    template<typename T>
    T * mallocHost(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocHost(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return static_cast<T *>(memory);
    }

    template<typename T>
    T * mallocDevice(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMalloc(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return static_cast<T *>(memory);
    }

    template<typename T>
    T * mallocManaged(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocManaged(&memory, size);
        assert(status == cudaSuccess);
        assert(memory != nullptr);
        return static_cast<T *>(memory);
    }

    inline
    void freeHost(void * memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t status = cudaFreeHost(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }

    inline
    void freeDevice(void * memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t status = cudaFree(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }

    inline
    void freeManaged(void * memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t status = cudaFree(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }
#endif
}