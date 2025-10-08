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
        if (memory == nullptr)
        {
           std::cerr << "malloc failed" << std::endl;
           abort();
        }
        return static_cast<T *>(memory);
    }

    inline
    void freeStd(void * memory) noexcept
    {
        if (memory != nullptr)
        {
            free(memory);
        }
        else
        {
            std::cerr << "free called on nullptr" << std::endl;
            abort();
        }
    }

#ifdef __CUDACC__
    template<typename T>
    T * mallocHost(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocHost(&memory, size);
        if (status != cudaSuccess or memory == nullptr)
        {
            std::cerr << "cudaMallocHost failed" << std::endl;
            abort();
        }
        return static_cast<T *>(memory);
    }

    template<typename T>
    T * mallocDevice(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMalloc(&memory, size);
        if (status != cudaSuccess or memory == nullptr)
        {
            std::cerr << "cudaMalloc failed" << std::endl;
            abort();
        }
        return static_cast<T *>(memory);
    }

    template<typename T = void>
    T * mallocManaged(i64 const size) noexcept
    {
        void * memory = nullptr;
        cudaError_t status = cudaMallocManaged(&memory, size);
        if (status != cudaSuccess or memory == nullptr)
        {
            std::cerr << "cudaMallocManaged failed" << std::endl;
            abort();
        }
        return static_cast<T *>(memory);
    }

    inline
    void freeHost(void * memory) noexcept
    {
        if (memory != nullptr)
        {
            cudaError_t status = cudaFreeHost(memory);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaFreeHost failed" << std::endl;
                abort();
            }
        }
        else
        {
            std::cerr << "cudaFreeHost called on nullptr" << std::endl;
            abort();
        }

    }

    inline
    void freeDevice(void * memory) noexcept
    {
        if (memory != nullptr)
        {
            cudaError_t status = cudaFree(memory);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaFree failed" << std::endl;
                abort();
            }
        }
        else
        {
            std::cerr << "cudaFree called on nullptr" << std::endl;
            abort();
        }
    }

    inline
    void freeManaged(void * memory) noexcept
    {
        if (memory != nullptr)
        {
            cudaError_t status = cudaFree(memory);
            if (status != cudaSuccess)
            {
                std::cerr << "cudaFree failed" << std::endl;
                abort();
            }
        }
        else
        {
            std::cerr << "cudaFree called on nullptr" << std::endl;
            abort();
        }
    }
#endif
}