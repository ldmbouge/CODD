#pragma once

#include <sys/mman.h>
#include <unistd.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "Types.hpp"
#include "MathUtils.hpp"
#include "DebugUtils.hpp"

namespace gfl
{

    inline
    i64 pageSize() noexcept
    {
        static i64 const size = scast<i64>(sysconf(_SC_PAGESIZE));
        return size;
    }

    // Heap

    template<typename T = u8>
    T* heapReserve(i64 const count) noexcept
    {
        assert(count > 0);
        void* memory = std::malloc(sizeof(T) * count);
        assert(memory != nullptr);
        return scast<T*>(memory);
    }

    template<typename T>
    T* heapRealloc(T* const ptr, i64 const count) noexcept
    {
        assert(ptr != nullptr);
        assert(count > 0);
        void* memory = std::realloc(ptr, sizeof(T) * count);
        assert(memory != nullptr);
        return scast<T*>(memory);
    }

    inline
    void heapRelease(void*& memory) noexcept
    {
        assert(memory != nullptr);
        std::free(memory);
        memory = nullptr;
    }

    template<typename T>
    void heapRelease(T*& memory) noexcept { heapRelease(rcast<void*&>(memory)); }

    // Virtual memory — reserve once, commit as needed, pointer never moves

    template<typename T>
    T* vmReserve(i64 const count) noexcept
    {
        assert(count > 0);
        i64 const bytes = roundUp<i64>(sizeof(T) * count, pageSize());
        void* p = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        assert(p != MAP_FAILED);
        return scast<T*>(p);
    }

    template<typename T>
    void vmCommit(T* const ptr, i64 const count) noexcept
    {
        assert(ptr != nullptr);
        assert(count > 0);
        i64 const bytes  = roundUp<i64>(sizeof(T) * count, pageSize());
        i32 const result = mprotect(ptr, bytes, PROT_READ | PROT_WRITE);
        assert(result == 0);
    }

    template<typename T>
    void vmRelease(T*& ptr, i64 const count) noexcept
    {
        assert(ptr != nullptr);
        assert(count > 0);
        i64 const bytes = roundUp<i64>(sizeof(T) * count, pageSize());
        munmap(ptr, bytes);
        ptr = nullptr;
    }

#ifdef __CUDACC__
    // CUDA

    template<typename T = u8>
    T* cudaReserveHost(i64 const count) noexcept
    {
        assert(count > 0);
        void* memory = nullptr;
        cudaError_t const status = cudaMallocHost(&memory, sizeof(T) * count);
        assert(status == cudaSuccess and memory != nullptr);
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseHost(void*& memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t const status = cudaFreeHost(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }

    template<typename T = u8>
    T* cudaReserveDevice(i64 const count) noexcept
    {
        assert(count > 0);
        void* memory = nullptr;
        cudaError_t const status = cudaMalloc(&memory, sizeof(T) * count);
        assert(status == cudaSuccess and memory != nullptr);
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseDevice(void*& memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t const status = cudaFree(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }

    template<typename T = u8>
    T* cudaReserveManaged(i64 const count) noexcept
    {
        assert(count > 0);
        void* memory = nullptr;
        cudaError_t const status = cudaMallocManaged(&memory, sizeof(T) * count);
        assert(status == cudaSuccess and memory != nullptr);
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseManaged(void*& memory) noexcept
    {
        assert(memory != nullptr);
        cudaError_t const status = cudaFree(memory);
        assert(status == cudaSuccess);
        memory = nullptr;
    }
#endif
}