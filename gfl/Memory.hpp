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

    GFL_HOST_DEVICE inline
    void printMemSize(i64 const size) noexcept
    {
        char const* const units[] = {" B", "KB", "MB", "GB"};
        auto uIdx = 0;
        auto dSize = static_cast<double>(size);
        while (dSize >= 1024 and uIdx < 3)
        {
            dSize /= 1024.0;
            uIdx += 1;
        }
        printf("%7.2f %s", dSize, units[uIdx]);
    }


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
        checkOrAbort(count > 0, "heapReserve failed: count must be > 0");
        void* memory = std::malloc(sizeof(T) * count);
        checkOrAbort(memory != nullptr, "heapReserve failed: null pointer returned");
        return scast<T*>(memory);
    }

    inline
    void heapRelease(void*& memory) noexcept
    {
        checkOrAbort(memory != nullptr, "heapRelease failed: null pointer");
        std::free(memory);
        memory = nullptr;
    }

    template<typename T>
    void heapRelease(T*& memory) noexcept { heapRelease(rcast<void*&>(memory)); }

    // Virtual memory — reserve once, commit as needed, pointer never moves

    template<typename T>
    T* vmReserve(i64 const count) noexcept
    {
        checkOrAbort(count > 0, "vmReserve failed: count must be > 0");
        void* p = mmap(nullptr, sizeof(T) * count,PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        checkOrAbort(p != MAP_FAILED, "vmReserve failed: mmap returned MAP_FAILED");
        return scast<T*>(p);
    }

    template<typename T>
    void vmCommit(T* const ptr, i64 const count) noexcept
    {
        checkOrAbort(ptr != nullptr, "vmCommit failed: null pointer");
        checkOrAbort(count > 0, "vmCommit failed: count must be > 0");
        i64 const bytes  = roundUp<i64>(sizeof(T) * count , pageSize());
        i32 const result = mprotect(ptr, bytes, PROT_READ | PROT_WRITE);
        checkOrAbort(result == 0, "vmCommit failed: mprotect failed");
    }

    template<typename T>
    void vmRelease(T*& ptr, i64 const count) noexcept
    {
        checkOrAbort(ptr != nullptr, "vmRelease failed: null pointer");
        checkOrAbort(count > 0, "vmRelease failed: count must be > 0");
        munmap(ptr, sizeof(T) * count);
        ptr = nullptr;
    }

#ifdef __CUDACC__
    // CUDA

    template<typename T = u8>
    T* cudaReserveHost(i64 const count) noexcept
    {
        checkOrAbort(count > 0, "cudaReserveHost failed: count must be > 0");
        void* memory = nullptr;
        cudaError_t const status = cudaMallocHost(&memory, sizeof(T) * count);
        checkOrAbort(status == cudaSuccess and memory != nullptr, "cudaReserveHost failed");
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseHost(void*& memory) noexcept
    {
        checkOrAbort(memory != nullptr, "cudaReleaseHost failed: null pointer");
        cudaError_t const status = cudaFreeHost(memory);
        checkOrAbort(status == cudaSuccess, "cudaReleaseHost failed");
        memory = nullptr;
    }

    template<typename T = u8>
    T* cudaReserveDevice(i64 const count) noexcept
    {
        checkOrAbort(count > 0, "cudaReserveDevice failed: count must be > 0");
        void* memory = nullptr;
        cudaError_t const status = cudaMalloc(&memory, sizeof(T) * count);
        checkOrAbort(status == cudaSuccess and memory != nullptr, "cudaReserveDevice failed");
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseDevice(void*& memory) noexcept
    {
        checkOrAbort(memory != nullptr, "cudaReleaseDevice failed: null pointer");
        cudaError_t const status = cudaFree(memory);
        checkOrAbort(status == cudaSuccess, "cudaReleaseDevice failed");
        memory = nullptr;
    }

    template<typename T = u8>
    T* cudaReserveManaged(i64 const count) noexcept
    {
        checkOrAbort(count > 0, "cudaReserveManaged failed: count must be > 0");
        void* memory = nullptr;
        cudaError_t const status = cudaMallocManaged(&memory, sizeof(T) * count);
        checkOrAbort(status == cudaSuccess and memory != nullptr, "cudaReserveManaged failed");
        return scast<T*>(memory);
    }

    inline
    void cudaReleaseManaged(void*& memory) noexcept
    {
        checkOrAbort(memory != nullptr, "cudaReleaseManaged failed: null pointer");
        cudaError_t const status = cudaFree(memory);
        checkOrAbort(status == cudaSuccess, "cudaReleaseManaged failed");
        memory = nullptr;
    }
#endif
}