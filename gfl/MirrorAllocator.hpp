#pragma once

#include "Malloc.hpp"
#include "StackAllocator.hpp"

#ifdef __CUDACC__
namespace gfl
{
    template<typename T>
    class MirrorPtr
    {
        public:
            T * h;
            T * d;

        public:
            constexpr MirrorPtr(T * const hostPtr = nullptr, T * const devicePtr = nullptr) noexcept : h(hostPtr), d(devicePtr) {}
            GFL_HOST_DEVICE
            constexpr T * operator->() const noexcept;
            template<typename U = T>
            GFL_HOST_DEVICE
            typename std::enable_if<not std::is_same<U,void>::value, U&>::type
            operator[](i32 const idx) const noexcept;

    };

    class MirrorAllocator
    {
        protected:
            StackAllocator hostAllocator;
            StackAllocator deviceAllocator;

        public:
            inline MirrorAllocator(i64 size) noexcept;
            inline ~MirrorAllocator() noexcept;
            template <typename T>
            MirrorPtr<T> allocate(i64 size = sizeof(T), i32 align = alignof(T)) noexcept;
            template <typename T>
            MirrorPtr<T> allocateArray(i64 size, i32 align = alignof(T)) noexcept;
            inline void clear() noexcept;
            inline MirrorPtr<void> getMem() const noexcept;
            inline MirrorPtr<void> getFreeMem() const noexcept;
            i64 calcFreeMemSize() const noexcept {return hostAllocator.calcFreeMemSize();}
            i64 calcUsedMemSize() const noexcept {return hostAllocator.calcUsedMemSize();}
            i64 calcTotalMemSize() const noexcept {return hostAllocator.calcTotalMemSize();}
    };

    template<typename T>
    GFL_HOST_DEVICE constexpr
    T *  MirrorPtr<T>::operator->() const noexcept
    {
        T * result;
#ifdef __CUDA_ARCH__
        result = d;
#else
        result = h;
#endif
        return result;
    }

    template<typename T>
    template<typename U>
    GFL_HOST_DEVICE
    typename std::enable_if<not std::is_same<U, void>::value, U&>::type
    MirrorPtr<T>::operator[](i32 const idx) const noexcept
    {
        return operator->()[idx];
    }

    inline
    MirrorAllocator::MirrorAllocator(i64 const size) noexcept:
        hostAllocator(mallocHost<void>(size), size),
        deviceAllocator(mallocDevice<void>(size), size)
    {}

    inline
    MirrorAllocator::~MirrorAllocator() noexcept
    {
        freeHost(hostAllocator.getMem());
        freeDevice(deviceAllocator.getMem());
    }

    template<typename T>
    MirrorPtr<T> MirrorAllocator::allocate(i64 const size, i32 const align) noexcept
    {
        T * const hostPtr = hostAllocator.allocate<T>(size, align);
        T * const devicePtr = deviceAllocator.allocate<T>(size, align);
        return MirrorPtr<T>(hostPtr, devicePtr);
    }

    template<typename T>
    MirrorPtr<T> MirrorAllocator::allocateArray(i64 const size, i32 const align) noexcept
    {
        T * const hostPtr = hostAllocator.allocateArray<T>(size, align);
        T * const devicePtr = deviceAllocator.allocateArray<T>(size, align);
        return MirrorPtr<T>(hostPtr, devicePtr);
    }

    inline
    void MirrorAllocator::clear() noexcept
    {
        hostAllocator.clear();
        deviceAllocator.clear();
    }

    inline
    MirrorPtr<void> MirrorAllocator::getMem() const noexcept
    {
        void * hostPtr = hostAllocator.getMem();
        void * devicePtr = deviceAllocator.getMem();
        return MirrorPtr<void>(hostPtr, devicePtr);
    }

    inline
    MirrorPtr<void> MirrorAllocator::getFreeMem() const noexcept
    {
        void * hostPtr = hostAllocator.getFreeMem();
        void * devicePtr = deviceAllocator.getFreeMem();
        return MirrorPtr<void>(hostPtr, devicePtr);
    }
}
#endif