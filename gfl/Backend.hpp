#pragma once

#ifdef __NVCC__
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <cuda/std/bit>
#include <cuda/std/initializer_list>
#include <cuda/std/limits>
#else
#include <optional>
#include <tuple>
#include <bit>
#include <initializer_list>
#include <limits>
#endif

#include "Common.hpp"

namespace gfl
{
#ifdef __CUDACC__
    namespace backend = cuda::std;
#else
    namespace backend = std;
#endif

    // Optional
    template<typename  T>
    using optional = backend::optional<T>;
    constexpr static auto nullopt = backend::nullopt;

    // Tuple
    template <typename... Types>
    using tuple = backend::tuple<Types...>;

    template <typename... Types>
    GFL_HOST_DEVICE
    backend::tuple<Types...> make_tuple(Types&&... args)
    {
        return backend::make_tuple(backend::forward<Types>(args)...);
    }

    // Bit
    template <typename T>
    GFL_HOST_DEVICE
    T rotl(T x, int s)
    {
        return backend::rotl(x, s);
    }

    // Initializer List
    template <typename T>
    using initializer_list = backend::initializer_list<T>;

    // Numeric Limits
    template <typename T>
    using numeric_limits = backend::numeric_limits<T>;
}
