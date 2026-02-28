#pragma once

#ifdef __CUDACC__
    #include <cuda/std/optional>
    #include <cuda/std/tuple>
    #include <cuda/std/limits>
#else
    #include <optional>
    #include <tuple>
    #include <limits>
#endif

namespace gfl
{
#ifdef __CUDACC__
    namespace backend = cuda::std;
#else
    namespace backend = std;
#endif

    constexpr inline auto nullopt = backend::nullopt;


    // Optional
    template<typename T>
    using optional = backend::optional<T>;

    // Tuple
    template<typename... Types>
    using tuple = backend::tuple<Types...>;
    using backend::make_tuple;
    using backend::tie;
    using backend::get;

    // Numeric Limits
    template<typename T>
    using numeric_limits = backend::numeric_limits<T>;

    GFL_HOST_DEVICE inline
    void abort()
    {
#ifdef __CUDA_ARCH__
        __trap();
#else
        std::abort();
#endif
    }
}