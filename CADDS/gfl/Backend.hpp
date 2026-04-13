#pragma once

#ifdef __CUDACC__
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#else
#include <limits>
#include <optional>
#include <tuple>
#endif

namespace gfl {
#ifdef __CUDACC__
namespace backend = cuda::std;
#else
namespace backend = std;
#endif

constexpr inline auto nullopt = backend::nullopt;

template<typename T>
using optional = backend::optional<T>;

template<typename... Types>
using tuple = backend::tuple<Types...>;
using backend::make_tuple;

template<typename T>
using numeric_limits = backend::numeric_limits<T>;
}