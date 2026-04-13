#pragma once

#include "Backend.hpp"
#include "FunQual.hpp"

namespace gfl {
template<typename C, typename P>
GFL_HOST_DEVICE
C filter(C const & c, P const p) {
  C out;
  for (auto const v : c) {
    if (p(v))
      out.insert(v);
  }
  return out;
}

template<typename C, typename T>
GFL_HOST_DEVICE
tuple<i32, i32> argmin(C const & c, T const & t) {
  i32 min = numeric_limits<i32>::max();
  i32 elt;
  for (auto const v : c) {
    auto const tv = t(v);
    min = (min < tv) ? min : tv;
    elt = (min < tv) ? elt : v;
  }
  return {elt, min};
}
} // namespace gfl