#pragma once

#include <cassert>

#include "Types.hpp"

namespace gfl
{
    class Interval
    {
        i32 const begin_;
        i32 const end_;

        public:
            bool isValid() const noexcept {return begin_ <= end_;}
            Interval(i32 begin, i32 end) : begin_(begin), end_(end)
            { assert(isValid());}
            i32 begin() const noexcept {return begin_;}
            i32 end() const noexcept {return end_;}
            i32 size() const noexcept {return end_ - begin_;}
    };
}