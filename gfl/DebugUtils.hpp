#pragma once
#include <cstdio>
#include <cstdlib>

namespace gfl
{
    inline
    void abortWithMsg(char const * const msg) noexcept
    {
        std::printf("%s\n", msg);
        std::fflush(stdout);
        std::abort();
    }

    inline
    void checkOrAbort(bool const condition, char const * const msg)
    {
        if (not condition) abortWithMsg(msg);
    }
}
