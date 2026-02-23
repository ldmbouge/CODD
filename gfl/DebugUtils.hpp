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

#ifdef __CUDACC__
    inline
    void checkCudaError(cudaError_t err, const char *file, const int line)
    {
        if (err != cudaSuccess)
        {
            const char * errorStr = NULL;
            errorStr = cudaGetErrorString(err);
            fprintf(stderr, "CUDA API error = %04d \"%s\" from %s:%i\n", err, errorStr, file, line);
            exit(EXIT_FAILURE);
        }
    }

    #define CHECK_CUDA_ERROR(err) gfl::checkCudaError(err, __FILE__, __LINE__)
    #define CHECK_LAST_CUDA_ERROR() CHECK_CUDA_ERROR(cudaGetLastError())
#endif

}
