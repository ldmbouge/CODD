#pragma once

#ifdef __CUDACC__
    #define GFL_HOST __host__
    #define GFL_DEVICE __device__
    #define GFL_HOST_DEVICE __host__ __device__
    #define GFL_GLOBAL __global__
#else
    #define GFL_HOST
    #define GFL_DEVICE
    #define GFL_HOST_DEVICE
    #define GFL_GLOBAL
#endif