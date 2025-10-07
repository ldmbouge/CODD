#include "tsptw_gpu_base.cuh"
#include "tsptw_model_3.hpp"

int main(int argc,char* argv[])
{
    return exec<TSPTW3>(argc, argv);
}