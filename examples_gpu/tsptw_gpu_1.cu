#include "tsptw_gpu_base.cuh"
#include "tsptw_model_1.hpp"

int main(int argc,char* argv[])
{
    return exec<TSPTW1>(argc, argv);
}