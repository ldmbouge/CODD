#include "codd.hpp"
#include "searchRelaxedFirst.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include "searchRestrictedOnlyNoQ.hpp"
#include "searchRestrictedFirstThreaded.hpp"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>

constexpr auto static ReadOnlyMemSize{48 * 1024}; // Cached in shared memory

template<typename TSPTWModel>
int exec(int argc,char* argv[])
{
    // Select GPU
    cudaSetDevice(0);

    // Parse arguments
    int width = 0;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    long long int gpu_width = std::numeric_limits<long long int>::max();
    std::string strategy;
    std::string instance;
    cxxopts::Options options("tsptw3gpu", "A C++ solver for the TSPTW");
    options.add_options("Available")
        ("w,width"   , "Non-negative integer specifying the maximum width", cxxopts::value(width))
        ("s,strategy", "Search strategy to use among XF,RF,RO,RONQ", cxxopts::value(strategy))
        ("g,gpu-width", "Minimum width to offload computation to GPU", cxxopts::value(gpu_width))
        ("h,help"    , "Show this help message and exit")
        ("i,instance", "Path to the instance file", cxxopts::value(instance))
        ("t,timeout", "Timeout in seconds", cxxopts::value(timeout));
    options.parse_positional({"instance"});
    options.custom_help("<OPTIONS>");
    options.positional_help("<INSTANCE>");
    auto const result = options.parse(argc, argv);

    // Option validation
    if (width <= 0)
    {
        std::cerr << "Negative or zero width" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (instance.empty())
    {
        std::cerr << "Missing instance file" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (result.count("help") > 0)
    {
        std::cout << options.help();
        exit(EXIT_SUCCESS);
    }

    auto * const readOnlyMem =
#ifdef __CUDACC__
            gfl::mallocManaged<void>(ReadOnlyMemSize);
#else
            gfl::mallocStd<void>(ReadOnlyMemSize);
#endif

    gfl::StackAllocator allocator(readOnlyMem, ReadOnlyMemSize);
    auto * const model = new (allocator) TSPTWModel();
    TSPTWModel::parseFile(model, instance, allocator);

    auto labels = typename TSPTWModel::Labels(0, model->n-1);

#ifdef __CUDACC__
    auto dd = DD<TSPTWModel>::makeDD(model, labels ,gpu_width);
#else
    auto dd = DD<TSPTW>::makeDD(model, labels);
#endif

    Bounds bnds([](const std::vector<int>& inc)  {});
    BAndB * engine = nullptr;

    if (strategy == "XF")
    {
        engine = new BAndBRelaxedFirst(dd, width);
    }
    else if(strategy == "RF")
    {
        engine = new BAndBRestrictedFirst(dd, width);
    }
    else if(strategy == "RFT")
    {
        engine = new BAndBRestrictedFirstThreaded(dd, width);
    }
    else if(strategy == "RO")
    {
        engine = new BAndBRestrictedOnly(dd, width);
    }
    else if(strategy == "RONQ")
    {
        engine = new BAndBRestrictedOnlyNoQ(dd, width);
    }
    else
    {
        std::cerr << "Unknown search strategy. Supported strategies are:" << std::endl
                  << "XF     RelaXed First" << std::endl
                  << "RF     Restricted First" << std::endl
                  << "RFT    Restricted First Threaded" << std::endl
                  << "RO     Restricted Only" << std::endl
                  << "RONQ   Restricted Only No Queue (CABS style)";
        exit(EXIT_FAILURE);
    }

    std::cout << "Instance file: " << instance << std::endl;
    std::cout << "Width: " << width << std::endl;
    std::cout << "Strategy: " << strategy << std::endl;
    engine->setTimeLimit([timeout](auto ms){return ms / 1000 > timeout;});
    engine->search(bnds);

    return EXIT_SUCCESS;
}