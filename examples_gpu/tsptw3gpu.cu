#include "tsptw_model.hpp"
#include "codd.hpp"
#include "searchRelaxedFirst.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include "searchRestrictedOnlyNoQ.hpp"
#include "searchRestrictedFirstThreaded.hpp"
#include <StackAllocator.hpp>
#include <cxxopts.hpp>
#include <Malloc.hpp>

void parseFile(TSPTW * const model, std::string const & instance, gfl::StackAllocator & allocator)
{
    auto m = model;

    std::ifstream file(instance);
    if (not file.good())
    {
        std::cerr << "File does not exist or could not be opened: " << instance << std::endl;
        exit(EXIT_FAILURE);
    }
    file >> m->n;
    m->d = Matrix<int, 2>(m->n, m->n, allocator);
    for (auto i = 0; i < m->n; i++)
    {
        for (auto j = 0; j < m->n; j++)
        {
            file >> m->d[i][j];
        }
    }
    m->tw = FArray<TSPTW::TimeWindow>(m->n, allocator);
    for (auto i = 0; i < m->n; i++)
    {
        int a, b;
        file >> a >> b;
        m->tw[i] = TSPTW::TimeWindow(a, b);
    }
    file.close();

    m->dInNS = FArray<int>(m->n, allocator);
    m->dIn = FArray<int>(m->n, allocator);
    m->dOut = FArray<int>(m->n, allocator);
    m->permIn = FArray<int>(m->n, allocator);
    m->permOut = FArray<int>(m->n, allocator);

    auto allCities = TSPTW::Labels(0, m->n - 1);
    for (auto j : allCities)
    {
        auto [e1, minIn] = argmin(allCities - j, [m,j](int k) { return m->d[k][j]; });
        auto [e2, minOut] = argmin(allCities - j, [m,j](int k) { return m->d[j][k]; });
        m->dIn[j] = m->dInNS[j] = minIn;
        m->dOut[j] = minOut;
        m->permIn[j] = j;
        m->permOut[j] = j;
    }
    mergeSortPerm(m->dIn.data(), m->permIn.data(), m->n, [](double a, double b) { return a < b; });   // From smallest to largest
    mergeSortPerm(m->dOut.data(), m->permOut.data(), m->n, [](double a, double b) { return a < b; }); // From smallest to largest
}

constexpr auto static ReadOnlyMemSize{24 * 1024}; // Cached in shared memory

int main(int argc,char* argv[])
{
    // Select GPU
    cudaSetDevice(0);

    // Parse arguments
    int width = 0;
    int timeout = std::numeric_limits<int>::max(); // 68 years
    std::string strategy;
    std::string instance;
    cxxopts::Options options("tsptw3gpu", "A C++ solver for the TSPTW");
    options.add_options("Available")
        ("w,width"   , "Non-negative integer specifying the maximum width", cxxopts::value(width))
        ("s,strategy", "Search strategy to use among XF,RF,RO,RONQ", cxxopts::value(strategy))
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
    auto * const model = new (allocator) TSPTW();
    parseFile(model, instance, allocator);

    auto labels = TSPTW::Labels(0, model->n-1);
    auto dd = DD<TSPTW>::makeDD(model, labels);
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