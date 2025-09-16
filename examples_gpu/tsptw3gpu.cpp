#include "codd.hpp"
#include "search.hpp"
#include "searchRestrictedFirst.hpp"
#include "searchRestrictedOnly.hpp"
#include "searchRestrictedOnlyNoQ.hpp"
#include "searchRestrictedFirstThreaded.hpp"
#include <getopt.h>
#include <StackAllocator.hpp>
#include "tsptw_model.hpp"

void skip(std::ifstream& f) {
   char ch = 0;
   std::string comment;
   do {
      ch = f.get();
      if (ch == '#') {
         std::getline(f,comment);
         ch = ' ';
      }
   } while (isspace(ch) && !f.eof());
   if (!f.eof())
      f.unget();
}

void parseFile(TSPTW * const model, const char* fName, gfl::StackAllocator & allocator)
{
    auto m = model;

    using namespace std;
    ifstream f(fName);
    skip(f);
    f >> m->n;
    m->d = Matrix<int, 2>(m->n, m->n, allocator);
    skip(f);
    for (auto i = 0; i < m->n; i++)
    {
        for (auto j = 0; j < m->n; j++)
        {
            f >> m->d[i][j];
        }
    }
    skip(f);
    m->tw = FArray<TSPTW::TimeWindow>(m->n, allocator);
    for (auto i = 0; i < m->n; i++)
    {
        int a, b;
        f >> a >> b;
        m->tw[i] = TSPTW::TimeWindow(a, b);
    }
    f.close();
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
    mergeSortPerm(&m->dIn[0], &m->permIn[0], m->n, [](double a, double b) { return a < b; });   // From smallest to largest
    mergeSortPerm(&m->dOut[0], &m->permOut[0], m->n, [](double a, double b) { return a < b; }); // From smallest to largest
}

constexpr auto static ReadOnlyMemSize{24 * 1024}; // Cached in shared memory

int main(int argc,char* argv[])
{
    const char* fName = nullptr;
    int w = -1;
    std::string solver = "XF"; // Default value

    const option long_opts[] = {
        {"instance", required_argument, 0, 'i'},
        {"width", required_argument, 0, 'w'},
        {"solver", required_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:w:s:h", long_opts, nullptr)) != -1)
    {
        switch (opt)
        {
        case 'i':
            fName = optarg;
            break;
        case 'w':
            w = std::atoi(optarg);
            if (w < 0)
            {
                std::cerr << "Error: Width must be a non-negative integer.\n";
                return -1;
            }
            break;
        case 's':
            solver = optarg;
            break;
        case 'h':
            std::cout << "Usage: " << argv[0] <<
                " --instance <file> --width <int> [--solver <name>] [--nolocal] [--nodom]\n"
                << "  -i, --instance <file>   Path to the instance file (required)\n"
                << "  -w, --width <int>       Non-negative integer specifying the max width (required)\n"
                << "  -s, --solver <name>     Name of the solver to use (optional, default: XF)\n"
                << "  -h, --help              Show this help message and exit\n";
            return 0;
        case '?':
        default:
            return -1;
        }
    }

    if (fName == nullptr || w < 0)
    {
        std::cerr << "Error: --instance and --width are required.\n";
        return -1;
    }

    std::cout << "Instance file: " << fName << '\n';
    std::cout << "Width: " << w << '\n';
    std::cout << "Solver: " << solver << '\n';

    gfl::StackAllocator allocator(malloc(ReadOnlyMemSize), ReadOnlyMemSize);
    auto * const model = new (allocator) TSPTW();
    parseFile(model, fName, allocator);
    auto labels = TSPTW::Labels(0, model->n-1);
    auto dd = DD<TSPTW, Minimize<double>>::makeDD(model, labels);
    Bounds bnds([](const std::vector<int>& inc)  {});

    if (solver == "XF")
    {
        BAndB engine(dd,w);
        engine.search(bnds);
    }
    else if (solver == "RF")
    {
        BAndBRestrictedFirst engine(dd,w);
        engine.search(bnds);
    }
    else if (solver == "RFT")
    {
        BAndBRestrictedFirstThreaded engine(dd,w);
        engine.search(bnds);
    }
    else if (solver == "RO")
    {
        BAndBRestrictedOnly engine(dd,w);
        engine.search(bnds);
    }
    else if (solver == "RONQ")
    {
        BAndBRestrictedOnlyNoQ engine(dd,w);
        engine.search(bnds);
    }
    else
    {
        std::cerr << "The solver " << solver << " is unknown.\n"
            << "This model only supports the following solvers:\n"
            << "XF     relaXed First\n"
            << "RF     Restricted First\n"
            << "RFT    Restricted First Threaded\n"
            << "RO     Restricted Only\n"
            << "RONQ   Restricted Only No Queue (CABS-style)";
        return -1;
    }

    return 0;
}