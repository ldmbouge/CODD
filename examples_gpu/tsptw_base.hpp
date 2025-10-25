#pragma once

#include "util.hpp"
#include "msort.hpp"
#include <fstream>
#include <StackAllocator.hpp>
#include <iostream>
#include <limits>

struct TSPTWBase
{
    // Instance data
    struct TimeWindow
    {
        int a, b;

        TimeWindow() noexcept : a(std::numeric_limits<int>::max()), b(std::numeric_limits<int>::min()) {}

        TimeWindow(int const a, int const b) noexcept : a(a), b(b) {}

        friend std::ostream &operator<<(std::ostream &os, TimeWindow const &t)
        {
            return os << "[" << t.a << "," << t.b << "]";
        }

    };

    constexpr static int depot = 0;
    int n;
    Matrix<int, 2> d;
    FArray <TimeWindow> tw;
    FArray<int> dInNS;
    FArray<int> dIn;
    FArray<int> dOut;
    FArray<int> permIn;
    FArray<int> permOut;

    static inline
    void parseFile(TSPTWBase * const model, std::string const & instance, gfl::StackAllocator & allocator)
    {
        auto m = model;

        std::ifstream file(instance);
        if (not file.good())
        {
            std::cerr << "File does not exist or could not be opened: " << instance << std::endl;
            exit(EXIT_FAILURE);
        }
        file >> m->n;
        new (&m->d) Matrix<int, 2>(m->n, m->n, allocator); // The = operator gives issues whith destruction
        for (auto i = 0; i < m->n; i++)
        {
            for (auto j = 0; j < m->n; j++)
            {
                file >> m->d[i][j];
            }
        }
        new (&m->tw) FArray<TSPTWBase::TimeWindow>(m->n, allocator);
        for (auto i = 0; i < m->n; i++)
        {
            int a, b;
            file >> a >> b;
            m->tw[i] = TSPTWBase::TimeWindow(a, b);
        }
        file.close();

        new (&m->dInNS) FArray<int>(m->n, allocator);
        new (&m->dIn) FArray<int>(m->n, allocator);
        new (&m->dOut) FArray<int>(m->n, allocator);
        new (&m->permIn) FArray<int>(m->n, allocator);
        new (&m->permOut) FArray<int>(m->n, allocator);

        auto allCities = NatSet<4>(0,m->n-1);
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
};