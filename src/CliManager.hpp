#pragma once

#include <cxxopts.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

class CliManager
{
public:
    int width;
    int timeout;
    bool use_gpu;
    long long int mem_size;
    std::string instance_path;

private:
    cxxopts::Options options_;

public:
    CliManager(std::string const& programName, std::string const& description);
    void parse(int argc, char* argv[]);

private:
    void validate();
};

inline
CliManager::CliManager(std::string const& programName, std::string const& description)
    : width(-1),
      timeout(std::numeric_limits<int>::max()),
      use_gpu(false),
      mem_size(-1),
      instance_path(),
      options_(programName, description)
{
    options_.add_options("Available")
        ("w,width", "DD width", cxxopts::value(width))
        ("h,help", "Show this help message and exit")
        ("m,memory", "Working memory in GB", cxxopts::value(mem_size))
        ("g,gpu", "Use GPU acceleration", cxxopts::value(use_gpu))
        ("i,instance", "Path to the instance file", cxxopts::value(instance_path))
        ("t,timeout", "Timeout in seconds", cxxopts::value(timeout));

    options_.parse_positional({"instance"});
    options_.custom_help("<OPTIONS>");
    options_.positional_help("<INSTANCE>");
}

inline
void CliManager::parse(int argc, char* argv[])
{
    auto const result = options_.parse(argc, argv);

    if (result.count("help") > 0)
    {
        std::cout << options_.help() << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    validate();
}

inline
void CliManager::validate()
{
    bool hasError = false;

    if (instance_path.empty())
    {
        std::cerr << "Error: Instance file is required" << std::endl;
        hasError = true;
    }
    else if (!std::filesystem::exists(instance_path))
    {
        std::cerr << "Error: Instance file does not exist: " << instance_path << std::endl;
        hasError = true;
    }
    else if (!std::filesystem::is_regular_file(instance_path))
    {
        std::cerr << "Error: Instance path is not a regular file: " << instance_path << std::endl;
        hasError = true;
    }
    else
    {
        std::ifstream f(instance_path);
        if (!f.is_open())
        {
            std::cerr << "Error: Instance file cannot be opened: " << instance_path << std::endl;
            hasError = true;
        }
    }

    if (width <= 0)
    {
        std::cerr << "Error: Width must be greater than 0" << std::endl;
        hasError = true;
    }

    if (timeout <= 0)
    {
        std::cerr << "Error: Timeout must be greater than 0" << std::endl;
        hasError = true;
    }

    if (mem_size <= 0)
    {
        std::cerr << "Error: Memory must be greater than 0" << std::endl;
        hasError = true;
    }

    if (hasError)
    {
        std::cerr << "\n" << options_.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}